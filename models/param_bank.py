from collections import Counter
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
import copy
from .pq_assignment import priority_queue_assignment


# Handle mixed precision compatibility.
if not hasattr(torch.cuda, 'amp') or not hasattr(torch.cuda.amp, 'autocast'):
    def autocast(enabled=False):
        return contextlib.nullcontext()
else:
    from torch.cuda.amp import autocast

class SharedParameters(nn.Module):
    def __init__(self, upsample_type, emb_size, num_params, init, init_blocks, params):
        super(SharedParameters, self).__init__()
        self._upsample_type = upsample_type
        self._emb_size = emb_size
        init2 = False
        if params is None:
            params = torch.Tensor(num_params)
            init2 = True
            init(params[:(num_params // 9) * 9].view(-1, 3, 3))

        if init2:
            self._params = nn.Parameter(params)
        else:
            self._params = params

        self._num_params = num_params

    def get_params(self, num_candidates, start, end):
        params = self._params[start:end]
        return params.view(num_candidates, -1)

    def __len__(self):
        return self._num_params

class ParameterCombiner(nn.Module):
    def __init__(self, learn_groups, num_candidates, layer_shape, num_params, upsample_type, upsample_window, coefficients, init, layer=None, trans=1):
        super(ParameterCombiner, self).__init__()
        self._learn_groups = learn_groups
        self._upsample_type = upsample_type
        self._upsample_window = upsample_window
        self._out_param_shape = layer_shape
        self.num_candidates = num_candidates
        self.layer = layer
        self.single_dims = list(np.ones(1, np.int32))
        assert isinstance(trans, int)
        self.trans = trans
        if self.trans > 1 and (self._out_param_shape[-1] != 3 or self._out_param_shape[-2] != 3):
            coefficients = coefficients[:num_candidates]
            self.trans = 1

        if coefficients is not None:
            coefficients = nn.Parameter(coefficients)

        self.coefficients = coefficients
        if coefficients is not None:
            self.mask = nn.Parameter(torch.ones(num_candidates), requires_grad=False)

        self._needs_resize = False


    def get_candidate_weights(self):
        coefficients = self.coefficients
        if self.layer is not None:
            coefficients = self.layer(coefficients.view(1, -1))

        if self._learn_groups:
            coefficients = F.softmax(coefficients.view(1, -1), 1)

        if self.trans > 1:
            return coefficients.view(self.num_candidates * self.trans, *self.single_dims)

        coefficients = coefficients.view(self.num_candidates, *self.single_dims)
        return coefficients * self.mask.view(self.num_candidates, *self.single_dims)

    def forward(self, x, upsample_layer=None):
        x = x.view(self.num_candidates, -1)

        if self.coefficients is not None:
            if self.trans > 1:
                flip = torch.rot90(x.view(-1, 3, 3), 3, [1, 2])
                if self.trans > 2:
                    flip = torch.cat((flip, torch.rot90(x.view(-1, 3, 3), 1, [1, 2]), torch.rot90(x.view(-1, 3, 3), 2, [1, 2])), 0)

                x = torch.cat((x, flip.view(self.num_candidates * (self.trans - 1), *self._out_param_shape)), 0)
            cand_weights = self.get_candidate_weights()
            cand_weights = cand_weights.to(x.get_device())
            x = (x * cand_weights).sum(0)

        return x


class ParameterGroups(nn.Module):
    def __init__(self, groups, share_type, bin_type, upsample_type, upsample_window, trans=1, max_params=0, max_candidates=2, param_group_bins=-1, separate_kernels=False, allocation_normalized=False, share_coeff=False, coeff_share_idxs=None, verbose=False):
        super(ParameterGroups, self).__init__()
        self._learn_groups = groups is None

        if self._learn_groups:
            num_groups = 1
            max_candidates = 4
            groups = 0
            assert share_type in ['wavg', 'emb']
        elif isinstance(groups, int):
            if groups <= 0:
                num_groups = 0
                self._layers = []
                self._upsample_type = upsample_type
                self._upsample_window = upsample_window
                self._share_type = share_type
                self._max_candidates = max_candidates
            else:
                num_groups = 1
        else:
            if isinstance(groups[0], list):
                num_groups = 0
                self._layers = []
                self._upsample_type = upsample_type
                self._upsample_window = upsample_window
                self._share_type = share_type
                self._max_candidates = max_candidates
            else:
                num_groups = len(set(groups))

        self._num_groups = num_groups
        self._group_ids = []
        self._max_params = max_params
        for group_id in range(num_groups):
            self.add_module('_param_group_' + str(group_id), ParameterBank(share_type, upsample_type, upsample_window, max_params, max_candidates, trans, self._learn_groups))

        self._groups = groups
        self._layer_count = 0
        self._max_width = 0
        self.param_group_bins = param_group_bins
        self.bin_type = bin_type
        self.separate_kernels = separate_kernels
        self.allocation_normalized = allocation_normalized
        self._share_coeff = share_coeff
        self._group_sizing = None
        self._coeff_share_idxs = coeff_share_idxs
        self._verbose = verbose

    def add_layer(self, layer):
        layer_id = self._layer_count
        group_id = self._groups
        self._layer_count += 1
        if isinstance(group_id, int):
            if not self._learn_groups and group_id <= 0:
                self._layers.append(layer)
                return None, group_id, layer_id

            group_id = 0
        elif isinstance(group_id[0], list):
            self._layers.append(layer)
            return None, group_id, layer_id
        elif not isinstance(group_id, int):
            group_id = group_id[layer_id]

        params = getattr(self, '_param_group_' + str(group_id))
        params.add_layer(layer, layer_id)
        return params, group_id, layer_id

    def get_params(self):
        out_params = []
        for width in range(self._max_width):
            for group_id in self._all_nets_group_ids:
                params = getattr(self, '_param_group_' + str(group_id) + '_width_' + str(width), None)
                if params is None:
                    out_params.append((None, None, None))
                    continue
                if params.single_layer:
                    out_params.append(params._layer)
                else:
                    if not hasattr(params, '_params'):
                        out_params.append((None, None, None))
                    else:
                        out_params.append((params._params._params, params._combs, (params._layer_group_idxs, params._coeff_share_idxs)))

        return out_params

    def setup_bank(self, shared_params, layer_shapes):
        # break up each network into groups and divvy up parameters according to layers in each group from all networks
        assert 'all_nets' in self.bin_type
        num_nets = len(layer_shapes[1])

        pretraining = False
        if self._coeff_share_idxs is not None:
            for share_idx in self._coeff_share_idxs.values():
                if share_idx['layer'] == 0: 
                    pretraining = True
                else: 
                    pretraining = False
                    break

        separated_layer_shapes = None
        if self.separate_kernels and isinstance(self._groups, int):
            # Move 1x1 to beginning, save index of 1x1 and force into separate groups below
            kernel_start_idxs = []
            separated_layer_shapes = []
            for i, net in enumerate(layer_shapes[1]):
                kernel_layers = {}
                for layer in net:
                    # Check size
                    kernel = ' '.join(str(k) for k in layer[2][2:])
                    if kernel not in kernel_layers:
                        kernel_layers[kernel] = []
                    kernel_layers[kernel].append(layer[1])

                separated_net_layer_shapes, net_kernel_start_idxs = [], []
                start_idx = 0
                for kernel in sorted(list(kernel_layers.keys())): 
                    net_kernel_start_idxs.append(start_idx)
                    separated_net_layer_shapes.extend(kernel_layers[kernel])
                    start_idx += len(kernel_layers[kernel])
                kernel_start_idxs.append(net_kernel_start_idxs)
                separated_layer_shapes.append(separated_net_layer_shapes)

            verbose_print('kernel_start_idxs {}'.format(kernel_start_idxs), self._verbose)

        depths = [len(net_layers) for net_layers in layer_shapes[1]]
        if isinstance(self._groups, int):
            layers_per_bin = self._get_layers_per_bin(depths, kernel_start_idxs)
            layers_to_bin = self._get_layers_to_bin(layers_per_bin, separated_layer_shapes=separated_layer_shapes)
            verbose_print('layers_per_bin {}'.format(layers_per_bin), self._verbose)
            verbose_print('layers_to_bin {}'.format(layers_to_bin), self._verbose)
        elif isinstance(self._groups[0], list):
            layers_to_bin = self._groups
            verbose_print('layers_to_bin {}'.format(layers_to_bin), self._verbose)

        all_nets_group_ids = []
        for net in layers_to_bin:
            all_nets_group_ids.extend(net)
        all_nets_group_ids = set(all_nets_group_ids)
        all_nets_groups = self._get_width_0_groups(layers_to_bin)

        # Assume depths are decreasing later
        for i in range(1,len(depths)): assert depths[i-1] >= depths[i]

        all_nets_groups_sizing = {}
        for net in list(all_nets_groups.keys()):
            all_nets_groups_sizing[net] = {'width_0': {}}

        if 'groupslim' in self.bin_type:
            all_nets_groups, all_nets_groups_sizing = self._create_stack_groups(num_nets, all_nets_groups,
                                                                                all_nets_groups_sizing,
                                                                                layer_shapes,
                                                                                all_nets_group_ids)
        else:
            all_nets_groups, all_nets_groups_sizing = self._create_slim_stack_groups(num_nets, all_nets_groups,
                                                                                    all_nets_groups_sizing,
                                                                                    layer_shapes,
                                                                                    all_nets_group_ids)
        groups = all_nets_groups['net_{}'.format(layer_shapes[0])]
        verbose_print("Groups: {}".format(groups), self._verbose)
        verbose_print("Group sizing: {}".format(all_nets_groups_sizing), self._verbose)

        layer2groups = self._init_groups(depths, groups, all_nets_groups_sizing, layer_shapes, pretraining)
        self._group_ids = list(groups['width_0'].keys())
        self._groups = layer2groups
        self._group_sizing = all_nets_groups_sizing
        self._all_nets_group_ids = sorted(list(all_nets_group_ids))

        if self._max_params > 0:
            # going to split up parameters based on the relative sizes
            # of the parameter groups
            max_num_groups = self._get_max_num_groups(all_nets_groups, all_nets_group_ids)
            group_sizes = np.zeros(max_num_groups)

            avail_params = self._max_params
            for group_id in self._all_nets_group_ids:
                for width in range(self._max_width):
                    # Max layer number of params in group across all nets
                    for net in all_nets_groups_sizing.keys():
                        num_params = 0
                        if 'width_{}'.format(width) in all_nets_groups_sizing[net] and group_id in all_nets_groups_sizing[net]['width_{}'.format(width)]:
                            for size in all_nets_groups_sizing[net]['width_{}'.format(width)][group_id]:
                                num_params = max(num_params, size[0])

                        curr_params = group_sizes[width * self._max_group_depth + group_id]
                        group_sizes[width * self._max_group_depth + group_id] = max(curr_params, num_params)

            total_params = np.sum(group_sizes)
            verbose_print('total_params {}'.format(total_params), self._verbose)
            assert total_params > 0
            if self.allocation_normalized:
                group_sizes = np.round((group_sizes / total_params) * avail_params).astype(np.int32)
            else:
                group_sizes = self._allocate_templates(group_sizes, all_nets_groups_sizing, all_nets_groups, all_nets_group_ids, avail_params, total_params, reverse_order=False, cap=8, min=0)

            for group_id, max_params in enumerate(group_sizes):
                if max_params == 0: continue
                width = group_id // self._max_group_depth
                group = group_id % self._max_group_depth
                if 'width_{}'.format(width) not in groups or group not in groups['width_{}'.format(width)]: continue
                param_bank = getattr(self, '_param_group_' + str(group) + '_width_' + str(width))

                if True:
                    assert max_params > 0
                    param_bank.set_num_params(max_params)

            verbose_print("Group Sizes: {}".format(group_sizes), self._verbose)
            verbose_print('Max width: {}'.format(self._max_width), self._verbose)

        setup = set()
        for group_id in self._group_ids:
            # Check if group has any layers for this network
            for layer_id, layer in enumerate(layer2groups):
                if str(group_id) + '_width_0' not in layer: continue

                for group_width in layer:
                    if group_width in setup: continue
                    else: setup.add(group_width)

                    width = int(group_width.split('_')[-1])

                    params = getattr(self, '_param_group_' + group_width)
                    local_params, coeffs, share_idxs = None, {}, None
                    if shared_params is not None:
                        if self._coeff_share_idxs is not None:
                            shared_net = self._coeff_share_idxs[layer_id]['net']
                            local_params, coeffs, _ = shared_params[shared_net][width * self._max_group_depth + group_id]

                            # If shared net doesn't have layers at this group_width then check other nets that are already setup
                            if local_params is None:
                                verbose_print('No teacher params for group {}, net {}, teacher {}'.format(group_width, layer_shapes[0], len(shared_params), self._coeff_share_idxs[layer_id]['net']), self._verbose)
                                for net_params in shared_params[1:layer_shapes[0]]:
                                    local_params, coeffs, share_idxs = net_params[width * self._max_group_depth + group_id]
                                    if local_params is not None: 
                                        break

                        else:
                            # Grab coefficients of largest net sharing
                            for net_params in shared_params:
                                local_params, coeffs, _ = net_params[width * self._max_group_depth + group_id]
                                if local_params is not None:
                                    break

                        if not self._share_coeff:
                            coeffs = {}

                    verbose_print('Group id: {}, Width {}'.format(group_width, width), self._verbose)
                    params.setup_bank(local_params, coeffs, share_idxs)

        if isinstance(self._groups, int):
            self._groups = np.zeros(self._layer_count, np.int32)


    def _get_layers_per_bin(self, depths, kernel_start_idxs=None):
        if 'depth' in self.bin_type:
            if self.separate_kernels:
                layers_per_bin, kernel_bins = [], []
                for i, depth in enumerate(depths):
                    net_lpb = []
                    for j, start_idx in enumerate(kernel_start_idxs[i]):
                        end_idx = depth - 1 if len(kernel_start_idxs[i]) == j + 1 else kernel_start_idxs[i][j+1] - 1
                        num_layers = end_idx + 1 - start_idx
                        # number of bins is based on largest net
                        if i == 0:
                            kernel_bins.append(round(self.param_group_bins * num_layers / depth))
                            if kernel_bins[j] == 0:
                                kernel_bins[j] = 1
                        if num_layers >= kernel_bins[j]:
                            kernel_lpb = [num_layers // kernel_bins[j]] * kernel_bins[j]
                            # If not evenly divisible then add one layer to each bin starting with the first
                            excess = num_layers % kernel_bins[j]
                            for k in range(excess): kernel_lpb[k] += 1
                        else:
                            kernel_lpb = ([1] * num_layers) + ([0] * (kernel_bins[j] - num_layers))

                        net_lpb.extend(kernel_lpb)
                    layers_per_bin.append(net_lpb)
            else:
                layers_per_bin = []
                for depth in depths:
                    net_lpb = [depth // self.param_group_bins] * self.param_group_bins
                    # If not evenly divisible then add one layer to each bin starting with the first
                    excess = depth % self.param_group_bins
                    for i in range(excess): net_lpb[i] += 1
                    layers_per_bin.append(net_lpb)

        elif 'lpb' in self.bin_type:
            if self.separate_kernels:
                layers_per_bin = []
                for i, depth in enumerate(depths):
                    net_lpb = []
                    for j, start_idx in enumerate(kernel_start_idxs[i]):
                        end_idx = depth - 1 if len(kernel_start_idxs[i]) == j + 1 else kernel_start_idxs[i][j+1] - 1
                        num_layers = end_idx + 1 - start_idx
                        kernel_lpb = [min(num_layers,self.param_group_bins)] * (num_layers // self.param_group_bins)

                        # If not evenly divisible then add final group with excess layers
                        excess = num_layers % self.param_group_bins
                        if excess > 0: kernel_lpb.append(excess)
                        net_lpb.extend(kernel_lpb)
                    layers_per_bin.append(net_lpb)

            else:
                layers_per_bin = []
                for depth in depths:
                    net_lpb = [self.param_group_bins] * (depth // self.param_group_bins)
                    # If not evenly divisible then add final group with excess layers
                    excess = depth % self.param_group_bins
                    if excess > 0: net_lpb.append(excess)
                    layers_per_bin.append(net_lpb)

        return layers_per_bin

    def _get_layers_to_bin(self, layers_per_bin, separated_layer_shapes=None):
        if self.separate_kernels:
            layers_to_bin = []
            for net, net_lpb in enumerate(layers_per_bin):
                layers_to_bin.append([0] * len(separated_layer_shapes[net]))
                separated_layer_idx = 0
                for i, bin_size in enumerate(net_lpb):
                    for layer in range(bin_size):
                        # Map back to original layer index
                        layers_to_bin[-1][separated_layer_shapes[net][separated_layer_idx]] = i
                        separated_layer_idx += 1
        else:
            layers_to_bin = []
            for net_lpb in layers_per_bin:
                layers_to_bin.append([])
                for i, bin in enumerate(net_lpb):
                    for _ in range(bin):
                        layers_to_bin[-1].append(i)

        return layers_to_bin

    def _get_width_0_groups(self, layers_to_bin):
        all_nets_groups = {}
        for i, net_layers_to_bin in enumerate(layers_to_bin):
            all_nets_groups['net_{}'.format(i)] = {'width_0': {}}
            for j, bin in enumerate(net_layers_to_bin):
                if bin not in all_nets_groups['net_{}'.format(i)]['width_0']:
                    all_nets_groups['net_{}'.format(i)]['width_0'][bin] = []
                all_nets_groups['net_{}'.format(i)]['width_0'][bin].append(j)

        return all_nets_groups

    def _create_slim_stack_groups(self, num_nets, all_nets_groups, all_nets_groups_sizing, layer_shapes, all_nets_group_ids):
        # For each group, remove excess channels from larger layers and add them to separate group
        for layer_group in all_nets_groups['net_0']['width_0'].keys():

            for layer_idx_in_group in range(len(all_nets_groups['net_0']['width_0'][layer_group])):

                group_layer_ids = []
                for _, net in all_nets_groups.items():
                    if layer_group not in net['width_0']: continue
                    layer_id = net['width_0'][layer_group][layer_idx_in_group]
                    if layer_id not in group_layer_ids: group_layer_ids.append(layer_id)

                group_layer_ids = sorted(group_layer_ids)
                sizes, sizes_layer_net_ids = self._get_all_group_sizes_and_ids(num_nets, all_nets_groups, layer_group, layer_shapes, group_layer_ids, groupslim=False, layer_idx_in_group=layer_idx_in_group)
                sorted_size_idxs = np.argsort(sizes).tolist()

                for layer_id in group_layer_ids:
                    for net in range(num_nets):

                        if layer_group not in all_nets_groups['net_{}'.format(net)]['width_0'] or \
                            layer_id != all_nets_groups['net_{}'.format(net)]['width_0'][layer_group][layer_idx_in_group]: continue

                        if layer_group not in all_nets_groups_sizing['net_{}'.format(net)]['width_0']:
                            all_nets_groups_sizing['net_{}'.format(net)]['width_0'][layer_group] = []
                        # assume layers have been added to groups in order throughout - need to align all_nets_groups_sizing and layer in all_nets_groups
                        assert len(all_nets_groups_sizing['net_{}'.format(net)]['width_0'][layer_group]) == layer_idx_in_group

                        # Sizing for width 0
                        # Layer shape at width 0 is the smallest width at this layer
                        smallest_layer_net = sizes_layer_net_ids[sorted_size_idxs[0]]
                        shape = layer_shapes[1][smallest_layer_net['net']][smallest_layer_net['layer']][2]
                        all_nets_groups_sizing['net_{}'.format(net)]['width_0'][layer_group].append([np.prod(shape), [shape]])

                        # If all sizes are the same then continue
                        if np.all(np.array(sizes) == sizes[0]): continue

                        # Loop through all widths up to this layer and add excess width to excess width groups
                        w = 1
                        for width in range(1,sorted_size_idxs.index(sizes_layer_net_ids.index({'net':net, 'layer':layer_id}))+1):
                            shape_layer_net = sizes_layer_net_ids[sorted_size_idxs[width]]
                            shape = layer_shapes[1][shape_layer_net['net']][shape_layer_net['layer']][2]

                            prev_layer_net = sizes_layer_net_ids[sorted_size_idxs[width-1]]

                            if layer_shapes[1][shape_layer_net['net']][shape_layer_net['layer']][2] == layer_shapes[1][prev_layer_net['net']][prev_layer_net['layer']][2]:
                                continue

                            if 'width_{}'.format(w) not in all_nets_groups_sizing['net_{}'.format(net)]:
                                all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(w)] = {}
                                all_nets_groups['net_{}'.format(net)]['width_{}'.format(w)] = {}

                            if layer_group not in all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(w)]:
                                all_nets_groups['net_{}'.format(net)]['width_{}'.format(w)][layer_group] = []
                                all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(w)][layer_group] = []
                            # Again assume layers have been added to groups in order throughout
                            assert len(all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(w)][layer_group]) <= layer_idx_in_group
                            assert len(all_nets_groups['net_{}'.format(net)]['width_{}'.format(w)][layer_group]) <= layer_idx_in_group

                            # Previous layers in group don't exist at width so add them with sizing 0
                            if len(all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(w)][layer_group]) < layer_idx_in_group:
                                while len(all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(w)][layer_group]) < layer_idx_in_group:
                                    all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(w)][layer_group].append(torch.Size([0,0,0,0]))
                                all_nets_groups['net_{}'.format(net)]['width_{}'.format(w)][layer_group] = all_nets_groups['net_{}'.format(net)]['width_0'][layer_group][:layer_idx_in_group]

                            # Subtract the previous widths from this width
                            prev_params, prev_in_ch, prev_out_ch = 0, 0, 0
                            for prev_width in range(w):
                                prev_params += all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(prev_width)][layer_group][layer_idx_in_group][0]
                                prev_shapes = all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(prev_width)][layer_group][layer_idx_in_group][1]
                                prev_out_ch += prev_shapes[0][0]
                                prev_in_ch = prev_shapes[0][1]

                            new_params = np.prod(shape) - prev_params
                            new_shapes = [torch.Size([shape[0] - prev_out_ch, shape[1], *shape[2:]])]
                            new_shape_params = np.prod(new_shapes[0])
                            if shape[1] > prev_in_ch:
                                prev_out_ch = min(prev_out_ch, shape[0])
                                new_shapes.append(torch.Size([prev_out_ch, shape[1] - prev_in_ch, *shape[2:]]))
                                new_shape_params = max(0, new_shape_params)
                                new_shape_params += np.prod(new_shapes[1])

                            all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(w)][layer_group].append([new_shape_params, new_shapes])
                            all_nets_groups['net_{}'.format(net)]['width_{}'.format(w)][layer_group].append(layer_id)
                            w += 1

        return all_nets_groups, all_nets_groups_sizing

    def _create_stack_groups(self, num_nets, all_nets_groups, all_nets_groups_sizing, layer_shapes, all_nets_group_ids):
        # For each group, remove excess channels from larger layers and add them to separate group
        for layer_group in all_nets_group_ids:

            group_layer_ids = []
            for _, net in all_nets_groups.items():
                if layer_group not in net['width_0']: continue
                for layer_id in net['width_0'][layer_group]:
                    if layer_id not in group_layer_ids: group_layer_ids.append(layer_id)

            group_layer_ids = sorted(group_layer_ids)
            sizes, sizes_layer_net_ids = self._get_all_group_sizes_and_ids(num_nets, all_nets_groups, layer_group, layer_shapes, group_layer_ids)
            sorted_size_idxs = np.argsort(sizes).tolist()

            for layer_id in group_layer_ids:
                for net in range(num_nets):

                    if layer_group not in all_nets_groups['net_{}'.format(net)]['width_0'] or \
                        layer_id not in all_nets_groups['net_{}'.format(net)]['width_0'][layer_group]: continue

                    layer_idx_in_group = all_nets_groups['net_{}'.format(net)]['width_0'][layer_group].index(layer_id)

                    if layer_group not in all_nets_groups_sizing['net_{}'.format(net)]['width_0']:
                        all_nets_groups_sizing['net_{}'.format(net)]['width_0'][layer_group] = []
                    # assume layers have been added to groups in order throughout - need to align all_nets_groups_sizing and layer in all_nets_groups
                    assert len(all_nets_groups_sizing['net_{}'.format(net)]['width_0'][layer_group]) == layer_idx_in_group

                    # Sizing for width 0
                    # Layer shape at width 0 is smallest width in group
                    smallest_layer_net = sizes_layer_net_ids[sorted_size_idxs[0]]
                    shape = layer_shapes[1][smallest_layer_net['net']][smallest_layer_net['layer']][2]

                    all_nets_groups_sizing['net_{}'.format(net)]['width_0'][layer_group].append([np.prod(shape), [shape]])

                    # If all sizes are the same then continue
                    if np.all(np.array(sizes) == sizes[0]): continue

                    # Loop through all widths up to this layer and add excess width to excess width groups
                    w = 1
                    for width in range(1,sorted_size_idxs.index(sizes_layer_net_ids.index({'net':net, 'layer':layer_id}))+1):
                        shape_layer_net = sizes_layer_net_ids[sorted_size_idxs[width]]
                        shape = layer_shapes[1][shape_layer_net['net']][shape_layer_net['layer']][2]

                        prev_layer_net = sizes_layer_net_ids[sorted_size_idxs[width-1]]

                        # If this shape is the same as previous width then skip
                        if layer_shapes[1][shape_layer_net['net']][shape_layer_net['layer']][2] == layer_shapes[1][prev_layer_net['net']][prev_layer_net['layer']][2]:
                            continue

                        if 'width_{}'.format(w) not in all_nets_groups_sizing['net_{}'.format(net)]:
                            all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(w)] = {}
                            all_nets_groups['net_{}'.format(net)]['width_{}'.format(w)] = {}

                        if layer_group not in all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(w)]:
                            all_nets_groups['net_{}'.format(net)]['width_{}'.format(w)][layer_group] = []
                            all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(w)][layer_group] = []
                        # Again assume layers have been added to groups in order throughout
                        assert len(all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(w)][layer_group]) <= layer_idx_in_group
                        assert len(all_nets_groups['net_{}'.format(net)]['width_{}'.format(w)][layer_group]) <= layer_idx_in_group

                        # Previous layers in group don't exist at width so add them with sizing 0
                        if len(all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(w)][layer_group]) < layer_idx_in_group:
                            while len(all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(w)][layer_group]) < layer_idx_in_group:
                                all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(w)][layer_group].append(torch.Size([0,0]))
                            all_nets_groups['net_{}'.format(net)]['width_{}'.format(w)][layer_group] = all_nets_groups['net_{}'.format(net)]['width_0'][layer_group][:layer_idx_in_group]

                        # Subtract the previous widths from this width
                        prev_params, prev_in_ch, prev_out_ch = 0, 0, 0
                        for prev_width in range(w):
                            prev_params += all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(prev_width)][layer_group][layer_idx_in_group][0]
                            prev_shapes = all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(prev_width)][layer_group][layer_idx_in_group][1]
                            prev_out_ch += prev_shapes[0][0]
                            prev_in_ch = prev_shapes[0][1]

                        new_params = np.prod(shape) - prev_params
                        new_shapes = [torch.Size([shape[0] - prev_out_ch, shape[1], *shape[2:]])]
                        new_shape_params = np.prod(new_shapes[0])
                        if shape[1] > prev_in_ch:
                            prev_out_ch = min(prev_out_ch, shape[0])
                            new_shapes.append(torch.Size([prev_out_ch, shape[1] - prev_in_ch, *shape[2:]]))
                            new_shape_params = max(0, new_shape_params)
                            new_shape_params += np.prod(new_shapes[1])

                        all_nets_groups_sizing['net_{}'.format(net)]['width_{}'.format(w)][layer_group].append([new_shape_params, new_shapes])
                        all_nets_groups['net_{}'.format(net)]['width_{}'.format(w)][layer_group].append(layer_id)
                        w += 1

        return all_nets_groups, all_nets_groups_sizing

    def _get_all_group_sizes_and_ids(self, num_nets, all_nets_groups, layer_group, layer_shapes, group_layer_ids, groupslim=True, layer_idx_in_group=None):
        assert groupslim or layer_idx_in_group is not None
        sizes, sizes_layer_net_ids = [], []
        for layer_id in group_layer_ids:
            for net in range(num_nets):
                if layer_group not in all_nets_groups['net_{}'.format(net)]['width_0']: continue
                if groupslim and layer_id not in all_nets_groups['net_{}'.format(net)]['width_0'][layer_group]: continue
                if not groupslim and layer_id != all_nets_groups['net_{}'.format(net)]['width_0'][layer_group][layer_idx_in_group]: continue

                if len(layer_shapes[1][net]) > layer_id:
                    sizes.append(np.prod(layer_shapes[1][net][layer_id][2]))
                    sizes_layer_net_ids.append({'net': net, 'layer': layer_id})
        return sizes, sizes_layer_net_ids

    def _init_groups(self, depths, groups, all_nets_groups_sizing, layer_shapes, pretraining):
        layer2groups = [[] for _ in range(depths[layer_shapes[0]])]
        learn_groups = False
        for group_id, _ in groups['width_0'].items():
            # loop through widths, if groups[width] has that group_id then create it
            for width in groups.keys():
                if group_id not in groups[width]: continue
                if not hasattr(self, '_param_group_' + str(group_id) + '_' + width):
                    share_type = self._share_type
                    self.add_module('_param_group_' + str(group_id) + '_' + width, 
                                    ParameterBank(share_type, self._upsample_type, self._upsample_window, 
                                                self._max_params, self._max_candidates, learn_groups, width=int(width.split('_')[-1]), group=str(group_id) + '_' + width, verbose=False, pretraining=pretraining)) #self._verbose))

                params = getattr(self, '_param_group_' + str(group_id) + '_' + width)
                group = groups[width][group_id]
                prev_layer_group_idx = -1
                for layer_group_idx, layer_id in enumerate(group):
                    if all_nets_groups_sizing['net_{}'.format(layer_shapes[0])][width][group_id][layer_group_idx] == torch.Size([0,0]):
                        continue

                    layer2groups[layer_id].append(str(group_id) + '_' + width)

                    # now let's set everything for each layer
                    # that is normally set elsewhere
                    layer = self._layers[layer_id]
                    layer.banks.append(params)
                    layer.group_id = group_id
                    layer_group_size = all_nets_groups_sizing['net_{}'.format(layer_shapes[0])][width][group_id][layer_group_idx]
                    layer.group_size.append(layer_group_size)
                    # set this on the layer
                    if self._coeff_share_idxs is not None:
                        coeff_share_idx = self._coeff_share_idxs[layer_id]['layer']
                    else:
                        coeff_share_idx = layer_group_idx
                    params.add_layer(layer, layer_id, layer_group_idx, coeff_share_idx)

        return layer2groups

    def _get_max_num_groups(self, all_nets_groups, all_nets_group_ids):
        self._max_group_depth = (np.max(list(all_nets_group_ids)) + 1)
        self._max_width = 0
        for net in all_nets_groups.keys():
            for width in all_nets_groups[net]:
                self._max_width = max(self._max_width, int(width.split('_')[-1]) + 1)
        return self._max_group_depth * self._max_width

    def _allocate_templates(self, group_sizes, all_nets_groups_sizing, all_nets_groups, all_nets_group_ids, avail_params, total_params, reverse_order=False, cap=8, min=2):
        # add templates to groups with most layers first where the number of templates is the max number of layers in that group
        group_sizes = np.round(group_sizes).astype(np.int32)
        excess_params = avail_params - total_params
        verbose_print('excess_params: {}'.format(excess_params), self._verbose)
        cands = 1
        while cands < cap and excess_params > 0:
            for group_id in sorted(all_nets_group_ids, reverse=reverse_order):
                for width in range(self._max_width):
                    # Max number of params in group across all nets
                    num_layers = 0
                    num_params = 0
                    for net in all_nets_groups_sizing.keys():
                        num_net_layers = 0
                        if 'width_{}'.format(width) in all_nets_groups_sizing[net] and group_id in all_nets_groups_sizing[net]['width_{}'.format(width)]:
                            for size in all_nets_groups_sizing[net]['width_{}'.format(width)][group_id]:
                                if size[0] > 0:
                                    num_net_layers += 1
                                num_params = max(num_params, size[0])

                            num_layers += num_net_layers

                    if max(min,num_layers) > cands and excess_params - num_params > 0:
                        curr_params = group_sizes[width * self._max_group_depth + group_id]
                        group_sizes[width * self._max_group_depth + group_id] = curr_params + int(num_params)
                        excess_params -= num_params
            cands += 1
        verbose_print('excess_params: {}'.format(excess_params), self._verbose)

        # If we still have remaining params then round robin templates with a cap
        if excess_params > 0:
            cands = 1
            while cands < cap and excess_params > 0:
                for group_id in sorted(all_nets_group_ids, reverse=reverse_order):
                    for width in range(self._max_width):
                        # Max number of params in group across all nets
                        num_params = 0
                        for net in all_nets_groups_sizing.keys():
                            if 'width_{}'.format(width) in all_nets_groups_sizing[net] and group_id in all_nets_groups_sizing[net]['width_{}'.format(width)]:
                                for size in all_nets_groups_sizing[net]['width_{}'.format(width)][group_id]:
                                    num_params = max(num_params, size[0])

                        if excess_params - num_params > 0 and num_params > 0:
                            curr_params = group_sizes[width * self._max_group_depth + group_id]
                            num_templates = curr_params / int(num_params)
                            if num_templates < cap:
                                group_sizes[width * self._max_group_depth + group_id] = curr_params + int(num_params)
                                excess_params -= num_params
                cands += 1
        verbose_print('excess_params: {}'.format(excess_params), self._verbose)

        return group_sizes

# Coefficient Analysis
    def get_grads(self, prev_grads=None, explore=False):
        grads = {}
        for width in range(self._max_width):
            for group_id in sorted(self._group_ids):
                group_str = str(group_id) + '_width_' + str(width)
                params = getattr(self, '_param_group_' + group_str, None)
                if params is None or not hasattr(params, '_params'):
                    continue
                if params.single_layer: verbose_print('single_layer {}, {}'.format(group_id, width), self._verbose)

                group_grads = []
                for layer in range(len(params._combs)):
                    if prev_grads is not None:
                        assert params._group == prev_grads[group_str][layer][0]._group

                        new_grad = params.get_comb(layer).coefficients.grad.clone().detach()
                        grad = new_grad + prev_grads[group_str][layer][1]
                        group_grads.append((params, grad))
                    else:
                        if explore:
                            print('---- group {}, width {}, layer {} - {} {} {}'.format(group_id, width, layer, params._layer_group_idxs, params._coeff_share_idxs, params.get_comb(layer).coefficients))
                        grad = params.get_comb(layer).coefficients.grad.clone().detach()
                        group_grads.append((params, grad))
                grads[group_str] = group_grads
        return grads


    def compare_grads(self, grads, threshold=0.5, num_iters=1):
        updated_params = []
        for width in range(self._max_width):
            for group_id in sorted(self._all_nets_group_ids):
                
                next_group = 0
                coeff_groups = {} # {i: new_group}
                queue = [] # [(sim_ij, i, j)]
                checked = set() # {(i,j)}

                group_str = str(group_id) + '_width_' + str(width)
                for net in range(len(grads)):
                    if group_str not in grads[net]: continue

                    for layer in range(len(grads[net][group_str])):

                        i = 'net_{}-layer_{}'.format(net, layer)
                        coeff_groups[i] = next_group

                        for comp_net in range(len(grads)):

                            if grads[comp_net] is None or group_str not in grads[comp_net]: continue

                            comp_layer = None
                            for comp_layer in range(len(grads[comp_net][group_str])):
                                j = 'net_{}-layer_{}'.format(comp_net, comp_layer)

                                if (i,j) in checked or (j,i) in checked: continue
                                if comp_layer == layer and net == comp_net: continue
                                if not grads[net][group_str][layer][0].get_comb(layer).coefficients is grads[comp_net][group_str][comp_layer][0].get_comb(comp_layer).coefficients: continue

                                sim = F.cosine_similarity(grads[net][group_str][layer][1], grads[comp_net][group_str][comp_layer][1], dim=0)

                                queue.append((sim, i, j))
                                checked.add((i,j))

                        next_group += 1
                
                if len(checked) == 0: continue

                queue.sort(key=lambda x: x[0])
                groups = priority_queue_assignment(threshold, coeff_groups, queue)

                # For each group make copy and pass in to each coeff
                group_2_layer = {}
                for i, group in groups.items():
                    if group not in group_2_layer:
                        group_2_layer[group] = []
                    group_2_layer[group].append(i)

                for i, (group, layers) in enumerate(group_2_layer.items()):
                    # Just keep the original
                    if i == 0: continue
                    
                    # Make copy
                    net = int(layers[0].split('-')[0].split('_')[1])
                    layer = int(layers[0].split('-')[1].split('_')[1])

                    curr_comb = grads[net][group_str][layer][0].get_comb(layer)
                    coeff = curr_comb.coefficients.clone()

                    # Assign copy to each layer
                    for netlayer in layers:
                        net = int(netlayer.split('-')[0].split('_')[1])
                        layer = int(netlayer.split('-')[1].split('_')[1])
                        grads[net][group_str][layer][0].update_layer_coeff(layer, curr_comb=curr_comb, coeff=coeff)
                    
                    updated_params.append(grads[net][group_str][layer][0].get_comb(layer).coefficients)
                
        return updated_params

    def forward(self, layer_id, layer_shape, coeff=None):
        layer_params = []
        for group_width in self._groups[layer_id]:
            params = getattr(self, '_param_group_' + str(group_width))
            layer_params.append(params(layer_id, layer_shape, coeff))
        return layer_params


class ParameterBank(nn.Module):
    def __init__(self, share_type, upsample_type, upsample_window, max_params=0, max_candidates=2, trans=1, learn_groups=False, init=nn.init.kaiming_normal_, emb_size=24, width=0, group=None, pretraining=False, verbose=False):
        super(ParameterBank, self).__init__()
        self._share_type = share_type
        self._upsample_type = upsample_type
        self._upsample_window = upsample_window
        self._trans = trans

        if max_params > 0:
            # later on we assume we have an even number
            # of params so enfource we get an even number
            remainder = max_params % 9
            if remainder:
                max_params += (9 - remainder)

        self._max_params = max_params
        self._max_candidates = max_candidates
        self._learn_groups = learn_groups
        self._emb_size = emb_size
        self._layers = []
        self._layerid2ind = {}
        self._layer_parms = []
        self._init = init
        self._layer_param_blocks = []
        self.single_layer = False
        self.width = width
        self._combs = []
        self._coeff_share_idxs = {}
        self._layer_group_idxs = {}
        self._group = group
        self._verbose = verbose
        self._pretraining = pretraining
        self._is_teacher = False

    def get_comb(self, i):
        if self._pretraining and self._is_teacher: idx = self._coeff_share_idxs[i]
        else: idx = self._layer_group_idxs[i]
        return self._combs[idx]

    def add_layer(self, layer, layer_id=None, layer_group_idx=None, coeff_share_idx=None):
        if layer_id is None:
            layer_id = len(self._layers)

        if layer_group_idx is not None:
            self._coeff_share_idxs[len(self._layers)] = coeff_share_idx
            self._layer_group_idxs[len(self._layers)] = layer_group_idx

        self._layerid2ind[layer_id] = len(self._layers)
        self._layers.append(layer)
        return layer_id

    def set_num_params(self, max_params):
        # this should be called before running setup_bank
        assert not hasattr(self, '_params')
        self._max_params = max_params

    def get_num_params(self):
        if hasattr(self, '_params'):
            return len(self._params)

        num_candidates = np.ones(len(self._layers), np.int64)
        params_by_layer, _ = self.params_per_layer(num_candidates)
        return np.sum(params_by_layer)

    def get_num_candidates(self, num_params=None):
        if num_params is None:
            num_params = self.get_num_params()

        num_candidates = np.zeros(len(self._layers), np.int64)
        for i, layer in enumerate(self._layers):
            num_candidates[i] = min(max(1, num_params // layer.get_group_size(self.width)[0]), self._max_candidates)

        verbose_print("num_candidates: {}".format(num_candidates), self._verbose)
        return num_candidates

    def params_per_layer(self, num_candidates=None):
        if num_candidates is None:
            num_candidates = self.get_num_candidates()

        params_by_layer = np.zeros(len(num_candidates), np.int64)
        for i, candidates, layer in zip(range(len(params_by_layer)), num_candidates, self._layers):
            params_by_layer[i] = candidates * layer.get_group_size(self.width)[0]

        return params_by_layer, num_candidates

    def setup_bank(self, shared_params, combs={}, share_idxs=None):
        self.single_layer = False#len(self._layers) == 1
        self._combs = combs if combs is not None else {}

        if share_idxs is not None:
            # if layers were shared with teacher but the teacher doesn't have layers at this width (none large enough),
                # but another depth net does, then any layers that were shared with teacher by both nets should be shared.
                # this can only happen with the 2 widest nets at different depths than each other and the teacher
            net_b_layer_group_idxs = share_idxs[0]
            net_b_coeff_share_idxs = share_idxs[1]
            new_coeff_share_idxs = {}
            # need to map self_coeff_share_idxs to new idxs via net_b_share_idxs
            # if any self._coeff_share_idxs[index_a] == net_b_coeff_share_idxs[index_b] for all a,b:
            #       change self._coeff_share_idxs[index_a] to net_b_layer_group_idxs[index_b]
            for index_a, value_a in self._coeff_share_idxs.items():
                new_value = False
                for index_b, value_b in net_b_coeff_share_idxs.items():
                    if value_a == value_b:
                        new_coeff_share_idxs[index_a] = net_b_layer_group_idxs[index_b]
                        new_value = True
                        break
                # if no layers share on other net then leave as is
                if not new_value:
                    new_coeff_share_idxs[index_a] = value_a

            self._coeff_share_idxs = new_coeff_share_idxs

        if self.single_layer:
            if shared_params is not None:
                self.add_module('_layer', shared_params)
                return

            for layer in self._layers:
                if isinstance(layer, SConv2d):
                    self.add_module('_layer', nn.Conv2d(layer.in_channels, layer.out_channels, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding))
                elif isinstance(layer, SLinear):
                    self.add_module('_layer',  nn.Linear(layer.in_channels, layer.out_channels))
                else:
                    raise('single shared layer not implemented')

            return

        if shared_params is not None:
            self._max_params = len(shared_params)

        if self._learn_groups:
            num_params, sharing_blocks = self.even_param_assignments()
        else:
            num_params, sharing_blocks = self.assign_layers_to_params()

        self.sharing_blocks = sharing_blocks
        assert self._max_params == num_params or shared_params is None

        sp = SharedParameters(self._upsample_type, self._emb_size, num_params, self._init, sharing_blocks, shared_params)

        self.add_module('_params', sp)

    def even_param_assignments(self):
        num_params = self._max_params
        assert num_params > 0
        set_candidate_number = 4
        num_candidates = np.ones(len(self._layers), np.int64) * set_candidate_number
        start_end_sets = {}
        global_start = 0
        layer_shapes = [str(l.get_group_size(self.width)[0]) for l in self._layers]
        most_common_shape = [int(d) for d in Counter(layer_shapes).most_common(1)[0][0].split()]
        self._layer_param_blocks = [None for _ in range(len(self._layers))]
        for shape, _ in Counter(layer_shapes).most_common():
            layer_ids = [i for i, layer in enumerate(layer_shapes) if layer == shape]
            for layer_id in layer_ids:
                candidates = num_candidates[layer_id]
                params_start = 0
                params_end = num_params - (num_params % candidates)
                self._layer_param_blocks[layer_id] = [candidates, params_start, params_end]

        sharing_blocks = None
        if self._share_type in ['wavg', 'wavg_slide', 'emb', 'emb_slide']:
            self.set_layer_coeff(num_candidates, num_params)

        return num_params, sharing_blocks

    def param_layer_assignment(self):
        num_params = self._max_params
        if num_params > 0:
            num_candidates = self.get_num_candidates(num_params)
        else:
            num_candidates = np.ones(len(self._layers), np.int64)
            params_by_layer, _ = self.params_per_layer(num_candidates)
            num_params = max(params_by_layer)
            num_candidates = self.get_num_candidates(num_params)

        params_by_layer, _ = self.params_per_layer(num_candidates)

        # setting coefficients also determines when not enugh params
        # are available for a layer
        if self._share_type in ['wavg', 'wavg_slide', 'emb', 'emb_slide']:
            self.set_layer_coeff(num_candidates, num_params)

        # let's get new layer sizes in case they changed after
        # setting coefficients
        params_by_layer, _ = self.params_per_layer(num_candidates)

        start_end_sets = {}
        global_start = 0
        layer_shapes = [str(l.get_group_size(self.width)[0]) for l in self._layers]
        most_common_shape = [int(d) for d in Counter(layer_shapes).most_common(1)[0][0].split()]
        self._layer_param_blocks = [None for _ in range(len(self._layers))]
        for shape, _ in Counter(layer_shapes).most_common():
            layer_ids = [i for i, layer in enumerate(layer_shapes) if layer == shape]
            for layer_id in layer_ids:
                layer_params, candidates = params_by_layer[layer_id], num_candidates[layer_id]
                if layer_params not in start_end_sets:
                    start_end_sets[layer_params] = {}
                    start_end_sets[layer_params]['num_can'] = candidates
                    can_start = int(np.ceil(global_start / float(layer_params)))
                    if can_start >= candidates:
                        global_start = 0
                        can_start = 0
                    else:
                        global_start += layer_params
                    start_end_sets[layer_params]['can'] = can_start

                params_start = start_end_sets[layer_params]['can'] * layer_params
                params_end = params_start + layer_params
                if params_end > num_params:
                    start_end_sets[layer_params]['can'] = 0
                    params_start = start_end_sets[layer_params]['can'] * layer_params
                    params_end = params_start + layer_params
                    if start_end_sets[layer_params]['can'] < 2:
                        params_end = min(params_end, num_params)


                start_end_sets[layer_params]['can'] += 1
                if self._share_type in ['sliding_window', 'avg_slide', 'wavg_slide', 'emb_slide', 'conv']:
                    if start_end_sets[layer_params]['can'] == start_end_sets[layer_params]['num_can']:
                        start_end_sets[layer_params]['can'] = 0

                verbose_print('layer_id {}, shape {}, can {}, start {}, end {}, num_params {}'.format(layer_id, shape, start_end_sets[layer_params]['can'], params_start, params_end, num_params), self._verbose)
                self._layer_param_blocks[layer_id] = [candidates, params_start, params_end]

        return num_params, num_candidates

    def assign_layers_to_params(self):
        need_sharing = self._max_params < 1 or self.get_num_params() < self._max_params
        num_params, num_candidates = self.param_layer_assignment()
        layer_shapes = [str(l.get_group_size(self.width)[0]) for l in self._layers]
        most_common_shape = [int(d) for d in Counter(layer_shapes).most_common(1)[0][0].split()]

        # sharing blocks give an order for parameter initialization
        # assumption is that some parameters will be initialized
        # mutliple times, but the most common one will be the last
        # and hopefully that is the best
        sharing_blocks = []
        for shape, _ in Counter(layer_shapes).most_common()[::-1]:
            layer_ids = [i for i, layer in enumerate(layer_shapes) if layer == shape]
            for layer_id in layer_ids:
                sharing_blocks.append([self._layer_param_blocks[layer_id], self._layers[layer_id].get_group_size(self.width)[0]])

        return num_params, sharing_blocks

    def set_layer_coeff(self, num_candidates, num_params):
        candidates_req = Counter(list(num_candidates))
        weight_params = {}
        for candidates, num_instances in candidates_req.items():

            if self._share_type in ['wavg', 'wavg_slide']:
                params = torch.zeros((num_instances, candidates))
            else:
                # right now emb doesn't care of the "blocks" for the candidates are the same or not
                # it could be this hurts performance, since only the layers operating on the same
                # blocks should share the same linear lay, but this seems to work well enough
                params = torch.zeros((num_instances, self._emb_size))
                latent = 64
                self.add_module('_coeff_proj_%i_candidates' % candidates, nn.Sequential(nn.Linear(self._emb_size, candidates)))#, nn.Softmax(dim=-1)))

            nn.init.orthogonal_(params)
            weight_params[candidates] = [0, params]

        copied_combs = {}
        self._is_teacher = len(self._combs) == 0
        for i, (layer, candidates) in enumerate(zip(self._layers, num_candidates)):
            combiner_params, comb_layer = None, None
            if True:
                ind, params = weight_params[candidates]
                combiner_params = params[ind]
                weight_params[candidates] = [ind+1, params]
                if self._share_type in ['emb', 'emb_slide']:
                    comb_layer = getattr(self, '_coeff_proj_%i_candidates' % candidates)

            assert candidates < 2 or combiner_params is not None
            
            # Pretraining coefficients are all 0 so we have to handle them separately
            if self._is_teacher and self._pretraining and len(copied_combs) == 0:
                # If we are pretraining and no coefficients for the group have been made then create them
                comb = ParameterCombiner(self._learn_groups, candidates, layer.get_group_size(self.width)[0], num_params, self._upsample_type, self._upsample_window, combiner_params, nn.init.orthogonal_, comb_layer, self._trans)
                copied_combs[self._coeff_share_idxs[i]] = comb
 
            elif self._is_teacher and self._pretraining and len(copied_combs) > 0:
                # If we are pretraining and coefficients for the group have been made then use them
                comb = copied_combs[0]

            elif self._coeff_share_idxs[i] not in self._combs:
                if self._is_teacher:
                    comb = ParameterCombiner(self._learn_groups, candidates, layer.get_group_size(self.width)[0], num_params, self._upsample_type, self._upsample_window, combiner_params, nn.init.orthogonal_, comb_layer, self._trans)
                    copied_combs[self._layer_group_idxs[i]] = comb

                else:
                    comb = ParameterCombiner(self._learn_groups, candidates, layer.get_group_size(self.width)[0], num_params, self._upsample_type, self._upsample_window, combiner_params, nn.init.orthogonal_, comb_layer, self._trans)

                    if self._coeff_share_idxs[i] == -1:
                        self._coeff_share_idxs[i] = np.max(list(self._coeff_share_idxs.values())) + 1

                    copied_combs[self._layer_group_idxs[i]] = comb

            else:
                comb = self._combs[self._coeff_share_idxs[i]]
                copied_combs[self._layer_group_idxs[i]] = comb

            layer.set_coefficients(comb, self.width)

        self._combs = copied_combs

    def update_layer_coeff(self, layer_idx, curr_comb=None, coeff=None):
        if curr_comb is None:
            curr_comb = self._combs[self._layer_group_idxs[layer_idx]]
            coeff = curr_comb.coefficients.clone()

        layer = self._layers[layer_idx]

        verbose_print('     Layer_id {}'.format(layer.layer_id), self._verbose)

        comb = ParameterCombiner(self._learn_groups, curr_comb.num_candidates, layer.get_group_size(self.width)[0], 0, self._upsample_type, self._upsample_window, coeff, nn.init.orthogonal_, None, self._trans)
        comb = comb.to(curr_comb.coefficients.get_device())

        self._combs[self._layer_group_idxs[layer_idx]] = comb
        layer.set_coefficients(comb, self.width)

        return layer.layer_id

    def forward(self, layer_id, coeff=None):
        ind = self._layerid2ind[layer_id]
        num_candidates, blocks_start, blocks_end = self._layer_param_blocks[ind]
        params = self._params.get_params(num_candidates, blocks_start, blocks_end)
        if coeff is not None:
            upsample_layer = None
            params = coeff(params, upsample_layer)
        elif self._share_type in ['avg', 'avg_slide']:
            params = params.mean(0)

        return params


def verbose_print(str, verbose):
    if verbose:
        print(str)
