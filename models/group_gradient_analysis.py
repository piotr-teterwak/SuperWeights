import torch
import numpy as np
import torch.nn.functional as F
from .pq_assignment import priority_queue_assignment



class Hook():
    # Used to grab SuperWeight gradients

    def __init__(self):
        self.grads = {}
        self.compute = False
        self.model_idx = -1

    def get_grad(self, group, layer_id):
        def hook(grad):
            if self.compute: 
                if self.model_idx not in self.grads: 
                    self.grads[self.model_idx] = {}
                if group not in self.grads[self.model_idx]:
                    self.grads[self.model_idx][group] = {}

                if layer_id in self.grads[self.model_idx][group]:
                    self.grads[self.model_idx][group][layer_id] = self.grads[self.model_idx][group][layer_id] + grad.clone().detach()
                else:
                    self.grads[self.model_idx][group][layer_id] = grad.clone().detach()

        return hook

    def update_model(self, model_idx):
        self.model_idx = model_idx
        self.compute = model_idx > -1
    
    def remove_grads(self):
        del self.grads

HOOK = Hook()


def assign_groups(grads, models, depths, next_group, threshold=0.8, 
                    coeff_threshold=None, concat_weightwgrad=False, verbose=False):

    # Split
    net_1_grads = {0: grads[0]}
    group2layer, next_group = split(net_1_grads, models, next_group, threshold, concat_weightwgrad, verbose)

    layer_coeff_share = {}
    unique_depths = set(depths)
    large_net_idx = 0

    for depth in sorted(unique_depths, reverse=True):
        depth_joined = depth == depths[0] 

        for i in range(1,len(depths)):
            if depths[i] != depth: continue
            if not depth_joined:
                net_depth_grads = {0: grads[0], i: grads[i]}
                group2layer, layer_coeff_share = join_large(net_depth_grads, group2layer, 
                                                        models, next_group, concat_weightwgrad,
                                                        verbose, layer_coeff_share, 
                                                        coeff_threshold=coeff_threshold)
                depth_joined = True
                large_net_idx = i

            elif coeff_threshold is not None:
                # Use coefficient theshold for whether to share coefficients
                net_depth_grads = {large_net_idx: grads[large_net_idx], i: grads[i]}
                group2layer, layer_coeff_share = join_large(net_depth_grads, group2layer, 
                                            models, next_group, concat_weightwgrad,
                                            verbose, layer_coeff_share,
                                            layer_idx_join=True, 
                                            coeff_threshold=coeff_threshold)

            else:
                net_depth_grads = {large_net_idx: grads[large_net_idx], i: grads[i]}
                group2layer, layer_coeff_share = join_large_same_depth(net_depth_grads, group2layer, 
                                                                models, next_group, layer_coeff_share, 
                                                                large_net_idx=large_net_idx)

    # Want consecutive group_ids
    prev_group, curr_group = -1, -1
    for group in range(next_group):
        for net in group2layer.keys():
            if group not in group2layer[net]: continue
            curr_group = group
            if curr_group == prev_group + 1: continue

            for layer_id in group2layer[net][group]:
                for module_name, module in models[net].named_modules():
                    if hasattr(module, 'layer_id') and layer_id == module.layer_id:
                        module.group_id = prev_group + 1
            
            group2layer[net][prev_group + 1] = group2layer[net][group]
            del group2layer[net][group]

        if curr_group != prev_group:
            prev_group = prev_group + 1
            curr_group = prev_group

    next_group = curr_group + 1

    # Convert group2layer to layer2group for use when setting up bank, and grab the minimum number of parameters needed for each group (largest layer)
    max_group_size = {}
    layer2group, group_ids = [], []
    for net in sorted(list(group2layer.keys())):
        layer2group.append([-1] * depths[net])
        group_ids.extend(list(group2layer[net].keys()))
        for group in group2layer[net].keys():
            if group not in max_group_size:
                max_group_size[group] = 0.0
            layerids = []
            for layer_id in group2layer[net][group]:
                for name, module in models[net].named_modules():
                    if hasattr(module, 'layer_id') and layer_id == module.layer_id:
                        layerids.append(module.layer_id) #
                        layer2group[-1][module.layer_id] = group
                        if torch.prod(torch.Tensor(module.shape)) > max_group_size[group]:
                            max_group_size[group] = torch.prod(torch.Tensor(module.shape))

            group2layer[net][group] = layerids #

    group_ids = np.unique(group_ids)
    num_groups = len(group_ids)

    next_group = np.max(group_ids)

    return group2layer, layer2group, num_groups, group_ids, max_group_size, layer_coeff_share


def split(grads, models, num_groups, threshold, concat_weightwgrad, verbose):

    group_2_layer = {}
    next_group = 0
    for group in range(num_groups):

        new_groups = {} # {i: new_group}
        queue = [] # [(sim_ij, i, j)]
        checked = set() # {(i,j)}

        for net_1 in grads.keys():
            if group not in grads[net_1]: continue
            if net_1 not in group_2_layer: group_2_layer[net_1] = {}

            for layer_1, grad_1 in grads[net_1][group].items():

                # Assign layer to new group
                i = 'net_{}-layer_{}'.format(net_1, layer_1)
                assert i not in new_groups
                new_groups[i] = next_group

                # Grab weights
                for name, module in models[net_1].named_modules():
                    if hasattr(module, 'layer_id') and layer_1 == module.layer_id:
                        # module.group_id = next_group # TODO
                        layer_1_weights = module.get_params()

                # Grab similarities
                for net_2 in grads.keys():
                    if group not in grads[net_2]: continue
                    for layer_2, grad_2 in grads[net_2][group].items():

                        j = 'net_{}-layer_{}'.format(net_2, layer_2)
                        # Don't compute if already computed
                        if (i,j) in checked: continue
                        # Don't compute for same layer
                        if net_1 == net_2 and layer_1 == layer_2: continue

                        # Grab weights
                        for name, module in models[net_2].named_modules():
                            if hasattr(module, 'layer_id') and layer_2 == module.layer_id:
                                layer_2_weights = module.get_params()

                        # grab smaller layer size and cut up larger layer to match
                        if grad_1.shape[0] > grad_2.shape[0]:
                            grad_1_crop = grad_1[:grad_2.shape[0],:]
                            layer_1_weights_crop = layer_1_weights[:grad_2.shape[0],:]
                            grad_2_crop = grad_2
                            layer_2_weights_crop = layer_2_weights
                        else:
                            grad_1_crop = grad_1
                            layer_1_weights_crop = layer_1_weights
                            grad_2_crop = grad_2[:grad_1.shape[0],:]
                            layer_2_weights_crop = layer_2_weights[:grad_1.shape[0],:]

                        if grad_1.shape[1] > grad_2.shape[1]:
                            grad_1_crop = grad_1_crop[:,:grad_2.shape[1]]
                            layer_1_weights_crop = layer_1_weights_crop[:,:grad_2.shape[1]]
                        else:
                            grad_2_crop = grad_2_crop[:,:grad_1.shape[1]]
                            layer_2_weights_crop = layer_2_weights_crop[:,:grad_1.shape[1]]

                        if concat_weightwgrad:
                            grad_1_flat = torch.cat([grad_1_crop.reshape(-1, 1) / torch.norm(grad_1_crop), layer_1_weights_crop.reshape(-1, 1) / torch.norm(layer_1_weights_crop)]).reshape(-1)
                            grad_2_flat = torch.cat([grad_2_crop.reshape(-1, 1) / torch.norm(grad_2_crop), layer_2_weights_crop.reshape(-1, 1) / torch.norm(layer_1_weights_crop)]).reshape(-1)
                        else:
                            grad_1_flat = torch.flatten(grad_1_crop)
                            grad_2_flat = torch.flatten(grad_2_crop)

                        sim = F.cosine_similarity(grad_1_flat, grad_2_flat, dim=0)

                        verbose_print('{}-{} - {}-{}, sim: {}'.format(net_1, layer_1, net_2, layer_2, sim), verbose)

                        queue.append((sim, i, j))
                        checked.add((i,j))

                next_group += 1

        # Assign to most similar groups
        queue.sort(key=lambda x: x[0])
        groups = priority_queue_assignment(threshold, new_groups, queue)

        # assign on layers and convert to group_2_layer
        for i, group in groups.items():
            net = int(i.split('-')[0].split('_')[1])
            layer = int(i.split('-')[1].split('_')[1])

            for name, module in models[net].named_modules():
                if hasattr(module, 'layer_id') and layer == module.layer_id:
                    module.group_id = group
                    break

            # convert to group_2_layer to work with legacy code
            if net not in group_2_layer:
                group_2_layer[net] = {}
            if group not in group_2_layer[net]:
                group_2_layer[net][group] = []
            
            group_2_layer[net][group].append(layer)

    return group_2_layer, next_group


def join_large_same_depth(grads, group_2_layer, models, num_groups, layer_coeff_share={}, large_net_idx=0):
    net_1 = large_net_idx
    for group in range(num_groups):
        if group not in grads[net_1]: continue

        for layer_group_idx_1, layer_1 in enumerate(sorted(list(grads[net_1][group].keys()))):
            # Grab large model group
            new_group_id = -1
            for _, module in models[net_1].named_modules():
                if hasattr(module, 'layer_id') and layer_1 == module.layer_id:
                    new_group_id = module.group_id

            for net_2 in grads.keys():
                if net_2 == net_1 or group not in grads[net_2]: continue
                if net_2 not in layer_coeff_share: 
                    layer_coeff_share[net_2] = {}
                for layer_group_idx_2, layer_2 in enumerate(sorted(list(grads[net_2][group].keys()))):
                    if layer_group_idx_1 != layer_group_idx_2: continue

                    if net_2 not in group_2_layer: group_2_layer[net_2] = {}
                    if new_group_id not in group_2_layer[net_2]: group_2_layer[net_2][new_group_id] = []
                    group_2_layer[net_2][new_group_id].append(layer_2)
                    for name, module in models[net_2].named_modules():
                        if hasattr(module, 'layer_id') and layer_2 == module.layer_id:
                            module.group_id = new_group_id
                            for new_layer_group_idx_1, new_layer_1 in enumerate(sorted(list(group_2_layer[net_1][new_group_id]))):
                                if new_layer_1 == layer_1:
                                    layer_coeff_share[net_2][layer_2] = {'layer': new_layer_group_idx_1, 'net': net_1}
                    break

    return group_2_layer, layer_coeff_share


def join_large(grads, group_2_layer, models, num_groups, concat_weightwgrad, verbose, layer_coeff_share={}, layer_idx_join=False, coeff_threshold=None):
    net_2 = 0

    for group in range(num_groups):
        for net_1 in grads.keys():
            if net_1 == 0 or group not in grads[net_1]: continue
            if net_1 not in layer_coeff_share:
                layer_coeff_share[net_1] = {}
            if net_1 not in group_2_layer: group_2_layer[net_1] = {}
            for layer_group_idx_1, layer_1 in enumerate(sorted(list(grads[net_1][group].keys()))):
                largest = None
                assert group in grads[net_2]

                for name, module in models[net_1].named_modules():
                    if hasattr(module, 'layer_id') and layer_1 == module.layer_id:
                        layer_1_weights = module.get_params()

                # Grab similarities
                for layer_group_idx_2, layer_2 in enumerate(sorted(list(grads[net_2][group].keys()))):
                    for name, module in models[net_2].named_modules():
                        if hasattr(module, 'layer_id') and layer_2 == module.layer_id:
                            layer_2_weights = module.get_params()

                    # grab smaller layer size and cut up larger layer to match
                    grad_1 = grads[net_1][group][layer_1]
                    grad_2 = grads[net_2][group][layer_2]

                    if grad_1.shape[0] > grad_2.shape[0]:
                        grad_1_crop = grad_1[:grad_2.shape[0],:]
                        layer_1_weights_crop = layer_1_weights[:grad_2.shape[0],:]
                        grad_2_crop = grad_2
                        layer_2_weights_crop = layer_2_weights
                    else:
                        grad_1_crop = grad_1
                        layer_1_weights_crop = layer_1_weights
                        grad_2_crop = grad_2[:grad_1.shape[0],:]
                        layer_2_weights_crop = layer_2_weights[:grad_1.shape[0],:]

                    if grad_1.shape[1] > grad_2.shape[1]:
                        grad_1_crop = grad_1_crop[:,:grad_2.shape[1]]
                        layer_1_weights_crop = layer_1_weights_crop[:,:grad_2.shape[1]]
                    else:
                        grad_2_crop = grad_2_crop[:,:grad_1.shape[1]]
                        layer_2_weights_crop = layer_2_weights_crop[:,:grad_1.shape[1]]

                    if concat_weightwgrad:
                        grad_1_flat = torch.cat([grad_1_crop.reshape(-1, 1) / torch.norm(grad_1_crop), 
                                                layer_1_weights_crop.reshape(-1, 1) / torch.norm(layer_1_weights_crop)]
                                                ).reshape(-1)
                        grad_2_flat = torch.cat([grad_2_crop.reshape(-1, 1) / torch.norm(grad_2_crop), 
                                                layer_2_weights_crop.reshape(-1, 1) / torch.norm(layer_1_weights_crop)]
                                                ).reshape(-1)
                    else:
                        grad_1_flat = torch.flatten(grad_1_crop)
                        grad_2_flat = torch.flatten(grad_2_crop)

                    sim = F.cosine_similarity(grad_1_flat, grad_2_flat, dim=0)

                    verbose_print('{}-{} - {}-{}, sim: {}'.format(net_1, layer_1, net_2, layer_2, sim), verbose)

                    if largest is None or largest[0] < sim:
                        for name, module in models[net_2].named_modules():
                            if hasattr(module, 'layer_id') and layer_2 == module.layer_id:
                                group_id = module.group_id
                                break

                        layer_group_idx = -1
                        for layer_group_idx_3, layer_3 in enumerate(group_2_layer[net_2][group_id]):
                            if layer_3 == layer_2:
                                layer_group_idx = layer_group_idx_3
                        largest = (sim, group_id, layer_2, layer_group_idx)
                    

                verbose_print('----> {}-{} - {}-{}, sim: {}'.format(net_1, layer_1, net_2, largest[2], largest[0]), verbose)
                for name, module in models[net_1].named_modules():
                    if hasattr(module, 'layer_id') and layer_1 == module.layer_id:

                        if layer_idx_join:
                            assert coeff_threshold is not None
                            # Analyze similarity to evaluate whether student should share coeff
                            grouped = False

                            for layer_group_idx_2, layer_2 in enumerate(sorted(list(grads[net_2][group].keys()))):
                                if layer_group_idx_1 != layer_group_idx_2: continue
                                for name_2, module_2 in models[net_2].named_modules():
                                    if hasattr(module_2, 'layer_id') and layer_2 == module_2.layer_id:
                                        module.group_id = module_2.group_id
                                        if module_2.group_id not in group_2_layer[net_1]:
                                            group_2_layer[net_1][module_2.group_id] = []
                                        
                                        # If the most similar layer is greater than threshold then don't share otherwise do 
                                        if largest[0] > coeff_threshold:
                                            for layer_group_idx_3, layer_3 in enumerate(group_2_layer[net_2][module_2.group_id]):
                                                if layer_3 == layer_2:
                                                    layer_coeff_share[net_1][layer_1] = {'layer': layer_group_idx_3, 'net': net_2}
                                                    break
                                        else:
                                            layer_coeff_share[net_1][layer_1] = {'layer': -1, 'net': net_2}
                                        
                                        grouped = True
                                        break
                                if grouped: break

                        else:
                            module.group_id = largest[1]
                            if largest[1] not in group_2_layer[net_1]:
                                group_2_layer[net_1][largest[1]] = []


                            if coeff_threshold is not None:
                                # Don't share coeff if < threshold
                                if largest[0] > coeff_threshold:
                                    layer_coeff_share[net_1][layer_1] = {'layer': largest[3], 'net': net_2}
                                else:
                                    layer_coeff_share[net_1][layer_1] = {'layer': -1, 'net': net_2}

                            else:
                                layer_coeff_share[net_1][layer_1] = {'layer': largest[3], 'net': net_2}
                                

                        group_2_layer[net_1][module.group_id].append(layer_1)
                        break

    return group_2_layer, layer_coeff_share


def verbose_print(str, verbose):
    if verbose:
        print(str)
