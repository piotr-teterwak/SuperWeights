from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import torch.autograd.profiler as profiler

from .group_gradient_analysis import *

# Compatibility with PyTorch v1.5.0 versus earlier.
if hasattr(torch.nn, '_VF'):
    from torch.nn import _VF
else:
    from torch import _VF


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

class SModule(nn.Module):
    def __init__(self, bank, in_features, out_features):
        super(SModule, self).__init__()
        self.in_channels = in_features
        self.out_channels = out_features
        self.group_size = []

    def get_group_size(self, width):
        return self.group_size[width]

    def param_shape(self):
        if hasattr(self, 'coefficients'):
            if self.coefficients is not None and hasattr(self.coefficients, '_in_param_shape'):
                return self.coefficients._in_param_shape
        return self.shape

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')

        child_lines = []
        for key, module in self._modules.items():
            if key == 'bank':
                continue
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)

        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def get_params(self):
        if hasattr(self, 'coefficients') and self.coefficients is not None:
            coeffs = self.coefficients
        else:
            coeffs = None

        layer = None
        for i, bank in enumerate(self.banks):
            coeffs_i = None
            if coeffs is not None:
                coeffs_i = coeffs[i]
            bank_params = bank(self.layer_id, coeffs_i)
            if len(self.group_size[i][1]) == 1:
                # Num input channels <= previous width
                if layer is not None:
                    layer = torch.cat((layer[:,:self.group_size[i][1][0][1]], bank_params.view(*self.group_size[i][1][0])), 0)
                # Width 0
                else:
                    layer = bank_params.view(*self.group_size[i][1][0])
            else:
                # num input channels > previous width
                if len(bank_params.shape) == 1: bank_params = torch.unsqueeze(bank_params, 0) 
                bank_n_params_inp = np.prod(self.group_size[i][1][1])
                bank_params_inp = bank_params[0,:bank_n_params_inp]
                layer_inp = bank_params_inp.view(*self.group_size[i][1][1])

                # Output channels could be <= 0
                if self.group_size[i][1][0][0] > 0:
                    bank_params_outp = bank_params[0,bank_n_params_inp:]
                    layer_outp = bank_params_outp.view(*self.group_size[i][1][0])
                    layer = torch.cat((torch.cat((layer, layer_inp), 1), layer_outp), 0)
                elif self.group_size[i][1][0][0] < 0:
                    layer = torch.cat((layer[:self.group_size[i][1][0][0],:], layer_inp), 1)
                else:
                    layer = torch.cat((layer, layer_inp), 1)

        if layer.shape != torch.Size(self.shape) and self.groups == 1:
            print('Incorrect weight shape of {}, should be {} for layer {}, group {}, with number of banks (width) {}'.format(layer.shape, self.shape, self.layer_id, self.group_id, len(self.banks)), self.groups)
            print('Attempting to reshape ...')
            
            # attempt to reshape
            layer = layer.view(self.shape)
            # If width 0 then we can change group size so we don't have to reshape each forward pass
            if len(self.banks) == 1:
                self.group_size[0][1][0] = self.shape
        return layer


class SConv2d(SModule):
    def __init__(self, bank, in_features, out_features, kernel_size, stride=1, padding=0, groups=1, bias=None):
        super(SConv2d, self).__init__(bank, in_features, out_features)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.groups = groups
        assert in_features % groups == 0
        self.shape = [out_features, in_features // groups, kernel_size, kernel_size]
        self.bank, self.group_id, self.layer_id = bank.add_layer(self)
        self.banks = []
        self.register_hook = False
        self.p = None

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, groups={groups}')
        s += ', parameter_group={group_id}'
        return s.format(**self.__dict__)

    def set_coefficients(self, coefficients, width):
        if coefficients is None:
            self.coefficients = None
        else:
            if hasattr(self, 'coefficients') and self.coefficients is not None:
                if len(self.coefficients) > width:
                    self.coefficients[width] = coefficients
                else:
                    self.coefficients.append(coefficients)
            else:
                self.add_module('coefficients', nn.ModuleList([coefficients]))

    def forward(self, input):        
        params = self.get_params()
        if self.register_hook:
            params.register_hook(HOOK.get_grad(self.group_id, self.layer_id))
            params.retain_grad()
        params = params.to(input.get_device())
        return F.conv2d(input, params, stride=self.stride, padding=self.padding, groups=self.groups)

class SLinear(SModule):
    def __init__(self, bank, in_features, out_features, bias=None):
        super(SLinear, self).__init__(bank, in_features, out_features)
        self.shape = [out_features, in_features]
        self.bank, self.group_id, self.layer_id = bank.add_layer(self)
        self.banks = []
        self.register_hook = False
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def set_coefficients(self, coefficients, width):
        if coefficients is None:
            self.coefficients = None
        else:
            if hasattr(self, 'coefficients') and self.coefficients is not None:
                if len(self.coefficients) > width:
                    self.coefficients[width] = coefficients
                else:
                    self.coefficients.append(coefficients)
            else:
                self.add_module('coefficients', nn.ModuleList([coefficients]))

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, paramter_group={}'.format(
            self.in_channels, self.out_channels, self.bias is not None, self.group_id
        )

    def forward(self, input):        
        params = self.get_params()
        if self.register_hook:
            params.register_hook(HOOK.get_grad(self.group_id, self.layer_id))
            params.retain_grad()

        return F.linear(input, params, bias=self.bias)
