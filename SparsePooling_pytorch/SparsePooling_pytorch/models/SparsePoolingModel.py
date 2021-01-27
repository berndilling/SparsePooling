import torch
import torch.nn as nn

from SparsePooling_pytorch.models import SparsePoolingLayers, Supervised

class SparsePoolingModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.update_params = True

        # Format: (layer_type, out_channels, kernel_size, p)    
        # self.architecture = [('SC', 100, 10, 5e-2)]
        self.architecture = [('SC', 100, 10, 5e-2, None), ('SFA',100, 1, 1e-1, 3)] #,('M') .. etc
        self.layers = nn.ModuleList([])
        
        in_channels = opt.in_channels_input
        for (layer_type, out_channels, kernel_size, p, timescale) in self.architecture:
            if layer_type=='SC':
                layer = SparsePoolingLayers.SC_layer(opt, in_channels, out_channels, kernel_size, p)
            elif layer_type=='SFA':
                layer = SparsePoolingLayers.SFA_layer(opt, in_channels, out_channels, kernel_size, p, timescale)
            else: 
                raise ValueError("layer type not implemented yet")
            
            self.layers.append(layer)
            in_channels = out_channels


    def forward(self, input):
        dparams = None
        pre = input
        for layer in self.layers:
            post = layer(pre).clone().detach()
            if self.update_params:
                dparams = layer.update_parameters(pre, post)
            pre = post
        
        return post, dparams

