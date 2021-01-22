import torch
import torch.nn as nn

from SparsePooling_pytorch.models import SparsePoolingLayers, Supervised

class SparsePoolingModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.update_params = True

        # Format: (layer_type, out_channels, kernel_size, p)    
        arch = [('SC', 32, 10, 1e-1)] # ,('SC',64, 3, 1e-1),('M') .. etc
        self.layers = nn.ModuleList([])
        
        in_channels = opt.in_channels_input
        for (layer_type, out_channels, kernel_size, p) in arch:
            if layer_type=='SC':
                layer = SparsePoolingLayers.SC_layer(opt, in_channels, out_channels, kernel_size, p)
            else:
                raise ValueError("layer type not implemented yet")
            
            self.layers.append(layer)
            in_channels = out_channels


    def forward(self, input):
        pre = input
        for layer in self.layers:
            post = layer(pre).clone().detach()
            if self.update_params:
                layer.update_parameters(pre, post)
            pre = post
        
        return post

