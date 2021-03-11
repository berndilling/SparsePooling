import torch
import torch.nn as nn
from IPython import embed

from SparsePooling_pytorch.models import SparsePoolingLayers, Supervised


class SparsePoolingModel(torch.nn.Module):
    def __init__(self, opt, do_update_params = True):
        super().__init__()

        self.update_params = do_update_params

        ###############################################################################################################
        ##   ATTENTION: architecture is written to log and needs to be adapted if old architectures are re-loaded!   ##
        ###############################################################################################################

        # architecture format: (layer_type, out_channels, kernel_size, p, timescale)
        # architecture = [('SC', 400, 10, 0.05, None), ('SFA', 10, 1, 0.1, 8)] #,('MaxPool', None, 2, None, None) .. etc
        # architecture = [('SC', 20, 10, 0.05, None), ('SFA', 2, 1, 1/2, 8)] # for bars
        # architecture = [('SC', 100, 3, 0.05, None), ('MaxPool', 100, 2, None, None), 
        #                 ('SC', 200, 3, 0.05, None), ('MaxPool', 200, 2, None, None), 
        #                 ('SC', 400, 3, 0.05, None), ('MaxPool', 400, 2, None, None)]
        # architecture = [('SC', 100, 3, 0.05, None), ('SFA', 100, 2, 0.18, 8), 
        #                 ('SC', 200, 3, 0.05, None), ('SFA', 200, 2, 0.18, 8), 
        #                 ('SC', 400, 3, 0.05, None), ('SFA', 400, 2, 0.18, 8)]
        # # sparsity 0.18 comes from 2x2 max-pooling: new sparsity = (4 choose 1)*0.95^3*0.05^1 + negligible terms (4 choose 2) etc
        architecture = [('BP', 100, 3, None, None), ('MaxPool', 100, 2, None, None), 
                        ('BP', 200, 3, None, None), ('MaxPool', 200, 2, None, None), 
                        ('BP', 400, 3, None, None), ('MaxPool', 400, 2, None, None)]
        # architecture = [('BP', 100, 3, None, None), ('SFA', 100, 2, 0.5, 8), 
        #                 ('BP', 200, 3, None, None), ('SFA', 200, 2, 0.5, 8), 
        #                 ('BP', 400, 3, None, None), ('SFA', 400, 2, 0.5, 8)]
        
        self.architecture = architecture
        self.layers = nn.ModuleList([])
        
        in_channels = opt.in_channels_input
        for (layer_type, out_channels, kernel_size, p, timescale) in self.architecture:
            if layer_type=='BP':
                layer = SparsePoolingLayers.BP_layer(opt, in_channels, out_channels, kernel_size, do_update_params = not do_update_params)
            elif layer_type=='SC':
                layer = SparsePoolingLayers.SC_layer(opt, in_channels, out_channels, kernel_size, p, do_update_params = do_update_params)
            elif layer_type=='SFA':
                layer = SparsePoolingLayers.SFA_layer(opt, in_channels, out_channels, kernel_size, p, timescale, do_update_params = do_update_params)
            elif layer_type=='MaxPool':
                layer = nn.MaxPool2d(kernel_size, stride=2)
            else:
                raise ValueError("layer type not implemented yet")
            
            self.layers.append(layer)
            in_channels = out_channels

        print(self.layers)


    def forward(self, input, up_to_layer=None):
        dparams = None
        if up_to_layer==None:
            up_to_layer = len(self.layers)
        
        pre = input
        if up_to_layer==-1: # return (reshaped/flattened) input image
            s = pre.shape # b, in_channels, x, y
            post = pre.reshape(s[0], s[1]*s[2]*s[3]).unsqueeze(-1).unsqueeze(-1) # b, in_channels*x*y
        else:
            for layer_idx, layer in enumerate(self.layers[:up_to_layer+1]):
                layer_type = self.architecture[layer_idx][0]
                if layer_type=='BP':
                    post = layer(pre)
                    if not layer.update_params:
                        post = post.clone().detach() # detach to avoid potential backprop to earlier layers
                else:
                    post = layer(pre).clone().detach() # detach to avoid potential backprop to earlier layers
                if self.update_params:
                    if (layer_type=='SC') or (layer_type=='SFA'):
                        if layer.update_params:
                            dparams = layer.update_parameters(pre, post)
                pre = post
        
        return post, dparams


    def set_update_params(self, update_model = True, update_BP = False, update_SC_SFA = True):
        if update_BP and update_SC_SFA:
            raise Exception("BP and SC/SFA layers should not be updated at the same time!")
        self.update_params = update_model
        for layer_idx, layer in enumerate(self.layers):
            layer_type = self.architecture[layer_idx][0]
            if layer_type=='BP':
                layer.update_params = update_BP
            if (layer_type=='SC') or (layer_type=='SFA'):
                layer.update_params = update_SC_SFA

