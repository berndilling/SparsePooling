from os import EX_CANTCREAT
from IPython.core.interactiveshell import ExecutionInfo
import torch.nn as nn
import torch
import numpy as np
from IPython import embed

class SparsePoolingLayer(nn.Module):
    def __init__(self, opt, in_channels, out_channels, kernel_size, p, stride=1, padding=0, do_update_params = True): # stride=max(kernel_size//2, 1)
        super(SparsePoolingLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.W_ff = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.W_rec = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.W_rec.weight.data = torch.zeros(self.W_rec.weight.shape) # initialize with zeros
        self.threshold = torch.nn.Parameter(0.1 * torch.randn(out_channels, requires_grad=True))
        
        self.nonlin = nn.ReLU(inplace=False) # ATTENTION: bias update assumes ReLU-like nonlin with real zeros
        
        self.epsilon = opt.epsilon
        self.tau = opt.tau
        self.p = p
        self.inference_recurrence = opt.inference_recurrence
        if self.inference_recurrence==2 and self.p != None:
            self.k = int(self.p * out_channels)

        self.update_params = do_update_params

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.learning_rate) 
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=opt.learning_rate)

    def forward(self, input, max_n_iter=20): # max_n_iter in units of 1/tau
        max_n_iter = max_n_iter * int(1. / self.tau)
        u_0 = self.W_ff(input) # b, c, x, y
        # a = self.nonlin(u_0).clone().detach() # b, c, x, y; forward path without biases
        a = self.nonlin(u_0 - self.threshold.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)).clone().detach() # b, c, x, y

        cur_device = input.get_device()
        if cur_device==-1:
            cur_device = None
        if self.inference_recurrence==1: # 1 - lateral recurrence within layer, 0 - no recurrence
            converged = False
            u = torch.zeros(u_0.shape, device=cur_device, requires_grad=False)
            u_old = torch.zeros(u_0.shape, device=cur_device, requires_grad=False)
            i = 0
            while not converged:
                u = ((1 - self.tau) * u + self.tau * (u_0 - self.W_rec(a))).detach() # detach is important, otherwise graph grows and RAM overflows
                # u = ((1 - self.tau) * u + self.tau * (u_0 - self.W_rec(a / self.out_channels))).detach() # attempt to lower number of iterations -> leads to less refined RFs
                a = self.nonlin(u - self.threshold.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)).clone().detach()
                converged = torch.norm(u - u_old) / torch.norm(u) < self.epsilon
                u_old = u.clone()
                i += 1
                if i > max_n_iter:
                    raise Exception("Surpassed maximum number of iterations ("+str(max_n_iter)+") in forward path..")
            # print("iterations: ", i)
        
        elif self.inference_recurrence==2: # 2 - k-winner-take-all
            a_topks, inds = torch.topk(a, self.k, dim=1) # k biggest elements: b, k, x, y
            a_topk = a_topks[:, -1, :, :] # k-th biggest element: b, 1, x, y
            mask = a >= a_topk.unsqueeze(1) # b, c, x, y
            a = torch.where(mask, a, torch.zeros(a.shape, device=cur_device, requires_grad=False)) # b, c, x, y
        
        return a.clone().detach()

    # manual implementation of SC learning rules: overwrite grads & update with opt.step()
    def update_parameters(self, pre, post):
        dparams = None
        if self.update_params:
            self.zero_grad()
            dW_ff, dthreshold, dW_rec = self.get_parameters_updates(pre, post)
            self.optimizer.step()
            self.postprocess_parameter_updates()
            dparams = (dW_ff, dthreshold, dW_rec)

        return dparams 

    def get_parameters_updates(self, pre, post):
        dW_ff = self.get_update_W_ff(pre, post)
        dthreshold = self.get_update_threshold_ff(post)
        dW_rec = self.get_update_W_rec(post)
        return dW_ff, dthreshold, dW_rec

    def get_update_W_ff(self, pre, post, power=2):
        # TODO set power back to power=2

        # pre: b, c_pre, x_pre, y_pre 
        # post: b, c_post, x_post, y_post
        # self.W_ff: c_post, c_pre, kernel_size, kernel_size

        # First: weight decay
        post_av = torch.mean(post, (0,2,3)) # c_post (average over batch, x_post, y_post)
        dW_ff_weight_decay = post_av.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)**power * self.W_ff.weight # c_post, c_pre, kernel_size, kernel_size

        # Second: (data-driven) outer product of pre and post (Hebb) and average over batch, x_post, y_post
        pre = (
            pre.unfold(2, self.W_ff.weight.shape[2], self.W_ff.stride[0]) # b, c_pre, x_post, y_pre, kernel_size
                    .unfold(3, self.W_ff.weight.shape[3], self.W_ff.stride[1]) # b, c_pre, x_post, y_post, kernel_size, kernel_size
        )
        dW_ff_data = (
            torch.einsum('bixykh,bjxy->jikh', pre, post)
            /(post.shape[0] * post.shape[-1] * post.shape[-2])
        ) # c_post, c_pre, kernel_size, kernel_size
        
        dW_ff = dW_ff_data - dW_ff_weight_decay # c_post, c_pre, kernel_size, kernel_size
        self.W_ff.weight.grad = -1 * dW_ff
        return dW_ff
        
    def get_update_threshold_ff(self, post, lr_factor_to_W_ff = 10.):
        # post: b, c_post, x_post, y_post
        # self.threshold: c_post

        # ATTENTION: this assumes ReLU-like nonlin with real zeros; otherwise sign operation doesn't work!

        # CAREFUL: this implements L0 penalty!
        dthreshold = torch.mean(torch.sign(post), (0,2,3)) - self.p # c_post
        
        # CAREFUL: this implements L1 penalty!
        # dthreshold = torch.mean(post, (0,2,3)) - self.p # c_post

        # L0.5 penalty
        #dthreshold = torch.mean(torch.sqrt(post), (0,2,3)) - self.p # c_post
        
        self.threshold.grad = -1 * lr_factor_to_W_ff * dthreshold # c_post
        return dthreshold

    def get_update_W_rec(self, post, post_trace = None, lr_factor_to_W_ff = 20., center = True):
        # post: b, c_post, x_post, y_post
        # self.W_rec: c_post, c_post, 1, 1

        if post_trace == None:
            post_av = torch.mean(post, (0,2,3)) # c_post (average over batch, x_post, y_post)
        else:
            post_av = torch.mean(post_trace, (0,2,3)) # c_post (average over batch, x_post, y_post)
        
        # First: weight decay
        dW_rec_weight_decay = torch.einsum('i,j->ij', post_av, post_av).unsqueeze(-1).unsqueeze(-1) * self.W_rec.weight # c_post, c_post, 1, 1
        
        # Second: (data-driven) decorrelation through anti-Hebb and average over batch, x_post, y_post
        if center:
            if post_trace == None:
                post_centered = post - post_av.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # b, c_post, x_post, y_post
            else:
                post_centered = post - post_trace # b(_red), c_post, x_post, y_post
        else:
            post_centered = post
        
        dW_rec_data = (
            torch.einsum('bixy,bjxy->ij', post_centered, post_centered)
            /(post.shape[0] * post.shape[-1] * post.shape[-2])
        ).unsqueeze(-1).unsqueeze(-1) # c_post, c_post, 1, 1
        
        dW_rec = dW_rec_data - dW_rec_weight_decay
        self.W_rec.weight.grad = -1 * lr_factor_to_W_ff * dW_rec  # c_post, c_post, 1, 1
        # postprocessing (no self inhib. + Dale's law) is done after updating
        return dW_rec

    def postprocess_parameter_updates(self):
        W_rec = self.W_rec.weight.squeeze().clone() # c_post, c_post
        
        # no self-inhibition: diag = zero
        W_rec = W_rec - torch.diag_embed(torch.diag(W_rec))
        
        # Dale's law: only one sign allowed in weight matrix (pos but eff. neg in forward)
        cur_device = W_rec.get_device()
        if cur_device==-1:
            cur_device = None
        zeros = torch.zeros(size=W_rec.shape, device=cur_device)
        W_rec = torch.where(W_rec > zeros, W_rec, zeros)
        
        self.W_rec.weight.data = W_rec.unsqueeze(-1).unsqueeze(-1) # c_post, c_post, 1, 1

#########################################################################################################################        
# Sub classes of SparsePoolingLayer

class BP_layer(SparsePoolingLayer): #  nn.Module
    def __init__(self, opt, in_channels, out_channels, kernel_size, do_update_params = True, padding=0):
        super(BP_layer, self).__init__(opt, in_channels, out_channels, kernel_size, None, do_update_params = do_update_params, padding=padding)
        # super(BP_layer, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0, bias=True)
        # self.nonlin = nn.ReLU(inplace=False)

    # overwrite parent's functions
    def forward(self, input):
        u_0 = self.W_ff(input) # b, c, x, y
        a = self.nonlin(u_0 - self.threshold.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)) # b, c, x, y
        return a
        # out = self.nonlin(self.conv(input))
        # return out


class SC_layer(SparsePoolingLayer):
    def __init__(self, opt, in_channels, out_channels, kernel_size, p, do_update_params=True, padding=0, stride=1):
        super(SC_layer, self).__init__(opt, in_channels, out_channels, kernel_size, p, do_update_params = do_update_params, padding=padding, stride=stride)
        
        # self.init_optimal_weight_bars()

    def init_optimal_weight_bars(self, size=10):
        print("Initialise SC layer with optimal weights for bars")
        X = torch.zeros(2 * size, 1, size, size) # n_img, n_channels, img_size, img_size
        for i in range(size):
            X[i, 0, i, :] = 1. # horizontal bars
            X[i+size, 0, :, i] = 1. # vertical bars
        self.W_ff.weight.data = X * 1.5

        self.W_rec.weight.data = torch.zeros(self.W_rec.weight.shape) # initialize with zeros
        self.threshold.data = 1.5 * torch.ones(self.threshold.shape)


class SFA_layer(SparsePoolingLayer):
    def __init__(self, opt, in_channels, out_channels, kernel_size, p, timescale, do_update_params = True, padding=0, stride=None):
        if not opt.classifying and opt.dataset_type != "moving":
            raise ValueError("Option --dataset_type must be 'moving' if you are training SFA layers")

        if stride==None:
            stride_v = kernel_size
        else:
            stride_v = stride

        super(SFA_layer, self).__init__(opt, in_channels, out_channels, kernel_size, p, stride=stride_v, do_update_params = do_update_params, padding=padding)
        self.timescale = timescale
        self.sequence_length = opt.sequence_length
        if self.timescale >= self.sequence_length:
            raise ValueError("layer timescale is greater than sequence length of data")
        self.subtract_mean = opt.subtract_mean_from_SFA_input

        self.init_trace_filter(out_channels)
        # self.nonlin = nn.Hardsigmoid()

        # self.init_optimal_weight_bars()

    def init_optimal_weight_bars(self, size=10):
        print("Initialise SFA layer with optimal weights for bars")
        X = torch.zeros(2, 2 * size, 1, 1)
        X[0,:size,0,0] = 1.
        X[0,size:,0,0] = -1.
        X[1,:size,0,0] = -1.
        X[1,size:,0,0] = 1.
        self.W_ff.weight.data = X * 0.15

        self.W_rec.weight.data = torch.zeros(self.W_rec.weight.shape) # initialize with zeros
        self.threshold.data = .5 * torch.ones(self.threshold.shape)

    # Careful: padding at the end of trace has to be cut off!
    def init_trace_filter(self, out_channels):
        self.Trace_filter = torch.nn.Conv1d(out_channels, out_channels, self.timescale, padding = 0, bias=False) # padding = self.timescale-1
        self.Trace_filter.weight.data = torch.zeros(self.Trace_filter.weight.data.shape) # out, out, timescale
        for i in range(out_channels):
            for t in range(self.timescale-1):
                self.Trace_filter.weight.data[i, i, self.timescale-t-2] = 1./self.timescale * np.exp( -t / self.timescale)
                # only diagonal elements: each channel is low-pass-filtered independently


    def calculate_trace_filter(self, post):
        s = post.shape # b'(=b*sequence_length), c_post, x_post, y_post
        post = post.reshape(-1, self.sequence_length, s[1], s[2], s[3]) # b, sequence_length, c_post, x_post, y_post
        post = post.permute(0, 3, 4, 2, 1) # b, x_post, y_post, c_post, sequence_length
        post = post.reshape(-1, s[1], self.sequence_length) # b * x_post * y_post, c_post, sequence_length

        post_tr = self.Trace_filter(post) # b * x_post * y_post, c_post, sequence_length_padded
        
        # post_tr = post_tr[:, :, :self.sequence_length] # cut off average over padding at the end of trace, b * x_post * y_post, c_post, sequence_length
        post_tr = post_tr.reshape(-1, s[2], s[3], s[1], self.sequence_length - self.timescale + 1) # b, x_post, y_post, c_post, sequence_length
        post_tr = post_tr.permute(0, 4, 3, 1, 2) # b, sequence_length, c_post, x_post, y_post
        #return post_tr.reshape(s)
        return post_tr.reshape(-1, s[1], s[2], s[3]) # b*(sequence_length-timescale+1), c_post, x_post, y_post

    def calculate_trace(self, post):
        s = post.shape # b'(=b*sequence_length), c_post, x_post, y_post
        post = post.reshape(-1, self.sequence_length, s[1], s[2], s[3]) # b, sequence_length, c_post, x_post, y_post
        post = post.unfold(1, self.timescale, 1) # b, sequence_length-timescale+1, c_post, x_post, y_post, timescale
        # Calculate trace by averaging over timescale-1 time steps. Current activation ([..., :-1]) not included in average)
        post_tr = torch.mean(post[:, :, :, :, :, :-1], (-1)) # b, sequence_length-timescale+1, c_post, x_post, y_post
        return post_tr.reshape(-1, s[1], s[2], s[3]) # b*(sequence_length-timescale+1), c_post, x_post, y_post

    # subtract batch mean
    def subtract_batch_mean(self, pre):
        pre_av = torch.mean(pre, (0,2,3)) # c_pre (average over batch, x_pre, y_pre)
        return pre - pre_av.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # b', c_pre, x_pre, y_pre
    
    # cut of beginning of sequence where no trace can be computed
    def cut_beginning(self, activity): 
        s = activity.shape # b'(=b*sequence_length), c, x, y
        activity = activity.reshape(-1, self.sequence_length, s[1], s[2], s[3]) # b, sequence_length, c, x, y
        activity = activity[:, self.timescale-1:, :, :, :] # b, sequence_length-timescale+1, c, x, y
        return activity.reshape(-1, s[1], s[2], s[3]) # b*(sequence_length-timescale+1), c, x, y

    # overwrite parent's functions
    def get_pre_and_trace(self, pre, post):
        post_trace = self.calculate_trace_filter(post)
        if self.subtract_mean:
            pre = self.subtract_batch_mean(pre)
        pre_cut = self.cut_beginning(pre)
        return pre_cut, post_trace
    def get_update_W_ff(self, pre, post, power=2): # 4
        pre_cut, post_trace = self.get_pre_and_trace(pre, post)
        return super().get_update_W_ff(pre_cut, post_trace, power=power)

    def get_update_W_rec(self, post, lr_factor_to_W_ff = 20.): # 20. # 0.5
        post_trace = self.calculate_trace_filter(post)
        post_cut = self.cut_beginning(post)
        return super().get_update_W_rec(post_cut, post_trace = post_trace, lr_factor_to_W_ff = lr_factor_to_W_ff)

    def get_update_threshold_ff(self, post, lr_factor_to_W_ff = 10.): # 10 # 1.
        post_cut = self.cut_beginning(post)
        return super().get_update_threshold_ff(post_cut, lr_factor_to_W_ff = lr_factor_to_W_ff)


class CSFA_layer(SFA_layer): # Contrastive SFA layer
    def __init__(self, opt, in_channels, out_channels, kernel_size, p, timescale, do_update_params = True, padding=0, stride=1): # stride=None
        if not opt.classifying and opt.dataset_type != "moving":
            raise ValueError("Option --dataset_type must be 'moving' if you are training SFA layers")

        super(CSFA_layer, self).__init__(opt, in_channels, out_channels, kernel_size, p, timescale, do_update_params = do_update_params, padding=padding, stride=stride)

    def sample_contrastive_samples(self, pre, post, cur_device, beginning_is_cut=False):
        if beginning_is_cut:
            seq_length = self.sequence_length - self.timescale + 1
        else:
            seq_length = self.sequence_length
        s_pre = pre.shape # b'(=b*sequence_length), c_pre, x, y
        s_post = post.shape # b', c_post, x, y
        
        pre = pre.reshape(-1, seq_length, s_pre[1], s_pre[2], s_pre[3]) # b, sequence_length, c_pre, x, y
        post = post.reshape(-1, seq_length, s_post[1], s_post[2], s_post[3]) # b, sequence_length, c_post, x, y
        b = pre.shape[0]
        
        rand_index = torch.randint(b, (b,), dtype=torch.long, device=cur_device) # upper limit, shape

        pre_contr = pre[rand_index, :, :, :, :] # b, sequence_length, c_pre, x, y
        post_contr = post[rand_index, :, :, :, :] # b, sequence_length, c_post, x, y

        # reshape back
        pre_contr = pre_contr.reshape(-1, s_pre[1], s_pre[2], s_pre[3]) # b'(=b*sequence_length), c_pre, x, y
        post_contr = post_contr.reshape(-1, s_post[1], s_post[2], s_post[3]) # b'(=b*sequence_length), c_post, x, y
        
        return pre_contr, post_contr

    # overwrite parent's (SFA_layer) functions
    def get_update_W_ff(self, pre, post, power=2):
        # TODO: refactor to reduce redundancy for _contr arrays!
        cur_device = pre.get_device()
        if cur_device==-1:
            cur_device = None
        
        # get traces and cut pre and post
        pre_cut, post_trace = super().get_pre_and_trace(pre, post) # (b', c_pre, x, y), (b', c_post, x, y)
        post_cut = super().cut_beginning(post) # (b', c_post, x, y)

        # create shuffled pres and posts for neg sample (traces stay the same). Use same permutation over batch for pre and post
        pre_contr, post_contr = self.sample_contrastive_samples(pre_cut, post_cut, cur_device, beginning_is_cut=True)

        # calculate updates (trace * rho' * pre) for pos and neg samples. 
        # get rho' = post_dev (derivative at activation, considering ReLU!)
        ones = torch.ones(size=post_cut.shape, dtype=torch.float32, device=cur_device)
        zeros = torch.zeros(size=post_cut.shape, dtype=torch.float32, device=cur_device)
        post_dev = torch.where(post_cut > 0., ones, zeros)
        post_contr_dev = torch.where(post_contr > 0., ones, zeros)

        # trace * rho'
        post_plast = post_trace * post_dev
        post_contr_plast = post_trace * post_contr_dev
        
        # get updates for pos and contrastive samples
        dW = SparsePoolingLayer.get_update_W_ff(self, pre_cut, post_plast, power=power)
        dW_contr = SparsePoolingLayer.get_update_W_ff(self, pre_contr, post_contr_plast, power=power)
        
        # add updates with opposite sign
        dW_eff = dW - dW_contr
    
        return dW_eff
