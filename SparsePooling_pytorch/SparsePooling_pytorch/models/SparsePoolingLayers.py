import torch.nn as nn
import torch
from IPython import embed

class SparsePoolingLayer(nn.Module):
    def __init__(self, opt, in_channels, out_channels, kernel_size, p):
        super(SparsePoolingLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        #self.W_ff = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=max(kernel_size//2, 1), padding=0, bias=False)
        self.W_ff = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0, bias=False)
        self.W_rec = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.W_rec.weight.data = torch.zeros(self.W_rec.weight.shape) # initialize with zeros
        self.threshold = torch.nn.Parameter(0.1 * torch.randn(out_channels, requires_grad=True))
        
        self.nonlin = nn.ReLU(inplace=False) # ATTENTION: bias update assumes ReLU-like nonlin with real zeros
        self.epsilon = opt.epsilon
        self.tau = opt.tau
        self.p = p

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.learning_rate) 
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=opt.learning_rate)

    def forward(self, input):
        u_0 = self.W_ff(input) # b, c, x, y
        a = self.nonlin(u_0).clone().detach()

        converged = False
        cur_device = input.get_device()
        if cur_device==-1:
            cur_device = None
        u = torch.zeros(u_0.shape, device=cur_device, requires_grad=False)
        u_old = torch.zeros(u_0.shape, device=cur_device, requires_grad=False)
        i = 0
        while not converged:
            u = ((1 - self.tau) * u + self.tau * (u_0 - self.W_rec(a))).detach() # detach is important, otherwise graph grows and RAM overflows
            a = self.nonlin(u - self.threshold.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)).clone().detach()
            converged = torch.norm(u - u_old) / torch.norm(u) < self.epsilon
            u_old = u.clone()
            i += 1

        # print("iterations: ", i)
        return a.clone().detach()

    # manual implementation of SC learning rules: overwrite grads & update with opt.step()
    def update_parameters(self, pre, post):
        self.zero_grad()
        dW_ff, dthreshold, dW_rec = self.get_parameters_updates(pre, post)
        self.optimizer.step()
        self.postprocess_parameter_updates()
        return (dW_ff, dthreshold, dW_rec)

    def get_parameters_updates(self, pre, post):
        dW_ff = self.get_update_W_ff(pre, post)
        dthreshold = self.get_update_threshold_ff(post)
        dW_rec = self.get_update_W_rec(post)
        return dW_ff, dthreshold, dW_rec

    def get_update_W_ff(self, pre, post, power=2): 
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
        dthreshold = torch.mean(torch.sign(post), (0,2,3)) - self.p # c_post
        self.threshold.grad = -1 * lr_factor_to_W_ff * dthreshold # c_post
        return dthreshold

    def get_update_W_rec(self, post, lr_factor_to_W_ff = 20.):
        # post: b, c_post, x_post, y_post
        # self.W_rec: c_post, c_post, 1, 1
        
        # First: weight decay
        post_av = torch.mean(post, (0,2,3)) # c_post (average over batch, x_post, y_post)
        dW_rec_weight_decay = torch.einsum('i,j->ij', post_av, post_av).unsqueeze(-1).unsqueeze(-1) * self.W_rec.weight # c_post, c_post, 1, 1
        
        # Second: (data-driven) decorrelation through anti-Hebb and average over batch, x_post, y_post
        post_centered = post - post_av.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # b, c_post, x_post, y_post
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
        

class SC_layer(SparsePoolingLayer):
    def __init__(self, opt, in_channels, out_channels, kernel_size, p):
        super(SC_layer, self).__init__(opt, in_channels, out_channels, kernel_size, p)

class SFA_layer(SparsePoolingLayer):
    def __init__(self, opt, in_channels, out_channels, kernel_size, p, timescale):
        if not opt.classifying and opt.dataset_type != "moving":
            raise ValueError("Option --dataset_type must be 'moving' if you are training SFA layers")

        super(SFA_layer, self).__init__(opt, in_channels, out_channels, kernel_size, p)
        self.timescale = timescale
        self.sequence_length = opt.sequence_length
        if self.timescale >= self.sequence_length:
            raise ValueError("layer timescale is greater than sequence length of data")

    def calculate_trace(self, post):
        s = post.shape # b'(=b*sequence_length), c_post, x_post, y_post
        post = post.reshape(-1, self.sequence_length, s[1], s[2], s[3]) # b, sequence_length, c_post, x_post, y_post
        post = post.unfold(1, self.timescale, 1) # b, sequence_length-timescale+1, c_post, x_post, y_post, timescale
        # TODO add weights to fake exponential average?
        post_tr = torch.mean(post, (-1)) # b, sequence_length-timescale+1, c_post, x_post, y_post
        return post_tr.reshape(-1, s[1], s[2], s[3]) # b*(sequence_length-timescale+1), c_post, x_post, y_post

    # center and cut of beginning of sequence where no trace can be computed
    def preprocess_pre(self, pre):
        s = pre.shape # b'(=b*sequence_length), c_pre, x_pre, y_pre
        pre_av = torch.mean(pre, (0,2,3)) # c_pre (average over batch, x_pre, y_pre)
        pre = pre - pre_av.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # b', c_pre, x_pre, y_pre
        pre = pre.reshape(-1, self.sequence_length, s[1], s[2], s[3]) # b, sequence_length, c_pre, x_pre, y_pre
        pre = pre[:, self.timescale-1:, :, :, :] # b, sequence_length-timescale+1, c_pre, x_pre, y_pre
        return pre.reshape(-1, s[1], s[2], s[3]) # b*(sequence_length-timescale+1), c_pre, x_pre, y_pre

    # overwrite parent's function
    def get_update_W_ff(self, pre, post, power=2):
        post_trace = self.calculate_trace(post)
        pre_centered = self.preprocess_pre(pre)
        return super().get_update_W_ff(pre_centered, post_trace, power=power)
    

