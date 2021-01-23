import torch.nn as nn
import torch.nn.functional as F
import torch

class SparsePoolingLayer(nn.Module):
    def __init__(self, opt, in_channels, out_channels, kernel_size, p):
        super(SparsePoolingLayer, self).__init__()
        self.W_ff = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=False)
        self.W_rec = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.W_rec.weight.data = torch.zeros(self.W_rec.weight.shape) # initialize with zeros
        self.threshold = torch.nn.Parameter(0.1 * torch.randn(out_channels, requires_grad=True))
        
        self.nonlin = nn.ReLU(inplace=False) # ATTENTION: bias update assumes ReLU-like nonlin with real zeros
        self.epsilon = opt.epsilon
        self.tau = opt.tau
        self.p = p

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.learning_rate) 
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=opt.learning_rate)

    def forward(self, input):
        u_0 = self.W_ff(input) # b, c, x, y
        a = self.nonlin(u_0).clone().detach()

        converged = False
        cur_device = input.get_device()
        u = torch.zeros(u_0.shape, device=cur_device)
        u_old = torch.zeros(u_0.shape, device=cur_device)
        i = 0
        while not converged:
            u = (1 - self.tau) * u + self.tau * (u_0 - self.W_rec(a))
            a = self.nonlin(u - self.threshold.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)).clone().detach()
            converged = torch.norm(u - u_old) / torch.norm(u) < self.epsilon
            u_old = u.clone()
            i += 1

        # print("iterations: ", i)
        return a.clone().detach()

    # manual implementation of SC learning rules: overwrite grads & update with opt.step()
    def update_parameters(self, pre, post):
        self.zero_grad()
        self.get_parameters_updates(pre, post)
        self.optimizer.step()
        self.postprocess_parameter_updates()

    def get_parameters_updates(self, pre, post):
        self.get_update_W_ff(pre, post)
        self.get_update_threshold_ff(post)
        self.get_update_W_rec(post)

    def get_update_W_ff(self, pre, post, power=4): 
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
        
        self.W_ff.weight.grad = -1 * (dW_ff_data - dW_ff_weight_decay) # c_post, c_pre, kernel_size, kernel_size
        
    def get_update_threshold_ff(self, post, lr_factor_to_W_ff = 10.):
        # post: b, c_post, x_post, y_post
        # self.threshold: c_post

        # ATTENTION: this assumes ReLU-like nonlin with real zeros; otherwise sign operation doesn't work!
        dthreshold = torch.mean(torch.sign(post), (0,2,3)) - self.p # c_post
        self.threshold.grad = -1 * lr_factor_to_W_ff * dthreshold # c_post

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
        
        self.W_rec.weight.grad = -1 * lr_factor_to_W_ff * (dW_rec_data - dW_rec_weight_decay) # c_post, c_post, 1, 1
        # postprocessing (no self inhib. + Dale's law) is done after updating

    def postprocess_parameter_updates(self):
        W_rec = self.W_rec.weight.squeeze().clone() # c_post, c_post
        
        # no self-inhibition: diag = zero
        W_rec = W_rec - torch.diag_embed(torch.diag(W_rec))
        
        # Dale's law: only one sign allowed in weight matrix (pos but eff. neg in forward)
        cur_device = W_rec.get_device()
        zeros = torch.zeros(size=W_rec.shape, device=cur_device)
        W_rec = torch.where(W_rec > zeros, W_rec, zeros)
        
        self.W_rec.weight.data = W_rec.unsqueeze(-1).unsqueeze(-1) # c_post, c_post, 1, 1
        

class SC_layer(SparsePoolingLayer):
    def __init__(self, opt, in_channels, out_channels, kernel_size, p):
        super(SC_layer, self).__init__(opt, in_channels, out_channels, kernel_size, p)

    # def update_parameters(self, pre, post):
    # overwrite with specific inputs

class SFA_layer(SparsePoolingLayer):
    def __init__(self, opt, in_channels, out_channels, kernel_size, p):
        super(SFA_layer, self).__init__(opt, in_channels, out_channels, kernel_size, p)

    

