import tqdm
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn import utils

class SpectralNorm:
    def __init__(self, name, bound=False):
        self.name = name
        self.bound = bound
    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)

        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()

        sigma = u @ weight_mat @ v

        if self.bound:
            weight_sn = weight / (sigma + 1e-6) * torch.clamp(sigma, max=1)

        else:
            weight_sn = weight / sigma
        return weight_sn, u

    @staticmethod
    def apply(module, name, bound):
        fn = SpectralNorm(name, bound)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)
        module.register_forward_pre_hook(fn)
        return fn


    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)





def spectral_norm(module, init=True, std=1, bound=False):
    if init:
        nn.init.normal_(module.weight, 0, std)
    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()
    SpectralNorm.apply(module, 'weight', bound=bound)
    return module

def Langevin_dynamics(samples, netD, n_steps=10000, step_lr=0.0002,noisel=0.2,train=True):
    # images = []
    neg_img = torch.autograd.Variable(samples.data).requires_grad_(True)
    # print(samples.data)
    for _ in range(n_steps):
        # images.append(neg_img)
        noise = torch.randn_like(neg_img,device='cuda')
        images_out = netD(neg_img)
        images_out.sum().backward()
        # print(neg_img.grad.norm(dim=1).size())
        # print("modulus of grad components:  norm {}".format(noise.norm()))
        # print(neg_img.grad.norm(dim=1).unsqueeze(1).expand(256, 2))
        noise = neg_img.grad.norm(dim=1).unsqueeze(1).expand(256, 2) * noise.normal_(0, noisel) * np.sqrt(step_lr)
        neg_img.data.add_(noise.data)
        # print("modulus of grad components: mean {}, max {} , norm {}".format(neg_img.grad.data.abs().mean(),
        #                                                                      neg_img.grad.data.abs().max(),
        #                                                                      neg_img.grad.norm()))
        # print(neg_img)
        neg_img.data.add_(step_lr, neg_img.grad.data)
        # print(neg_img)
        # neg_img.data.add_(1.0,noise)
        neg_img.grad.detach_()
        neg_img.grad.zero_()
    if not train:
        neg_img = neg_img.to('cpu')
    # print(samples.data)
    # print(neg_img.data)

    return neg_img


