import torch.nn as nn
import torch.nn.functional as F
import torch


class MaxPoolSpatial2D(nn.Module):
    def __init__(self, ksize, stride=None, pad=0):
        super().__init__()
        self.ksize = ksize
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        return plane_group_spatial_max_pooling(x, self.ksize, self.stride, self.pad)
    

def plane_group_spatial_max_pooling(x, ksize, stride=None, pad=0):
    xs = x.size()
    x = x.view(xs[0], xs[1] * xs[2], xs[3], xs[4])
    x = F.max_pool2d(input=x, kernel_size=ksize, stride=stride, padding=pad)
    x = x.view(xs[0], xs[1], xs[2], x.size()[2], x.size()[3])
    return x


class MaxPoolRotation2D(nn.Module):
    def forward(self, x):
        xs = x.size() # bs, ch, stab, img_dim, img_dim
        x = torch.max(x, dim=2)[0]
        return x
    
