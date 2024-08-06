import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn_act=True, max_pool=False):
        super().__init__()
        padding = 1 if kernel_size==3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False if bn_act else True)
        self.bn = nn.BatchNorm2d(out_channels) 
        self.act = nn.LeakyReLU(0.1, True)
        self.bn_act = bn_act
        self.max_pool = max_pool
    
    def forward(self, x):
        if self.bn_act:
            out = self.act(self.bn(self.conv(x)))
        else:
            out = self.conv(x)
        if self.max_pool:
            out = F.max_pool2d(out, kernel_size=2, stride=2)
        return out