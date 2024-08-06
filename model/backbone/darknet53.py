import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.cnn_block import CNNBlock
import numpy as np

config = [  
    (32, 3, 1),
    (64, 3, 2),
    ['R', 1], #Residual block
    (128, 3, 2),
    ['R', 2],
    (256, 3, 2),
    ['R', 8],
    (512, 3, 2),
    ['R', 8],
    (1024, 3, 2),
    ['R', 4]
]

class ResidualBlock(nn.Module):
    def __init__(self, channels, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            self.blocks += [
                nn.Sequential(CNNBlock(channels, channels//2, kernel_size=1),
                              CNNBlock(channels//2, channels, kernel_size=3))
            ]
    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x
    
class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList()
        in_channels = 3
        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                self.layers.append(CNNBlock(in_channels, out_channels, kernel_size, stride))
                in_channels = out_channels
            elif isinstance(module, list):
                if module[0]=='R':
                   self.layers.append(ResidualBlock(in_channels, module[1]))

    def forward(self, x):
        intermediate_feature_maps = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, ResidualBlock) and layer.num_blocks==8:
                intermediate_feature_maps.append(x)
        return x, intermediate_feature_maps
    
    def load_weight(self, weight_path):
        with open(weight_path, 'rb') as f:
            header = np.fromfile(f, count=5, dtype=np.int32)
            weights = np.fromfile(f, dtype=np.float32)
        ptr = 0
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                conv = layer
            if isinstance(layer, nn.BatchNorm2d):
                bn = layer
                ptr = self._load_conv_bn(weights, ptr, conv, bn)
        print('Load darknet53 weights succesfully')
    
    def _load_conv_bn(self, weights, ptr, conv_model, bn_model):
        num_b = bn_model.bias.numel()

        bn_bias = torch.from_numpy(weights[ptr:ptr + num_b])
        bn_model.bias.data.copy_(bn_bias.view_as(bn_model.bias.data))
        ptr = ptr + num_b

        bn_weight = torch.from_numpy(weights[ptr:ptr + num_b])
        bn_model.weight.data.copy_(bn_weight.view_as(bn_model.weight.data))
        ptr = ptr + num_b

        bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_b])
        bn_model.running_mean.copy_(bn_running_mean.view_as(bn_model.running_mean))
        ptr = ptr + num_b

        bn_running_var = torch.from_numpy(weights[ptr:ptr + num_b])
        bn_model.running_var.copy_(bn_running_var.view_as(bn_model.running_var))
        ptr = ptr + num_b

        num_w = conv_model.weight.numel()
        conv_weight = torch.from_numpy(weights[ptr:ptr + num_w])
        conv_model.weight.data.copy_(conv_weight.view_as(conv_model.weight.data))
        ptr = ptr + num_w

        return ptr