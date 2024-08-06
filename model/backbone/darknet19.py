import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.backbone.cnn_block import CNNBlock

config = [
    # Unit1 (2)
    (32, 3, True),
    (64, 3, True),
    # Unit2 (3)
    (128, 3, False),
    (64, 1, False),
    (128, 3, True),
    # Unit3 (3)
    (256, 3, False),
    (128, 1, False),
    (256, 3, True),
    # Unit4 (5)
    (512, 3, False),
    (256, 1, False),
    (512, 3, False),
    (256, 1, False),
    (512, 3, True),
    # Unit5 (5)
    (1024, 3, False),
    (512, 1, False),
    (1024, 3, False),
    (512, 1, False),
    (1024, 3, False),
]
    
class Darknet19(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 3
        for out_channels, kernel_size, max_pool in config:
            layers.append(CNNBlock(in_channels, out_channels, kernel_size, max_pool=max_pool))
            in_channels = out_channels
        
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

    def load_weight(self, weight_path):
        with open(weight_path, 'rb') as f:
            header = np.fromfile(f, count=4, dtype=np.int32)
            weights = np.fromfile(f, dtype=np.float32)

        ptr = 0
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                conv = module
            if isinstance(module, nn.BatchNorm2d):
                bn = module
                ptr = self._load_conv_bn(weights, ptr, conv, bn)
        assert ptr==weights.shape[0]
    
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