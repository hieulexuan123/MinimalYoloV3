import torch
import torch.nn as nn
from model.backbone.cnn_block import CNNBlock
from model.backbone.darknet53 import Darknet53

class NeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.model = nn.Sequential(
            CNNBlock(in_channels, out_channels, kernel_size=1),
            CNNBlock(out_channels, out_channels*2, kernel_size=3),
            CNNBlock(out_channels*2, out_channels, kernel_size=1),
            CNNBlock(out_channels, out_channels*2, kernel_size=3),
            CNNBlock(out_channels*2, out_channels, kernel_size=1)
        )
    def forward(self, x):
        return self.model(x)
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.model = nn.Sequential(
            CNNBlock(in_channels, out_channels, kernel_size=1),
            nn.Upsample(scale_factor=2)
        )
    def forward(self, x):
        return self.model(x)

class YoloLayer(nn.Module):
    def __init__(self, channels, num_classes, anchors, stride, is_training):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchors = anchors
        self.stride = stride
        self.is_training = is_training
        
        self.model = nn.Sequential(
            CNNBlock(channels, channels*2, kernel_size=3),
            CNNBlock(channels*2, (num_classes+5)*self.num_anchors, kernel_size=1, bn_act=False)
        )
    
    def set_training(self, is_training: bool):
        self.is_training = is_training

    def forward(self, x):
        prediction = self.model(x)
        nB = prediction.shape[0]
        nG = prediction.shape[2]
        device = prediction.device
        
        prediction = prediction.reshape(nB, self.num_anchors, self.num_classes+5, nG, nG).permute(0, 1, 3, 4, 2) #(B, n_anchors, S, S, 5+n_classes)
        
        # Get outputs
        cx = torch.sigmoid(prediction[..., 0])  # Center x
        cy = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Object confidence (objectness)
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Class prediction
        
        # Calculate offsets for each grid
        grid_x = torch.arange(nG, dtype=torch.float, device=device).repeat(nG, 1).view(
            [1, 1, nG, nG])
        grid_y = torch.arange(nG, dtype=torch.float, device=device).repeat(nG, 1).t().view(
            [1, 1, nG, nG])
        
        anchors = torch.tensor(self.anchors, dtype=torch.float, device=device)
        anchor_w = anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
        
        # Add offset and scale with anchors
        pred_boxes = torch.zeros_like(prediction[..., :4], device=device)
        pred_boxes[..., 0] = (cx + grid_x) * self.stride
        pred_boxes[..., 1] = (cy + grid_y) * self.stride
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h

        pred = (pred_boxes.reshape(nB, -1, 4),
                pred_conf.reshape(nB, -1, 1),
                pred_cls.reshape(nB, -1, self.num_classes))
        output = torch.cat(pred, -1)
        
        if self.is_training:
            return prediction
        else:
            return output

class YOLOv3(nn.Module):
    def __init__(self, num_classes=20, is_training=True):
        super().__init__()
        anchors = {'scale1': [(10, 13), (16, 30), (33, 23)],
                   'scale2': [(30, 61), (62, 45), (59, 119)],
                   'scale3': [(116, 90), (156, 198), (373, 326)]}
        strides = {'scale1': 8,
                   'scale2': 16,
                   'scale3': 32}

        self.backbone = Darknet53()
        self.neck3 = NeckBlock(1024, 512)
        self.yolo_layer3 = YoloLayer(channels=512, num_classes=num_classes, anchors=anchors['scale3'], stride=strides['scale3'], is_training=is_training)
        
        self.upsample2 = UpsampleBlock(512, 256)
        self.neck2 = NeckBlock(768, 256)
        self.yolo_layer2 = YoloLayer(channels=256, num_classes=num_classes, anchors=anchors['scale2'], stride=strides['scale2'], is_training=is_training)
        
        self.upsample1 = UpsampleBlock(256, 128)
        self.neck1 = NeckBlock(384, 128)
        self.yolo_layer1 = YoloLayer(channels=128, num_classes=num_classes, anchors=anchors['scale1'], stride=strides['scale1'], is_training=is_training)
        
        self.is_training = is_training
        
    def set_training(self, is_training: bool):
        self.is_training = is_training
        self.yolo_layer1.set_training(is_training)
        self.yolo_layer2.set_training(is_training)
        self.yolo_layer3.set_training(is_training)

    def forward(self, x):
        with torch.no_grad():
            x, inter_feature_maps = self.backbone(x)
        
        scale3 = self.neck3(x)
        yolo_output3 = self.yolo_layer3(scale3)
        
        scale2 = self.upsample2(scale3)
        scale2 = torch.cat([scale2, inter_feature_maps[1]], dim=1)
        scale2 = self.neck2(scale2)
        yolo_output2 = self.yolo_layer2(scale2)
        
        scale1 = self.upsample1(scale2)
        scale1 = torch.cat([scale1, inter_feature_maps[0]], dim=1)
        scale1 = self.neck1(scale1)
        yolo_output1 = self.yolo_layer1(scale1)
        
        if self.is_training:
            yolo_outputs = [yolo_output1, yolo_output2, yolo_output3]
        else:
            yolo_outputs = [yolo_output1, yolo_output2, yolo_output3]
            yolo_outputs = torch.cat(yolo_outputs, 1).detach().cpu()
        return yolo_outputs