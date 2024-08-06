import torch
import torch.nn as nn
import numpy as np
from utils import build_targets

class YOLOLossv3(nn.Module):
    def __init__(self, ignore_thres=0.5, obj_scale=1, no_obj_scale=100, coord_scale=1, cls_scale=1):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        #self.bce_loss = nn.BCELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ignore_thres = ignore_thres
        self.obj_scale = obj_scale
        self.no_obj_scale = no_obj_scale
        self.coord_scale = coord_scale
        self.cls_scale = cls_scale
    
    def forward(self, prediction, targets, anchors, stride):
        num_anchors = len(anchors)
        
        cx = torch.sigmoid(prediction[..., 0])  # Center x
        cy = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        #pred_conf = torch.sigmoid(prediction[..., 4])  # Object confidence (objectness)
        #pred_cls = torch.sigmoid(prediction[..., 5:])  # Class prediction
        pred_conf = prediction[..., 4]
        pred_cls = prediction[..., 5:]
        
        scaled_anchors = torch.as_tensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors],
                                         dtype=torch.float, device=prediction.device)
        anchor_w = scaled_anchors[:, 0:1].view((1, num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, num_anchors, 1, 1))
        
        obj_mask, no_obj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
            prediction=prediction,
            target=targets,
            anchors=scaled_anchors,
            ignore_thres=self.ignore_thres,
            device=prediction.device
        )

        # Loss: Mask outputs to ignore non-existing objects (except with conf. loss)
        loss_x = self.mse_loss(cx[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(cy[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_bbox = (loss_x + loss_y + loss_w + loss_h) * self.coord_scale
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_no_obj = self.bce_loss(pred_conf[no_obj_mask], tconf[no_obj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.no_obj_scale * loss_conf_no_obj
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask]) * self.cls_scale
        loss_layer = loss_bbox + loss_conf + loss_cls
    
        return loss_layer