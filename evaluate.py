import csv
import os
import time

import torch
import torch.utils.data
import numpy as np
import tqdm

from voc_dataset import VOCDataset, custom_collate_fn
from utils import *

def evaluate(model, path, iou_thres, conf_thres, nms_thres, image_size, batch_size, num_workers, device):
    model.eval()
    model.set_training(False)

    dataset = VOCDataset(path, image_size, augment=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             collate_fn=custom_collate_fn)

    labels = []
    sample_metrics = []  # List[Tuple] -> [(TP, confs, pred)]
    entire_time = 0
    for images, targets in tqdm.tqdm(dataloader, desc='Evaluate method', leave=False):
        if targets is None:
            continue

        # Extract labels
        labels.extend(targets[:, 1].tolist())

        # Rescale targets
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= image_size

        # Predict objects
        start_time = time.time()
        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)
            outputs = non_max_suppression(outputs, conf_thres, nms_thres)
        entire_time += time.time() - start_time

        # Compute true positives, predicted scores and predicted labels per batch
        sample_metrics.extend(get_batch_statistics(outputs, targets, iou_thres))

    # Concatenate sample statistics
    if len(sample_metrics) == 0:
        true_positives, pred_scores, pred_labels = np.array([]), np.array([]), np.array([])
    else:
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

    # Compute AP
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    # Compute inference time and fps
    inference_time = entire_time / dataset.__len__()
    fps = 1 / inference_time

    # Export inference time to miliseconds
    inference_time *= 1000

    return precision, recall, AP, f1, ap_class, inference_time, fps