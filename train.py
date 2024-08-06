import torch
import os 
import argparse
import random
from tqdm import tqdm
from voc_dataset import VOCDataset, custom_collate_fn
from torch.utils.data import DataLoader
from model import YOLOv3
from loss import YOLOLossv3
from utils import *
from evaluate import evaluate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=416, help="Training image width and height")
    parser.add_argument("--multiscale_training", type=bool, default=True, help="allow for multi-scale training")
    parser.add_argument("--batch_size", type=int, default=64, help="The number of images per batch")
    parser.add_argument("--init_lr", type=int, default=0.001, help="Initial learning rate")
    parser.add_argument("--start_epoch", type=int, default=0, help="If start_epoch > 0, training starts from a checkpoint")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--grad_accum", type=int, default=1, help="Number of batches for grad accumulation")
    parser.add_argument("--pretrained_weights", type=str, require=True, help="Path to load the pretrained weight")
    parser.add_argument("--ckpt_path", type=str, default='checkpoints', help="Checkpoint path to save the progress")
    parser.add_argument("--train_path", type=str, default="dataset/train.txt", help="the path to training set")
    parser.add_argument("--val_path", type=str, default="dataset/2007_test.txt", help="the path to val set")
    parser.add_argument("--class_path", type=str, default="dataset/voc_classes.txt", help="the path to the txt file of class labels")

    args = parser.parse_args()
    return args

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_names = load_classes(args.class_path)
    num_classes = len(class_names)

    model = YOLOv3(num_classes=num_classes).to(device)
    model.apply(init_weights_normal)
    if args.pretrained_weights.endswith('.pth'):
        model.load_state_dict(torch.load(args.pretrained_weights))
    else:
        model.backbone.load_weight(args.pretrained_weights)

    dataset = VOCDataset(args.train_path, args.image_size, augment=True)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            collate_fn=custom_collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # loss
    criterion = YOLOLossv3(obj_scale=1, no_obj_scale=100, coord_scale=1, cls_scale=1)

    loss_log = tqdm.tqdm(total=0, position=2, bar_format='{desc}', leave=False)

    best_mAP = 0

    # Train code.
    for epoch in tqdm.tqdm(range(args.start_epoch, args.num_epochs), desc='Epoch'):
        model.train()
        model.set_training(True)
        
        for batch_idx, (images, targets) in enumerate(tqdm.tqdm(dataloader, desc='Batch', leave=False)):
            step = len(dataloader) * epoch + batch_idx
            if args.multiscale_training and (step+1)%10==0:
                dataset.image_size = random.choice(range(320, 608 + 1, 32))

            images = images.to(device)
            targets = targets.to(device)

            yolo_output1, yolo_output2, yolo_output3 = model(images)
            loss1 = criterion(yolo_output1, targets, [(10, 13), (16, 30), (33, 23)], stride=8)
            loss2 = criterion(yolo_output2, targets, [(30, 61), (62, 45), (59, 119)], stride=16)
            loss3 = criterion(yolo_output3, targets, [(116, 90), (156, 198), (373, 326)], stride=32)
            loss = loss1 + loss2 + loss3        
            loss.backward()

            if step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_log.set_description_str('Loss: {:.6f}'.format(loss.item()))


        scheduler.step()

        precision, recall, AP, f1, ap_class, inference_time, fps = evaluate(model,
                                                    path=args.valid_path,
                                                    iou_thres=0.5,
                                                    conf_thres=0.5,
                                                    nms_thres=0.5,
                                                    image_size=args.image_size,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    device=device)
        print('val_precision', precision.mean(), precision)
        print('val_recall', recall.mean(), recall)
        print('val_mAP', AP.mean(), AP)
        print('val_f1', f1.mean(), f1)
        print('ap_class', ap_class)
        print('inference_time', inference_time)
        print('fps', fps)

        os.makedirs(args.ckpt_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.ckpt_path, 'yolov3_latest.pth'))
        print('Save latest model')
        
        if AP.mean() > best_mAP:
            best_mAP = AP.mean()
            torch.save(model.state_dict(), os.path.join(args.ckpt_path, 'yolov3_best.pth'))
            print('Save best model')

if __name__ == "__main__":
    args = parse_args()
    train(args)
    
