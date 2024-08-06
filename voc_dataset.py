import torch
import albumentations as A
from torchvision import transforms
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class VOCDataset(Dataset):
    def __init__(self, list_path, image_size=416, augment=False):
        self.image_size = image_size
        self.augment = augment
        
        with open(list_path, 'r') as file:
            self.image_files = file.readlines() 
        self.label_files = [path.replace('/kaggle/input/pascal-voc-2007-and-2012/', '').replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                                .replace('JPEGImages', 'labels') for path in self.image_files]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx].strip()
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label_path = self.label_files[idx].strip()
        if os.path.exists(label_path):
            labels_boxes = np.loadtxt(label_path).reshape(-1, 5)
            labels = labels_boxes[:, 0].tolist()
            boxes = labels_boxes[:, 1:].tolist()

        #Augmentation
        if self.augment:
            h_orig, w_orig = img.shape[0], img.shape[1]
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                #A.RandomCrop(height=int(0.9*h_orig), width=int(0.9*w_orig), p=0.5),
                #A.MotionBlur(blur_limit=5, p=0.5),
                A.ColorJitter(brightness=1.5, saturation=1.5, hue=0.1, p=0.5),
                A.LongestMaxSize(max_size=int(self.image_size)),
                A.PadIfNeeded(
                    min_height=int(self.image_size),
                    min_width=int(self.image_size),
                    border_mode=cv2.BORDER_CONSTANT,
                ),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.3))           
        else: 
            transform = A.Compose([
                A.LongestMaxSize(max_size=int(self.image_size)),
                A.PadIfNeeded(
                    min_height=int(self.image_size),
                    min_width=int(self.image_size),
                    border_mode=cv2.BORDER_CONSTANT,
                ),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.3))
        transformed = transform(image=img, bboxes=boxes, labels=labels)
        
        img = transformed['image']
        boxes = transformed['bboxes']
        labels = transformed['labels']

        #Convert img to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform(img)

        #Create target tensor
        if len(boxes)>0:
            target = torch.zeros((len(boxes), 6))
            boxes = torch.tensor(boxes)
            labels = torch.tensor(labels)
            target[:, 2:] = boxes
            target[:, 1] = labels
        else:
            target = None

        return img, target
    
def custom_collate_fn(batch):
    imgs, targets = list(zip(*batch))
    targets = list(targets)
    for i, boxes in enumerate(targets):
        if boxes is not None:
            boxes[:, 0] = i #sample idx
    targets = [boxes for boxes in targets if boxes is not None]

    targets = torch.cat(targets, dim=0)
    imgs = torch.stack(imgs, dim=0)
    return imgs, targets
        
        



        
