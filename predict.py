import torch
from torchvision import transforms
import albumentations as A

import argparse
import os
import cv2
import matplotlib.pyplot as plt

from model import YOLOv3
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to img to be predicted")
    parser.add_argument("--image_size", type=int, default=416)
    parser.add_argument("--pretrained_weights", type=str, default='checkpoints/yolov3_voc_best.pth', help="Path to load the pretrained weight")
    parser.add_argument("--class_path", type=str, default="dataset/voc_classes.txt", help="the path to the txt file of class labels")
    parser.add_argument("--conf_thres", type=int, default=0.5, help="Confidence threshold")
    parser.add_argument("--nms_thres", type=int, default=0.5, help="Non max suppression threshold")

    return parser.parse_args()

def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_names = load_classes(args.class_path)
    num_classes = len(class_names)

    model = YOLOv3(num_classes=num_classes, is_training=False)
    model.load_state_dict(torch.load(args.pretrained_weights, map_location=device))

    img = cv2.imread(args.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
                A.LongestMaxSize(max_size=int(args.image_size)),
                A.PadIfNeeded(
                    min_height=int(args.image_size),
                    min_width=int(args.image_size),
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                ),
            ])
    transformed_img = (transform(image=img))['image']
    transformed_img = transforms.ToTensor()(transformed_img)
    transformed_img = transformed_img.unsqueeze(0)

    with torch.no_grad():
        prediction = model(transformed_img)

    prediction = (non_max_suppression(prediction, args.conf_thres, args.nms_thres))[0]
    
    if prediction is not None:
        cmap = np.array(plt.colormaps.get_cmap('Paired').colors)
        cmap_rgb = np.multiply(cmap, 255).astype(np.int32).tolist()

        prediction = rescale_boxes_original(prediction, args.image_size, (img.shape[1], img.shape[0]))

        for x1, y1, x2, y2, obj_conf, cls_conf, cls_pred in prediction:
            color = cmap_rgb[int(cls_pred) % len(cmap_rgb)]

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            text = '{} {:.1f}'.format(class_names[int(cls_pred)], obj_conf.item() * 100)
            font_scale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
            text_x = int(x1)
            text_y = int(y1) - text_size[1]

            cv2.rectangle(img, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), color, -1)
            cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    
    filename = args.image_path.split('/')[-1]
    folder = args.image_path.replace(f'/{filename}', '')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(folder, f'predicted_{filename}'), img)


if __name__ == "__main__":
    args = parse_args()
    predict(args)
