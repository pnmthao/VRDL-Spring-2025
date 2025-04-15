import os
import json
import torch
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-w', "--model_weight", type=str,
                    default="Resnet50_v2.pth")
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

model = torch.load(f"{args.model_weight}", weights_only=False)
model = model.to(device)

# Transform for the images
transform = T.Compose([
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # no need cuz FasterRCNN has already
])


def test_img(idx, device):
    with Image.open(f"nycu-hw2-data/test/{idx}.png") as img:
        img = transform(img.convert('RGB'))
        img = img.unsqueeze(0)
        return img.to(device)


coco_gt = COCO("nycu-hw2-data/valid.json")
thresh1, thresh2 = 0, 0.7
n_samples = 13068
results, pred_labels = [], []
datestr = f"{datetime.now().strftime('%Y%m%d__%H%M%S')}"

with torch.no_grad():
    for idx in tqdm(range(n_samples)):
        image = test_img(idx+1, device)
        output = model(image)

        boxes = output[0]["boxes"].to('cpu').detach().numpy()
        labels = output[0]["labels"].to('cpu').detach().numpy()
        scores = output[0]["scores"].to('cpu').detach().numpy()

        boxes = boxes[scores > thresh1]
        labels = labels[scores > thresh1]
        scores = scores[scores > thresh1]

        boxes = np.round(boxes).astype(float)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            boxes[i] = [min(x1, x2), min(y1, y2), abs(x1-x2), abs(y1-y2)]
            if labels[i] == 0:
                continue  # it's background
            results.append({
                "image_id": idx+1,
                "bbox": boxes[i].tolist(),
                "score": float(scores[i]),
                "category_id": int(labels[i])
            })

        digits = [(box, label.item()-1) for box, label, score in zip(
            boxes, labels, scores) if score > thresh2 and label > 0]
        digits.sort(key=lambda x: x[0][0])  # Sort left to right
        answer = ''.join(str(d) for _, d in digits)

        if not digits:
            answer = -1  # No digits detected
        pred_labels.append([idx+1, answer])

    df = pd.DataFrame(pred_labels, columns=['image_id', 'pred_label'])
    df.to_csv(f'pred.csv', index=False)
    with open(f'pred.json', 'w') as f:
        json.dump(results, f)

    os.system(f"zip -j {datestr}.zip pred.csv pred.json")
