import os
import time
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, ConcatDataset

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from torch import optim
from tqdm.auto import tqdm
import numpy as np
from datetime import datetime
import argparse
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()

parser.add_argument('-e', "--epochs", type=int, default=10)
parser.add_argument('-l', "--lr", type=float, default=0.001)
parser.add_argument('-m', "--model_name", type=str, default="Resnet50_v2")
parser.add_argument('-w', "--model_weight", type=str, default="")
parser.add_argument('-b', "--batch_size", type=int, default=6)
parser.add_argument("--clip", type=float, default=0)

args = parser.parse_args()
print(args)


def create_model(model_name, num_classes, pretrained=True, coco_model=False):
    print(model_name)
    if 'Eff' in model_name:
        if model_name == 'EffB0':
            # Load the pretrained EfficientNetB0 large features.
            backbone = torchvision.models.efficientnet_b0(
                weights='DEFAULT').features

            # We need the output channels of the last convolutional layers from
            # the features for the Faster RCNN model.
            backbone.out_channels = 1280  # 1280 for EfficientNetB0.

        elif model_name == 'EffB4':
            # Load the pretrained EfficientNetB0 large features.
            backbone = torchvision.models.efficientnet_b4(
                weights='DEFAULT').features

            # We need the output channels of the last convolutional layers from
            # the features for the Faster RCNN model.
            backbone.out_channels = 1792

        # Generate anchors using the RPN. Here, we are using 5x3 anchors.
        # Meaning, anchors with 5 different sizes and 3 different aspect
        # ratios.
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        # Feature maps to perform RoI cropping.
        # If backbone returns a Tensor, `featmap_names` is expected to
        # be [0]. We can choose which feature maps to use.
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )

        # Final Faster RCNN model.
        model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

        return model

    if model_name == 'Mobilev3FPN':
        # Load Faster RCNN pre-trained model
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights='DEFAULT',
        )
        if coco_model:  # Return the COCO pretrained model for COCO classes.
            return model, coco_model

        # get the number of input features
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)

        return model

    if 'Resnet50' in model_name:
        # Load Faster RCNN pre-trained model
        if model_name == 'Resnet50':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights='DEFAULT',
            )
        elif model_name == 'Resnet50_v2':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights='DEFAULT',
            )
        if coco_model:  # Return the COCO pretrained model for COCO classes.
            return model, coco_model

        # Get the number of input features
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)

        return model

    # if 'resnet1' in model_name: # for resnet101 (8), resnet152 (8), resnext50_32x4d,
    # resnext101_32x8d (6), wide_resnet101_2 (8), wide_resnet50_2 (8)
    # Step 1: Create ResNet-101 + FPN backbone
    backbone = resnet_fpn_backbone(
        backbone_name=model_name, weights='IMAGENET1K_V2')

    # Step 2: Customize anchor generator (optional, recommended defaults)
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # Step 3: ROI Align pooling
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    # Step 4: Define Faster R-CNN model
    num_classes = 11  # 10 digits + 1 background
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model


class DigitCocoDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = f"{self.image_dir}/{img_info['file_name']}"

        image = Image.open(img_path).convert("RGB")
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

        boxes, labels = [], []

        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(ann['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        return idx, image, boxes, labels


def collate_fn(batch):
    indices, images, boxes, labels = zip(*batch)

    # Handle padding if needed, or return images and targets as they are
    return indices, images, boxes, labels


# Transform for the images
transform = T.Compose([
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # no need cuz FasterRCNN has already
])

batch_size = args.batch_size
train_dataset = DigitCocoDataset('nycu-hw2-data/train/',
                                 'nycu-hw2-data/train.json', transform=transform)
valid_dataset = DigitCocoDataset('nycu-hw2-data/valid/',
                                 'nycu-hw2-data/valid.json', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=2, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=2, collate_fn=collate_fn)

# For final run, concat all train and valid as training data
# concat_dataset = ConcatDataset([train_dataset, valid_dataset])
# train_loader = DataLoader(concat_dataset, batch_size=batch_size,
#                           shuffle=True, num_workers=6, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

model_name = args.model_name

if args.model_weight:
    model = torch.load(f"{args.model_weight}", weights_only=False)
else:
    model = create_model(model_name, num_classes=11,
                         pretrained=True, coco_model=False)

model = model.to(device)

print('#parameters:', sum(p.numel() for p in model.parameters()))
print('#trainable :', sum(p.numel()
      for p in model.parameters() if p.requires_grad))

lr, momentum, weight_decay = args.lr, 0.9, 0.0005
epochs = args.epochs
optimizer = optim.SGD(model.parameters(), lr=lr,
                      momentum=momentum, weight_decay=weight_decay)


def lr_lambda(epoch):
    if epoch < 3:
        return 1.0      # 0.001 * 1 = 0.001
    elif epoch < 10:
        return 1.0      # 0.001 * 0.5 = 0.0005
    else:
        return 1.0     # 0.0005 * 0.25 = 0.0001


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


notes = "__pretrained" if args.model_weight else ""
datestr = f"{datetime.now().strftime('%Y%m%d__%H%M%S')}"
result_dir = f"results/{model_name}_SGD_LambdaLR_epoch{epochs}_lr{lr}_bs{batch_size}_clip{args.clip}{notes}/{datestr}"
os.makedirs(result_dir, exist_ok=True)
os.system(f"cp 413540004.py {result_dir}/")  # backup code
# Create a SummaryWriter instance
writer = SummaryWriter(
    log_dir=f"tensorboard/{datestr}_{result_dir.split('/')[1]}")

coco_gt = COCO("nycu-hw2-data/valid.json")
thresh = 0.5
best_val_map, best_epoch = -1, -1
num_steps, step = 500, 0
for epoch in tqdm(range(epochs)):
    model.train()
    loss_total = loss_classifier = loss_box_reg = loss_objectness = loss_rpn_box_reg = 0
    loss_steps_total = loss_steps_classifier = loss_steps_box_reg = loss_steps_objectness = loss_steps_rpn_box_reg = 0

    for i, (_, images, boxes, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # Move to default device
        images = list(image.to(device) for image in images)
        targets = [{'boxes': b.to(
            device), 'labels': l.to(device)} for b, l in zip(boxes, labels)]

        # Forward prop.
        loss_dict = model(images, targets)
        loss_dict['loss_total'] = sum(loss for loss in loss_dict.values())
        loss_classifier += loss_dict['loss_classifier'].item()
        loss_box_reg += loss_dict['loss_box_reg'].item()
        loss_objectness += loss_dict['loss_objectness'].item()
        loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
        loss_total += loss_dict['loss_total'].item()

        loss_steps_classifier += loss_dict['loss_classifier'].item()
        loss_steps_box_reg += loss_dict['loss_box_reg'].item()
        loss_steps_objectness += loss_dict['loss_objectness'].item()
        loss_steps_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
        loss_steps_total += loss_dict['loss_total'].item()

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        if args.clip:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clip)
        optimizer.step()

        if (i+1) % num_steps == 0:  # log every $step steps
            print(
                f"\nEpoch: {epoch}, Step: {(i+1)*batch_size}/{len(train_dataset)}:\n\
                    loss_classifier:  {loss_steps_classifier/num_steps:.6f}\n\
                    loss_box_reg:     {loss_steps_box_reg/num_steps:.6f}\n\
                    loss_objectness:  {loss_steps_objectness/num_steps:.6f}\n\
                    loss_rpn_box_reg: {loss_steps_rpn_box_reg/num_steps:.6f}\n\
                    loss_total:       {loss_steps_total/num_steps:.6f}")
            writer.add_scalar('train_steps/loss_classifier',
                              loss_steps_classifier/num_steps, step)
            writer.add_scalar('train_steps/loss_box_reg',
                              loss_steps_box_reg/num_steps, step)
            writer.add_scalar('train_steps/loss_objectness',
                              loss_steps_objectness/num_steps, step)
            writer.add_scalar('train_steps/loss_rpn_box_reg',
                              loss_steps_rpn_box_reg/num_steps, step)
            writer.add_scalar('train_steps/loss_total',
                              loss_steps_total/num_steps, step)

            step += num_steps
            loss_steps_total = loss_steps_classifier = loss_steps_box_reg = loss_steps_objectness = loss_steps_rpn_box_reg = 0

        # break
    writer.add_scalar('train/loss_classifier',
                      loss_classifier/len(train_loader), epoch)
    writer.add_scalar('train/loss_box_reg',
                      loss_box_reg/len(train_loader), epoch)
    writer.add_scalar('train/loss_objectness',
                      loss_objectness/len(train_loader), epoch)
    writer.add_scalar('train/loss_rpn_box_reg',
                      loss_rpn_box_reg/len(train_loader), epoch)
    writer.add_scalar('train/loss_total',
                      loss_total/len(train_loader), epoch)

    scheduler.step()
    model.eval()
    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

    with torch.no_grad():
        val_start = time.time()
        results = []
        for (indices, images, _, _) in tqdm(valid_loader, total=len(valid_loader)):
            images = list(image.to(device) for image in images)

            # without targets will return the prediction instead
            outputs = model(images)
            for i in range(len(outputs)):
                boxes = outputs[i]["boxes"].to('cpu').detach().numpy()
                labels = outputs[i]["labels"].to('cpu').detach().numpy()
                scores = outputs[i]["scores"].to('cpu').detach().numpy()

                boxes = boxes[scores > thresh]
                labels = labels[scores > thresh]
                scores = scores[scores > thresh]

                boxes = np.round(boxes).astype(float)

                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[j]
                    boxes[j] = [min(x1, x2), min(y1, y2),
                                abs(x1-x2), abs(y1-y2)]
                    if labels[j] == 0:  # it's background
                        continue
                    results.append({
                        "image_id": indices[i]+1,
                        "bbox": boxes[j].tolist(),
                        "score": float(scores[j]),
                        "category_id": int(labels[j])
                    })
            # break

        # Evaluation
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAP_score = coco_eval.stats[0]

        if mAP_score > best_val_map:
            best_val_map = mAP_score
            best_epoch = epoch
            best_model = deepcopy(model)
        writer.add_scalar('val/mAP', mAP_score, epoch)
        print(
            f'\nEpoch {epoch}: Val_mAP = {mAP_score:.8f}, Best_Val_mAP = {best_val_map:.8f}')

torch.save(
    best_model, f"{result_dir}/Faster_RCNN-{model_name}_mAP{best_val_map:.6f}.pth")
