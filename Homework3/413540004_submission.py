import os
import json
import argparse
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm

import pycocotools.mask as mask_util
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import detection_utils

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, default='path/to/model')
parser.add_argument("--config_path", type=str, default='path/to/config')

args = parser.parse_args()
print(args)


# Inference should use the config with parameters that are used in training
cfg = get_cfg()
cfg.merge_from_file(args.config_path)
cfg.MODEL.WEIGHTS = os.path.join(args.model_path)
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
cfg.TEST.DETECTIONS_PER_IMAGE = 800

# print(cfg)
predictor = DefaultPredictor(cfg)

total_instances, prediction = [], []
with open('hw3-data/test_image_name_to_ids.json') as f:
    dataset_dict = json.load(f)
datestr = f"{datetime.now().strftime('%Y%m%d__%H%M%S')}"

for img in tqdm(dataset_dict):
    id = img["id"]
    im = detection_utils.read_image(
        f"hw3-data/test_release/{img['file_name']}", format='BGR')
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")
    instance_num = len(instances)
    total_instances.append(instance_num)

    # Get prediction results: class labels, bounding boxes, masks, scores
    categories = instances.pred_classes
    boxes = instances.pred_boxes
    masks = instances.pred_masks
    scores = instances.scores

    # Encode binary masks into RLE format
    instances.pred_masks_rle = [
        mask_util.encode(np.asfortranarray(mask)) for mask in masks]

    # Decode 'counts'
    for rle in instances.pred_masks_rle:
        rle['counts'] = rle['counts'].decode('utf-8')

    instances.remove('pred_masks')

    for i in range(instance_num):
        pred = {}
        pred['image_id'] = id
        bbox = instances.pred_boxes[i].tensor.numpy().tolist()[0]
        pred['bbox'] = bbox
        pred['score'] = float(scores[i])  # Confidence score
        pred['category_id'] = int(categories[i])
        # RLE-encoded segmentation
        pred['segmentation'] = instances.pred_masks_rle[i]
        prediction.append(pred)

with open("test-results.json", "w") as f:
    json.dump(prediction, f)
os.system(f"zip -j {datestr}.zip test-results.json")


print(total_instances)
print("Total instances: ", sum(total_instances))
print("Finised.")
