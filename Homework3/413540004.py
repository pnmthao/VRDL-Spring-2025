import os
import yaml
import json
import argparse
from datetime import datetime
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, detection_utils
from detectron2.data.build import build_detection_train_loader
from detectron2 import model_zoo

import copy
import torch
import numpy as np
import detectron2.data.transforms as T

parser = argparse.ArgumentParser()

parser.add_argument('-e', "--epochs", type=int, default=20)
parser.add_argument('-l', "--lr", type=float, default=0.01)
parser.add_argument('-m', "--model_name", type=str,
                    default="mask_rcnn_R_101_DC5_3x")
parser.add_argument('-b', "--batch_size", type=int, default=8)

args = parser.parse_args()
print(args)

setup_logger()
DatasetCatalog.register("dataset_train", lambda: load_data())


def load_data():
    with open('hw3-data/train.json', 'r') as f:
        data = json.load(f)
    return data


cfg = get_cfg()
model_path = f'COCO-InstanceSegmentation/{args.model_name}.yaml'
cfg.merge_from_file(model_zoo.get_config_file(model_path))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)

cfg.DATASETS.TRAIN = ("dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 8
cfg.SOLVER.IMS_PER_BATCH = args.batch_size
cfg.SOLVER.CHECKPOINT_PERIOD = 750
cfg.SOLVER.BASE_LR = args.lr
cfg.SOLVER.MAX_ITER = 2500
cfg.SOLVER.WARMUP_FACTOR = 1.0 / 10000
cfg.SOLVER.WARMUP_ITERS = 90
cfg.SOLVER.GAMMA = 0.75
cfg.SOLVER.STEPS = tuple([500, 750, 1000, 1500, 2000])

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # 4 classes (1-4) + 1 background
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2

cfg.INPUT.MIN_SIZE_TEST = 1000
cfg.INPUT.MAX_SIZE_TEST = 1333
cfg.TEST.EVAL_PERIOD = 0
cfg.TEST.DETECTIONS_PER_IMAGE = 800
cfg.TEST.AUG["ENABLED"] = True
cfg.TEST.AUG.MIN_SIZES = (1500, 1600, 1700)

datestr = f"{datetime.now().strftime('%Y%m%d__%H%M%S')}"
output_path = f"tensorboard/{args.model_name}/bs{args.batch_size}_lr{args.lr}/{datestr}"

cfg.OUTPUT_DIR = output_path
print(cfg)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
with open(f'{cfg.OUTPUT_DIR}/config.yaml', 'w') as fp:
    yaml.dump(cfg, fp, default_flow_style=False)


class Trainer(DefaultTrainer):  # SGD default
    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        dataloader = build_detection_train_loader(cfg, mapper=mapper)
        return dataloader


def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = detection_utils.read_image(dataset_dict['file_name'], format='BGR')
    aug_input = T.StandardAugInput(image)
    aug_transform = [
        T.RandomBrightness(0.9, 1.1),
        T.RandomCrop('relative', (0.5, 0.5)),
        T.ResizeShortestEdge(
            short_edge_length=608,
            max_size=800,
            sample_style='choice'),
        T.RandomFlip(prob=0.5)
    ]
    transforms = aug_input.apply_augmentations(aug_transform)
    image = aug_input.image
    image_shape = image.shape[:2]
    dataset_dict['image'] = torch.as_tensor(
        np.ascontiguousarray(image.transpose(2, 0, 1)))

    annos = [
        detection_utils.transform_instance_annotations(
            annotation,
            transforms,
            image_shape)
        for annotation in dataset_dict.pop('annotations')
    ]

    instances = detection_utils.annotations_to_instances(
        annos, image_shape, mask_format='bitmask'
    )

    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
    dataset_dict["instances"] = detection_utils.filter_empty_instances(
        instances)

    return dataset_dict


trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print("Finised.")
