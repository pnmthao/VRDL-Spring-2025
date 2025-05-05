import cv2
import json
import numpy as np
from glob import glob
import skimage.io as sio
from pycocotools import mask as mask_util

annotations = []

for idx, folder_name in enumerate(glob(f"hw3-data/train/*")):
    img_path = f"{folder_name}/image.tif"
    img_height, img_width, _ = cv2.imread(img_path).shape

    img_annot = []
    for ins_path in glob(f"{folder_name}/class*.tif"):
        instance = sio.imread(ins_path)

        for mask in np.unique(instance):
            if mask == 0:
                continue

            rle_encode = mask_util.encode(np.asfortranarray(instance == mask))
            rle_encode['counts'] = rle_encode['counts'].decode('utf-8')

            img_annot.append({
                'category_id': int(ins_path.split('class')[-1][0]),
                'segmentation': rle_encode,
                'bbox_mode': 1,  # BoxMode.XYWH_ABS
                'bbox': mask_util.toBbox(rle_encode).tolist()  # [x, y, w, h]
            })

    annotations.append({
        'image_id': idx+1,
        'file_name': img_path,
        'height': img_height,
        'width': img_width,
        'annotations': img_annot
    })

with open(f'hw3-data/train__annot.json', 'w') as f:
    json.dump(annotations, f)
