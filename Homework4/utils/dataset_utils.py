import os
import random
import numpy as np
from PIL import Image
from glob import glob

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from utils.utils import random_augmentation, crop_img


class PromptDataset(Dataset):
    def __init__(self, args, mode="train"):
        super(PromptDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.sn_ids = []
        self.de_type = self.args.de_type
        self.mode = mode
        print(self.de_type)

        self.de_dict = {'derain': 0, 'desnow': 1}

        self._init_ids()
        self._merge_ids()

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'desnow' in self.de_type:
            self._init_sn_ids()

        random.shuffle(self.de_type)

    def _init_rs_ids(self):
        temp_ids = glob("hw4-data/train/degraded/rain*")
        if self.mode == "train":
            temp_ids = temp_ids[:int(len(temp_ids) * 0.9)]
        else:
            temp_ids = temp_ids[int(len(temp_ids) * 0.9):]
        self.rs_ids = [{"clean_id": x, "de_type": 0} for x in temp_ids]
        self.rs_ids = self.rs_ids * (5 if self.mode == "train" else 1)

        print("Total Rainy Ids : {}".format(len(self.rs_ids)))

    def _init_sn_ids(self):
        temp_ids = glob("hw4-data/train/degraded/snow*")
        if self.mode == "train":
            temp_ids = temp_ids[:int(len(temp_ids) * 0.9)]
        else:
            temp_ids = temp_ids[int(len(temp_ids) * 0.9):]
        self.sn_ids = [{"clean_id": x, "de_type": 1} for x in temp_ids]
        self.sn_ids = self.sn_ids * (5 if self.mode == "train" else 1)

        print("Total Rainy Ids : {}".format(len(self.sn_ids)))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size,
                        ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size,
                        ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, filename, type="rain"):
        # hw4-data/train/degraded/rain-405.png
        filename = filename.replace("degraded", "clean")
        gt_name = filename.replace(f"{type}-", f"{type}_clean-")

        return gt_name

    def _merge_ids(self):
        self.sample_ids = []

        if "derain" in self.de_type:
            self.sample_ids += self.rs_ids
        if "desnow" in self.de_type:
            self.sample_ids += self.sn_ids

        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        if de_id == 0:  # rain
            degrad_img = crop_img(
                np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(sample["clean_id"], type="rain")
            clean_img = crop_img(
                np.array(Image.open(clean_name).convert('RGB')), base=16)
        elif de_id == 1:  # snow
            degrad_img = crop_img(
                np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(sample["clean_id"], type="snow")
            clean_img = crop_img(
                np.array(Image.open(clean_name).convert('RGB')), base=16)

        degrad_patch, clean_patch = random_augmentation(
            *self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)

        return degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class TestDataset(Dataset):
    def __init__(self, args):
        super(TestDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_degraded_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_degraded_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception(
                    'The input directory does not contain any image files')
            self.degraded_ids += [root + id_ for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print("Total Images : {}".format(name_list))

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(
            np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)

        name = self.degraded_ids[idx].split('/')[-1]
        degraded_img = self.toTensor(degraded_img)

        return name, degraded_img

    def __len__(self):
        return self.num_img
