import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime
import lightning.pytorch as pl
from model import PromptIR
from torch.utils.data import DataLoader
from utils.utils import save_image_tensor
from utils.dataset_utils import TestDataset


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--test_path', type=str,
                        default="hw4-data/test/degraded/", help='save path of test images')
    parser.add_argument('--output_path', type=str,
                        default="output", help='output save path')
    parser.add_argument('--ckpt_dir', type=str,
                        default="train_ckpt", help='checkpoint save path')
    parser.add_argument('--ckpt_name', type=str, default="",
                        help='checkpoint save path')
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    model = PromptIRModel.load_from_checkpoint(
        f"{args.ckpt_dir}/{args.ckpt_name}").cuda()
    model.eval()

    testset = TestDataset(args)
    testloader = DataLoader(testset, batch_size=1,
                            pin_memory=True, shuffle=False, num_workers=4)
    os.makedirs(args.output_path, exist_ok=True)
    with torch.no_grad():
        for (degraded_name, degrad_patch) in tqdm(testloader):
            degrad_patch = degrad_patch.cuda()
            restored = model(degrad_patch)

            save_image_tensor(
                restored, f"{args.output_path}/{degraded_name[0]}")

    output_npz = f"{args.output_path}/pred.npz"

    # Initialize dictionary to hold image arrays
    images_dict = {}

    # Loop through all files in the folder
    for filename in os.listdir(args.test_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(args.test_path, filename)

            # Load image and convert to RGB
            image = Image.open(file_path).convert('RGB')
            img_array = np.array(image)

            # Rearrange to (3, H, W)
            img_array = np.transpose(img_array, (2, 0, 1))

            # Add to dictionary
            images_dict[filename] = img_array

    # Save to .npz file
    np.savez(output_npz, **images_dict)
    print(f"Saved {len(images_dict)} images to {output_npz}")

    zip_path = f"{args.output_path}/{datetime.now().strftime('%Y%m%d__%H%M%S')}.zip"
    os.system(f"zip -j {zip_path} {output_npz}")
    print("Finised.")
