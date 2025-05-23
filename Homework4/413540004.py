import piq
import torch
import torch.nn as nn
import torch.optim as optim

from model import PromptIR
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from utils.dataset_utils import PromptDataset
from utils.utils import AverageMeter, compute_psnr
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.schedulers import LinearWarmupCosineAnnealingLR

import argparse


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()  # MAE
        # self.loss_fn = nn.MSELoss() # L2

        self.ssim_loss_fn = piq.SSIMLoss(data_range=1.)
        self.psnr = AverageMeter()

    def restoration_loss(self, restored_patch, clean_patch):
        # import piq
        # MAE(L1)/MSE(L2) ensures pixel accuracy.

        # SSIM promotes structural and perceptual fidelity.

        # Fine-tune the weight (0.1) for SSIM depending on your use case.

        pixel_loss = self.loss_fn(restored_patch, clean_patch)
        ssim_loss = self.ssim_loss_fn(torch.clamp(restored_patch, 0, 1),
                                      torch.clamp(clean_patch, 0, 1))

        total_loss = pixel_loss + 0.1 * ssim_loss  # Adjust 0.1 based on results
        return total_loss, ssim_loss

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        (degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        # loss = self.loss_fn(restored, clean_patch)
        loss, ssim_loss = self.restoration_loss(restored, clean_patch)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_ssim_loss", ssim_loss)
        return loss

    def validation_step(self, batch, batch_idx):

        (degrad_patch, clean_patch) = batch

        restored = self.net(degrad_patch)
        # loss = self.loss_fn(restored, clean_patch)
        loss, ssim_loss = self.restoration_loss(restored, clean_patch)

        temp_psnr, N = compute_psnr(restored, clean_patch)
        self.psnr.update(temp_psnr, N)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_ssim_loss", ssim_loss, prog_bar=True, sync_dist=True)
        return temp_psnr

    def on_validation_epoch_end(self):
        self.log('val_psnr', self.psnr.avg, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.psnr.reset()

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()[0]
        self.logger.experiment.add_scalar('lr', lr, self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=15, max_epochs=150)

        return [optimizer], [scheduler]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Parameters

    parser.add_argument('--epochs', type=int, default=200,
                        help='maximum number of epochs to train the total model.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size to use per GPU")
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate of encoder.')
    parser.add_argument('--de_type', nargs='+', default=['derain', 'desnow'],
                        help='which type of degradations is training and testing for.')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='patchsize of input.')
    parser.add_argument('--num_workers', type=int,
                        default=4, help='number of workers.')
    parser.add_argument("--ckpt_dir", type=str, default="train_ckpt",
                        help="Name of the Directory where the checkpoint is to be saved")

    args = parser.parse_args()

    logger = TensorBoardLogger(save_dir="logs/")

    trainset = PromptDataset(args, mode="train")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir, save_top_k=1, monitor='val_psnr', mode='max')

    trainloader = DataLoader(trainset, batch_size=args.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=args.num_workers)

    validset = PromptDataset(args, mode="valid")
    validloader = DataLoader(validset, batch_size=1,
                             pin_memory=True, shuffle=False, num_workers=args.num_workers)

    model = PromptIRModel()

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="gpu", devices=1,
                         strategy="ddp_find_unused_parameters_true", logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model=model,
                train_dataloaders=trainloader,
                val_dataloaders=validloader)
    print("Finised.")
