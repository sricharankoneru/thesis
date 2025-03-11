import os
import random
import numpy as np
import rasterio

import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

#from torch.optim.lr_scheduler import StepLR
#from torch.optim.lr_scheduler import ExponentialLR
#from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchmetrics.segmentation import MeanIoU
import torchvision.ops as ops

import wandb


class DeepLab(pl.LightningModule):
    def __init__(self, arch, encoder_name, encoder_weights, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=out_classes,  **kwargs
        )
        self.metric = MeanIoU(num_classes = 2, include_background = True, per_class = False)
        self.loss_fn = lambda logits, targets: ops.sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean")
        self.logged_image = False
        self.random_batch_idx = None
        self.output_dir = os.path.join(os.getcwd(), 'jupyter/predictions/resnet-101-B8-35/predict')

    def training_step(self, batch, batch_idx):
        loss, iou = self._shared_eval_step(batch, batch_idx)
        metrics = {"loss": loss, "iou": iou}
        self.log_dict(metrics, on_epoch = True, prog_bar = True, logger = True)
        return metrics

    def validation_step(self, batch, batch_idx):
        loss, iou = self._shared_eval_step(batch, batch_idx) 
        metrics = {"val_loss": loss, "val_iou": iou}
        self.log_dict(metrics, on_epoch = True, prog_bar = True, logger= True)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)

        pred_prob = torch.nn.functional.sigmoid(y_hat)
        pred_binary = (pred_prob > 0.5).int()
        y_binary = y.int()

        miou = self.metric(pred_binary, y_binary)

        if not self.logged_image and batch_idx == self.random_batch_idx:
            rand_idx = random.randint(0, len(x)-1)
            x_img = x[rand_idx].cpu().detach().numpy()
            y_img = y[rand_idx].cpu().detach().numpy()
            y_hat_img = pred_binary[rand_idx].cpu().detach().numpy()

            x_img = np.transpose(x_img, (1, 2, 0))
            y_img = np.squeeze(y_img, axis=0)
            y_hat_img = np.squeeze(y_hat_img, axis=0)

            wandb.log({
                'input_image': wandb.Image(x_img, caption="Input Image"),
                'ground_truth': wandb.Image(y_img, caption="Ground Truth Mask"),
                'predicted_mask': wandb.Image(y_hat_img, caption="Predicted Mask")
            })
            self.logged_image = True

        return loss, miou

    def on_validation_epoch_start(self):
        self.logged_image = False
        self.random_batch_idx = random.randint(0, self.trainer.num_val_batches[0] - 1)  # Random batch index

    def test_step(self, batch, batch_idx):
        x, y, metas, image_names = batch
        y_hat = self.model(x)

        loss = self.loss_fn(y_hat, y)
        pred_prob = torch.nn.functional.sigmoid(y_hat)

        pred_binary = (pred_prob > 0.5).int()
        y_binary = y.int()

        miou = self.metric(pred_binary, y_binary)

        metrics = {"loss": loss, "iou": miou}
        self.log_dict(metrics, on_epoch = True, prog_bar = True, logger = True, batch_size = x.shape[0])

        binary_image = pred_binary.cpu().numpy()
        pred_mask = (binary_image * 255).astype(np.uint8)

        for i in range(pred_mask.shape[0]):
            pred = pred_mask[i, 0, :, :]  # Assuming (B, 1, H, W) shape for prediction
            meta = metas[i]  # Metadata for the corresponding image

            # Define output file path
            output_filename = image_names[i].replace("input_", "mask_")
            output_filepath = os.path.join(self.output_dir, output_filename)

            with rasterio.open(
                output_filepath, 'w', 
                driver='GTiff', 
                height=pred.shape[0], 
                width=pred.shape[1], 
                count=1, 
                dtype=pred.dtype, 
                crs=meta['crs'], 
                transform=meta['transform']
            ) as dst:
                dst.write(pred, 1)  # Write the mask to the first band

        return metrics
    
    def predict_step(self, batch, batch_idx):
        x, metas, image_names = batch
        y_hat = self.model(x)

        pred_prob = torch.nn.functional.sigmoid(y_hat)
        pred_binary = (pred_prob > 0.5).int()
        binary_image = pred_binary.cpu().numpy()
        pred_mask = (binary_image * 255).astype(np.uint8)

        for i in range(pred_mask.shape[0]):
            pred = pred_mask[i, 0, :, :]  # Assuming (B, 1, H, W) shape for prediction
            meta = metas[i]  # Metadata for the corresponding image

            # Define output file path
            output_filename = image_names[i].replace("input_", "mask_")
            output_filepath = os.path.join(self.output_dir, output_filename)

            with rasterio.open(
                output_filepath, 'w', 
                driver='GTiff', 
                height=pred.shape[0], 
                width=pred.shape[1], 
                count=1, 
                dtype=pred.dtype, 
                crs=meta['crs'], 
                transform=meta['transform']
            ) as dst:
                dst.write(pred, 1)  # Write the mask to the first band

        return pred_mask


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        #scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        #scheduler = ExponentialLR(optimizer, gamma=0.95)
        ##scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, min_lr=1e-6, patience=3)
        
        ##return {
            ##'optimizer': optimizer,
            ##'lr_scheduler': {
                ##'scheduler': scheduler,
                ##'monitor': 'val_loss',  # metric to monitor for plateau
            #}
        #}
    
        return optimizer#], [scheduler]