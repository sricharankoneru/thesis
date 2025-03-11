import os
import torch
import rasterio
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

#from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from model import DeepLab
import argparse

# Argument parser to handle batch_size and backbone model from the shell script
parser = argparse.ArgumentParser(description="Training Script for DeepLabV3+")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation")
parser.add_argument("--backbone", type=str, default="resnet101", help="Backbone model (e.g., resnet50, resnet101)")

args = parser.parse_args()

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.image_folder = os.path.join(root_dir, 'tiles')
        self.image_files = os.listdir(self.image_folder)
        self.mask_folder = os.path.join(root_dir, 'masks') 
        self.transforms = transforms    

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)
        mask_name = image_name.replace('input_', 'mask_')
        mask_path = os.path.join(self.mask_folder, mask_name)

        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3])

        image_normalized = image.astype(np.float32) / 255.0
        
        image_tensor = torch.from_numpy(image_normalized).float()

        with rasterio.open(mask_path) as src:
            mask = src.read([1])

        mask_tensor = torch.from_numpy(mask).float()

        if self.transforms:
            image = image_tensor.permute(1, 2, 0).numpy()  # (512, 512, 4)
            mask = mask_tensor.squeeze().numpy()
            augmented = self.transforms(image=image, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask'].unsqueeze(0)

        return image_tensor, mask_tensor


train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),          # Horizontal flip
    A.VerticalFlip(p=0.5),            # Vertical flip
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ToTensorV2()                      # Convert to tensor
], additional_targets={'mask': 'mask'})  # Ensure the mask is handled correctly

# Define validation transformations
val_transforms = A.Compose([
    ToTensorV2()                      # Convert to tensor only
], additional_targets={'mask': 'mask'})

curr_path = os.getcwd()
train_dataset_path = os.path.join(curr_path, 'Dataset/Dataset/train')
train_dataset = SegmentationDataset(train_dataset_path, transforms = train_transforms)

val_dataset_path = os.path.join(curr_path, 'Dataset/Dataset/val')
val_dataset = SegmentationDataset(val_dataset_path, transforms = val_transforms)

#indices = list(range(len(dataset)))
#train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

#train_subset = torch.utils.data.Subset(dataset, train_indices)
#val_subset = torch.utils.data.Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=6)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=6)

early_stop_callback = EarlyStopping(
    monitor = 'val_iou',
    patience = 10,
    min_delta = 0.001,
    mode = 'max',
    verbose = True
)

checkpoint_callback = ModelCheckpoint(
    monitor = 'val_iou',
    mode = 'max',
    save_top_k = 1,
    verbose = True,
    filename = 'ResNet-retrained-checkpoint',
    save_weights_only = True
)

wandblogger = WandbLogger(project = "Thesis")

model = DeepLab("DeepLabV3Plus", args.backbone, "imagenet", in_channels=3, out_classes=1)

trainer = pl.Trainer(
    logger = wandblogger,
    callbacks=[early_stop_callback],
    devices = 1,
    min_epochs = 50,
    max_epochs = 100,
    accelerator = 'gpu'
)

trainer.fit(
    model,
    train_loader,
    val_loader
)