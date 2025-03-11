import os
import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from train import SegmentationDataset  # Make sure to import your SegmentationDataset class

# Define the paths to your dataset
curr_path = os.getcwd()
train_dataset_path = os.path.join(curr_path, 'Dataset/Dataset/train')
val_dataset_path = os.path.join(curr_path, 'Dataset/Dataset/val')

# Initialize the dataset
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

train_dataset = SegmentationDataset(train_dataset_path, transforms=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=6)

# 1. Print the shapes of tensors for a batch
for images, masks in train_loader:
    print(f'Batch of images shape: {images.shape}')  # (batch_size, channels, height, width)
    print(f'Batch of masks shape: {masks.shape}')    # (batch_size, channels, height, width)
    break  # Only do this for the first batch

# 2. Visualize the images and masks for the first batch
def visualize(images, masks):
    num_images = min(images.shape[0], 5)  # Display up to 5 images
    fig, axs = plt.subplots(num_images, 2, figsize=(10, num_images * 4))
    
    for i in range(num_images):
        # Convert to numpy for visualization
        img = images[i].permute(1, 2, 0).numpy()  # (H, W, C)
        mask = masks[i].squeeze().numpy()          # (H, W)

        axs[i, 0].imshow(img)
        axs[i, 0].set_title('Image')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(mask, cmap='gray')
        axs[i, 1].set_title('Mask')
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

# Visualize the first batch
visualize(images, masks)

# 3. Run through the entire dataset to check for any loading issues
for images, masks in train_loader:
    # Just run through the dataset
    print(f'Loaded batch of images and masks with shapes: {images.shape}, {masks.shape}')
