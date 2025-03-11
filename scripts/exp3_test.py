import os
import torch
import rasterio
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from exp1_model import DeepLab

class SegmentationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_folder = os.path.join(root_dir, 'tiles')
        self.image_files = os.listdir(self.image_folder)
        self.mask_folder = os.path.join(root_dir, 'masks')        

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)
        mask_name = image_name.replace('input_', 'mask_')
        mask_path = os.path.join(self.mask_folder, mask_name)

        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3, 4]) 
            meta = src.meta 

        # Compute NDVI: (NIR - Red) / (NIR + Red)
        nir = image[3].astype(np.float32)  # NIR channel (4th band)
        red = image[2].astype(np.float32)  # Red channel (3rd band)
        ndvi = (nir - red) / (nir + red + 1e-8)  # Add small constant to avoid division by zero

        # Replace the 4th channel with NDVI
        image_with_ndvi = np.stack([image[0], image[1], image[2], ndvi], axis=0)

        # Normalize the image (excluding NDVI) to [0, 1]
        image_normalized = image_with_ndvi.astype(np.float32)
        image_normalized[:3] /= 255.0  # Normalize only the RGB channels
        
        image_tensor = torch.from_numpy(image_normalized).float()

        with rasterio.open(mask_path) as src:
            mask = src.read([1])

        mask_tensor = torch.from_numpy(mask).float()
        
        return image_tensor, mask_tensor, meta, image_name
    

def custom_collate(batch):
    # Unzip the batch items
    images, masks, metas, image_names = zip(*batch)
    
    # Stack images and masks (assuming they are tensors)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    
    # Collect metadata in a list (assuming it's a dictionary or similar object)
    metas = list(metas)

    image_names = list(image_names)
    
    return images, masks, metas, image_names

curr_path = os.getcwd()
test_path = os.path.join(curr_path, 'Dataset/test')

test_dataset = SegmentationDataset(test_path)

test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8, collate_fn=custom_collate)

checkpoint_path = os.path.join(curr_path, 'Thesis/ki2yflmb/checkpoints/Xception71-best-checkpoint-C4-B8-NDVI-WA.ckpt')
checkpoint = torch.load(checkpoint_path)  

model = DeepLab("DeepLabV3Plus", "tu-xception71", in_channels=4, out_classes=1)

model_state_dict = checkpoint['state_dict']
model.load_state_dict(model_state_dict)


trainer = pl.Trainer(
    accelerator = 'gpu'
)

trainer.test(model, test_dataloader)