import os
import torch
import rasterio
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from model import DeepLab

class SegmentationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_folder = os.path.join(root_dir, 'predict_1')
        self.image_files = os.listdir(self.image_folder)
  
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)

        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3]) 
            meta = src.meta  

        # Convert to float32
        image_normalized = image.astype(np.float32) / 255.0
        
        image_tensor = torch.from_numpy(image_normalized).float()
       
        return image_tensor, meta, image_name

def custom_collate(batch):
    # Unzip the batch items
    images, metas, image_names = zip(*batch)
    
    images = torch.stack(images, dim=0)
    
    # Collect metadata in a list (assuming it's a dictionary or similar object)
    metas = list(metas)

    image_names = list(image_names)
    
    return images, metas, image_names

curr_path = os.getcwd()
predict_path = os.path.join(curr_path, 'Dataset/predict')

predict_dataset = SegmentationDataset(predict_path)

predict_dataloader = DataLoader(predict_dataset, batch_size=16, shuffle=False, num_workers=8, collate_fn=custom_collate)

checkpoint_path = os.path.join(curr_path, 'Thesis/05vkn26q/checkpoints/Exception-71_B8.ckpt')
checkpoint = torch.load(checkpoint_path)  

model = DeepLab("DeepLabV3Plus", "tu-xception71", "imagenet", in_channels=3, out_classes=1)

model_state_dict = checkpoint['state_dict']
model.load_state_dict(model_state_dict)


trainer = pl.Trainer(
    accelerator = 'gpu'
)

trainer.predict(model, predict_dataloader)