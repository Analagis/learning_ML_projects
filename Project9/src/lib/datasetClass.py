import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


TRAIN_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    ToTensorV2()
])

class KSLDataset(Dataset):
    def __init__(self, df, images_dir, transform = TRAIN_TRANSFORM):
        self.df = df          
        self.images_dir = images_dir        
        self.transform = transform   

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["img_IDS"]                
        target = row["target"]                   

        img_path = os.path.join(self.images_dir, img_name+".jpg")
        image = cv2.imread(img_path)             
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        target = torch.tensor(target, dtype=torch.float32)
        
        return image, target
