import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import cv2

class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, transform_target=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.transform_target = transform_target

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            # Read using OpenCV
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError("cv2.imread returned None (possibly corrupt image)")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image = Image.fromarray(image) 
        except Exception as e:
            print(f"[WARNING] Failed to load image {img_path}: {e}")
            return None


        if self.transform:
            image = self.transform(image)
        if self.transform_target:
            label = self.transform_target

        return image, label
