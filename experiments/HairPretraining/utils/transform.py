import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import read_image
import torch
import torchvision.transforms as T


positive_transform = T.Compose([
    T.RandomRotation(degrees=(-15, 15)),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
])

negative_transform = T.Compose([     # Zoom nhẹ hơn positive
    T.RandomHorizontalFlip(p=0.5),                     # Flip như positive
    T.ColorJitter(brightness=0.1, contrast=0.1,         # Thay đổi màu rất nhẹ
                  saturation=0.1, hue=0.02),
])


def get_train_transform(size, mean, std):
    """
    Trả về chuỗi transform dùng cho ảnh huấn luyện.
    Args:
        opt: đối tượng chứa các tham số như opt.size, opt.mean, opt.std
    Returns:
        train_transform: torchvision.transforms.Compose
    """
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        #transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.Resize(size),
        transforms.ToTensor(),
        normalize
    ])

    return train_transform

def get_test_transform(size, mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        normalize
    ])

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

class PositiveMaskingTransform:
    """
    Custom transform to apply masking on a batch of positive images (tensors) after SimCLR transforms.
    Assumes input is a torch.Tensor (B, C, H, W) in [0,1] range, with black background (0).
    Masks 10-20% of hair-containing patches by setting them to 0, independently for each image in the batch.
    
    Args:
    - patch_size: int, size of each patch (e.g., 32 for 32x32 patches)
    - mask_ratio_range: tuple, range for random mask ratio (e.g., (0.1, 0.2) for 10-20%)
    - threshold: float, mean pixel value to consider a patch as containing hair
    """
    def __init__(self, patch_size=32, mask_ratio_range=(0.1, 0.2), threshold=0.01):
        self.patch_size = patch_size
        self.mask_ratio_range = mask_ratio_range
        self.threshold = threshold

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply masking to the input batch tensor.
        
        Args:
        - images: torch.Tensor (B, C, H, W), the batch of positive images after SimCLR transforms
        
        Returns:
        - masked_images: torch.Tensor (B, C, H, W), the masked versions
        """
        if not isinstance(images, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        B, C, H, W = images.shape
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w
        
        # Unfold to extract patches: (B, C, num_h, num_w, patch_h, patch_w)
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        
        # Reshape for mean calculation: (B, C, num_patches, patch_h, patch_w)
        patches = patches.contiguous().view(B, C, num_patches, self.patch_size, self.patch_size)
        
        # Identify hair patches: mean > threshold, averaged over C, patch_h, patch_w
        patch_means = patches.mean(dim=[1, 3, 4])  # (B, num_patches)
        hair_masks = patch_means > self.threshold  # (B, num_patches)
        
        masked_images = images.clone()
        
        for b in range(B):
            hair_indices = torch.nonzero(hair_masks[b]).squeeze(-1)  # Indices for this batch item
            if len(hair_indices) == 0:
                continue  # No hair, skip
            
            # Random mask ratio for this image
            mask_ratio = torch.empty(1, device=images.device).uniform_(*self.mask_ratio_range).item()
            num_mask = int(len(hair_indices) * mask_ratio)
            if num_mask == 0:
                continue
            
            # Random select indices to mask
            mask_indices = hair_indices[torch.randperm(len(hair_indices), device=images.device)[:num_mask]]
            
            # Mask selected patches (set to 0)
            for idx in mask_indices:
                ph = idx // num_patches_w
                pw = idx % num_patches_w
                masked_images[b, :, ph*self.patch_size:(ph+1)*self.patch_size, pw*self.patch_size:(pw+1)*self.patch_size] = 0.0
        
        return masked_images

# Example usage in SimCLR pipeline:
# Assume simclr_transform is your Compose for SimCLR (returns tensor)
# In data loader or forward:
# positives = simclr_transform(batch_pil_images)  # torch.Tensor (B, C, H, W), e.g., [256, 3, 224, 224]
# masking_transform = PositiveMaskingTransform()
# masked_positives = masking_transform(positives)
# Example usage in SimCLR pipeline:
# Assume simclr_transform is your Compose for SimCLR (returns tensor)
# In data loader or forward:
# positive = simclr_transform(pil_image)  # torch.Tensor (C, H, W)
# masking_transform = PositiveMaskingTransform()
# masked_positive = masking_transform(positive)