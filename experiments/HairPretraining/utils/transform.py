import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import read_image
import torch
import torchvision.transforms as T


positive_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Zoom nhẹ
    T.RandomHorizontalFlip(p=0.5),               # Lật ngang nhẹ
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
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
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