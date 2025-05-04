import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import OrderedDict
from functools import lru_cache


def GetDataLoaderDict(dataset_dict, batch_size):
    """Create optimized DataLoader dictionary from dataset dictionary."""
    # Determine optimal number of workers based on CPU cores
    num_workers = 4  # num_workers for each domain and split(train/val/test)

    dataloader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 4,
    }

    # Create DataLoader dictionary with optimized settings
    dataloader_dict = {}
    for dataset_name in dataset_dict.keys():
        if 'train' in dataset_name:
            dataloader_dict[dataset_name] = DataLoader(
                dataset_dict[dataset_name],
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                **dataloader_kwargs
            )
        else:
            dataloader_dict[dataset_name] = DataLoader(
                dataset_dict[dataset_name],
                batch_size=batch_size * 2,  # Larger batch size for validation/testing
                shuffle=False,
                drop_last=False,
                **dataloader_kwargs
            )

    return dataloader_dict


class MetaDataset(Dataset):
    """Basic dataset for RGB images with domain labels."""

    def __init__(self, imgs, labels, domain_label, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.domain_label = domain_label
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_class_label = self.labels[index]

        # Use OpenCV for faster image loading
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            # Apply albumentations transform
            transformed = self.transform(image=img)
            img = transformed["image"]

        return img, img_class_label, self.domain_label

    def __len__(self):
        return len(self.imgs)


class CachedMetaDataset(Dataset):
    """Memory-efficient caching dataset implementation."""

    def __init__(self, imgs, labels, domain_label, transform=None, cache_size=1000):
        self.imgs = imgs
        self.labels = labels
        self.domain_label = domain_label
        self.transform = transform
        self.cache_size = min(cache_size, len(imgs))

        # Image cache using OrderedDict with LRU behavior
        self.img_cache = OrderedDict()

        # Preload most frequent images for efficiency
        if self.cache_size > 0:
            print(f"Preloading {self.cache_size} images into memory cache...")
            # Prioritize first images as they may be accessed frequently
            for i in range(min(self.cache_size, len(self.imgs))):
                self._load_and_cache_image(i)

    def _load_and_cache_image(self, index):
        """Load an image and store in cache."""
        if index in self.img_cache:
            return self.img_cache[index]

        # Load image with OpenCV (faster than PIL)
        img_path = self.imgs[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Update cache - remove oldest item if at capacity
        if len(self.img_cache) >= self.cache_size:
            self.img_cache.popitem(last=False)  # Remove oldest item

        self.img_cache[index] = img
        return img

    def __getitem__(self, index):
        img_class_label = self.labels[index]

        # Try to get from cache, or load if not present
        try:
            img = self._load_and_cache_image(index)
        except Exception as e:
            print(f"Error loading image {self.imgs[index]}: {e}")
            # Provide fallback/default image in case of error
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]

        return img, img_class_label, self.domain_label

    def __len__(self):
        return len(self.imgs)
