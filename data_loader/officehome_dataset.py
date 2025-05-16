import os
import torch
import numpy as np
from functools import lru_cache
import random
from collections import defaultdict
from data_loader.meta_dataset import MetaDataset, GetDataLoaderDict, CachedMetaDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs.default import officehome_path

# Using Albumentations for faster data_loader augmentation
transform_train = A.Compose([
    A.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
    A.HueSaturationValue(hue_shift_limit=0.4, sat_shift_limit=0.4, val_shift_limit=0.4),
    A.ToGray(p=0.1),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

transform_test = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Office-Home domain name mapping
officehome_name_dict = {
    'p': 'Product',
    'a': 'Art',
    'c': 'Clipart',
    'r': 'Real_World',
}

# Dataset split type mapping
split_dict = {
    'train': 'train',
    'val': 'val',
    'test': 'test',
}


def generate_dataset_splits(root_path):
    """
    Generate dataset splits for OfficeHome dataset if they don't exist.
    This creates train, validation, and test split files for each domain.

    Args:
        root_path: Root path of the dataset
    """
    # Create splits directory if it doesn't exist
    splits_dir = os.path.join(root_path, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    # Process each domain
    for domain_code, domain_name in officehome_name_dict.items():
        domain_path = os.path.join(root_path, domain_name)
        if not os.path.isdir(domain_path):
            print(f"Warning: Domain directory {domain_path} not found. Skipping.")
            continue

        print(f"Processing domain: {domain_name}")

        # Check if split files already exist
        all_splits_exist = all(
            os.path.exists(os.path.join(splits_dir, f"{domain_name}_{split}.txt"))
            for split in split_dict.values()
        )

        if all_splits_exist:
            print(f"Split files for {domain_name} already exist. Skipping.")
            continue

        # Dictionary to store images by class
        class_images = defaultdict(list)

        # Find all class directories
        class_dirs = [d for d in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, d))]
        class_dirs.sort()  # Ensure consistent class ordering

        # Create class to label mapping
        class_to_label = {cls: idx for idx, cls in enumerate(class_dirs)}

        # Find all images in each class
        for class_name in class_dirs:
            class_path = os.path.join(domain_path, class_name)

            # Find all images in this class directory
            for root, _, files in os.walk(class_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Get relative path for image
                        img_path = os.path.join(domain_name, class_name, file)
                        class_images[class_name].append(img_path)

        # Create a list of all images with their labels
        all_images = []
        for class_name, images in class_images.items():
            label = class_to_label[class_name]
            for img_path in images:
                all_images.append((img_path, label))

        # Shuffle the images
        random.seed(42)  # For reproducibility
        random.shuffle(all_images)

        train_split = int(len(all_images) * 0.7)

        train_images = all_images[:train_split]
        val_images = all_images[train_split:]
        test_images = all_images
        # Write split files
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }

        for split_name, images in splits.items():
            split_file = os.path.join(splits_dir, f"{domain_name}_{split_name}.txt")
            with open(split_file, 'w') as f:
                for img_path, label in images:
                    f.write(f"{img_path} {label}\n")

    print("Split generation complete!")


class OfficeHome_SingleDomain():
    def __init__(self, root_path, domain_name='a', split='test', train_transform=None,
                 use_cache=True, cache_size=1000):
        # Domain name validation
        if domain_name in officehome_name_dict.keys():
            self.domain_name = officehome_name_dict[domain_name]
            self.domain_label = list(officehome_name_dict.keys()).index(domain_name)
        else:
            raise ValueError('domain_name should be in a p c r')

        self.root_path = root_path
        self.split = split

        # Create splits directory path
        splits_dir = os.path.join(root_path, 'splits')

        # Generate splits if they don't exist
        if not os.path.exists(splits_dir) or not os.path.exists(
                os.path.join(splits_dir, f"{self.domain_name}_{split_dict[self.split]}.txt")):
            print(f"Split files not found. Generating dataset splits...")
            generate_dataset_splits(root_path)

        self.split_file = os.path.join(
            root_path,
            'splits',
            f'{self.domain_name}_{split_dict[self.split]}.txt'
        )

        self.transform = train_transform if train_transform is not None else transform_test

        # Preload image paths and labels
        imgs, labels = self._read_txt_cached(self.split_file, self.root_path)

        # Choose dataset implementation based on caching preference
        if use_cache:
            self.dataset = CachedMetaDataset(
                imgs, labels, self.domain_label, self.transform, cache_size=cache_size
            )
        else:
            self.dataset = MetaDataset(imgs, labels, self.domain_label, self.transform)

    @staticmethod
    @lru_cache(maxsize=0)  # Cache results to avoid repeated file parsing
    def _read_txt_cached(txt_path, root_path):
        imgs = []
        labels = []
        with open(txt_path, 'r') as f:
            txt_component = f.readlines()

        # Preprocess all file paths at once
        for line_txt in txt_component:
            line_txt = line_txt.strip().split(' ')
            imgs.append(os.path.join(root_path, line_txt[0]))
            labels.append(int(line_txt[1]))

        return imgs, labels


class OfficeHome_FedDG():
    def __init__(self, test_domain, batch_size, root_path=officehome_path, use_cache=False, cache_size=250):
        self.batch_size = batch_size
        self.domain_list = list(officehome_name_dict.keys())
        self.test_domain = test_domain

        self.site_dataset_dict = {}
        self.site_dataloader_dict = {}

        # Initialize datasets for all domains
        for domain_name in self.domain_list:
            self.site_dataloader_dict[domain_name], self.site_dataset_dict[domain_name] = \
                self._create_single_site(domain_name, root_path, self.batch_size, use_cache, cache_size)

        self.test_dataset = self.site_dataset_dict[self.test_domain]['test']
        self.test_dataloader = self.site_dataloader_dict[self.test_domain]['test']

    def _create_single_site(self, domain_name, root_path, batch_size=16, use_cache=False, cache_size=250):
        dataset_dict = {
            'train': OfficeHome_SingleDomain(
                root_path=root_path,
                domain_name=domain_name,
                split='train',
                train_transform=transform_train,
                use_cache=use_cache,
                cache_size=cache_size
            ).dataset,
            'val': OfficeHome_SingleDomain(
                root_path=root_path,
                domain_name=domain_name,
                split='val',
                use_cache=use_cache,
                cache_size=cache_size
            ).dataset,
            'test': OfficeHome_SingleDomain(
                root_path=root_path,
                domain_name=domain_name,
                split='test',
                use_cache=use_cache,
                cache_size=cache_size
            ).dataset,
        }

        # Get optimized dataloaders
        dataloader_dict = GetDataLoaderDict(dataset_dict, batch_size)
        return dataloader_dict, dataset_dict

    def GetData(self):
        return self.site_dataloader_dict, self.site_dataset_dict