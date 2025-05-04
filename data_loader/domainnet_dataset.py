import os
import torch
import numpy as np
import random
from functools import lru_cache
from data_loader.meta_dataset import MetaDataset, GetDataLoaderDict, CachedMetaDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs.default import domainnet_path

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


domainnet_name_dict = {
    'c': 'clipart',
    'i': 'infograph',
    'p': 'painting',
    'q': 'quickdraw',
    'r': 'real',
    's': 'sketch',
}

# Dataset split type mapping - now only train and test
split_dict = {
    'train': 'train',
    'test': 'test',
}


class DomainNet_SingleDomain():
    def __init__(self, root_path, domain_name='c', split='test', train_transform=None,
                 use_cache=True, cache_size=1000, val_ratio=0.2, seed=42):
        # Domain name validation
        if domain_name in domainnet_name_dict.keys():
            self.domain_name = domainnet_name_dict[domain_name]
            self.domain_label = list(domainnet_name_dict.keys()).index(domain_name)
        else:
            raise ValueError('domain_name should be in c i p q r s')

        self.root_path = root_path
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed

        # Handle validation split specially
        if self.split == 'val':
            # For validation, we'll use a portion of training data_loader
            self.split_file = os.path.join(
                root_path,
                'splits',
                f'{self.domain_name}_train.txt'
            )
            train_imgs, train_labels = self._read_txt_cached(self.split_file, self.root_path)
            imgs, labels = self._split_for_validation(train_imgs, train_labels)
        else:
            # For train and test, use the official files
            self.split_file = os.path.join(
                root_path,
                'splits',
                f'{self.domain_name}_{split_dict[self.split]}.txt'
            )
            imgs, labels = self._read_txt_cached(self.split_file, self.root_path)

            # If it's training data_loader, remove validation portion
            if self.split == 'train':
                imgs, labels = self._split_for_training(imgs, labels)

        self.transform = train_transform if train_transform is not None else transform_test

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

    def _split_for_validation(self, all_imgs, all_labels):
        # Create paired data_loader
        paired_data = list(zip(all_imgs, all_labels))

        # Shuffle with fixed seed for reproducibility
        random.seed(self.seed)
        random.shuffle(paired_data)

        # Calculate split index - take last val_ratio portion for validation
        split_idx = int(len(paired_data) * (1 - self.val_ratio))

        # Get only validation portion
        val_data = paired_data[split_idx:]

        # Unzip back to separate lists
        val_imgs, val_labels = zip(*val_data) if val_data else ([], [])

        return list(val_imgs), list(val_labels)

    def _split_for_training(self, all_imgs, all_labels):
        # Create paired data_loader
        paired_data = list(zip(all_imgs, all_labels))

        # Shuffle with fixed seed for reproducibility
        random.seed(self.seed)
        random.shuffle(paired_data)

        # Calculate split index - take first (1-val_ratio) portion for training
        split_idx = int(len(paired_data) * (1 - self.val_ratio))

        # Get only training portion
        train_data = paired_data[:split_idx]

        # Unzip back to separate lists
        train_imgs, train_labels = zip(*train_data) if train_data else ([], [])

        return list(train_imgs), list(train_labels)


class DomainNet_FedDG():
    def __init__(self, test_domain, batch_size, root_path=domainnet_path, use_cache=False,
                 cache_size=1000, val_ratio=0.2, seed=42):
        self.batch_size = batch_size
        self.domain_list = list(domainnet_name_dict.keys())
        self.test_domain = test_domain
        self.val_ratio = val_ratio
        self.seed = seed

        self.site_dataset_dict = {}
        self.site_dataloader_dict = {}

        # Initialize datasets for all domains
        for domain_name in self.domain_list:
            self.site_dataloader_dict[domain_name], self.site_dataset_dict[domain_name] = \
                self._create_single_site(domain_name, root_path, self.batch_size, use_cache, cache_size)

        self.test_dataset = self.site_dataset_dict[self.test_domain]['test']
        self.test_dataloader = self.site_dataloader_dict[self.test_domain]['test']

    def _create_single_site(self, domain_name, root_path, batch_size=16, use_cache=False, cache_size=1000):
        dataset_dict = {
            'train': DomainNet_SingleDomain(
                root_path=root_path,
                domain_name=domain_name,
                split='train',
                train_transform=transform_train,
                use_cache=use_cache,
                cache_size=cache_size,
                val_ratio=self.val_ratio,
                seed=self.seed
            ).dataset,
            'val': DomainNet_SingleDomain(
                root_path=root_path,
                domain_name=domain_name,
                split='val',
                use_cache=use_cache,
                cache_size=cache_size,
                val_ratio=self.val_ratio,
                seed=self.seed
            ).dataset,
            'test': DomainNet_SingleDomain(
                root_path=root_path,
                domain_name=domain_name,
                split='test',
                use_cache=use_cache,
                cache_size=cache_size,
                val_ratio=self.val_ratio,
                seed=self.seed
            ).dataset,
        }

        # Get optimized dataloaders
        dataloader_dict = GetDataLoaderDict(dataset_dict, batch_size)
        return dataloader_dict, dataset_dict

    def GetData(self):
        return self.site_dataloader_dict, self.site_dataset_dict
