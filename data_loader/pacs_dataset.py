import os
import torch
import numpy as np
from functools import lru_cache
from data_loader.meta_dataset import MetaDataset, GetDataLoaderDict, CachedMetaDataset
from configs.default import pacs_path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Using Albumentations for faster data_loader augmentation
transform_train = A.Compose([
    A.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
    A.HorizontalFlip(),
    # ColorJitter in albumentations needs to be broken down into individual transforms
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
    A.HueSaturationValue(hue_shift_limit=0.4, sat_shift_limit=0.4, val_shift_limit=0.4),
    # Use ToGray instead of RandomGrayscale
    A.ToGray(p=0.1),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

transform_test = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# PACS domain name mapping
pacs_name_dict = {
    'p': 'photo',
    'a': 'art_painting',
    'c': 'cartoon',
    's': 'sketch',
}

# Dataset split type mapping
split_dict = {
    'train': 'train',
    'val': 'crossval',
    'test': 'test',
}


class PACS_SingleDomain():
    def __init__(self, root_path=pacs_path, domain_name='p', split='test', train_transform=None,
                 use_cache=False, cache_size=1000):
        # Domain name validation
        if domain_name in pacs_name_dict.keys():
            self.domain_name = pacs_name_dict[domain_name]
            self.domain_label = list(pacs_name_dict.keys()).index(domain_name)
        else:
            raise ValueError('domain_name should be in p a c s')

        self.root_path = os.path.join(root_path, 'raw_images')
        self.split = split
        self.split_file = os.path.join(
            root_path,
            'Train val splits and h5py files pre-read',
            f'{self.domain_name}_{split_dict[self.split]}_kfold.txt'
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
            labels.append(int(line_txt[1]) - 1)

        return imgs, labels


class PACS_FedDG():
    def __init__(self, test_domain, batch_size, use_cache=False, cache_size=1000):
        self.batch_size = batch_size
        self.domain_list = list(pacs_name_dict.keys())
        self.test_domain = test_domain
        self.train_domain_list = self.domain_list.copy()
        self.train_domain_list.remove(self.test_domain)

        self.site_dataset_dict = {}
        self.site_dataloader_dict = {}

        # Initialize datasets for all domains
        # for domain_name in self.train_domain_list:
        for domain_name in self.domain_list:
            self.site_dataloader_dict[domain_name], self.site_dataset_dict[domain_name] = \
                self._create_single_site(domain_name, self.batch_size, use_cache, cache_size)

        # self.test_dataset = self.site_dataset_dict[self.test_domain]['test']
        # self.test_dataloader = self.site_dataloader_dict[self.test_domain]['test']

    def _create_single_site(self, domain_name, batch_size=16, use_cache=False, cache_size=1000):
        dataset_dict = {
            'train': PACS_SingleDomain(
                domain_name=domain_name,
                split='train',
                train_transform=transform_train,
                use_cache=use_cache,
                cache_size=cache_size
            ).dataset,
            'val': PACS_SingleDomain(
                domain_name=domain_name,
                split='val',
                use_cache=use_cache,
                cache_size=cache_size
            ).dataset,
            'test': PACS_SingleDomain(
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
