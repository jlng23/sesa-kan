# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Union, Tuple
import torchvision.transforms as T
import albumentations as A
import numpy as np
from PIL import Image, ImageFile

class RadiographSexAgeDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        fold_nums: List[int],
        transforms: Union[A.Compose, T.Compose],
        fold_txt_dir: str='splits',
        use_sex: bool = True,
        use_age: bool = True,
        use_1st: bool = True,
        bia_paper: bool = True,
        age_division: int = 100
    ):
        super().__init__()

        self.root_dir = root_dir
        self.fold_nums = fold_nums
        self.transforms = transforms

        self.use_sex = use_sex
        self.use_age = use_age
        self.use_1st = use_1st
        self.bia_paper = bia_paper

        self.age_division = age_division

        self.sex_labels = {
            'F': 0,
            'M': 1
        }

        if isinstance(self.transforms, A.Compose):
            self.albumentations = True
            print('Using albumentations package.')
        else:
            self.albumentations = False
            print('Using torchvision package.')

        if not use_sex and not use_age:
            raise Exception('At least sex or age input must be True.')

        # labels
        self.filepaths = []
        for i in fold_nums:
            filepath = os.path.join(root_dir, fold_txt_dir, f'{i:02d}.txt')
            with open(filepath) as txt_file:
                for line in txt_file:
                    img_relpath = line.strip()

                    if not self.use_1st and img_relpath.startswith('1st-set/'):
                        continue

                    filename = img_relpath.split('/')[-1]
                    sex, _, years, months = self._get_attributes(filename)

                    if use_sex and sex not in ['M', 'F']:
                        continue
                    if use_age and years == 'YNA':
                        continue
                    if use_age and months == 'MNA':
                        print('Skipping file with missing months:', filename)
                        continue

                    self.filepaths.append(os.path.join(root_dir, img_relpath))

        # this maybe useful later for reproducibility
        self.filepaths.sort()
        print(f'\nLoaded {len(self.filepaths)} images.')

    def _get_attributes(self, filename: str) -> Tuple[str, str, str, str]:
        splits      = Path(filename).stem.split('-')
        sex, frac   = splits[10], splits[11]
        years, months = splits[12], splits[13]

        return sex, frac, years, months

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int):
        # image and label
        filepath = self.filepaths[index]
        filename = filepath.split('/')[-1]

        # get the labels
        sex, _, years, months = self._get_attributes(filename)

        image = Image.open(filepath)
        image = image.convert('RGB')
        
        # apply transforms
        if self.albumentations:
            image = np.array(image)
            img_tensor = self.transforms(image=image)["image"]            
            
        else:
            img_tensor = self.transforms(image)

        label = []

        # assert sex in ['F', 'M']
        if self.use_sex and not self.use_age:
            if sex == 'F':
                label = [1, 0]
            elif sex == 'M':
                label = [0, 1]
        elif self.use_sex and self.use_age:
            sex_label = self.sex_labels[sex]
            label.append(sex_label)

            assert years != 'YNA'
            assert months != 'MNA'
            label.append(int(years[1:]) / self.age_division)
            # TODO: add months
            # label.append(months)
        elif self.use_age and not self.use_sex:
            assert years != 'YNA'
            assert months != 'MNA'
            label.append(int(years[1:]) / self.age_division)

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img_tensor, label_tensor
    
def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset

def build_dataset_age(is_train, args, transform = None):
    
    train_folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    val_folds = [26, 27, 28, 29, 30]
    
    folds = {
        'train': train_folds,
        'val':   val_folds
    }
    # root = os.path.join(args.data_path, 'train' if is_train else 'val')
    subset = 'train' if is_train else 'val'
    
    if transform is None:
        transform = build_transform(is_train, args)
        
    dataset = RadiographSexAgeDataset(
                args.data_path,
                folds[subset],
                transforms=transform,
                use_sex=True,
                use_age=True,
                use_1st=False,
                bia_paper=False
            )

    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
