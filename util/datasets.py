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
import pandas as pd


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
        use_2nd: bool = True,
        bia_paper: bool = True,
        age_division: int = 100,
        remove: List[str] = None
    ):
        super().__init__()

        self.root_dir = root_dir
        self.fold_nums = fold_nums
        self.transforms = transforms

        self.use_sex = use_sex
        self.use_age = use_age
        self.use_1st = use_1st
        self.use_2nd = use_2nd
        self.bia_paper = bia_paper
        self.remove = remove

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
        TOLERANCIA = 95
        self.filepaths = []
        for i in fold_nums:
            filepath = os.path.join(root_dir, fold_txt_dir, f'{i:02d}.txt')
            with open(filepath) as txt_file:
                for line in txt_file:
                    img_relpath = line.strip()

                    if not self.use_1st and img_relpath.startswith('1st-set/'):
                        continue
                    if not self.use_2nd and img_relpath.startswith('2nd-set/'):
                        continue

                    # filename = img_relpath.split('/')[-1]
                    # sex, prob, years, months = self._get_attributes(filename)

                    # if use_sex and sex not in ['M', 'F']:
                    #     # print('Skipping file with missing sex:', filename)
                    #     continue
                    
                    # if sex == "NA":
                    #     # print('Skipping file with missing sex:', filename)
                    #     continue
                  
                    # if use_age and years == 'YNA':
                    #     # print('Skipping file with missing years:', filename)
                    #     continue
                    # if use_age and months == 'MNA':
                    #     print('Skipping file with missing months:', filename)
                    #     continue
                    
                    # if '.' in prob:
                    #     prob = float(prob)*100
                    # prob = int(prob)

                    # if prob < TOLERANCIA:
                    #     print('Skipping file with missing prob:', filename)
                    #     continue
                    filename = img_relpath.split('/')[-1]
                    fname = filename.split('-')
                    if fname[0].startswith('.'):
                        continue
                    sex, prob, year = fname[10:13]

                    if sex == "NA":
                      continue

                    if year == 'YNA':
                      # print(year)
                      continue
                    if '.' in prob:
                        prob = float(prob)*100
                    prob = int(prob)

                    if prob < TOLERANCIA:
                        continue
                    self.filepaths.append(os.path.join(root_dir, img_relpath))
        if self.remove is not None:
            # Contar o número de elementos antes da remoção
            n1 = len(self.filepaths)        
            # Remover os elementos pelos índices
            self.filepaths = [valor for indice, valor in enumerate(self.filepaths) if valor.split('/')[-1] not in self.remove]          
            print(f"Removed {n1-len(self.filepaths)} images")
          
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
            # assert months != 'MNA'
            label.append(int(years[1:]) / self.age_division)
            # TODO: add months
            # label.append(months)
        elif self.use_age and not self.use_sex:
            assert years != 'YNA'
            # assert months != 'MNA'
            label.append(int(years[1:]) / self.age_division)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return img_tensor, label_tensor
    

class Eval_RadiographSexAgeDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        fold_nums: List[int],
        transforms: Union[A.Compose, T.Compose],
        fold_txt_dir: str='splits',
        use_sex: bool = True,
        use_age: bool = True,
        use_1st: bool = True,
        use_2nd: bool = True,
        bia_paper: bool = True,
        age_division: int = 100,
        remove: List[str] = None
    ):
        super().__init__()

        self.root_dir = root_dir
        self.fold_nums = fold_nums
        self.transforms = transforms

        self.use_sex = use_sex
        self.use_age = use_age
        self.use_1st = use_1st
        self.use_2nd = use_2nd
        self.bia_paper = bia_paper
        self.remove = remove

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
        TOLERANCIA = 95
        self.filepaths = []
        for i in fold_nums:
            filepath = os.path.join(root_dir, fold_txt_dir, f'{i:02d}.txt')
            with open(filepath) as txt_file:
                for line in txt_file:
                    img_relpath = line.strip()

                    if not self.use_1st and img_relpath.startswith('1st-set/'):
                        continue
                    if not self.use_2nd and img_relpath.startswith('2nd-set/'):
                        continue
                    
                    filename = img_relpath.split('/')[-1]
                    fname = filename.split('-')
                    if fname[0].startswith('.'):
                        continue
                    sex, prob, year = fname[10:13]

                    if sex == "NA":
                      continue

                    if year == 'YNA':
                      # print(year)
                      continue
                    if '.' in prob:
                        prob = float(prob)*100
                    prob = int(prob)

                    if prob < TOLERANCIA:
                        continue
                    self.filepaths.append(os.path.join(root_dir, img_relpath))
        if self.remove is not None:
            # Contar o número de elementos antes da remoção
            n1 = len(self.filepaths)        
            # Remover os elementos pelos índices
            self.filepaths = [valor for indice, valor in enumerate(self.filepaths) if valor.split('/')[-1] not in self.remove]          
            print(f"Removed {n1-len(self.filepaths)} images")
          
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
        names = []
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
            # assert months != 'MNA'
            label.append(int(years[1:]) / self.age_division)
            names.append(filename)
            # TODO: add months
            # label.append(months)
        elif self.use_age and not self.use_sex:
            assert years != 'YNA'
            # assert months != 'MNA'
            label.append(int(years[1:]) / self.age_division)
            
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return img_tensor, label_tensor, names
   
def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset

def build_dataset_multi(is_train, args, folds, transform = None, use_sex=True, use_age = True, use_1st=False, use_2nd = True):
    
    
    # root = os.path.join(args.data_path, 'train' if is_train else 'val')
    subset = 'train' if is_train else 'val'
    
    if transform is None:
        transform = build_transform(is_train, args)
        
    df_outliers = pd.read_csv('/home/ivisionlab/Documentos/github/mae/demo/remove_outliers/df_remove_filenames.csv')  
    remove = df_outliers.filenames.to_list()
    
    dataset = RadiographSexAgeDataset(
                args.data_path,
                folds,
                transforms=transform,
                use_sex=use_sex,
                use_age=use_age,
                use_1st=use_1st,
                use_2nd=use_2nd, 
                bia_paper=False,
                remove=remove
            )

    return dataset

def build_dataset_multi_eval(is_train, args, folds, transform = None, use_sex=True, use_age = True, use_1st=False, use_2nd = True):
    
    
    # root = os.path.join(args.data_path, 'train' if is_train else 'val')
    subset = 'train' if is_train else 'val'
    
    if transform is None:
        transform = build_transform(is_train, args)
        
        
    df_outliers = pd.read_csv('/home/ivisionlab/Documentos/github/mae/demo/remove_outliers/df_remove_filenames.csv')  
    remove = df_outliers.filenames.to_list()
    
    dataset = Eval_RadiographSexAgeDataset(
                args.data_path,
                folds,
                transforms=transform,
                use_sex=use_sex,
                use_age=use_age,
                use_1st=use_1st,
                use_2nd=use_2nd, 
                bia_paper=False,
                remove=remove
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
