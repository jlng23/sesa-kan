import os
from pathlib import Path
from typing import List, Union, Tuple

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset

import torchvision.transforms as T

import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
#  sys.path.append('..')
from preprocessing.utils import crop_image, remove_border, filter_files_bia_paper


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
        age_division: int = 100
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
                        print('Using 1st-set.')
                        continue
                    if not self.use_2nd and img_relpath.startswith('2nd-set/'):
                        continue

                    if self.bia_paper and filter_files_bia_paper(img_relpath):
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

        # copy = (10, 3, 224, 224)
        # img_tensor = img_tensor.unsqueeze(0).expand(*copy)       

        label = []

        assert sex in ['F', 'M']
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

class RadiographSexAgeDataset_names(Dataset):
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
        self.bia_paper = bia_paper

        self.age_division = age_division

        self.sex_labels = {
            'F': 0,
            'M': 1
        }

        self.remove = remove
        
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

                    if self.bia_paper and filter_files_bia_paper(img_relpath):
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
        if self.remove is not None:
            # Contar o número de elementos antes da remoção
            n1 = len(self.filepaths)
            # Remover os elementos pelos índices
            self.filepaths = [valor for indice, valor in enumerate(self.filepaths) if valor.split('/')[-1] not in self.remove]
            n2 = len(self.filepaths)
            print(f"{n1-n2} images were removed")
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
            
        # if 4D == True:    
        #     copy = (10, 3, 224, 224)
        #     img_tensor = img_tensor.unsqueeze(0).expand(*copy)       

        label = []
        filenames_list = []
        
        assert sex in ['F', 'M']
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
            filenames_list.append(filename)
            # TODO: add months
            # label.append(months)
        elif self.use_age and not self.use_sex:
            assert years != 'YNA'
            assert months != 'MNA'
            label.append(int(years[1:]) / self.age_division)
            
            
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img_tensor, label_tensor, filenames_list

class DoubleTransformsSexAgeDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        fold_nums: List[int],
        transform_image: Union[A.Compose, T.Compose],
        transform_attention: Union[A.Compose, T.Compose],
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
        self.transform_image = transform_image
        self.transform_attention = transform_attention

        self.use_sex = use_sex
        self.use_age = use_age
        self.use_1st = use_1st
        self.bia_paper = bia_paper

        self.age_division = age_division

        self.sex_labels = {
            'F': 0,
            'M': 1
        }

        if isinstance(self.transform_image, A.Compose):
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

                    if self.bia_paper and filter_files_bia_paper(img_relpath):
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
            full_img_tensor = self.transform_image(image=image)["image"]
            crop_img_tensor = self.transform_attention(image=image)["image"]
        else:
            full_img_tensor = self.transform_image(image)
            crop_img_tensor = self.transform_attention(image)

        label = []

        assert sex in ['F', 'M']
        if self.use_sex and not self.use_age:
            if sex == 'F':
                label = [1, 0]
            elif sex == 'M':
                label = [0, 1]
        elif self.use_sex and self.use_sex:
            sex_label = self.sex_labels[sex]
            label.append(sex_label)

            assert years != 'YNA'
            assert months != 'MNA'
            label.append(int(years[1:]) / self.age_division)
            # TODO: add months
            # label.append(months)

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return (full_img_tensor, crop_img_tensor), label_tensor


class RadiographSexDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        fold_nums: list,
        transforms,
        albumentations_package: bool=True,
        crop_side: str=None,
        border: int=0
    ):
        super().__init__()

        self.root_dir = root_dir
        self.fold_nums = fold_nums
        self.transforms = transforms
        self.albumentations = albumentations_package
        self.crop_side = crop_side

        # labels
        self.filepaths = []
        for fold_num in self.fold_nums:
            foldpath = os.path.join(self.root_dir, f'fold-{fold_num:02d}')
            for filename in os.listdir(foldpath):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    self.filepaths.append(os.path.join(foldpath, filename))

        # this maybe useful later for reproducibility
        self.filepaths.sort()

    def __len__(self) -> int:
        return len(self.filepaths)

    def _getitem_albumentations(self, image):
        # Read image with OpenCV2 and convert it from BGR (OpenCV2) to RGB (most common format)
        image = np.array(image)

        # apply transformation with albumentations package
        if self.transforms is not None:
            img_tensor = self.transforms(image=image)["image"]

        return img_tensor

    def _getitem_torchvision(self, image):

        if self.transforms is not None:
            img_tensor = self.transforms(image)

        return img_tensor

    def __getitem__(self, index: int):
        # image and label
        filepath = self.filepaths[index]

        # get label
        filename = filepath.split('/')[-1]
        gender = filename.split('-')[7]

        assert gender in ['F', 'M']
        if gender == 'F':
            label = 0
        else:
            label = 1

        label_tensor = torch.tensor(label, dtype=torch.int64)

        image = Image.open(filepath)
        image = remove_border(image)
        if self.crop_side:
            image = crop_image(image, self.crop_side)

        # apply transforms
        if self.albumentations:
            img_tensor = self._getitem_albumentations(image)
        else:
            img_tensor = self._getitem_torchvision(image)

        return img_tensor, label_tensor


class FullRadiographSexDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        fold_nums: list,
        transforms,
        fold_txt_dir: str='splits',
        albumentations_package: bool=True
    ):
        super().__init__()

        self.root_dir = root_dir
        self.fold_nums = fold_nums
        self.transforms = transforms
        self.albumentations = albumentations_package

        # labels
        self.filepaths = []
        for i in fold_nums:
            filepath = os.path.join(root_dir, fold_txt_dir, f'{i:02d}.txt')
            with open(filepath) as txt_file:
                for line in txt_file:
                    img_relpath = line.strip()
                    filename = img_relpath.split('/')[-1]
                    sex = filename.split('-')[10]
                    if sex not in ['M', 'F']:
                        continue
                    self.filepaths.append(os.path.join(root_dir, img_relpath))

        # this maybe useful later for reproducibility
        self.filepaths.sort()

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int):
        # image and label
        filepath = self.filepaths[index]
        filename = filepath.split('/')[-1]

        # get the labels
        sex = filename.split('-')[10]
        # age = filename.split('-')[-2][1:]
        # months = filename.split('-')[-1][1:3]

        assert sex in ['F', 'M']
        if sex == 'F':
            label = 0
        else:
            label = 1

        label_tensor = torch.tensor(label, dtype=torch.int64)

        image = Image.open(filepath)
        image = image.convert('RGB')

        # apply transforms
        if self.albumentations:
            image = np.array(image)
            img_tensor = self.transforms(image=image)["image"]
        else:
            raise Exception('Not implemented yet.')

        return img_tensor, label_tensor
