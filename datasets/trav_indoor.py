import json
import os
from collections import namedtuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np


class IndoorTrav(Dataset):
    """
    Use positive to train, challenging as validation
    Parameters:
    root (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
    split (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
    scenes: select from [elb, erb, nh, uc, woh]
    transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts, like a struct
    IndoorTravClass = namedtuple('IndoorTravClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        IndoorTravClass('ground', 6, 0, 'void', 0, False, True, (81, 0, 81)),
        IndoorTravClass('road', 7, 1, 'flat', 1, False, False, (128, 64, 128)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, root, split='train', scenes=[], transform=None):
        self.root = os.path.expanduser(root)
        self.scenes = scenes
        self.scene_dirs = [os.path.join(self.root, x) for x in self.scenes]
        if split == 'train':
            self.images_dirs = [os.path.join(x, 'positive') for x in self.scene_dirs]
        else:  # 'val'
            self.images_dirs = [os.path.join(x, 'challenging') for x in self.scene_dirs]

        # self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train",'
                             ' or split="val"')

        if not all([os.path.isdir(x) for x in self.images_dirs]):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        for scene in self.images_dirs:
            img_dir = os.path.join(scene, 'images')
            target_dir = os.path.join(scene, 'labels')

            for filename in os.listdir(img_dir):
                abs_img_path = os.path.join(img_dir, filename)
                target_name = filename.rstrip('.jpg') + '.png'
                abs_label_path = os.path.join(target_dir, target_name)
                if os.path.exists(abs_img_path) and os.path.exists(abs_label_path):
                    self.images.append(abs_img_path)
                    self.targets.append(abs_label_path)

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        try:
            image = Image.open(self.images[index]).convert('RGB')  # [640,480]
            target = Image.open(self.targets[index])  # [480, 640]
        except IOError:
            raise RuntimeError(f'Cannot open image: {self.images[index]} or label: {self.targets[index]}')

        if self.transform:
            image, target = self.transform(image, target)
        target = (target / 255).int()
        target = self.encode_target(target)
        return image, target, self.images[index]

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)


def get_norm_values():
    """
    calculate normalization mean and std values for my custom dataset
    all 5 scenes mean, std: 
    (tensor([0.5174, 0.4857, 0.5054]), tensor([0.2726, 0.2778, 0.2861]))
    """
    data_transforms = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(f'/home/qiyuan/2023spring/segmentation_indoor_images', transform=data_transforms)

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0)

    # VAR[X] = E[X**2] - E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    print(f'mean, std: {mean, std}')  # (tensor([0.5174, 0.4857, 0.5054]), tensor([0.2726, 0.2778, 0.2861]))
    return mean, std


if __name__ == '__main__':
    get_norm_values()
