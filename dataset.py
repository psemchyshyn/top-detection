'''
Dataset logic
'''

import torch
import numpy as np
import glob
import cv2
import os
import json
from PIL import Image
from torch.utils.data import Dataset
from utility import labelme2mask
from augmentations import get_tta_augs


class TopDetectionDataset(Dataset):
    def __init__(self, names, path2images, path2labels, image_h, image_w, transforms=None, tta=False):
        super(TopDetectionDataset, self).__init__()
        self.names = names
        self.transforms = transforms
        self.image_h = image_h
        self.image_w = image_w
        self.names = names
        self.tta = tta # defines whether to use test-time augmentations
        self.path2images = path2images
        self.path2labels = path2labels

        self.tta_transforms = get_tta_augs()

        self.transforms = transforms

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = self.names[idx]
        image_path = os.path.join(self.path2images, f"{name}.png")
        image = Image.open(image_path)

        annotation_path = os.path.join(self.path2labels, f"{name}.json")
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        mask_roof, mask_height = labelme2mask(annotation, image.width, image.height)
        image = np.asarray(image)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask_roof=mask_roof, mask_height=mask_height)
            image = transformed["image"]
            mask_roof = transformed["mask_roof"]
            mask_height = transformed["mask_height"]
        else:
            image = cv2.resize(image, (self.image_h, self.image_w))
            mask_roof = cv2.resize(mask_roof, (self.image_h, self.image_w))
            mask_height = cv2.resize(mask_height, (self.image_h, self.image_w))

        if self.tta:
            images = []
            for _ in range(5):
                image_i = self.tta_transforms(image=image)['image']
                image_i = torch.from_numpy(image_i).permute(2, 0, 1)
                images.append(image_i)
            image = torch.stack(images)

        else:
            image = torch.from_numpy(image).permute(2, 0, 1)

        return {"name": name, "image": image, "mask_roof": (torch.from_numpy(mask_roof).unsqueeze(0)).long(), "mask_height": (torch.from_numpy(mask_height).unsqueeze(0)).float()}

class TestDataset(Dataset):
    def __init__(self, path2images, image_h, image_w, tta=False, ext="png"):
        super(TestDataset, self).__init__()
        self.image_h = image_h
        self.image_w = image_w
        self.tta = tta
        self.tta_transforms = get_tta_augs()
        self.path2images = glob.glob(f"{path2images}/*.{ext}")

    def __len__(self):
        return len(self.path2images)

    def __getitem__(self, idx):
        image = cv2.resize(cv2.imread(self.path2images[idx]), (self.image_w, self.image_h))
        name = self.path2images[idx].split(os.sep)[-1].split('.')[0]

        if self.tta:
            images = []
            for _ in range(5):
                image_i = self.tta_transforms(image=image)['image']
                image_i = torch.from_numpy(image_i).permute(2, 0, 1)
                images.append(image_i)
            image = torch.stack(images)

        else:
            image = torch.from_numpy(image).permute(2, 0, 1)

        return {"name": name, "image": image}
