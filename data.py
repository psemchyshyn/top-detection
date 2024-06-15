'''
Data object which complies with logic of the torch-lightining Trainer
'''


import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from augmentations import get_augmentations
from dataset import TopDetectionDataset, TestDataset

class DataWrapper(pl.LightningDataModule):
    def __init__(self, config, test='test'):
        super().__init__()
        self.test = test
        self.config = config['data']
        self.batch_size = self.config['batch_size']
        
        self.names = self.get_names()
        if self.config['train_size'] > 1:
            self.train_names = self.names
            self.val_names = self.names[-500:]
        else:
            self.train_names = self.names[:int(self.config['train_size']*len(self.names))]
            self.val_names = self.names[int(self.config['train_size']*len(self.names)):]

        self.train_transforms = get_augmentations(self.config['image_h'], self.config['image_w'], self.config['train_augs'])
        self.val_transforms = get_augmentations(self.config['image_h'], self.config['image_w'], self.config['val_augs'])

        self.train_dataset = TopDetectionDataset(self.train_names, self.config['image_path'], self.config['label_path'], self.config['image_h'], self.config['image_w'], transforms=self.train_transforms)
        self.val_dataset = TopDetectionDataset(self.val_names, self.config['image_path'], self.config['label_path'], self.config['image_h'], self.config['image_w'], transforms=self.val_transforms, tta=self.config['tta'])
        self.test_dataset = TestDataset(self.config['test_data_path'], self.config['image_h'], self.config['image_w'], tta=self.config['tta'])

    def get_names(self):
        names = os.listdir(self.config['image_path'])
        names = list(map(lambda x: x.split('.')[0], names))
        return names
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True)

    def test_dataloader(self):
        if self.test == 'train':
            return DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True)
        elif self.test == 'val':
            return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True)
        else:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True)
