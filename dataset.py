from torch.utils.data import Dataset, DataLoader
import os
import h5py
from config import Config
from transforms import transform_augment
import torch


class AffectNet(Dataset):
    def __init__(self, split='training', transform=None):
        super(AffectNet, self).__init__()
        self.split = split
        self.transform = transform
        self.data = h5py.File(os.path.join(Config.affectnet_path, split + '.h5'))
        self.length = len(self.data['labels'])

    def __getitem__(self, index):
        pixel, label, landmark = self.data['pixels'][index], self.data['labels'][index], self.data['landmarks'][index]
        pixel = pixel[..., ::-1]
        if self.transform is not None:
            if self.split == 'training':
                pixel, landmark = self.transform(pixel, landmark)
            elif self.split == 'validation':
                pixel = self.transform(pixel)
        return pixel, label, landmark

    def __len__(self):
        return self.length


def affectnet_loader():
    trainset = AffectNet(split='training', transform=transform_augment())
    trainloader = DataLoader(trainset, shuffle=True, batch_size=Config.batch_size, num_workers=Config.num_workers)
    return trainloader
