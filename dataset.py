import os
import pickle
from collections import namedtuple
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import lmdb
from PIL import Image


CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])


class ImageFileDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        dirs, filename = os.path.split(path)
        _, class_name = os.path.split(dirs)
        filename = os.path.join(class_name, filename)

        return sample, target, filename


class LMDBDataset(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))

        return torch.from_numpy(row.top), torch.from_numpy(row.bottom), row.filename

class ImageDataset(Dataset):
    def __init__(self, file, prefix, size):
        self.size = size
        self.prefix = prefix
        with open(file, 'r') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def _transform(self, img):
        img_ = transforms.RandomResizedCrop(self.size, scale=(0.8, 1.0), ratio=(0.9, 1.1))(img)
        img_ = transforms.ToTensor()(img_)
        return img_

    def __getitem__(self, index):
        img_file = self.data[index]
        img_x = Image.open(os.path.join(self.prefix, img_file))
        img_y = Image.open(os.path.join(self.prefix, img_file.replace(img_file[-5], 't')))
        seed = np.random.randint(2147483647)
        random.seed(seed)
        img_x = self._transform(img_x)
        random.seed(seed)
        img_y = self._transform(img_y)
        return img_x, img_y
