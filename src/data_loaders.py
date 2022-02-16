import os

import h5py
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
import torch

from os import listdir
from os.path import isfile, join


class Dsprites(Dataset):
    def __init__(self):
        data = np.load('./dataset/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding='bytes', mmap_mode='r')
        self.train_data = data['imgs'][0:len(data['imgs'])]

    def __getitem__(self, item):
        x = torch.from_numpy(self.train_data[item]).unsqueeze(0).float()  # load lazily
        return x

    def __len__(self):
        return len(self.train_data)


class CelebA(Dataset):
    def __init__(self):
        workdir = os.getcwd()
        self.transforms = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        self.train_data = datasets.ImageFolder(workdir+'/dataset/CelebA/')

    def __getitem__(self, i):
        return self.transforms(self.train_data[i][0])

    def __len__(self):
        return len(self.train_data)


class BoxHead(Dataset):
    def __init__(self, dataset='boxheadsimple2'):
        # read boxhead datasets from hdf5 and make a np array
        workdir = os.getcwd()
        path = workdir + '/dataset/'+dataset+'/'
        filenames = [f for f in listdir(path) if isfile(join(path, f))]

        all_images_in_batch = []
        all_images = []
        for filename in filenames:
            with h5py.File(path + filename, "r") as f:
                list_of_data = list(f.keys())
                for k in list_of_data:
                    group = f[k]
                    images = group["data"][()]
                    all_images_in_batch.append(images)
                    all_images = np.reshape(np.array(all_images_in_batch), (-1, 64, 64, 3))
        all_images = (all_images.astype('float64') / 255.0)
        all_images = np.einsum('abcd->adbc', all_images)
        self.train_data = all_images

    def __getitem__(self, item):
        x = torch.from_numpy(self.train_data[item]).float()  # load lazily
        return x

    def __len__(self):
        return len(self.train_data)
