import os

import h5py
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
import torch

from os import listdir
from os.path import isfile, join
import pickle


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


class BoxHeadWithLabels(Dataset):
    def __init__(self, dataset='boxheadsimple2'):
        # read boxhead datasets from hdf5 and make a np array
        workdir = os.getcwd()
        path = workdir + '/dataset/'+dataset+'/'
        filenames = [f for f in listdir(path) if isfile(join(path, f))]

        all_images_in_batch = []
        all_images = []
        self.all_labels = []
        for filename in filenames:
            with h5py.File(path + filename, "r") as f:
                list_of_data = list(f.keys())
                for k in list_of_data:
                    group = f[k]
                    images = group["data"][()]
                    all_images_in_batch.append(images)
                    all_images = np.reshape(np.array(all_images_in_batch), (-1, 64, 64, 3))

                    labels = pickle.loads(group["labels"][()].item())
                    for label in labels:
                      self.all_labels.append(label)


        all_images = (all_images.astype('float64') / 255.0)
        all_images = np.einsum('abcd->adbc', all_images)
        test_set = []
        for i, image in enumerate(all_images):
          test_set.append({'image':image, 'labels':label})
        self.train_data = test_set

    def __getitem__(self, item):
        x = torch.from_numpy(self.train_data[item]['image']).float()
        label = self.train_data[item]['labels']  # load lazily
        return x, label

    def __len__(self):
        return len(self.train_data)

    def get_all_labels(self):
      labels_by_parameter= []
      azimuths = []
      floor_colors = []
      wall_colors = []
      eye1_colors = []
      eye2_colors = []
      eye3_colors = []
      eye4_colors = []
      box_colors = []
      box_sizes = []

      for label in self.all_labels:
        azimuths.append(label['azimuth'])
        floor_colors.append(label['bg_color'][0])
        wall_colors.append(label['bg_color'][1])
        eye1_colors.append(label['eye_color'][0])
        eye2_colors.append(label['eye_color'][1])
        eye3_colors.append(label['eye_color'][2])
        eye4_colors.append(label['eye_color'][3])
        box_colors.append(label['color'])
        box_sizes.append(label['scale'][0])

      labels_by_parameters = {'azimuths':azimuths, 'floor_colors':floor_colors,'wall_colors':wall_colors,'eye1_colors':eye1_colors,'eye2_colors':eye2_colors,'eye3_colors':eye3_colors,'eye4_colors':eye4_colors, 'box_colors':box_colors,'box_sizes':box_sizes}
      return labels_by_parameters


class BoxHeadNoRotation(Dataset):
    def __init__(self, dataset='boxheadsimple2'):
        # read boxhead datasets from hdf5 and make a np array
        workdir = os.getcwd()
        path = workdir + '/dataset/'+dataset+'/'
        filenames = [f for f in listdir(path) if isfile(join(path, f))]

        all_images_in_batch = []
        all_images = []
        self.all_labels = []
        for filename in filenames:
            with h5py.File(path + filename, "r") as f:
                list_of_data = list(f.keys())
                for k in list_of_data:
                    group = f[k]
                    images = group["data"][()]
                    labels = pickle.loads(group["labels"][()].item())
                  
                    for idx, label in enumerate(labels):
                      if label['azimuth'] == 0.0:
                        self.all_labels.append(label)
                        all_images_in_batch.append(images[idx])
                    all_images = np.reshape(np.array(all_images_in_batch), (-1, 64, 64, 3))



        all_images = (all_images.astype('float64') / 255.0)
        all_images = np.einsum('abcd->adbc', all_images)
        test_set = []
        for i, image in enumerate(all_images):
          test_set.append({'image':image, 'labels':self.all_labels[i]})
        self.train_data = test_set

    def __getitem__(self, item):
        x = torch.from_numpy(self.train_data[item]['image']).float()
        return x

    def __len__(self):
        return len(self.train_data)



class BoxHeadNoRotationWithLabels(Dataset):
    def __init__(self, dataset='boxheadsimple2'):
        # read boxhead datasets from hdf5 and make a np array
        workdir = os.getcwd()
        path = workdir + '/dataset/'+dataset+'/'
        filenames = [f for f in listdir(path) if isfile(join(path, f))]

        all_images_in_batch = []
        all_images = []
        self.all_labels = []
        for filename in filenames:
            with h5py.File(path + filename, "r") as f:
                list_of_data = list(f.keys())
                for k in list_of_data:
                    group = f[k]
                    images = group["data"][()]
                    labels = pickle.loads(group["labels"][()].item())
                  
                    for idx, label in enumerate(labels):
                      if label['azimuth'] == 0.0:
                        self.all_labels.append(label)
                        all_images_in_batch.append(images[idx])
                    all_images = np.reshape(np.array(all_images_in_batch), (-1, 64, 64, 3))



        all_images = (all_images.astype('float64') / 255.0)
        all_images = np.einsum('abcd->adbc', all_images)
        test_set = []
        for i, image in enumerate(all_images):
          test_set.append({'image':image, 'labels':self.all_labels[i]})
        self.train_data = test_set

    def __getitem__(self, item):
        x = torch.from_numpy(self.train_data[item]['image']).float()
        label = self.train_data[item]['labels']  # load lazily
        return x, label

    def __len__(self):
        return len(self.train_data)

    def get_all_labels(self):
      labels_by_parameter= []
      azimuths = []
      floor_colors = []
      wall_colors = []
      eye1_colors = []
      eye2_colors = []
      eye3_colors = []
      eye4_colors = []
      overall_eye_color = []
      box_colors = []
      box_sizes = []

      for label in self.all_labels:
        azimuths.append(label['azimuth'])
        floor_colors.append(label['bg_color'][0])
        wall_colors.append(label['bg_color'][1])
        eye1_colors.append(label['eye_color'][0])
        eye2_colors.append(label['eye_color'][1])
        eye3_colors.append(label['eye_color'][2])
        eye4_colors.append(label['eye_color'][3])
        overall_eye_color.append(statistics.mean(label['eye_color']))
        box_colors.append(label['color'])
        box_sizes.append(label['scale'][0])

      labels_by_parameters = {'azimuths':azimuths, 'floor_colors':floor_colors,'wall_colors':wall_colors,'eye1_colors':eye1_colors,'eye2_colors':eye2_colors,'eye3_colors':eye3_colors,'eye4_colors':eye4_colors, 'box_colors':box_colors,'box_sizes':box_sizes, 'overall_eye_color':overall_eye_color}
      return labels_by_parameters