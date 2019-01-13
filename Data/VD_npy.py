from torch.utils.data import Dataset
from skimage import io, transform
from sklearn import preprocessing
import numpy as np
import h5py
import torch
import os

class VD_npy(Dataset):

    def __init__(self, data_folder, phase, label_types, data_transform=None):
        self.do_norm = False
        self.label_types = label_types
        self.files = {label_type: np.load(os.path.join(data_folder, label_type, phase + '.npy'), mmap_mode='r') for label_type in self.label_types}
        self.targets = {label_type: self.files[label_type][
            :, -1] for label_type in self.label_types}
        if self.do_norm:
            self.data = {label_type: preprocessing.normalize(self.files[label_type][:, :-1], norm='l2') for label_type in self.label_types}

    def __getitem__(self, index):
        inputs_data = {label_type: () for label_type in self.label_types}
        for label_type in self.label_types:
            feas = self.data[label_type][index] if self.do_norm else self.files[label_type][index, :-1]
            target = self.targets[label_type][index]
            inputs_data[label_type] = torch.from_numpy(feas), int(target)
            #print inputs_data
        return inputs_data

    def __len__(self):
        return len(self.files[self.label_types[0]][:, -1])
