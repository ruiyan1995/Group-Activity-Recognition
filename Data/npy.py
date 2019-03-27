from torch.utils.data import Dataset
from skimage import io, transform
from sklearn import preprocessing
import numpy as np
import h5py
import torch
import os

class npy(Dataset):
    def __init__(self, data_folder, phase, label_type, data_transform=None):
        self.label_type = label_type
        self.data = np.load(os.path.join(data_folder, label_type, phase + '.npy'), mmap_mode='r')

    def __getitem__(self, index):
        fea = self.data[index, :-1]
        target = self.data[index, -1]
        return torch.from_numpy(fea), int(target)

    def __len__(self):
        return len(self.data[:, -1])
