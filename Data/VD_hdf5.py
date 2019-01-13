"""
"""
from torch.utils.data import Dataset
from skimage import io, transform
import numpy as np
import h5py
import torch
class VD_hdf5(Dataset):
	def __init__(self, data_root, phase, label_type):
		self.hdf5_file = data_root + 'volleyball.hdf5'
		self.file = h5py.File(self.hdf5_file, 'r')
		self.phase = phase
		# It can not be use item by item. I do not kown why?
		self.targets = self.file[self.phase + '_labels'][:]

	def __getitem__(self, index):
		img = self.file[self.phase + '_img'][index,:,:]
		#target = self.file[self.phase + '_labels'][index]
		target = self.targets[index]
		if target<0 or target>8:
			print 'index:', index, 'target_value:', target
		return torch.from_numpy(img), int(target)

	def __len__(self):
		return len(self.file[self.phase + '_img'])