"""
"""
from torch.utils.data import Dataset
from skimage import io, transform
import numpy as np
import h5py
import torch
class VD_npy(Dataset):
	def __init__(self, data_root, phase, label_type):
		self.file = np.load(data_root + 'volleyball_' + label_type +'_' + phase + '.npy', mmap_mode = 'r')
		self.phase = phase
		# it can not be use item by item. I do not kown why?
		self.targets = self.file[:,-1]

	def __getitem__(self, index):
		feas = self.file[index,:-1]
		target = self.targets[index]
		if target<0 or target>8:
			print 'index:', index, 'target_value:', target

		return torch.from_numpy(feas), int(target)

	def __len__(self):
		return len(self.file[self.phase + '_feas'])