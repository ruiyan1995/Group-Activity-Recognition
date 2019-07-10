"""
 Load data from img source.
"""
from torch.utils.data import Dataset
from skimage import io, transform
import numpy as np
from PIL import Image
import os
class img(Dataset):
	def __init__(self, data_root, phase, label_type, data_transform=None):
		self.data_transform = data_transform
		self.txt_file = os.path.join(data_root, phase + '_' + label_type + '.txt')

		lines = open(self.txt_file)
		self.path_list = []
		self.labels_list = []
		for i, line in enumerate(lines):
			img_path = line.split('\n')[0].split('\t')[0]
			label = line.split('\n')[0].split('\t')[1]
			if ('error' in label) or ('NA' in label):
				pass
			else:
				self.path_list.append(img_path)
				self.labels_list.append(label)

	def __getitem__(self, index):
		img = Image.open(self.path_list[index])
		target = self.labels_list[index]
		if self.data_transform is not None:
			img = self.data_transform(img)
		return img, int(target)

	def __len__(self):
		return len(self.labels_list)
