"""
 Load data of CAD dataset from img source.
"""
from torch.utils.data import Dataset
from skimage import io, transform
import numpy as np

class CAD_img(Dataset):
	def __init__(self, data_root, phase, label_type):
		
		self.txt_file = data_root + phase + '_' + label_type + '.txt'

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
		img = io.imread(self.path_list[index])
		img = transform.resize(img,(224,224),mode='constant')
		img = img.transpose((2, 0, 1))
		target = self.labels_list[index]
		return img, int(target)

	def __len__(self):
		return len(self.labels_list)