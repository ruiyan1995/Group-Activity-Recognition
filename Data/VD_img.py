"""
"""
from torch.utils.data import Dataset
from skimage import io, transform
import numpy as np
import os
from PIL import Image
class VD_img(Dataset):
	def __init__(self, dataset_folder, phase, label_types, data_transform=None):
		self.label_types = label_types
		self.data_transform = data_transform
		lines = {label_type: open(os.path.join(dataset_folder, phase + '_' + label_type + '.txt')) for label_type in self.label_types}

		self.path_list = {label_type: [] for label_type in self.label_types}
		self.labels_list = {label_type: [] for label_type in self.label_types}

		for label_type in label_types:
			for i, line in enumerate(lines[label_type]):
				img_path = line.split('\n')[0].split('\t')[0]
				label = line.split('\n')[0].split('\t')[1]

				if 'error' in label:
					pass
				else:
					self.path_list[label_type].append(img_path)
					self.labels_list[label_type].append(label)

	def __getitem__(self, index):
		
		'''
		inputs_data = {label_type: () for label_type in self.label_types}
		for label_type in self.label_types:
			img = io.imread(self.path_list[label_type][index])
			img = transform.resize(img,(224,224),mode='constant')
			img = img.transpose((2, 0, 1))
			target = self.labels_list[label_type][index]
			inputs_data[label_type] = img, int(target)
		return inputs_data
		'''

		inputs_data = {label_type: () for label_type in self.label_types}
		for label_type in self.label_types:
			img = Image.open(self.path_list[label_type][index])

			if self.data_transform is not None:
				img = self.data_transform(img)

			target = self.labels_list[label_type][index]
			inputs_data[label_type] = img, int(target)
		return inputs_data





	def __len__(self):
		return len(self.labels_list[self.label_types[0]])