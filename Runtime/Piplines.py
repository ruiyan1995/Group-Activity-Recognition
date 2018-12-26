"""
	Common Piplines
"""
from abc import ABCMeta, abstractmethod
from Configs import Dataset_Configs, Model_Configs, Solver_Configs
from torchvision import transforms
import Solver
import Models
import Dataset
import torch
class Piplines(object):
	"""docstring for Piplines"""
	def __init__(self, dataset_root, dataset_name, stage, mode):
		super(Piplines, self).__init__()
		self.dataset_name = dataset_name
		self.stage = stage
		# Dataset configs:
		self.dataset_confs = Dataset_Configs(dataset_root, dataset_name, stage, mode).configuring()
		print self.dataset_confs
		# Model configs:
		self.model_confs = Model_Configs(dataset_name, stage).configuring()

		if torch.cuda.is_available():
			self.net = self.__loadModel(self.model_confs).cuda()
		else:
			self.net = self.__loadModel(self.model_confs)

		# Solver configs:
		self.solver_confs = Solver_Configs(dataset_name, stage, self.net, self.dataset_confs).configuring()
		#self.confs = {'dataset':dataset_confs, 'model':model_confs, 'solver_confs':solver_confs}

	#@abstractmethod
	def __loadModel(self, model_confs):
		net = Models.resnet50_LSTM(pretrained=True, model_confs=model_confs)
		print net
		return net

	def trainval(self):
		data_loaders, data_sizes = self.__loadData(self.dataset_confs)
		solver = Solver.Solver(data_loaders, data_sizes, self.net, self.solver_confs)
		solver.train_model()

	def test(self):
		#self.data_loaders = __loadData(self.dataset_confs, phases = ['test'])
		pass


	def __loadData(self, dataset_confs, phases = ['trainval', 'test']):
		data_transforms = {
			'trainval': transforms.Compose([
				transforms.RandomResizedCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	        ]),
	        'test': transforms.Compose([
	        	transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			]),
		}
		dataset = {phase: eval('Dataset.'+self.dataset_name+'_'+dataset_confs.data_type)(dataset_confs.dataset_folder, phase, dataset_confs.label_types, data_transforms[phase]) for phase in phases}
		data_loaders = {phase: torch.utils.data.DataLoader(dataset[phase], batch_size=dataset_confs.batch_size[phase], shuffle=False, num_workers=16) for phase in phases}
		data_sizes = {phase: len(dataset[phase]) for phase in phases}
		return data_loaders, data_sizes

