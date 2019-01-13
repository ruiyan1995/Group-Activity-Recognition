"""
	Solver_Configs
"""
import argparse
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
class Solver_Configs(object):
	"""docstring for Solver_Configs"""
	def __init__(self, dataset_name, stage, net, dataset_confs):
		super(Solver_Configs, self).__init__()
		self.dataset_name = dataset_name
		self.stage = stage
		self.net = net
		self.dataset_confs = dataset_confs
		self.confs_dict = {
			'VD':{
				'action':{
					'num_epochs': 20,
					'lr_scheduler': {'step_size': 5, 'gamma': 0.1},
					'optimizer': {'method':'SGD','lr':0.001,'arg':0.9}
				},
				'activity':{
					'num_epochs': 100,
					'lr_scheduler': {'step_size': 10, 'gamma': 0.1},
					'optimizer': {'method':'Adam','lr':0.0001,'arg':(0.9,0.9)}
				},
				'semantic':{
					'num_epochs': 50,
					'lr_scheduler': {'step_size': 10, 'gamma': 0.1},
					'optimizer': {'method':'Adam','lr':0.01,'arg':(0.9,0.9)}
					#'optimizer': {'method':'SGD','lr':0.001,'arg':0.9}
				}
				
			},

			'CAD':{
				'action':{
					'num_epochs': 20,
					'lr_scheduler': {'step_size': 5, 'gamma': 0.1},
					'optimizer': {'method':'SGD','lr':0.001,'arg':0.9}
				},
				'activity':{
					'num_epochs': 100,
					'lr_scheduler': {'step_size': 10, 'gamma': 0.1},
					'optimizer': {'method':'Adam','lr':0.0001,'arg':(0.9,0.9)}
				},
				'semantic':{
					'num_epochs': 100,
					'lr_scheduler': {'step_size': 10, 'gamma': 0.1},
					'optimizer': {'method':'Adam','lr':0.0001,'arg':(0.9,0.9)}
				}
				
			}
		}

	def configuring(self):
		solver_confs = self.confs_dict[self.dataset_name][self.stage]
		parser = argparse.ArgumentParser()
		parser.add_argument('--num_epochs', type=int, default=solver_confs['num_epochs'])
		parser.add_argument('--gpu', type=bool, default=torch.cuda.is_available(), help='*****')

		criterion = nn.CrossEntropyLoss()
		parser.add_argument('--criterion', type=type(criterion), default=criterion)

		optim = solver_confs['optimizer']
		optimizer = eval('torch.optim.'+optim['method'])(self.net.parameters(), optim['lr'], optim['arg'])
		parser.add_argument('--optimizer', type=type(optimizer), default=optimizer)

		# Decay LR by a factor of 0.1 every 7 epochs
		lr_sch = solver_confs['lr_scheduler']
		exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_sch['step_size'], gamma=lr_sch['gamma'])
		parser.add_argument('--exp_lr_scheduler', type=type(exp_lr_scheduler), default=exp_lr_scheduler)
		parser.add_argument('--dataset_name', type=str, default=self.dataset_name)
		parser.add_argument('--stage', type=str, default=self.stage)
		parser.add_argument('--label_types', type=type(self.dataset_confs.label_types), default=self.dataset_confs.label_types)

		return parser.parse_args()
