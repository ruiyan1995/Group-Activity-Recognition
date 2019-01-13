from Piplines import *
import Models
from torch.autograd import Variable
import os
import numpy as np
class Action_Level(Piplines):
	"""docstring for Action_Level"""
	def __init__(self, dataset_root, dataset_name, mode):
		super(Action_Level, self).__init__(dataset_root, dataset_name, 'action', mode)

	def loadModel(self, model_confs):
		net = Models.resnet50_LSTM(pretrained=True, model_confs=model_confs)
		print net
		return net

	def extractFeas(self, feas_prefix = 'resnet50_LSTM_action_'):
		# args
		dataset_confs = Configs.Dataset_Configs(self.dataset_root, self.dataset_name).configuring()
		K = dataset_confs.num_players
		feas_size = 5048
		feas_size = 3000
		
		# through model
		for phase in ['trainval', 'test']:
			batch_size = 10
			dataset_size = self.data_sizes[phase]
			data_loader = self.data_loaders[phase]
			print phase, dataset_size/K
			#create data_file
			filename = os.path.join(self.dataset_root, self.dataset_name, 'feas', feas_prefix + phase + '.npy') 
			feas = np.zeros([dataset_size/K, feas_size*K+1])
			np.save(filename, feas)
			print 'The features files are created at ' + filename + '\n'

			feas = np.load(filename, mmap_mode = 'r+')
			i = 0
			flag = True
			for data in data_loader:
	            # get the inputs
				inputs, labels = data.values()[0]
				print batch_size*i,'/',dataset_size/K
	            # wrap them in Variable
				inputs = Variable(inputs.float().cuda()) if self.solver_confs.gpu else Variable(inputs.float())
	            # forward
				try:
				    fea, _ = self.net.forward(inputs)
				except RuntimeError as e:
				    if 'out of memory' in str(e):
				        print('| WARNING: ran out of memory')
				        if hasattr(torch.cuda, 'empty_cache'):
				            torch.cuda.empty_cache()
				    else:
				        raise e
				if flag: # even
					#print fea.size()
					feas[i*batch_size:(i+1)*batch_size,:feas_size*K/2] = fea.data.cpu().numpy()
					feas[i*batch_size:(i+1)*batch_size,-1] = labels.cpu().numpy()[:10]
					flag = False
				else: # old
					#print fea.size()
					feas[i*batch_size:(i+1)*batch_size,feas_size*K/2:-1] = fea.data.cpu().numpy()
					feas[i*batch_size:(i+1)*batch_size,-1] = labels.cpu().numpy()[:10]
					flag = True
					i = i+1
		print 'Done, the action features are saved at ' + filename + '\n'