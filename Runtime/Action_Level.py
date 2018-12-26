from Piplines import *
import Dataset
import Models
class Action_Level(Piplines):
	"""docstring for Action_Level"""
	def __init__(self, dataset_root, dataset_name, mode):
		super(Action_Level, self).__init__(dataset_root, dataset_name, 'action', mode)

		
	def extractFeas(self, save_folder):
		pass
		print 'Done, the features files are saved at ' + save_folder + '\n'


	def __loadModel(self, model_confs):
		net = models.resnet_LSTM(pretrained=True, model_confs=model_confs)
		return net
