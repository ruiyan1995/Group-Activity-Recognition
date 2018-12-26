from Piplines import *
from Dataset import VD_img, CAD_img
class Activity_Level(Piplines):
	"""docstring for Action_Level"""
	def __init__(self, dataset_root, dataset_name):
		super(Activity_Level, self).__init__(dataset_root, dataset_name, 'activity')

		
	def extractFeas(self, save_folder):
		pass
		print 'Done, the features files are saved at ' + save_folder + '\n'


	def __loadModel(self, model_confs):
		net = models.alexNet_LSTM(pretrained=True, model_confs=model_confs)
		return net
