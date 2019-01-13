from Piplines import *
class Activity_Level(Piplines):
	"""docstring for Action_Level"""
	def __init__(self, dataset_root, dataset_name, mode):
		super(Activity_Level, self).__init__(dataset_root, dataset_name, 'activity', mode)

		
	def extractFeas(self, save_folder):
		pass
		print 'Done, the features files are saved at ' + save_folder + '\n'


	def loadModel(self, model_confs):
		net = Models.one_to_all(pretrained=False, model_confs=model_confs)
		print net
		return net
