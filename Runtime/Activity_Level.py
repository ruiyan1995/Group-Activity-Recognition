from Piplines import *
class Activity_Level(Piplines):
	"""docstring for Action_Level"""
	def __init__(self, dataset_root, dataset_name, mode):
		super(Activity_Level, self).__init__(dataset_root, dataset_name, 'activity', mode)


	def loadModel(self, pretrained=False):
		if 'trainval' in self.mode:
			pretrained=False
			net = Models.PCTDM(pretrained=pretrained, model_confs=self.model_confs)
		return net
