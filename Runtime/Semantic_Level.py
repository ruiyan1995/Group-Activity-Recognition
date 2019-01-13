import Data
import Models
from Piplines import *
import os
class Semantic_Level(Piplines):
	"""docstring for Semantic_Level"""
	def __init__(self, dataset_root, dataset_name, mode):
		self.dataset_root = dataset_root
		self.dataset_name = dataset_name
		super(Semantic_Level, self).__init__(dataset_root, dataset_name, 'semantic', mode)

		
	def shareAttentions(self, save_folder):
		pass
		print 'Done, the features files are saved at ' + save_folder + '\n'


	def loadModel(self, model_confs):
		net = Models.semantic_CLS(pretrained=False, model_confs=model_confs)
		print net
		return net