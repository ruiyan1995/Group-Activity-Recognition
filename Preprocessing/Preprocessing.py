"""
	Dataset_preprocessing
"""

class Preprocessing(object):
	"""Preprocessing dataset, e.g., track, split and anonatation.

	Attributes:
		dataset_root: 
		dataset_name: 
	"""
	def __init__(self, dataset_root, dataset_name):
		super(Preprocessing, self).__init__()
		self.dataset_root = dataset_root
		self.dataset_name = dataset_name
		# track the persons
		eval(self.dataset_name+'_track')(self.dataset_root)

		