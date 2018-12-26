"""
	Model_Configs
"""
import argparse
class Model_Configs(object):
	"""docstring for Model_Configs"""
	def __init__(self, dataset_name, stage):
		super(Model_Configs, self).__init__()
		self.dataset_name = dataset_name
		self.stage = stage
		self.confs_dict = {
			'VD':{
				'num_frames': 10,
				'num_classes':{'action':9, 'activity':8},
				'num_players': 12,
				'num_groups': 2,
			},
			'CAD':{
				'num_frames': 10,
				'num_classes':{'action':5, 'activity':4},
				'num_players': 5,
				'num_groups': 1,
			}

		}

	def configuring(self):
		parser = argparse.ArgumentParser()
		model_confs = self.confs_dict[self.dataset_name]
		parser.add_argument('--num_classes', type=int, default=model_confs['num_classes'][self.stage])
		parser.add_argument('--num_players', type=int, default=model_confs['num_players'])
		parser.add_argument('--num_groups', type=int, default=model_confs['num_groups'])
		parser.add_argument('--num_frames', type=int, default=model_confs['num_frames'])
		return parser.parse_args()