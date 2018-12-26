"""
	Dataset_Configs
"""
import argparse
import os
class Dataset_Configs(object):
	"""docstring for Dataset_Configs"""
	def __init__(self, dataset_root, dataset_name, stage, mode):
		super(Dataset_Configs, self).__init__()
		self.dataset_root = dataset_root
		self.dataset_name = dataset_name
		self.stage = stage
		self.mode = mode

		self.confs_dict = {
			'action':{
				'data_type': 'img', # you can set it as the case may be, such as 'img', 'npy', 'hdf5', and so on.
				'cur_folder': {
					'trainval_action':'person_imgs',
					'extract_action_feas': 'person_imgs',
					'frame_trainval_activity': 'imgs'
				},
				'label_types': {'trainval_action':['action'], 'extract_action_feas':['activity'], 'frame_trainval_activity':['frame_activity']},
				'batch_size': {'trainval':80,'test':10}
			},
			'activity':{
				'data_type': 'npy',
				'cur_folder': {
					'CNNLSTM': 'feas'
				},
				'label_types': {'CNNLSTM':['CNNLSTM_action_feas']},
				'batch_size': {'trainval':2000,'test':10}
			}
		}


	def configuring(self):
		dataset_confs = self.confs_dict[self.stage]
		parser = argparse.ArgumentParser()
		parser.add_argument('--dataset_folder', type=str, default=os.path.join(self.dataset_root, self.dataset_name, dataset_confs['cur_folder'][self.mode]), help='')
		parser.add_argument('--batch_size', type=dict, default=dataset_confs['batch_size'])

		parser.add_argument('--data_type', type=str, default=dataset_confs['data_type'], choices=['img', 'hdf5', 'npy'], help='the story type for data')
		parser.add_argument('--label_types', type=list, default=dataset_confs['label_types'][self.mode], choices=[['action'], ['action_feas'], ['temporal_action_feas'], ['CNNLSTM_action_feas'], ['action_feas', 'frame_feas']], help='the label types for data')
		return parser.parse_args()



		
