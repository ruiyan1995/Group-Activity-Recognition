"""
	Data_Configs
"""
import argparse
import os


class Data_Configs(object):
    """docstring for Dataset_Configs"""

    def __init__(self, dataset_root, dataset_name, stage, mode):
        super(Data_Configs, self).__init__()
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.stage = stage
        self.mode = mode

        self.confs_dict = {
            'action': {
                # you can set it as the case may be, such as 'img', 'npy',
                # 'hdf5', and so on.
                'data_type': 'img',
                'cur_folder': {
                    'trainval_action': 'person_imgs',
                    'extract_action_feas': 'person_imgs',
                    'frame_trainval_activity': 'imgs'
                },
                'label_types': {'trainval_action': ['action'], 'extract_action_feas': ['activity'], 'frame_trainval_activity': ['frame_activity']},
                'batch_size': {'trainval_action': {'trainval': 100, 'test': 10}, 'extract_action_feas': {'trainval': 60, 'test': 60} }
            },
            'activity': {
                'data_type': 'npy',
                'cur_folder': {
                    'trainval_activity': 'feas'
                },
                'label_types': {'trainval_activity': ['activity']},
                'batch_size': {'trainval_activity': {'trainval': 1000, 'test': 10}}
            },
            'semantic': {
                'data_type': 'npy',
                'cur_folder': {'trainval_activity': 'feas'},
                'label_types': {'trainval_activity': ['TF_IDF']},
                'batch_size': {'trainval': 10, 'test': 1000}
            }
        }

    def configuring(self):
        dataset_confs = self.confs_dict[self.stage]
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_folder', type=str, default=os.path.join(
            self.dataset_root, self.dataset_name, dataset_confs['cur_folder'][self.mode]), help='')
        parser.add_argument('--batch_size', type=dict,
                            default=dataset_confs['batch_size'][self.mode])

        parser.add_argument('--data_type', type=str, default=dataset_confs[
                            'data_type'], choices=['img', 'hdf5', 'npy'], help='the story type for data')
        parser.add_argument('--label_types', type=list, default=dataset_confs['label_types'][self.mode], choices=[['action'], ['action_feas'], [
                            'temporal_action_feas'], ['CNNLSTM_action_feas'], ['action_feas', 'frame_feas']], help='the label types for data')
        return parser.parse_args()
