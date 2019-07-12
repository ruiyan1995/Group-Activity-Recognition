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
                'data_type': {
                    'trainval_action': 'img',
                    'extract_action_feas': 'img',
                },
                'cur_folder': {
                    'trainval_action': 'imgs_ranked',
                    'extract_action_feas': 'imgs_ranked',
                },
                'label_type': {'trainval_action': 'action', 'extract_action_feas': 'activity'},
                'batch_size': {
                    'trainval_action': {
                        'VD':{'trainval': 300, 'test': 10},
                        'CAD':{'trainval': 300, 'test': 10}
                    }, 
                    'extract_action_feas': {
                        'VD':{'trainval': 120, 'test': 120},
                        'CAD':{'trainval': 50, 'test': 50}
                    }
                }
            },
            'activity': {
                'data_type': {
                    'trainval_activity': 'npy',
                },
                'cur_folder': {
                    'trainval_activity': 'feas',
                },
                'label_type': {'trainval_activity': 'activity', 'end_to_end':'activity'},
                'batch_size': {'trainval_activity': {'trainval': 500, 'test': 10}}
            }
        }

    def configuring(self):
        dataset_confs = self.confs_dict[self.stage]
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_folder', type=str, default=os.path.join(
            self.dataset_root, self.dataset_name, dataset_confs['cur_folder'][self.mode]), help='')
        parser.add_argument('--batch_size', type=dict,
                            default=dataset_confs['batch_size'][self.mode][self.dataset_name])

        parser.add_argument('--data_type', type=str, default=dataset_confs[
                            'data_type'][self.mode], choices=['img', 'hdf5', 'npy'], help='the story type for data')
        parser.add_argument('--label_type', type=str, default=dataset_confs['label_type'][self.mode], help='the label type for data')
        return parser.parse_args()
