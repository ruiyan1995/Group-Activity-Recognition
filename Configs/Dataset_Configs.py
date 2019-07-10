"""
    Pre_Configs
"""
import argparse
import os


class Dataset_Configs(object):
    """docstring for Pre_Configs"""

    def __init__(self, dataset_root, dataset_name):
        super(Dataset_Configs, self).__init__()
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.confs_dict = {
            'VD': {
                'num_players': 12,
                'num_classes':{'action':9, 'activity':8},
                'num_videos': 55,
                'action_list': ['blocking', 'digging', 'falling', 'jumping', 'moving', 'setting', 'spiking', 'standing', 'waiting'],
                'activity_list': ['l-pass', 'r-pass', 'l_set', 'r_set', 'l-spike', 'r_spike', 'l_winpoint', 'r_winpoint'],
                'splits': {
                    'trainval': [0, 1, 2, 3, 6, 7, 8, 10, 12, 13, 15, 16, 17, 18, 19,
                                 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 36, 38, 39,
                                 40, 41, 42, 46, 48, 49, 50, 51, 52, 53, 54],
                    'test': [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
                }
            },
            'CAD': {
                'num_players': 5,
                'num_classes':{'action':5, 'activity':4},
                'num_videos': 44,
                'action_list': ['Walking', 'Crossing', 'Waiting', 'Queuing', 'Talking'],
                'activity_list': ['Moving', 'Waiting', 'Queuing', 'Talking'],
                'splits': {
                    'trainval': [7, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                 21, 22, 23, 24, 26, 27, 30, 31, 32, 33,
                                 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
                    'test': [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 25, 28, 29]
                }
            }
        }

    def configuring(self):
        dataset_confs = self.confs_dict[self.dataset_name]
        parser = argparse.ArgumentParser()
        parser.add_argument('--num_players', type=int,
                            default=dataset_confs['num_players'])
        parser.add_argument('--num_classes', type=dict,
                            default=dataset_confs['num_classes'])
        parser.add_argument('--num_videos', type=dict,
                            default=dataset_confs['num_videos'])
        parser.add_argument('--action_list', type=list,
                            default=dataset_confs['action_list'])
        parser.add_argument('--activity_list', type=list,
                            default=dataset_confs['activity_list'])
        parser.add_argument('--splits', type=dict,
                            default=dataset_confs['splits'])

        return parser.parse_args()
