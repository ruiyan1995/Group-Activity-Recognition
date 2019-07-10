"""
	Solver_Configs
"""
import argparse
import torch
import torch.nn as nn
from torch.optim import lr_scheduler


class Solver_Configs(object):
    """docstring for Solver_Configs"""

    def __init__(self, dataset_name, data_loaders, data_sizes, net, stage, mode, dataset_confs):
        super(Solver_Configs, self).__init__()
        self.dataset_name = dataset_name
        self.dataset_confs = dataset_confs
        
        self.data_loaders = data_loaders
        self.data_sizes = data_sizes
        self.stage = stage
        self.net = net
        self.mode = mode

        self.confs_dict = {
            'VD': {
                'action': {
                    'num_epochs': {
                        'trainval_action': 10,
                        'extract_action_feas': 0
                    },
                    'lr_scheduler': {
                        'trainval_action': {'step_size': 5, 'gamma': 0.1},
                        'extract_action_feas': {'step_size': 5, 'gamma': 0.1}
                    },
                    'optimizer': {
                        'trainval_action': {'method': 'SGD', 'lr': 0.001, 'arg': 0.9},
                        'extract_action_feas': {'method': 'SGD', 'lr': 0.001, 'arg': 0.9}
                    }
                },
                'activity': {
                    'num_epochs': {
                        'trainval_activity': 20
                    },
                    'lr_scheduler': {
                        'trainval_activity': {'step_size': 10, 'gamma': 0.1}
                    },
                    'optimizer': {
                        'trainval_activity': {'method': 'Adam', 'lr': 0.0001, 'arg': (0.9, 0.9)}
                    }
                }
            },

            'CAD': {
                'action': {
                    'num_epochs': {
                        'trainval_action': 20,
                        'extract_action_feas': 20
                    },
                    'lr_scheduler': {
                        'trainval_action': {'step_size': 5, 'gamma': 0.1},
                        'extract_action_feas': {'step_size': 5, 'gamma': 0.1}
                    },
                    'optimizer': {
                        'trainval_action': {'method': 'SGD', 'lr': 0.001, 'arg': 0.9},
                        'extract_action_feas': {'method': 'SGD', 'lr': 0.001, 'arg': 0.9}
                    }
                },
                'activity': {
                    'num_epochs': {
                        'trainval_activity': 50
                    },
                    'lr_scheduler': {
                        'trainval_activity': {'step_size': 10, 'gamma': 0.1}
                    },
                    'optimizer': {
                        'trainval_activity': {'method': 'Adam', 'lr': 0.0001, 'arg': (0.9, 0.9)}
                    }
                }
            }
        }

    def configuring(self):
        solver_confs = self.confs_dict[self.dataset_name][self.stage]
        parser = argparse.ArgumentParser()
        parser.add_argument('--num_epochs', type=int,
                            default=solver_confs['num_epochs'][self.mode])
        parser.add_argument('--gpu', type=bool,
                            default=torch.cuda.is_available(), help='*****')

        criterion = nn.CrossEntropyLoss()

        parser.add_argument(
            '--criterion', type=type(criterion), default=criterion)

        optim = solver_confs['optimizer'][self.mode]
        optimizer = eval(
            'torch.optim.' + optim['method'])(self.net.parameters(), optim['lr'], optim['arg'])
        parser.add_argument(
            '--optimizer', type=type(optimizer), default=optimizer)

        # Decay LR by a factor of 0.1 every 7 epochs
        lr_sch = solver_confs['lr_scheduler'][self.mode]
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_sch[
                                               'step_size'], gamma=lr_sch['gamma'])
        parser.add_argument(
            '--exp_lr_scheduler', type=type(exp_lr_scheduler), default=exp_lr_scheduler)
        
        parser.add_argument('--data_loaders', type=type(self.data_loaders), default=self.data_loaders)
        parser.add_argument('--data_sizes', type=type(self.data_sizes), default=self.data_sizes)
        parser.add_argument('--dataset_name', type=str, default=self.dataset_name)
        parser.add_argument('--stage', type=str, default=self.stage)
        parser.add_argument('--mode', type=str, default=self.mode)

        return parser.parse_args()
