"""
	Common Piplines
"""
from abc import ABCMeta, abstractmethod
from torchvision import transforms
import torch

import Configs
import Data
import Models
import Solver


class Piplines(object):
    """docstring for Piplines"""

    def __init__(self, dataset_root, dataset_name, stage, mode):
        super(Piplines, self).__init__()
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.stage = stage
        # Dataset configs:
        self.data_confs = Configs.Data_Configs(
            dataset_root, dataset_name, stage, mode).configuring()
        print self.data_confs
        self.data_loaders, self.data_sizes = self.loadData(self.data_confs)
        # Model configs:
        self.model_confs = Configs.Model_Configs(
            dataset_name, stage).configuring()

        if torch.cuda.is_available():
            self.net = self.loadModel(self.model_confs).cuda()
        else:
            self.net = self.loadModel(self.model_confs)

        # Solver configs:
        self.solver_confs = Configs.Solver_Configs(
            dataset_name, stage, self.net, self.data_confs).configuring()

    def loadModel(self, model_confs):
        raise NotImplementedError
        #net = Models.resnet50_LSTM(pretrained=True, model_confs=model_confs)
        # print net
        # return net

    def trainval(self):
        solver = Solver.Solver(self.stage, self.data_loaders, self.data_sizes,
                               self.net, self.solver_confs)
        solver.train_model()

    def test(self):
        #self.data_loaders = __loadData(self.dataset_confs, phases = ['test'])
        pass

    def loadData(self, data_confs, phases=['trainval', 'test']):
        if data_confs.data_type == 'img':
            data_transforms = {
                'trainval': transforms.Compose([
                    transforms.Resize(224),
                    # transforms.RandomResizedCrop(224),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'test': transforms.Compose([
                    transforms.Resize(224),
                    # transforms.CenterCrop(224),
                    transforms.ToTensor()
                    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }
        else:
            data_transforms = None

        dataset = {phase: eval('Data.' + self.dataset_name + '_' + data_confs.data_type)(
            data_confs.dataset_folder, phase, data_confs.label_types, data_transforms[phase] if data_transforms else None) for phase in phases}
        data_loaders = {phase: torch.utils.data.DataLoader(dataset[phase], batch_size=data_confs.batch_size[
                                                           phase], shuffle=False, num_workers=8) for phase in phases}
        data_sizes = {phase: len(dataset[phase]) for phase in phases}
        return data_loaders, data_sizes
