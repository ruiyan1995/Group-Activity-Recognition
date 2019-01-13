"""
	Dataset_preprocessing
"""
import Configs
from VD_Semantic import *

class Processing(object):
    """Preprocessing dataset, e.g., track, split and anonatation."""

    def __init__(self, dataset_root, dataset_name, operation=None):
        super(Processing, self).__init__()
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name

        # Pre configs:
        dataset_confs = Configs.Dataset_Configs(dataset_root, dataset_name).configuring()
        print dataset_confs
        model_confs = Configs.Model_Configs(dataset_name, 'action').configuring()
        # track or semantic
        if operation == None:
            print 'Please choose one of the operation! track or semantic...'
        else:
            eval(self.dataset_name + '_' +
                 operation)(self.dataset_root, dataset_confs, model_confs)
