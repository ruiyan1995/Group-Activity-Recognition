"""
	Dataset_preprocessing
"""
import Configs
from VD_Track import *
from CAD_Track import *
from Rank_MI import *

class Processing(object):
    """Preprocessing dataset, e.g., track, split and anonatation."""

    def __init__(self, dataset_root, dataset_name, operation=None, ranked=False):
        super(Processing, self).__init__()
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name

        # Pre configs:
        dataset_confs = Configs.Dataset_Configs(dataset_root, dataset_name).configuring()
        #print dataset_confs
        model_confs = Configs.Model_Configs(dataset_name, 'action').configuring()
        # track
        if operation == None:
            print 'Please choose one of the operation! Track'
        else:
            eval(self.dataset_name + '_' + str.capitalize(operation))(self.dataset_root, dataset_confs, model_confs, ranked)
