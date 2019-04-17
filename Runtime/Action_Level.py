from Piplines import *
import Models
from torch.autograd import Variable
import os
import numpy as np
class Action_Level(Piplines):
    """docstring for Action_Level"""
    def __init__(self, dataset_root, dataset_name, mode):
        super(Action_Level, self).__init__(dataset_root, dataset_name, 'action', mode)

    def loadModel(self, pretrained=False):
        if 'trainval'in self.mode:
            pretrained=True
        net = Models.alexNet_LSTM(pretrained, model_confs=self.model_confs)
        return net

    def extractFeas(self):
        # args
        self.net.eval()
        self.net.load_state_dict(torch.load('./weights/VD/action/best_wts.pkl'))
        dataset_confs = Configs.Dataset_Configs(self.dataset_root, self.dataset_name).configuring()
        K = dataset_confs.num_players
        feas_size = 7096

        with torch.no_grad():
            # through model
            for phase in ['trainval', 'test']:
                batch_size = 10
                dataset_size = self.data_sizes[phase]
                data_loader = self.data_loaders[phase]
                print phase, dataset_size/K
                #create data_file
                filename = os.path.join(self.dataset_root, self.dataset_name, 'feas', 'activity', phase + '.npy')
                feas = np.zeros([dataset_size/K, feas_size*K+1])
                np.save(filename, feas)
                print 'The features files are created at ' + filename + '\n'

                feas = np.load(filename, mmap_mode = 'r+')
                i = 0
                flag = True
                for data in data_loader:
                    # get the inputs
                    inputs, labels = data
                    print batch_size*i,'/',dataset_size/K
                    # wrap them in Variable
                    inputs = Variable(inputs.float().cuda()) if torch.cuda.is_available() else Variable(inputs.float())
                    # forward
                    try:
                        fea, _ = self.net.forward(inputs)
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print('| WARNING: ran out of memory')
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            raise e
                    feas[i*batch_size:(i+1)*batch_size,:-1] = fea.data.cpu().numpy()
                    feas[i*batch_size:(i+1)*batch_size,-1] = labels.cpu().numpy()[:10]
                    i = i+1
            print 'Done, the action features are saved at ' + filename + '\n'
