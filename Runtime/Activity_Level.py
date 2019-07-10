from Piplines import *
class Activity_Level(Piplines):
    """docstring for Activity_Level"""
    def __init__(self, dataset_root, dataset_name, mode):
        super(Activity_Level, self).__init__(dataset_root, dataset_name, 'activity', mode)


    def loadModel(self, pretrained=False):
        if 'trainval' in self.mode:
            pretrained=False
            net = Models.PCTDM(pretrained=pretrained, model_confs=self.model_confs)
        return net
    
    def evaluate(self):
        pretrained_dict = torch.load('./weights/'+self.dataset_name+'/activity/best_wts.pkl')
        self.net = Models.PCTDM(pretrained=False, model_confs=self.model_confs)
        self.net.load_state_dict(pretrained_dict)
        self.net.eval()
        self.solver.evaluate()

