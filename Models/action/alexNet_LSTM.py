import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet_LSTM', 'alexNet_LSTM']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet_LSTM(nn.Module):

    def __init__(self, model_confs):
        super(AlexNet_LSTM, self).__init__()
        self.do_LSTM = True
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        if self.do_LSTM:
            self.LSTM = nn.LSTM(input_size=4096, hidden_size=3000, num_layers=1, batch_first=True)
            self.classifier = nn.Linear(3000, model_confs.num_classes)
        else:
            self.classifier = nn.Linear(4096, model_confs.num_classes)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)

        x = self.fc(x)
        x = x.view(x.size(0)/10, 10, x.size(1))
        cnn_feas = x.contiguous()

        if self.do_LSTM:
            x, (h,c) = self.LSTM(x)
            lstm_feas = x.contiguous()
            # get feas
            feas = torch.cat((cnn_feas, lstm_feas), -1)
        else:
            # get feas
            feas = cnn_feas

        ###########################################
        feas = torch.transpose(feas, 0, 1)
        feas = feas.contiguous()
        feas = feas.view(feas.size(0),-1)
        #print feas.size()
        ###########################################

        out = self.classifier(x)
        out = out.view(out.size(0)*out.size(1),-1)

        return feas, out
        #return out


def alexNet_LSTM(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet_LSTM(**kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = torch.load('/home/ubuntu/MM/Run/VD_CNNLSTM.pkl')
        #pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.split('.')[0] != 'classifier' }
        '''for k,v in pretrained_dict.items():
            print k.split('.')[0]'''
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model
