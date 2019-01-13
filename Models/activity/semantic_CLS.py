import torch.nn as nn
import torch
import torch.nn.functional as F

__all__ = ['semantic_CLS', 'Semantic_CLS']


class Semantic_CLS(nn.Module):

    def __init__(self, model_confs):
        super(Semantic_CLS, self).__init__()
        self.do_attention = False
        self.fc = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(9 * model_confs.num_groups, 128),
            nn.Tanh(),
            #nn.ReLU(inplace=True),
        )
        self.attention_fun = nn.Sequential(
            #nn.Dropout(0.9),
            nn.Linear(9, 9),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(128, model_confs.num_classes)

    def forward(self, x):
        if self.do_attention:
            lx, rx = torch.chunk(x, 2, 1)
            l_alpha = F.softmax(torch.squeeze(self.attention_fun(lx)))
            r_alpha = F.softmax(torch.squeeze(self.attention_fun(rx)))
            lx = l_alpha * lx
            rx = r_alpha * rx
            x = torch.cat((lx, rx), 1)

        x = self.fc(x)
        out = self.classifier(x)
        return None, out.view(-1, out.size(-1))


def semantic_CLS(pretrained=False, model_confs=None, **kwargs):
    model = Semantic_CLS(model_confs, **kwargs)
    return model
