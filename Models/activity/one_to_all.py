import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import utils
from torch.autograd import Variable
__all__ = ['One_to_All', 'one_to_all']



class One_to_All(nn.Module):
    def __init__(self, model_confs):
        super(One_to_All, self).__init__()
        self.input_size = 2048 + 3000
        #self.input_size = 7096
        self.hidden_size = 1000
        self.num_players = model_confs.num_players
        self.num_classes = model_confs.num_classes
        self.do_attention = False
        self.do_one_to_all = False
        self.do_early_pooling = False
        self.interaction = False

        if self.interaction:
            self.Bi_Lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers = 1, batch_first = True, bidirectional=True)
            self.early_pooling = nn.MaxPool2d((2,1), stride = (2,1))
            self.fc_last = nn.Linear(self.hidden_size*(2+2*(not self.do_early_pooling)), self.num_classes)
        else:
            self.fc_last = nn.Linear(self.input_size*2, self.num_classes)
        
        
        if self.do_attention:
            #self.attention_fun = nn.Linear(1000, 1)
            fea_size = self.hidden_size*(1+(not self.do_early_pooling))
            self.attention_fun = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(fea_size, 1),
                #nn.Tanh()
                #nn.ReLU(inplace=True)
            )
            if self.do_one_to_all:
                self.Intra_Group_LSTM = nn.LSTM(fea_size, fea_size, num_layers = 1, batch_first = True)
        else:
            self.pool = nn.MaxPool2d((6, 1), stride = (6, 1))

    def forward(self, x):
        x = x.view(x.size(0), self.num_players, x.size(1)/self.num_players)
        # x = (batch, seq_len=K, input_size)
        # Ranking


        if self.interaction:
            lstm_out, (h, c) = self.Bi_Lstm(x) # x = (batch, seq_len, input_size)
            x = lstm_out.contiguous()
        
        x = x.view(x.size(0), 1, x.size(1), x.size(2)) # x = (batch, 1, seq_len, input_size)
        
        # do pooling: batch, 1, time_step(K_players), feas_size
        #print x.size()
        
        # early_pooling for bilstm, means that do not concat the double outputs of bilstm, just pooling them!!
        if self.do_early_pooling:
            #print x.size()
            x = x.view(x.size(0), x.size(1), x.size(2)*2, -1)
            #print x.size()
            x = self.early_pooling(x)
            #print x.size()

        #print x.size()
        # Intra-group Model
        if self.do_attention:
            # do intra-group attention
            # x(N,1,12,1000)
            x = torch.squeeze(x)
            # x(N,12,1000)
            lx, rx = torch.chunk(x, 2, 1)
            #print lx.size(),rx.size()
            lgamma = F.softmax(torch.squeeze(self.attention_fun(lx))).view(-1,1,6)
            rgamma = F.softmax(torch.squeeze(self.attention_fun(rx))).view(-1,1,6)
            
            if self.do_one_to_all:
            # one to all LSTM, output last node in each group
                lgamma, rgamma = lgamma.view(lgamma.size(0),-1,1), rgamma.view(rgamma.size(0),-1,1)
                lgamma.expand(lgamma.size(0),6,1000)
                rgamma.expand(rgamma.size(0),6,1000)
                #print lgamma.expand(lgamma.size(0),6,1000)*lx
                group1, _ = self.Intra_Group_LSTM(lx*lgamma)
                group2, _ = self.Intra_Group_LSTM(rx*rgamma)
                #print group1[:,-1,:].size(), group2[:,-1,:].size()
                x = torch.cat((group1[:,-1,:], group2[:,-1,:]), 1)
            else:
                x = torch.cat((torch.bmm(lgamma, lx), torch.bmm(rgamma, rx)), 2)
            
            x = torch.squeeze(x)

        else:
            # do intra-group pooling
            x = torch.cat(torch.chunk(x, 2, 2), 3)
            x = self.pool(x)
            x = x.view(x.size(0), -1)

        out = self.fc_last(x)
        return x, out


def one_to_all(pretrained=False, **kwargs):
    model = One_to_All(**kwargs)
    if pretrained:
        pretrained_dict = torch.load('/home/ubuntu/MM/models/ranked_one_to_all/best_test_acc.pkl')
        model.load_state_dict(pretrained_dict)
        '''for k,v in pretrained_dict.items():
            print k
        print model'''
    return model