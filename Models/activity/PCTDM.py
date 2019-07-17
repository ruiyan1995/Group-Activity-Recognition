import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import utils
import math
from torch.autograd import Variable
__all__ = ['PCTDM', 'pCTDM']



class pCTDM(nn.Module):
    def __init__(self, model_confs):
        super(pCTDM, self).__init__()
        self.input_size = 7096
        self.hidden_size = 1000
        self.num_players = model_confs.num_players
        self.num_classes = model_confs.num_classes
        self.num_groups = model_confs.num_groups
        self.do_attention = True
        self.do_one_to_all = True
        self.do_early_pooling = True
        self.interaction = True

        if self.interaction:
            self.Bi_Lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers = 1, batch_first = True, bidirectional=True)
            self.early_pooling = nn.MaxPool2d((2,1), stride = (2,1))
            self.fc_last = nn.Linear(self.hidden_size*(1+(not self.do_early_pooling))*self.num_groups, self.num_classes)
        else:
            self.fc_last = nn.Linear(self.input_size*self.num_groups, self.num_classes)
        
        
        if self.do_attention:
            #self.attention_fun = nn.Linear(1000, 1)
            fea_size = self.hidden_size*(1+(not self.do_early_pooling))
            
            self.att_source_weights = nn.Sequential(
                nn.Linear(fea_size, fea_size, bias=True),
            )
            self.att_context_weights = nn.Sequential(
                nn.Linear(fea_size, fea_size, bias=True),
            )
            self.att_extra_weights = nn.Sequential(
                nn.Linear(fea_size, 1, bias=True),
            )
            
            
            if self.do_one_to_all:
                self.Intra_Group_LSTM = nn.LSTM(fea_size, fea_size, num_layers = 1, batch_first = True)
        else:
            self.pool = nn.MaxPool2d((self.num_players/self.num_groups, 1), stride = (self.num_players/self.num_groups, 1))
    
    def get_att_weigths(self, x_s, context=None):
        gammas = {}
        if context is not None:
            context = torch.unsqueeze(context, 1).repeat(1, self.num_players/self.num_groups, 1)
            for g in xrange(self.num_groups):
                gammas[g] = F.softmax(torch.squeeze(self.att_extra_weights(
                   F.tanh(self.att_source_weights(x_s[g])+self.att_context_weights(context))))).view(-1,1,self.num_players/self.num_groups)
        else:
            for g in self.num_groups:
                gammas[g] = F.softmax(torch.squeeze(F.tanh(self.att_source_weights(x_s[g])))).view(-1,1,self.num_players/self.num_groups)

        return gammas
    
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
            x_s = torch.chunk(x, self.num_groups, 1)
            context = torch.mean(x, 1)
            gammas = self.get_att_weigths(x_s, context)
            
            group_feas = {}
            if self.do_one_to_all:
            # one to all LSTM, output last node in each group
                for g in xrange(self.num_groups):
                    gammas[g] = gammas[g].view(gammas[g].size(0),-1,1)
                    group_feas[g], _ = self.Intra_Group_LSTM(x_s[g] + x_s[g]*gammas[g])
                    group_feas[g] = group_feas[g][:,-1,:]
                x = torch.cat(tuple(group_feas.values()), 1)
            else:
                #x = torch.cat((torch.bmm(lgamma, lx), torch.bmm(rgamma, rx)), 2)
                for g in xrange(self.num_groups):
                    gammas[g] = gammas[g].view(gammas[g].size(0),-1,1)
                    group_feas[g]= torch.bmm(gammas[g], x_s[g])
                x = torch.cat(tuple(group_feas.values()), 2)

            x = torch.squeeze(x)

        else:
            # do intra-group pooling
            x = torch.cat(torch.chunk(x, self.num_groups, 2), 3)
            x = self.pool(x)
            x = x.view(x.size(0), -1)

        out = self.fc_last(x)
        return x, out


def PCTDM(pretrained=False, **kwargs):
    model = pCTDM(**kwargs)
    if pretrained:
        pretrained_dict = torch.load('****.pkl')
        model.load_state_dict(pretrained_dict)
    return model
