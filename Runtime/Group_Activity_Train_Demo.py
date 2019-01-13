"""
	Group_Activity Train_Demo
"""
import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import os
import Solver
from torch.utils.data import DataLoader, TensorDataset

import models
from dataset import VolleyballDataset_hdf5, VolleyballDataset_img,  CADataset_img

import utils
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
use_gpu = torch.cuda.is_available()
K_players = 12


### Step 1: Person Level
def Person_level():
    # Person_level
    Le1_num_epochs = 100
    Le1_batch_size = {'trainval':300,'test':10}# 500, 10
    Le1_step_size = 10
    Le1_gamma = 0.1
    #dataset_Class = 'VolleyballDataset_img'
    dataset_Class = 'CADataset_img'
    # dataset
    if dataset_Class == 'VolleyballDataset_img':
        Num_classes = 9
        #data_root = '/home/ubuntu/caffe-lstm/examples/deep-activity-rec/eclipse-project/fused_data/'
        data_root = '/home/ubuntu/GCNN/dataset/VD/person_imgs/'
    else:
        Num_classes = 5
        data_root = '/home/ubuntu/GCNN/dataset/CAD/person_imgs/'


    #image_datasets = {x: VolleyballDataset(Volleyball_data_root, x, 'action') for x in ['trainval', 'test']}
    image_datasets = {x: eval(dataset_Class)(data_root, x, 'action') for x in ['trainval', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=Le1_batch_size[x], shuffle=False, num_workers=8) for x in ['trainval', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['trainval', 'test']}
    print dataset_sizes['trainval'], dataset_sizes['test']
    cnn_lstm = models.alexNet_LSTM(pretrained=True, num_classes=Num_classes)
    print cnn_lstm

    if use_gpu:
	   cnn_lstm = cnn_lstm.cuda()
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(cnn_lstm.parameters(), lr=0.001, momentum=0.9)
    #optimizer = torch.optim.Adam(cnn_lstm.parameters(), lr=0.0001, betas=(0.9,0.9))
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    cnn_lstm = utils.train_model(cnn_lstm, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler,
                       use_gpu=use_gpu, num_epochs=100)
    #torch.save(cnn_lstm.state_dict(), 'Volleyball_CNN.pkl')

### Step 2: Group Level
def Group_level():
    # Group_level
    # init
    num_epochs = 50
    batch_size = {'trainval':500,'test':10}# 500, 10
    step_size = 10
    gamma = 0.1
    # data
    train_dataFile = '/home/ubuntu/MM/CNNLSTM_features_trainval.npy'
    test_dataFile = '/home/ubuntu/MM/CNNLSTM_features_test.npy'
    # model
    #net = models.Group_level.bilstm_Maxpooling(input_size = 7096, hidden_size = 1000, K_players = K_players)
    net = models.B2(input_size = 7096, hidden_size = 1000, K_players = K_players)
    if use_gpu:
        net = net.cuda()
    print net
    # define a loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # softmax + nll_loss
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9,0.9))

    solver = Solver.Solver(net, criterion, optimizer, step_size = step_size, gamma = gamma, num_epochs = num_epochs, batch_size = batch_size, use_gpu = use_gpu, scene = False)
    solver.set_Data(dataFile = {'trainval':train_dataFile, 'test': test_dataFile})
    solver.batch_train_model(save_prefix = '/home/ubuntu/MM/models/temp/')

### Step 3: Scene Level
def Scene_level():
    # init
    num_epochs = 100
    batch_size = {'trainval':200,'test':10}# 500, 10
    step_size = 10
    gamma = 0.1
    # data
    train_dataFile = '/home/ubuntu/MM/Group_Level_feas_trainval.npy'
    test_dataFile = '/home/ubuntu/MM/Group_Level_feas_test.npy'
    # model
    #net = models.Group_level.bilstm_Maxpooling(input_size = 7096, hidden_size = 1000, K_players = K_players)
    #pretrained=True
    net = models.scene_lstm(input_size = 2000, hidden_size = 2000, K_players = K_players)
    if use_gpu:
        net = net.cuda()

    # define a loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # softmax + nll_loss
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9,0.9))

    solver = Solver.Solver(net, criterion, optimizer, step_size = step_size, gamma = gamma, num_epochs = num_epochs, batch_size = batch_size, use_gpu = use_gpu, scene = True)
    solver.set_Data(dataFile = {'trainval':train_dataFile, 'test': test_dataFile})
    solver.batch_train_model(save_prefix = '/home/ubuntu/MM/models/Scene_level/')

if __name__ == '__main__':
    Person_level()
    #Group_level()
    #Scene_level()