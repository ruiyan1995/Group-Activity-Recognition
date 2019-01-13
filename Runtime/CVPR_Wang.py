"""
CVPR_Wang
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import os
import Solver
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn import datasets, svm, metrics
import models
from VolleyballDataset import *
import utils
import matplotlib.pyplot as plt
import itertools
use_gpu = torch.cuda.is_available()
K_players = 12

class_names = ['lpass', 'rpass', 'lset', 'rset', 'lspike', 'rspikle', 'lwin', 'rwin']





def label_map(labels):
	Map = [0,5,1,2,3,4,6,7]
	for i in range(labels.shape[0]):
		labels[i] = Map[labels[i]]
	return labels

def normlize(matrix):
	matrix = np.asarray(matrix, dtype=float)
	return np.round(matrix/np.sum(matrix, 1).reshape(-1,1)*100, decimals=2)

def Group_level_train():
	# Group_level_train
    # init
    num_epochs = 100
    batch_size = {'trainval':1000,'test':10}# 500, 10
    step_size = 10
    gamma = 0.1
    # data
    train_dataFile = '/home/ubuntu/MM/CNNLSTM_features_trainval.npy'
    test_dataFile = '/home/ubuntu/MM/CNNLSTM_features_test.npy'
    # model
    net = models.CVPR_Wang.group_network(input_size = 7096, hidden_size = 1000, K_players = K_players)
    if use_gpu:
        net = net.cuda()

    # define a loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # softmax + nll_loss
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9,0.9))

    solver = Solver.Solver(net, criterion, optimizer, step_size = step_size, gamma = gamma, num_epochs = num_epochs, batch_size = batch_size, use_gpu = use_gpu)
    solver.set_Data(dataFile = {'trainval':train_dataFile, 'test': test_dataFile})
    solver.batch_train_model(save_prefix = '/home/ubuntu/MM/models/CVPR_Wang/')

def Group_level_test():
    """
        Given a video, output a group activity label
    """

    # data: read 120 frames data of a video from *.npy
    testdata_File = '/home/ubuntu/MM/CNNLSTM_features_test.npy'
    testdata = np.load(testdata_File, mmap_mode = 'r')

    batchsize = 10

    # model: bilstm_maxpooling
    net = models.CVPR_Wang.group_network(pretrained=True, input_size=7096, hidden_size=1000, K_players=K_players)
    net = net.cuda()
    
    preds = []
    labels = []
    # forward
    for i in range(len(testdata)/batchsize):
        data = testdata[i*batchsize:batchsize*(i+1)]
        inputs = torch.from_numpy(data[:,:-1]).float()
        label = data[0,-1]
        if use_gpu:
            inputs = inputs.cuda()
        inputs = Variable(inputs, requires_grad=False)
        #print inputs.size()
        _, outputs = net.forward(inputs)
        probs = F.softmax(outputs.data)
        #print torch.sum(probs,1)
        #print probs
        _, pred = torch.max(torch.mean(outputs.data, 0).view(1, -1), 1)
        preds.append(pred.cpu().numpy())
        labels.append(label)
        #break
    ### step 5: result
    preds = np.asarray(preds, dtype=int)
    preds = label_map(preds)
    labels = np.asarray(labels, dtype=int)
    labels = label_map(labels)
    preds, labels = preds.reshape(-1,1), labels.reshape(-1,1)
    print("Classification report for classifier \n %s" % (metrics.classification_report(labels, preds)))
    print("Confusion matrix:\n%s" % normlize(metrics.confusion_matrix(labels, preds)))
    print np.sum(preds == labels), preds.shape, labels.shape
    print np.sum(preds == labels) / float(labels.shape[0])
    # Compute confusion matrix
    cnf_matrix = metrics.confusion_matrix(labels, preds)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    utils.plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    utils.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

    plt.show()

if __name__ == '__main__':
	Group_level_test()