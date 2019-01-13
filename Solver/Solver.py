# -*- coding:utf-8 -*-
import torch
from torch.autograd import Variable
import numpy as np
import time
import os
import gc
import sys
from torch.optim import lr_scheduler
import h5py


class Solver:
    # Setting basic parameters during Class init...

    def __init__(self, stage, data_loaders, dataset_sizes, net, solver_confs):
        self.stage = stage # action or activity or semantic
        # data args
        self.data_loaders = data_loaders
        self.dataset_sizes = dataset_sizes

        # net args
        self.net = net

        # model training args
        self.gpu = solver_confs.gpu
        self.num_epochs = solver_confs.num_epochs
        self.optimizer = solver_confs.optimizer
        self.criterion = solver_confs.criterion
        self.scheduler = solver_confs.exp_lr_scheduler
        self.label_types = solver_confs.label_types
        self.save_prefix = os.path.join(
            '/home/ubuntu/GAR/weights', solver_confs.dataset_name, solver_confs.stage, '_'.join(self.label_types))
        print self.save_prefix

    def training(self, inputs, labels, phase):
        # Get the inputs, and wrap them in Variable
        inputs = {label_type: Variable(
            inputs[label_type], requires_grad=False) for label_type in self.label_types}
        labels = {label_type: Variable(
            labels[label_type], requires_grad=False) for label_type in self.label_types}
        #labels = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
        # Zero the parameter gradients
        self.optimizer.zero_grad()
        # Forward
        if len(inputs.values()) > 1:
            _, outputs = self.net(inputs.values()[0], h_f=inputs.values()[1])
        else:
            _, outputs = self.net(inputs.values()[0])
        if phase == 'test' and self.stage != 'semantic':
            _, preds = torch.max(torch.mean(outputs.data, 0).view(1, -1), 1)
        else:
            _, preds = torch.max(outputs.data, 1)

        # print 'outputs size:', outputs.size()
        # print 'labels size:', labels.size()
        # print outputs, labels
        loss = self.criterion(outputs, labels.values()[0])
        # print loss
        # Backward + optimize(update parameters) only if in training phase
        if phase == 'trainval':
            loss.backward()
            self.optimizer.step()
        # ? Why is [0]? I need to find source code of '_WeightLoss'
        return preds, loss.data[0]

    def train_model(self):
        best_model_wts = self.net.state_dict()
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and evaluate phase
            for phase in ['trainval', 'test']:
                since = time.time()
                if phase == 'trainval':
                    self.scheduler.step()
                    self.net.train(True)  # Set model to training mode
                else:
                    self.net.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0.0

                # Iterate over data.
                for data in self.data_loaders[phase]:
                    # get the inputs
                    inputs = {label_type: data[label_type][
                        0].float() for label_type in self.label_types}
                    labels = {label_type: data[label_type][1]
                              for label_type in self.label_types}
                    if self.gpu:
                        for label_type in self.label_types:
                            inputs[label_type] = inputs[label_type].cuda()
                            labels[label_type] = labels[label_type].cuda()
                    '''inputs, labels = data
	                # wrap them in Variable
	                inputs = inputs.float()
	                if self.gpu:
	                    inputs = inputs.cuda()
	                    labels = labels.cuda()'''
                    preds, loss = self.training(inputs, labels, phase)

                    # statistics
                    running_loss += loss
                    #print preds, labels.values()[0]
                    running_corrects += torch.sum(preds == labels.values()[0])

                epoch_loss = float(running_loss) / (self.dataset_sizes[phase])
                epoch_acc = float(running_corrects) / \
                    (self.dataset_sizes[phase])

                # display related Info(Loss, Acc, Time, etc.)
                print('Epoch: {} phase: {} Loss: {} Acc: {}'.format(
                    epoch, phase, epoch_loss, epoch_acc))
                time_elapsed = time.time() - since
                print('Running this epoch in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))

    # deep copy the model
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    #best_model_wts = model.state_dict()
                    # deep copy the model
                    torch.save(self.net.state_dict(),
                               self.save_prefix + '.pkl')
        print('Best test Acc: {:4f}'.format(best_acc))
