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
import utils

class Solver:
    def __init__(self, net, model_confs, solver_confs):
        # data args
        self.data_loaders = solver_confs.data_loaders
        self.data_sizes = solver_confs.data_sizes
        self.T = model_confs.num_frames
        # net args
        self.net = net
        # model training args
        self.gpu = solver_confs.gpu
        self.num_epochs = solver_confs.num_epochs
        self.optimizer = solver_confs.optimizer
        self.criterion = solver_confs.criterion
        self.scheduler = solver_confs.exp_lr_scheduler
        self.save_path = os.path.join('./weights', solver_confs.dataset_name, solver_confs.stage)
        if not(os.path.exists(self.save_path)):
            os.makedirs(self.save_path)

    def training(self, inputs, labels, phase):
        # Get the inputs, and wrap them in Variable
        inputs, labels = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
        # Zero the parameter gradients
        self.optimizer.zero_grad()
        # Forward
        _, outputs = self.net(inputs)

        if phase == 'test':
            _, preds = torch.max(torch.mean(outputs.data, 0).view(1, -1), 1)
        else:
            _, preds = torch.max(outputs, 1)

        loss = self.criterion(outputs, labels)

        # Backward + optimize(update parameters) only if in training phase
        if phase == 'trainval':
            loss.backward()
            self.optimizer.step()    
        
        # statistics
        self.running_loss += loss.item()
        self.running_corrects += torch.sum(preds == labels)

    def train_model(self, phases=['trainval', 'test']):
        best_model_wts = self.net.state_dict()
        best_acc = 0.0
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)
            # Each epoch has a training and evaluate phase
            for phase in phases:
                since = time.time()
                if phase == 'trainval':
                    self.scheduler.step()
                    self.net.train()  # Set model to training mode
                else:
                    self.net.eval()  # Set model to evaluate mode
                
                self.running_loss = 0.0
                self.running_corrects = 0.0

                # Iterate over data.
                for data in self.data_loaders[phase]:
                    # get the inputs
                    inputs, labels = data[0].float(), data[1]

                    if self.gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    self.training(inputs, labels, phase)
                
                epoch_loss = float(self.running_loss) / (self.data_sizes[phase])
                epoch_acc = float(self.running_corrects) / \
                    (self.data_sizes[phase])

                # display related Info(Loss, Acc, Time, etc.)
                print('Epoch: {} phase: {} Loss: {} Acc: {}'.format(
                    epoch, phase, epoch_loss, epoch_acc))
                time_elapsed = time.time() - since
                print('Running this epoch in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))

                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(self.net.state_dict(),
                               self.save_path + 'best_wts.pkl')
        print('Best test Acc: {:4f}'.format(best_acc))

        
    def evaluate(self):
        with torch.no_grad():
            for i, data in enumerate(self.data_loaders['test']):
                inputs = data[0].cuda()
                target = data[1][0]
                # compute output
                _, outputs = self.model(inputs)
                probs = F.softmax(outputs.data)
                #print probs
                _, pred = torch.max(torch.mean(outputs.data, 0).view(1, -1), 1)
                self.preds.append(pred.cpu().numpy())
                self.targets.append(target)

        ### show result
        preds = np.asarray(self.preds, dtype=int)
        #preds = label_map(preds)
        targets = np.asarray(self.targets, dtype=int)
        #labels = label_map(labels)
        preds, targets = preds.reshape(-1,1), targets.reshape(-1,1)
        
        print("Classification report for classifier \n %s" % (metrics.classification_report(targets, preds)))
        print("Confusion matrix:\n%s" % utils.normlize(metrics.confusion_matrix(targets, preds)))
        print np.sum(preds == targets) / float(targets.shape[0])

        # Compute confusion matrix
        cnf_matrix = metrics.confusion_matrix(targets, preds)
        print cnf_matrix
