# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import itertools

def loadData(file, batchSize, ):
	'''data = np.load(file, mmap_mode='r')
	dims = data.shape # data include fea,label, so the dims = fea_dim + 1
	X = data[]
	Y = data[dims]'''

def write_txt(txtFile, content_str, mode):
    with open(txtFile, mode) as f:
        f.write(content_str)

def trainning(net, criterion, optimizer, x_val, y_val):
    x_val = Variable(x_val.cuda(), requires_grad=False)
    y_val = Variable(y_val.cuda(), requires_grad=False)
    # Reset gradient
    optimizer.zero_grad()
    # Forward
    out = net(x_val)
    # print pro
    #loss = criterion(pro[:, 1], y_val.unsqueeze(1))
    loss = criterion(out, y_val)
    # Backward
    loss.backward()
    # Update parameters
    optimizer.step()
    return loss.data[0]


def predict(net, x_val):
    x = Variable(x_val.cuda(), requires_grad=False)
    output = net.forward(x)

    return output.cpu().data.numpy()[:, 1], output.cpu().data.numpy().argmax(axis=1)
    # return output.cpu().data.numpy().argmax(axis=1)


'''def extract_features(net, x_val):
    x = Variable(x_val.cuda(), requires_grad=False)
    fc, output = net.forward(x)
    return fc.cpu().data.numpy()'''


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def block_shuffle(List, block_length):
    """
        Args
    """
    base_indx = np.arange(0, len(List), block_length)
    np.random.shuffle(base_indx)
    indx = base_indx
    for i in range(block_length-1):
        new_indx = base_indx + i + 1
        indx = np.column_stack((indx, new_indx))
    indx = indx.reshape(-1)
    #print indx
    #print List.type()
    #shuffled_List = List[indx]
    shuffled_List = type(List)(map(lambda i:List[i], indx))
    return shuffled_List


def get_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    matrix = matrix*100/matrix.astype(np.float).sum(axis=1).reshape(-1, 1)
    return matrix

def extract_features(model, dataloaders, dataset_sizes, batch_size, K_players, fea_dim, prefix, use_gpu = True):

    for phase in ['trainval', 'test']:
        print phase, dataset_sizes[phase]/K_players
        feas = np.zeros([dataset_sizes[phase]/K_players, fea_dim+1])
        np.save(prefix + phase + '.npy', feas)
        feas = np.load(prefix + phase + '.npy', mmap_mode = 'r+')
        i = 0
        for data in dataloaders[phase]:
            # get the inputs
            inputs, labels = data

            print batch_size*i
            # wrap them in Variable
            inputs = inputs.float()
            if use_gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)
            # forward
            fea, _ = model.forward(inputs)
            fea = fea.data.cpu().numpy()
            #print feas.shape,fc.size()
            #print feas[i*batch_size:(i+1)*batch_size,:-1].shape
            #feas[:,:-1] = fc[0]
            feas[i*batch_size:(i+1)*batch_size,:-1] = fea
            feas[i*batch_size:(i+1)*batch_size,-1] = labels.cpu().numpy()[:10]
            i = i+1
        #np.save(prefix + phase + '.npy', feas)

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, use_gpu=True, num_epochs=25):
    
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['trainval', 'test']:
            since = time.time()
            if phase == 'trainval':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                inputs = inputs.float()
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                _, outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                # backward + optimize only if in training phase
                if phase == 'trainval':
                    loss.backward()
                    optimizer.step()

                #print preds
                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

                #print float(running_corrects) / dataset_sizes[phase]
                #print running_corrects/dataset_sizes[phase]

            epoch_loss = float(running_loss) / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]


            # display related Info(Loss, Acc, Time, etc.)
            print('Epoch: {} phase: {} Loss: {} Acc: {}'.format(epoch, phase, epoch_loss, epoch_acc))
            time_elapsed = time.time() - since
            print('Training this epoch in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
            
            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(model.state_dict(), 'CAD_CNN.pkl')


    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]*100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f' if normalize else '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')

def Min_Max_Normlize(vec):
    """
        Args: vec
    """
    Max, Min = torch.max(vec), torch.min(vec)
    eps = 0.0001
    return (vec - Min) / (Max - Min) 


