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
from sklearn import datasets, svm, metrics
from collections import deque
import pickle
from PIL import Image, ImageDraw


def normlize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return np.round(matrix/np.sum(matrix, 1).reshape(-1,1)*100, decimals=2)

def save_pkl(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_pkl(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def write_txt(txtFile, content_str, mode):
    with open(txtFile, mode) as f:
        f.write(content_str)

def annotation_parse(line, action_list=[], activity_list=[]):
    keywords = deque(line.strip().split(' '))
    frame_id = keywords.popleft().split('.')[0]
    activity = activity_list.index(keywords.popleft())
    Rects = []
    while keywords:
        x = int(keywords.popleft())
        y = int(keywords.popleft())
        w = int(keywords.popleft())
        h = int(keywords.popleft())
        action = action_list.index(keywords.popleft())
        Rects.append([x,y,w,h,action,activity])
    Rects = np.asarray(Rects)
    # sort Rects by the first col
    Rects = Rects[np.lexsort(Rects[:,::-1].T)]
    return frame_id, Rects


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
    for i in range(block_length - 1):
        new_indx = base_indx + i + 1
        indx = np.column_stack((indx, new_indx))
    indx = indx.reshape(-1)
    # print indx
    # print List.type()
    #shuffled_List = List[indx]
    shuffled_List = type(List)(map(lambda i: List[i], indx))
    return shuffled_List



def get_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    matrix = matrix * 100 / matrix.astype(np.float).sum(axis=1).reshape(-1, 1)
    return matrix



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='',
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
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

