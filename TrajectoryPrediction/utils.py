import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import gc

'''
Helper functions for training the model.
'''

def save_model(model, epoch, val_acc):
    torch.save(model.state_dict(), './model_' + str(epoch) + '_' + str(int(val_acc*10000)) + '.pt')
    
def save_optimizer(optimizer, epoch, val_acc):
    torch.save(optimizer.state_dict(), './optimizer_' + str(epoch) + '_' + str(int(val_acc*10000)) + '.pt')

def calc_acc(pred, label, total):
    tol = 1
    correct = (np.linalg.norm(abs(pred-label), axis=3) < tol)
    acc = np.sum(correct) / total
    return acc / total

def calc_dist(pred, label, total):
    '''
    Calculates the average distance between predicted and ground truth points. 
    inputs
        pred: predicted point (x,y) [batch, agents, seq_len, 2]
        label: ground truth position [batch, agents, seq_len, 2]
        total: number of active points
    outputs
        average distance per point
    '''
    dist = np.sum(np.linalg.norm(pred - label, ord=2, axis=-1))
    return dist / total


def nllLoss(target, mean, stddev, corr, mask):
    '''
    Calculates the negative log likelihood loss of the bivariate Gaussian distribution evaluated
    at the target x,y point.
    inputs
        target: ground truth/labels [batch, agents, seq_len, 2]
        mean: distribution predicted mean in x & y from network [batch, agents, seq_len, 2]
        stddev: distribution predicted std deviation in x & y from network [batch, agents, seq_len, 2]
        corr: distribution predicted correlation from network [batch, agents, seq_len]
    '''
    torch.autograd.set_detect_anomaly(True)
    x1, x2 = target[:,:,:,0], target[:,:,:,1]
    m1, m2 = mean[:,:,:,0], mean[:,:,:,1]
    std1, std2 = stddev[:,:,:,0].clamp(min=1e-8), stddev[:,:,:,1].clamp(min=1e-8)
    
    Z = pow(x1-m1,2)/pow(std1,2) + pow(x2-m2,2)/pow(std2,2) - (2*corr*(x1-m1)*(x2-m2))/(std1*std2)
    N = 1/(2*np.pi*std1*std2*torch.sqrt(1-pow(corr,2))) * torch.exp((-Z)/(2*(1-pow(corr,2))).clamp(min=1e-6))
    
    L = - torch.log(N.clamp(min=1e-8))

    if L.isnan().any(): print('L nan', torch.sum(L.isnan()))
    if abs(L).isinf().any(): print('L inf', torch.sum(abs(L).isinf()))

    L = torch.sum(L.masked_fill_(mask,0), dim=2)
    torch.autograd.set_detect_anomaly(True)
    
    return L

def state_dict_helper(filename):
    # original saved file with DataParallel
    state_dict = torch.load(filename)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict
