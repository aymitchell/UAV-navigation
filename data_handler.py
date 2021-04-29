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

import params

'''
Handles data loading and preprocessing.
'''

def load_data(train_filename, val_filename):
    # load data from files
    train_data = np.load(train_filename, allow_pickle=True)['train_data']
    val_data = np.load(val_filename, allow_pickle=True)['val_data']
    return train_data, val_data

def collate_pad(batch_data):
    '''
    Pad the input batch data with -1 to all have 
    same number of agents and same sequence length. 
    '''
    data = []
    maxlen = max([d.shape[1] for d in batch_data])
    for i in range(len(batch_data)):
        batchsize = batch_data[i].shape[0]
        agents = batch_data[i].shape[1]
        feats = batch_data[i].shape[2]
        temp = np.concatenate([batch_data[i],-1*np.ones((batchsize,maxlen-agents,feats))],axis=1)
        data.append(torch.tensor(temp).float())
    return rnn.pad_sequence(data, batch_first=True, padding_value=-1)

class TrajectoryDataset(Dataset):
    # trajectory dataset class for custom dataloader
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

def create_dataloaders(train_data, val_data, collate_fn, batch_size):
    trainset = TrajectoryDataset(train_data)
    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, drop_last=False)

    valset = TrajectoryDataset(val_data)
    valloader = DataLoader(valset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, drop_last=False)

    return trainloader, valloader

def get_dataloaders():
    params = get_params()
    batch_size = params.batch_size

    # function to call from training file
    train_data, val_data = load_data('train_data.npz', 'val_data.npz')
    train_labels, val_labels = labels(train_data, val_data)

    trainloader, valloader = create_dataloaders(train_data, val_data, collate_pad, batch_size)
    
    return trainloader, valloader