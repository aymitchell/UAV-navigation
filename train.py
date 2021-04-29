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

from utils import *
from data_handler import *
from model import *


'''
Train the model.
'''

def train(model, criterion, optimizer, trainloader, valloader, startepoch, nepochs, save):
    print('Training...')
    print(time.ctime())
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(startepoch, startepoch+nepochs):
        print('Starting Epoch %d:' %(epoch), time.ctime())
        start_epoch = time.time()
        model.train()
        if epoch < 40: tprob = 0.1
        if epoch >= 40: tprob = 0.15
        if epoch >= 60: tprob = 0.2
        if epoch >= 100: tprob = 0.3
        if epoch >= 140: tprob = 0.4
        if epoch >= 180: tprob = 0.5
        if epoch >= 220: tprob = 0.55
        if epoch >= 260: tprob = 0.6
        if epoch >= 300: tprob = 0.65
        if epoch >= 340: tprob = 0.7
        if epoch >= 380: tprob = 0.75
        if epoch >= 420: tprob = 0.8
        if epoch >= 460: tprob = 0.85
        if epoch >= 500: tprob = 0.9
        
        if epoch >= 120: optimizer.param_groups[0]['lr'] = 5e-3
        if epoch >= 200: optimizer.param_groups[0]['lr'] = 1e-3
        if epoch >= 280: optimizer.param_groups[0]['lr'] = 5e-4
        if epoch >= 360: optimizer.param_groups[0]['lr'] = 1e-4
        if epoch >= 400: optimizer.param_groups[0]['lr'] = 5e-5
        if epoch >= 440: optimizer.param_groups[0]['lr'] = 1e-5
        if epoch >= 520: optimizer.param_groups[0]['lr'] = 5e-6


        train_loss, train_dist = [], []
        for batch_num, (data_in) in enumerate(trainloader):
            start_batch = time.time()
            
            means, stddevs, corrs = [], [], []
            
            optimizer.zero_grad()
            means = model(data_in, tprob)

            mask = (data_in[:,:,1:,:] < 0)
            
            loss = criterion(means[:,:,:-1,:], data_in[:,:,1:,:].to(device))
            loss = loss.masked_fill_(mask.to(device),0).sum()
            loss = (loss / (mask < 1).sum()).clamp(max=1e18)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            dist = calc_dist(data_in[:,:,1:,:].detach().cpu(), means[:,:,:-1,:].detach().cpu(),\
                             mask.detach().cpu())
            
            gc.collect()
            torch.cuda.empty_cache()
            train_loss.append(loss.detach().cpu().item())
            train_dist.append(dist.item())
            del loss, mask, means, data_in

            if epoch == 0 and batch_num == 0: 
                print('Single Batch Time: %d min %d sec' %((time.time()-start_batch)//60, (time.time()-start_batch)%60)) 
        
        val_loss, val_dist = 0, 0
        if epoch % 4 == 0:
            val_loss, val_dist = validate(model, criterion, valloader)

        if save:
            torch.save(model.state_dict(), 'model_'+str(epoch)+'.pt')
            torch.save(optimizer.state_dict(), 'optimizer_'+str(epoch)+'.pt')
        
        stop_epoch = time.time()
        min_epoch = (stop_epoch - start_epoch) // 60
        sec_epoch = (stop_epoch - start_epoch) % 60

        print("Epoch: %d, Run Time: %d min, %d sec" %(epoch, min_epoch, sec_epoch))
        print('Train Loss: {:.3f}, Train Avg Dist: {:.2f}'.format(np.mean(train_loss), np.mean(train_dist)))
        if (epoch % 4) == 0:
            print('Val Loss: {:.3f}, Val Avg Dist: {:.2f}'.format(val_loss, val_dist))
        print("==============================================")

def validate(model, criterion, valloader):
    model.eval()
    val_loss, val_dist = [], []
    for batch_num, (data_in) in enumerate(valloader):
        train_loss, train_dist = [], []
        means, stddevs, corrs = [], [], []

        means = model(data_in, tprob=1)

        mask = (data_in[:,:,1:,:] < 0)

        loss = criterion(means[:,:,:-1,:], data_in[:,:,1:,:].to(device))
        loss = loss.masked_fill_(mask.to(device),0).sum()
        loss = loss / (mask < 1).sum()

        dist = calc_dist(data_in[:,:,1:,:].detach().cpu(), means[:,:,:-1,:].detach().cpu(),\
                         mask.detach().cpu())

        gc.collect()
        torch.cuda.empty_cache()
        val_loss.append(loss.detach().cpu().item())
        val_dist.append(dist.item())
        del loss, mask, means, data_in
    
    return np.mean(val_loss), np.mean(val_dist)

def run(startepoch, nepochs, modelfilename, optfilename, save):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    torch.autograd.set_detect_anomaly(True)
    gc.collect()
    torch.cuda.empty_cache()

    # get the data
    trainloader, valloader = get_dataloaders()

    # define the model
    model = SocialLSTM()
    if modelfilename is not None:
        model.load_state_dict(state_dict_helper(modelfilename))
    model = nn.DataParallel(model)
    # model = model.to(torch.cuda.current_device())
    model = model.to(device)

    # define the criterion and optimizer
    criterion = nn.L1Loss(reduction='none')
    learning_rate = 3e-2
    weight_decay = 5e-5
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optfilename is not None:
        optimizer.load_state_dict(torch.load(optfilename))

    train(model, criterion, optimizer, trainloader, valloader, startepoch, nepochs, save)

if __name__ == "__main__":
    params = get_params()

    startepoch, nepochs = params.startepoch, params.nepochs
    modelfilename, optfilename = params.load_model, params.load_optimizer

    modelfilename = 'model_0.pt'
    optfilename = 'optimizer_0.pt'

    save = False

    startepoch = 0
    nepochs = 50

    run(startepoch, nepochs, modelfilename, optfilename, save)
