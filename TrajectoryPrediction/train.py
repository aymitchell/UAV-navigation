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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Training...')
    print(time.ctime())
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(startepoch, startepoch+nepochs):
        print('Starting Epoch %d:' %(epoch), time.ctime())
        start_epoch = time.time()
        model.train()
        train_loss, train_dist = [], []
        for batch_num, (data_in) in enumerate(trainloader):
            data_in = data_in.to(device)
            start_batch = time.time()
                        
            optimizer.zero_grad()
            means, stddevs, corrs = model(data_in, tprob)

            mask = (data_in[:,:,1:,:] < 0)
            loss = criterion(data_in[:,:,1:,:].to(device), means[:,:,:-1,:],\
                             stddevs[:,:,:-1,:], corrs[:,:,:-1,:], mask[:,:,:,0].to(device))
            loss = loss.sum()
            loss = (loss / (mask < 1).sum()).clamp(max=1e18)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            
            dist = calc_dist(data_in[:,:,1:,:].detach().cpu(), means[:,:,:-1,:].detach().cpu(),\
                             mask.detach().cpu())
            
            gc.collect()
            torch.cuda.empty_cache()
            train_loss.append(loss.detach().cpu().item())
            train_dist.append(dist.item())
            del loss, mask, means, stddevs, corrs, data_in

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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    val_loss, val_dist = [], []
    for batch_num, (data_in) in enumerate(valloader):
        data_in = data_in.to(device)

        means, stddevs, corrs = [], [], []

        means, stddevs, corrs = model(data_in, tprob=1)

        mask = (data_in[:,:,1:,:] < 0)
        loss = criterion(data_in[:,:,1:,:].to(device), means[:,:,:-1,:],\
                         stddevs[:,:,:-1,:], corrs[:,:,:-1,:], mask[:,:,:,0].to(device))
        loss = loss.sum()
        loss = loss / (mask < 1).sum()

        dist = calc_dist(data_in[:,:,1:,:].detach().cpu(), means[:,:,:-1,:].detach().cpu(),\
                         mask.detach().cpu())

        gc.collect()
        torch.cuda.empty_cache()
        val_loss.append(loss.detach().cpu().item())
        val_dist.append(dist.item())
        del loss, mask, means, stddevs, corrs, data_in
    
    return np.mean(val_loss), np.mean(val_dist), np.mean(val_evaldist)#, np.mean(val_finaldist)

def nllLoss(target, mean, stddev, corr, mask):
    x1, x2 = target[:,:,:,0], target[:,:,:,1]
    m1, m2 = mean[:,:,:,0], mean[:,:,:,1]
    std1, std2 = stddev[:,:,:,0].clamp(min=1e-6), stddev[:,:,:,1].clamp(min=1e-6)
    corr = corr[:,:,:].squeeze(-1)
    
    Z = pow((x1-m1)/std1,2) + pow((x2-m2)/std2,2) - 2*corr*(((x1-m1)*(x2-m2))/(std1*std2))
    N = (1 / (2*np.pi*std1*std2*torch.sqrt(1-pow(corr,2).clamp(max=1-1e-6))))
    a = torch.log(N.clamp(min=1e-6))
    b = (-Z/(2*(1-pow(corr,2))))
    L = a + b
    L = L.masked_fill_(mask,0)
    return -L

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
        model.load_state_dict(modelfilename, map_location=device)
    model = model.to(device)

    # define the criterion and optimizer
    criterion = nllLoss
    learning_rate = 3e-2
    weight_decay = 5e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
    nepochs = 100

    run(startepoch, nepochs, modelfilename, optfilename, save)
