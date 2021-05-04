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
Defines the SocialLSTM model.
'''

class SocialLSTM(nn.Module):
    def __init__(self):
        super(SocialLSTM, self).__init__()
        self.hsize = 256
        self.esize = 32
        self.rsize = 64
        self.embed_r = nn.Linear(in_features=2, out_features=self.rsize)
        self.embed_e = nn.Linear(in_features=self.hsize, out_features=self.esize) 
        self.lstm1 = nn.LSTMCell(input_size=(self.esize*16)+self.rsize, hidden_size=self.hsize)
        self.predict = nn.Linear(in_features=self.hsize+self.rsize, out_features=64)
        self.predict2 = nn.Linear(in_features=64, out_features=32)
        self.predict_mean = nn.Linear(in_features=32, out_features=2)
        self.predict_stddev = nn.Linear(in_features=32, out_features=2)
        self.predict_corr = nn.Linear(in_features=32, out_features=1)
        
        self.dropout = nn.Dropout(p=0.01)
    
    def social(self, data, hiddens):
        '''
        data : [batch, agents, 2]
        hiddens : [batch, agents, 128]
        '''
        batch, agents = data.shape[0], data.shape[1]
        dist = 32
        pad = int(dist/2)
        step = int(dist/4)
        inactive = (data[:,:,0] < 0)
        xmin = torch.tensor([step,0,-step,-2*step,step,0,-step,-2*step,step,0,-step,-2*step,step,0,-step,-2*step]).reshape(16,1,1).to(device)
        xmax = torch.tensor([2*step,step,0,-step,2*step,step,0,-step,2*step,step,0,-step,2*step,step,0,-step]).reshape(16,1,1).to(device)
        ymin = torch.tensor([-2*step,-2*step,-2*step,-2*step,-step,-step,-step,-step,0,0,0,0,step,step,step,step]).reshape(16,1,1).to(device)
        ymax = torch.tensor([-step,-step,-step,-step,0,0,0,0,step,step,step,step,2*step,2*step,2*step,2*step]).reshape(16,1,1).to(device)
        H = []
        masks = []
        for b in range(data.shape[0]):
            # [quad, agents, agents]
            mask = (((data[b].unsqueeze(1)-data[b])[:,:,0]).unsqueeze(0).repeat(16,1,1) >= xmin) * \
                   (((data[b].unsqueeze(1)-data[b])[:,:,0]).unsqueeze(0).repeat(16,1,1) <= xmax) * \
                   (((data[b].unsqueeze(1)-data[b])[:,:,1]).unsqueeze(0).repeat(16,1,1) >= ymin) * \
                   (((data[b].unsqueeze(1)-data[b])[:,:,1]).unsqueeze(0).repeat(16,1,1) <= ymax)
            
            inact = inactive[b].unsqueeze(0).repeat(16,1) # [quad,agents]
            mask[inact.unsqueeze(-1).repeat(1,1,data.shape[1])] = 0
            mask[inact.unsqueeze(-2).repeat(1,data.shape[1],1)] = 0
            mask = mask * (torch.diag(torch.ones(data.shape[1])) < 1).unsqueeze(0).repeat(16,1,1).to(device).float()
            H.append(torch.bmm(mask,hiddens[b].unsqueeze(0).repeat(16,1,1)))#.cpu())
        H = torch.stack(H, dim=0)#.to(device)
        return H
            
    def forward(self, data, tprob):
        '''
        data : [batch, agents, frames, 2]
        '''
        data = data.to(device)
        maxlen = data.shape[2]
        batch = data.shape[0]
        agents = data.shape[1]
        if tprob == 1:
            teach_prob = np.zeros((maxlen))
            for i in range(0,maxlen,20):
                if i > (maxlen-20): end = -1
                else: end = i+20
                teach_prob[i+8:end] = 1
        else:
            teach_prob = np.random.binomial(n=1,p=tprob,size=maxlen)
            teach_prob[:8] = 0

        hiddens1 = torch.zeros((batch*agents,self.hsize)).to(device)
        states1 = torch.zeros((batch*agents,self.hsize)).to(device)
        means, stddevs, corrs = [], [], []
        for t in range(maxlen):
            if teach_prob[t]:
                xin = mean.masked_fill_((data[:,:,t,:] < 0),-1)
            else:
                xin = data[:,:,t,:]
            
            r = F.relu(self.embed_r(xin))
            r = self.dropout(r)
            H = self.social(xin, hiddens1.reshape(batch,agents,self.hsize))
            e = F.relu(self.embed_e(H))
            e = e.reshape(batch, agents, -1)
            e = self.dropout(e)
            inp = torch.cat([r,e],dim=-1).reshape(-1,self.rsize+(self.esize*16))
            hiddens1, states1 = self.lstm1(inp, (hiddens1,states1))
            
            inp2 = torch.cat([r,hiddens1.reshape(data.shape[0],data.shape[1],self.hsize)],dim=-1)
            out = self.predict(inp2)
            out = self.predict2(out)
            mean = self.predict_mean(out)
            stddev = torch.exp(self.predict_stddev(out))
            corr = torch.tanh(self.predict_corr(out))
            means.append(mean)
            stddevs.append(stddev)
            corrs.append(corr)
            
        means = torch.stack(means, dim=2)
        stddevs = torch.stack(stddevs, dim=2)
        corrs = torch.stack(corrs, dim=2)
        return means, stddevs, corrs