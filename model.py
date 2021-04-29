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
        self.hsize = 128
        self.embed_r = nn.Linear(in_features=2, out_features=64)
        self.embed_e = nn.Linear(in_features=self.hsize, out_features=self.hsize*2) 
        self.lstm1 = nn.LSTMCell(input_size=64+self.hsize*2, hidden_size=self.hsize)
        self.predict_mean = nn.Linear(in_features=self.hsize+64, out_features=64)
        self.predict_mean2 = nn.Linear(in_features=64+2, out_features=32)
        self.predict_mean3 = nn.Linear(in_features=32, out_features=2)
        
        self.dropout = nn.Dropout(p=0.01)
    
    def social(self, data, hiddens):
        '''
        data : [batch, agents, 2]
        hiddens : [batch, agents 128]
        '''
        dist = 32
        inactive = (data[:,:,0] < 0)
        distances = []
        for b in range(data.shape[0]):
            distances.append(data[b].unsqueeze(1)-data[b])
        distances = torch.sqrt(torch.sum(pow(torch.stack(distances,dim=0),2),dim=-1)) # [batch, agent, agent, 2]
        mask = (distances < dist).float()
        mask[inactive.unsqueeze(-1).repeat(1,1,data.shape[1])] = 0
        mask[inactive.unsqueeze(-2).repeat(1,data.shape[1],1)] = 0
        mask = mask * (torch.diag(torch.ones(data.shape[1])).repeat(data.shape[0],1,1).to(device) < 1).float()
        H = torch.bmm(mask,hiddens)
        return H
    
    def forward(self, data, tprob):
        '''
        data : [batch, agents, frames, 2]
        tprob : teacher forcing probability
        '''
        data = data.to(device)
        maxlen = data.shape[2]
        batch = data.shape[0]
        agents = data.shape[1]
        if tprob == 1:
            teach_prob = np.zeros((maxlen))
            for i in range(0,maxlen,20):
                teach_prob[i+8:i+20] = 1
        else:
            teach_prob = np.random.binomial(n=1,p=tprob,size=maxlen)
            teach_prob[:8] = 0
        hiddens1 = torch.zeros((batch*agents,self.hsize)).to(device)
        means = []
        for t in range(maxlen):
            if teach_prob[t] < 1:
                xin = data[:,:,t,:]
            else:
                xin = mean
            
            r = F.relu(self.embed_r(xin))
            r = self.dropout(r)
            H = self.social(xin, hiddens1.reshape(batch,agents,self.hsize))
            e = F.relu(self.embed_e(H))
            e = self.dropout(e)
            inp = torch.cat([r,e],dim=-1).reshape(-1,64+self.hsize*2)
            hiddens1, _ = self.lstm1(inp)
            
            inp2 = torch.cat([r,hiddens1.reshape(data.shape[0],data.shape[1],self.hsize)],dim=-1)
            mean = F.relu(self.predict_mean(inp2))
            mean = self.predict_mean2(torch.cat([mean,xin],dim=-1))
            mean = self.predict_mean3(mean)
            means.append(mean)
            
        means = torch.stack(means, dim=2)
        return means