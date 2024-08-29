#!/usr/bin/env python3
# -*- coding:utf-8 -*-
 
'''
@author: Qun Yang
@license: (C) Copyright 2024, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 27/05/24 10:06 pm
@description: 
@version: 1.0
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


class Window_wiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(Window_wiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        X = self.w1(x)
        X = F.tanh(X)
        X = self.dropout(X)
        X = F.tanh(self.w2(X))
        return X
    
class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.BatchNorm1d(size)
    
    def forward(self, x, sublayer):
        x = x.to(self.device)
        # BatchNorm1d
        X = x.reshape(-1, x.size(2))
        X = self.norm(X)
        X = X.reshape(-1, x.size(1), x.size(2))
        # print(X)
        X = sublayer(X)  # Attention layer
        X = self.dropout(X)
        return x + X
    

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout, num_sublayers):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(p=dropout)
        self.sublayers = nn.ModuleList([SubLayerConnection(size, dropout) for _ in range(num_sublayers)])
    
    def forward(self, x):
    
        for i in range(len(self.sublayers) - 1):
            x = self.sublayers[i](x, lambda x: self.self_attn(x, x, x))
    
        return self.sublayers[-1](x, self.feed_forward)
    