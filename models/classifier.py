#!/usr/bin/env python3
# -*- coding:utf-8 -*-
 
'''
@author: Qun Yang
@license: (C) Copyright 2024, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 27/05/24 10:10 pm
@description: 
@version: 1.0
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention
from .encoder import Window_wiseFeedForward, EncoderLayer


class Extractor(nn.Module):
    def __init__(self, config):
        super(Extractor, self).__init__()

        d_model = {'time': config['win_len'],
                   'hist': config['hist_bin'],
                   'fft': config['win_len'] // 2}

        self.d_model = d_model[config['feature']]
        
        self.attention = MultiHeadAttention(config['head'], self.d_model, config['dropout'], config['n_att'])
        self.feed_forward = Window_wiseFeedForward(self.d_model, config['d_ff'], config['dropout'])
        self.encoder_layer = EncoderLayer(self.d_model, self.attention, self.feed_forward, 
                                          config['dropout'], config['n_sublayers'])
        self.encoders = nn.ModuleList([self.encoder_layer for _ in range(config['n_encoders'])])
        self.norm = nn.BatchNorm1d(self.d_model)

    def forward(self, x):
        for encoder in self.encoders: 
            x = encoder(x)
        X = x.reshape(-1, x.size(2))
        X = self.norm(X)
        x = X.reshape(-1, x.size(1), x.size(2))
        return x


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()

        self.extractor = Extractor(config)
        
        output_dims = {'time': config['win_len'],
                       'hist': config['hist_bin'],
                       'fft': config['win_len'] // 2}
        
        self.layer1 = nn.Linear(output_dims[config['feature']], 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, config['n_class'])

        self.z = None

    def forward(self, x):
        x = self.extractor(x)
        # print(x.shape)
        self.z = torch.max(x, dim=1).values
        x = torch.max(x, dim=1).values
        # x = x.view(x.size(0), -1)
        x = F.tanh(self.layer1(x))
        x = F.tanh(self.layer2(x))
        x = F.tanh(self.layer3(x))
        x = self.layer4(x)
        return x
