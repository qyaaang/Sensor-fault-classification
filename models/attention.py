#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@author: Qun Yang
@license: (C) Copyright 2024, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 27/05/24 09:55 pm
@description: 
@version: 1.0
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def attention(query, key, value, mask=None, dropout=None):
    
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout, num_att):
        super(MultiHeadAttention, self).__init__()
        
        self.head = head 
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(p=dropout)
        self.d_k = embedding_dim // head
        self.linears = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(num_att)])
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)
        batch_size = query.size(0)
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                             for model, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)  
        
        return self.linears[-1](x)
