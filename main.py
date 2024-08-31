#!/usr/bin/env python3
# -*- coding:utf-8 -*-
 
'''
@author: Qun Yang
@license: (C) Copyright 2024, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 27/05/24 08:40 pm
@description: 
@version: 1.0
'''


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.sensor_dataset import SensorDataset
from sklearn.model_selection import train_test_split
from utils.data_aug import DataAug
from models.classifier import Classifier
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random

from train import train
from test import test, test_sub_seq


class Expr:

    def __init__(self, train_config, test_config):
        self.train_config = train_config
        self.test_config = test_config

        dataset = torch.load('./data/{}.pt'.format(train_config['dataset']))

        X = dataset['samples']
        y = dataset['labels']

        if train_config['data_aug']:
            data = DataAug(X, y)
            X, y = data.transform()

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_dataset(X, y)

        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.split_dataset(X, y)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        train_dataset = SensorDataset(X_train, y_train, train_config)
        val_dataset = SensorDataset(X_val, y_val, train_config)
        test_dataset = SensorDataset(X_test, y_test, train_config)

        self.train_dataloader = DataLoader(train_dataset, 
                                           batch_size=train_config['batch'],
                                           num_workers=2,
                                           shuffle=True)

        self.val_dataloader = DataLoader(val_dataset, 
                                         batch_size=train_config['batch'],
                                         num_workers=2,
                                         shuffle=False)
        
        self.test_dataloader = DataLoader(test_dataset, 
                                          batch_size=train_config['batch'],
                                          num_workers=2,
                                          shuffle=False)
        
        self.model = Classifier(train_config).to(self.device)
        self.init_model()

    def split_dataset(self, X, y):
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def init_model(self):
        for param in self.model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def run(self):
    
        train(self.model, self.train_config, self.train_dataloader, self.val_dataloader)

        model_save_path = './checkpoints/{}_{}_BATCH_{}_WIN_{}_HIST_{}_HEAD_{}_AUG_{}.pt'.format(self.train_config['dataset'],
                                                                                                 self.train_config['feature'],
                                                                                                 self.train_config['batch'],
                                                                                                 self.train_config['win_len'],
                                                                                                 self.train_config['hist_bin'], 
                                                                                                 self.train_config['head'],
                                                                                                 self.train_config['data_aug'])
        # test            
        self.model.load_state_dict(torch.load(model_save_path, map_location=self.device))
        best_model = self.model.to(self.device)

        result_save_path = './results/{}_{}_BATCH_{}_WIN_{}_HIST_{}_HEAD_{}_AUG_{}'.format(self.train_config['dataset'],
                                                                                           self.train_config['feature'],
                                                                                           self.train_config['batch'],
                                                                                           self.train_config['win_len'],
                                                                                           self.train_config['hist_bin'], 
                                                                                           self.train_config['head'],
                                                                                           self.train_config['data_aug'])

        if not os.path.exists(result_save_path):
            os.mkdir(result_save_path)

        all_test_inputs, all_test_outputs, all_test_labels, all_attn_scores, all_z = test(best_model, self.train_config, self.test_dataloader)

        np.save('{}/test_dataset.npy'.format(result_save_path), self.X_test)
        np.save('{}/all_test_inputs.npy'.format(result_save_path), all_test_inputs.numpy())
        np.save('{}/all_test_outputs.npy'.format(result_save_path), all_test_outputs.numpy())
        np.save('{}/all_test_labels.npy'.format(result_save_path), all_test_labels.numpy())
        np.save('{}/all_attn_scores.npy'.format(result_save_path), all_attn_scores.numpy())
        np.save('{}/all_z.npy'.format(result_save_path), all_z.numpy())


        # results = test_sub_seq(best_model, self.test_config, self.X_test, self.y_test)

        # np.save('./animation/results.npy'.format(result_save_path), results)

    
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', help='dataset', type=str, default='IPC_SHM')
    parser.add_argument('-feature', help='time, hist, fft', type=str, default='hist') 
    parser.add_argument('-batch', help='batch size', type=int, default=200)
    parser.add_argument('-seq_len', help='sequence length', type=int, default=72000)
    parser.add_argument('-win_len', help='window length', type=int, default=2000)
    parser.add_argument('-hist_bin', help='', type=int, default=64)
    parser.add_argument('-d_ff', help='feed forward dimension', type=int, default=128)
    parser.add_argument('-n_class', help='number of fault types', type=int, default=7)
    parser.add_argument('-dropout', help='', type=float, default=0)
    parser.add_argument('-head', help='', type=int, default=8)
    parser.add_argument('-n_att', help='', type=int, default=4)
    parser.add_argument('-n_sublayers', help='', type=int, default=2)
    parser.add_argument('-n_encoders', help='', type=int, default=4)
    parser.add_argument('-random_seed', help='random seed', type=int, default=0)
    parser.add_argument('-data_aug', action='store_true')
    parser.add_argument('-softmax', action='store_true')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    train_config = {
        'dataset': args.dataset,
        'feature': args.feature,
        'batch': args.batch,
        'seq_len': args.seq_len,
        'win_len': args.win_len,
        'hist_bin': args.hist_bin,
        'd_ff': args.d_ff,
        'n_class': args.n_class,
        'dropout': args.dropout,
        'head': args.head,
        'n_att': args.n_att,
        'n_sublayers': args.n_sublayers,
        'n_encoders': args.n_encoders,
        'data_aug': args.data_aug,
        'softmax': args.softmax
    }

    test_config = {
        'dataset': args.dataset,
        'feature': args.feature,
        'batch': args.batch,
        'seq_len': args.win_len,
        'win_len': args.win_len,
        'hist_bin': args.hist_bin,
        'd_ff': args.d_ff,
        'n_class': args.n_class,
        'dropout': args.dropout,
        'head': args.head,
        'n_att': args.n_att,
        'n_sublayers': args.n_sublayers,
        'n_encoders': args.n_encoders,
        'data_aug': args.data_aug,
        'softmax': args.softmax
    }

    expr = Expr(train_config, test_config)

    expr.run()
