#!/usr/bin/env python3
# -*- coding:utf-8 -*-
 
'''
@author: Qun Yang
@license: (C) Copyright 2024, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 27/05/24 08:14 pm
@description: 
@version: 1.0
'''


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SensorDataset(Dataset):

    def __init__(self, data, label, train_config):
        super(SensorDataset, self).__init__()
        self.train_config = train_config

        process = {'time': self.process,
                   'hist': self.process_hist,
                   'fft': self.process_fft}

        self.X = process[train_config['feature']](data)
        self.y = label
    
    def process_hist(self, data):
        # seq_len = self.train_config['seq_len'] // self.train_config['win_len']
        d_model = self.train_config['hist_bin']

        def pad_and_window(data):

            N, T = data.shape
            window_size = self.train_config['win_len']
            pad_length = (window_size - (T % window_size)) % window_size  # Compute the padding length
            padded_tensor = F.pad(data, (0, pad_length), 'constant', 0)  # Pad the tensor with zeros
            
            # Reshape the tensor into windows of size window_size
            num_windows = (T + pad_length) // window_size
            reshaped_tensor = padded_tensor.view(N, num_windows, window_size)
            
            return reshaped_tensor

        data = pad_and_window(data)

        # print(data.shape)

        # data = data.reshape((data.shape[0], seq_len, -1))

        # Initialize a tensor to store histogram features
        # features = torch.zeros((data.shape[0], seq_len, d_model))
        features = torch.zeros((data.shape[0], data.shape[1], d_model))

        # Calculate histogram features
        for i in range(data.shape[0]):

            histogram_features = torch.zeros((data.shape[1], d_model - 3))
            for j in range(data.shape[1]):
                histogram = torch.histc(data[i, j, :], bins=d_model - 3, min=0, max=1)
                histogram_features[j, :] = histogram / data.shape[1]  # Normalize by the total number of data points

            # Calculate mean, range, and standard deviation
            mean_values = torch.mean(data[i, :, :], dim=-1, keepdim=True)
            range_values = torch.max(data[i, :, :], dim=-1, keepdim=True).values - torch.min(data[i, :, :], dim=-1, keepdim=True).values  # Calculate range
            std_dev_values = torch.std(data[i, :, :], dim=-1, keepdim=True)

            # Normalize each feature
            normalized_mean_values = mean_values / (torch.max(mean_values) + 1e-6)
            normalized_range_values = range_values / (torch.max(range_values) + 1e-6)
            normalized_std_dev_values = std_dev_values / (torch.max(std_dev_values) + 1e-6)

            # Combine normalized features
            combined_normalized_features = torch.cat([
                normalized_mean_values,
                normalized_range_values,
                normalized_std_dev_values,
                histogram_features
            ], dim=-1)

            features[i, :, :] = combined_normalized_features

            # has_nan = torch.isnan(normalized_range_values).any()
            # features = torch.nan_to_num(features, nan=0.0)

            # print(has_nan)

        return features
    
    def process_fft(self, data):
        seq_len = self.train_config['seq_len'] // self.train_config['win_len']
        d_model = self.train_config['win_len'] // 2

        data = data.reshape((data.shape[0], seq_len, -1))

        features = torch.zeros((data.shape[0], seq_len, d_model))

        # Calculate psd features
        for i in range(data.shape[0]):
            for j in range(seq_len):
                X = torch.fft.fft(data[i, j, :])
                psd = (X * X.conj()).real / self.train_config['win_len']
                psd = psd[:self.train_config['win_len'] // 2]
                features[i, j, :] = psd

        # Normalization
        min_values = torch.min(data, dim=-1, keepdim=True).values
        max_values = torch.max(data, dim=-1, keepdim=True).values
        range_values = max_values - min_values
        normalized_psd = (psd - min_values) / (range_values + 1e-6)
        # normalized_psd = psd / (max_values + 1e-6)
        # print(normalized_psd)
        return normalized_psd
    
    def process(self, data):
        seq_len = self.train_config['seq_len'] // self.train_config['win_len']
        d_model = self.train_config['win_len']

        data = data.reshape((data.shape[0], seq_len, -1))

        features = torch.zeros((data.shape[0], seq_len, d_model))

        # Normalization
        min_values = torch.min(data, dim=-1, keepdim=True).values
        max_values = torch.max(data, dim=-1, keepdim=True).values
        range_values = max_values - min_values
        normalized_data = (data - min_values) / (range_values + 1e-6)
        # normalized_data = data / (max_values + 1e-6)
        # print(normalized_data)
        return normalized_data

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.X.shape[0]
