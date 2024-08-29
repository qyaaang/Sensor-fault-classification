#!/usr/bin/env python3
# -*- coding:utf-8 -*-
 
'''
@author: Qun Yang
@license: (C) Copyright 2024, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 16/06/24 11:14 am
@description: 
@version: 1.0
'''


import torch
from sklearn.preprocessing import LabelEncoder
from scipy.io import loadmat
import h5py
import numpy as np
import os


def read_data(root_path, days=20):
    test_days = sorted(os.listdir(root_path))
    all_data = []

    for i in range(days):
        one_day_data = []
        data_path = os.path.join(root_path, test_days[i])
        data_files = sorted(os.listdir(data_path))

        for j in range(len(data_files)):
            mat_data = loadmat(os.path.join(data_path, data_files[j]))
            raw_data = mat_data['data']
            print('Reading {}'.format(data_files[j]))
            one_day_data.append(raw_data.T)

        one_day_data = np.stack(one_day_data, axis=1)
        all_data.append(one_day_data)

    all_data = np.concatenate(all_data, axis=1)
    all_data = all_data.reshape(-1, all_data.shape[2])
    all_data = np.nan_to_num(all_data, nan=0)
    return all_data


def read_label(label_path, days=20):
    label_data = []

    with h5py.File(label_path, 'r') as file:
        refs = file['info']['label']['manual'][:, 0]
        for ref in refs:
            ref_data = file[ref]
            if isinstance(ref_data, h5py.Dataset):
                label_data.append(ref_data[()])

    label_data = np.array(label_data) - 1
    label_data = label_data[:, :24 * days, :]
    labels = label_data.reshape(-1, 1)

    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(labels)
    num_classes = np.max(encoded_Y) + 1
    one_hot_label = np.eye(num_classes)[encoded_Y]
    return one_hot_label


def read_balance_data(data_path, label_path):
    data = loadmat(data_path)
    label = loadmat(label_path)

    data = data['result'].T
    data = np.nan_to_num(data, nan=0)

    label = label['matrix']
    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(label)
    num_classes = np.max(encoded_Y) + 1
    one_hot_label = np.eye(num_classes)[encoded_Y]

    return data, one_hot_label


if __name__ == '__main__':
    
    if not os.path.exists('../data/IPC_SHM'):
        os.mkdir('../data/IPC_SHM')
    output_dir = '../data/IPC_SHM'

    dataset = 'balance'

    if dataset == 'balance':
        file_path = '../data/raw/IPC_SHM/newsampledata.mat'
        label_path = '../data/raw/IPC_SHM/newsamplelabel.mat'
        X, y = read_balance_data(file_path, label_path)
        data = dict()
        data['samples'] = torch.from_numpy(X).to(torch.float32)
        data['labels'] = torch.from_numpy(y).to(torch.float32)
        torch.save(data, os.path.join(output_dir, 'IPC_SHM_balance.pt'))

    else:
        file_path = '../data/raw'
        label_path = '../data/raw/label.mat'
        days = 20
        X = read_data(file_path, days=days)
        y = read_label(label_path, days=days)

        print(X.shape)
        print(y.shape)

        data = dict()
        data['samples'] = torch.from_numpy(X).to(torch.bfloat16)
        data['labels'] = torch.from_numpy(y).to(torch.bfloat16)
        torch.save(data, os.path.join(output_dir, 'IPC_SHM.pt'))
