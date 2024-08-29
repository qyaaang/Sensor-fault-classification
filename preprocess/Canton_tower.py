#!/usr/bin/env python3
# -*- coding:utf-8 -*-
 
'''
@author: Qun Yang
@license: (C) Copyright 2024, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 04/08/24 11:32 am
@description: 
@version: 1.0
'''


import torch
from inject_fault import random_faults, malfunction_faults, drift_faults, bias_faults
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm


def make_normal_data(file_path):
    if not os.path.exists('../data/Canton_Tower/normal'):
        os.mkdir('../data/Canton_Tower/normal')
    accdata_files = sorted(os.listdir(file_path))

    for accdata_file in tqdm(accdata_files, desc='Making normal data'):
        
        all_clean_data = pd.read_csv(os.path.join(file_path, accdata_file), sep='\s+', header=None)
        all_clean_data = np.array(all_clean_data)

        np.save('../data/Canton_Tower/normal/{}.npy'.format(os.path.splitext(accdata_file)[0]), all_clean_data)


       
def make_random_faults(file_path):
    if not os.path.exists('../data/Canton_Tower/random'):
        os.mkdir('../data/Canton_Tower/random')
    accdata_files = sorted(os.listdir(file_path))

    for accdata_file in tqdm(accdata_files, desc='Making random faults'):
        
        all_clean_data = pd.read_csv(os.path.join(file_path, accdata_file), sep='\s+', header=None)
        # print(all_data)
        all_faulty_data = np.zeros_like(np.array(all_clean_data))

        for sensor_id in all_clean_data.columns:
            data = np.array(all_clean_data[sensor_id])
            faulty_data = random_faults(data, 0.001, 8.5)
            all_faulty_data[:, sensor_id] = faulty_data
            # print(data)
            # print(faulty_data)
            # print(np.linalg.norm(faulty_data - data))

            # fig, ax = plt.subplots(figsize=(10, 4))
            # ax.plot(data, c='k', ls='-', lw=1.0, label='Normal')
            # ax.plot(faulty_data, c='r', ls='--', lw=0.5, label='Faulty')

            # ax.legend()
            # fig.savefig(f'canton_tower_random_faults_{sensor}.jpg')
            # break
    
        np.save('../data/Canton_Tower/random/{}.npy'.format(os.path.splitext(accdata_file)[0]), all_faulty_data)


def make_malfunction_faults(file_path):
    if not os.path.exists('../data/Canton_Tower/malfunction faults'):
        os.mkdir('../data/Canton_Tower/malfunction faults')
    accdata_files = sorted(os.listdir(file_path))

    for accdata_file in tqdm(accdata_files, desc='Making malfunction faults'):
        
        all_clean_data = pd.read_csv(os.path.join(file_path, accdata_file), sep='\s+', header=None)
        all_clean_data = np.array(all_clean_data)
        all_faulty_data = malfunction_faults(all_clean_data, 1.5)

        # fig, ax = plt.subplots(figsize=(10, 4))
        # ax.plot(all_clean_data[:, 0], c='k', ls='-', lw=1.0, label='Normal')
        # ax.plot(all_faulty_data[:, 0], c='r', ls='--', lw=0.5, label='Faulty')

        # ax.legend()
        # fig.savefig('canton_tower_malfunction_faults.jpg')
        
        np.save('../data/Canton_Tower/malfunction faults/{}.npy'.format(os.path.splitext(accdata_file)[0]), all_faulty_data)


def make_drift_faults(file_path):
    if not os.path.exists('../data/Canton_Tower/drift'):
        os.mkdir('../data/Canton_Tower/drift')
    accdata_files = sorted(os.listdir(file_path))

    for accdata_file in tqdm(accdata_files, desc='Making drift faults'):
        
        all_clean_data = pd.read_csv(os.path.join(file_path, accdata_file), sep='\s+', header=None)
        all_clean_data = np.array(all_clean_data)
        all_faulty_data = drift_faults(all_clean_data, i_d=[-2, -1, 1, 2], n_d=1)

        # fig, ax = plt.subplots(figsize=(10, 4))
        # ax.plot(all_clean_data[:, 0], c='k', ls='-', lw=1.0, label='Normal')
        # ax.plot(all_faulty_data[:, 0], c='r', ls='--', lw=0.5, label='Faulty')

        # ax.legend()
        # fig.savefig('canton_tower_drift_faults.jpg')
        
        np.save('../data/Canton_Tower/drift/{}.npy'.format(os.path.splitext(accdata_file)[0]), all_faulty_data)


def make_bias_faults(file_path):
    if not os.path.exists('../data/Canton_Tower/bias'):
        os.mkdir('../data/Canton_Tower/bias')
    accdata_files = sorted(os.listdir(file_path))

    for accdata_file in tqdm(accdata_files, desc='Making bias faults'):
        
        all_clean_data = pd.read_csv(os.path.join(file_path, accdata_file), sep='\s+', header=None)
        all_clean_data = np.array(all_clean_data)
        all_faulty_data = bias_faults(all_clean_data, i_b=[1.5], n_b=0)

        # fig, ax = plt.subplots(figsize=(10, 4))
        # ax.plot(all_clean_data[:, 0], c='k', ls='-', lw=1.0, label='Normal')
        # ax.plot(all_faulty_data[:, 0], c='r', ls='--', lw=0.5, label='Faulty')

        # ax.legend()
        # fig.savefig('canton_tower_bias_faults.jpg')
        
        np.save('../data/Canton_Tower/bias/{}.npy'.format(os.path.splitext(accdata_file)[0]), all_faulty_data)


def read_data(file_path):
    faulty_types = ['normal', 'random', 'malfunction', 'drift', 'bias']

    all_data = []
    all_labels = []

    for label_idx, faulty_type in enumerate(faulty_types):

        for accdata_file in os.listdir(os.path.join(file_path, faulty_type)):
            data = np.load(os.path.join(file_path, faulty_type, accdata_file))
            labels = np.full(data.shape[1], label_idx)
            all_data.append(data.T)
            all_labels.append(labels)

    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)

    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(y)
    num_classes = np.max(encoded_Y) + 1
    one_hot_label = np.eye(num_classes)[encoded_Y]
    # print(X.shape)
    # print(one_hot_label.shape)

    return X, one_hot_label




if __name__ == '__main__':
    if not os.path.exists('../data/Canton_Tower'):
        os.mkdir('../data/Canton_Tower')
    output_dir = '../data/Canton_Tower'

    file_path = '../data/raw/Canton_Tower/Acc data'

    rc={'font.family': 'Arial',
            # 'mathtext.fontset': 'stix',
            'figure.constrained_layout.use': True,
            'text.usetex': False,
            'figure.dpi': 300
            }
    matplotlib.rcParams.update(rc)

    # make_normal_data(file_path)

    # make_random_faults(file_path)

    # make_malfunction_faults(file_path)

    # make_drift_faults(file_path)
    
    # make_bias_faults(file_path)

    X, y = read_data(output_dir)
    # print(y.shape)
    data = dict()
    data['samples'] = torch.from_numpy(X).to(torch.float32)
    data['labels'] = torch.from_numpy(y).to(torch.float32)
    torch.save(data, os.path.join(output_dir, 'Canton_Tower.pt'))