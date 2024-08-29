#!/usr/bin/env python3
# -*- coding:utf-8 -*-
 
'''
@author: Qun Yang
@license: (C) Copyright 2024, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 05/08/24 02:55 pm
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


accelerometers = ['AC1E402', 'AC1E103', 'AC1N101', 'AC1U402', 'AC1U101', 'AC1N102', 	
                  'AC1E102', 'AC1U102', 'AC1U401', 'AC1N401', 'AC1N402', 'AC1N301',  	
                  'AC1E301', 'AC1N403', 'AC1N103', 'AC1E101', 'AC1E401', 'AC1E403',  	
                  'ACRE102', 'ACRN103', 'ACRV102', 'ACRN402', 'ACRU402', 'ACRN101',  	
                  'ACRE402', 'ACRE103', 'ACRE101', 'ACRU101', 'ACRN102', 'ACRN401',  	
                  'ACRN313', 'ACRE403', 'ACRN403', 'ACRE313', 'ACRU401', 'ACRE401', 	
                  'ACBE209']

linear_pots = ['LPRN401', 'LPRN213', 'LPRN201', 'SPBU401', 'SPBU209', 'SPBU105',  	
               'LPRN102', 'LPRE102', 'LPRE201', 'LPRE101', 'LPRN203', 'LPRN208',  	
               'LPRN207', 'LPRN103', 'LPRN216', 'LPRN204', 'LPRE203', 'LP1E201',  	
               'LP1N201', 'LP1E209', 'LP1E212', 'LP1N206']


def extract_number(filename):
    # 使用正则表达式提取文件名中的数字
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')


def make_normal_data(file_path):
    if not os.path.exists('../data/UCSD_Disp/normal'):
        os.mkdir('../data/UCSD_Disp/normal')
    test_names = sorted(os.listdir(file_path), key=extract_number)

    for test_name in tqdm(test_names, desc='Making normal data'):
        for data_file in os.listdir(os.path.join(file_path, test_name)):
            all_normal_data = pd.read_excel(os.path.join(file_path, test_name, data_file), skiprows=[0, 1, 3])
            all_normal_data.columns = all_normal_data.columns.str.strip()
            all_normal_data = all_normal_data[linear_pots]
            all_normal_data = np.array(all_normal_data)
        
        np.save('../data/UCSD_Disp/normal/{}.npy'.format(test_name), all_normal_data)
            
        


def make_random_faults(file_path):
    if not os.path.exists('../data/UCSD_Disp/random'):
        os.mkdir('../data/UCSD_Disp/random')
    test_names = sorted(os.listdir(file_path), key=extract_number)

    for test_name in tqdm(test_names, desc='Making random faults'):
        for data_file in os.listdir(os.path.join(file_path, test_name)):
            all_normal_data = pd.read_excel(os.path.join(file_path, test_name, data_file), skiprows=[0, 1, 3])
            all_normal_data.columns = all_normal_data.columns.str.strip()
            all_normal_data = all_normal_data[linear_pots]
            all_faulty_data = np.zeros_like(np.array(all_normal_data))

            # print(all_acc_data)

            for i, sensor in enumerate(all_normal_data.columns):
                normal_data = np.array(all_normal_data[sensor])
                faulty_data = random_faults(normal_data, 0.0005, 2.5)
                all_faulty_data[:, i] = faulty_data
                # print(data)
                # print(faulty_data)
                # print(np.linalg.norm(faulty_data - data))

        #         fig, ax = plt.subplots(figsize=(10, 4))
        #         ax.plot(normal_data, c='k', ls='-', lw=1.0, label='Normal')
        #         ax.plot(faulty_data, c='r', ls='--', lw=0.5, label='Faulty')

        #         ax.legend()
        #         fig.savefig('UCSD_Disp_random_faults.jpg')
        #         break
        #     break
        # break
            
        np.save('../data/UCSD_Disp/random/{}.npy'.format(test_name), all_faulty_data)


def make_malfunction_faults(file_path):
    if not os.path.exists('../data/UCSD_Disp/malfunction'):
        os.mkdir('../data/UCSD_Disp/malfunction')
    test_names = sorted(os.listdir(file_path), key=extract_number)
    
    for test_name in tqdm(test_names, desc='Making malfunction faults'):
        for data_file in os.listdir(os.path.join(file_path, test_name)):
            all_normal_data = pd.read_excel(os.path.join(file_path, test_name, data_file), skiprows=[0, 1, 3])
            all_normal_data.columns = all_normal_data.columns.str.strip()
            all_normal_data = all_normal_data[linear_pots]
            all_normal_data = np.array(all_normal_data)
            all_faulty_data = malfunction_faults(all_normal_data, 5.5)

        #     fig, ax = plt.subplots(figsize=(10, 4))
        #     ax.plot(all_normal_data[:, 0], c='k', ls='-', lw=1.0, label='Normal')
        #     ax.plot(all_faulty_data[:, 0], c='r', ls='--', lw=0.5, label='Faulty')

        #     ax.legend()
        #     fig.savefig('UCSD_Disp_malfunction_faults.jpg')
        #     break
        # break

        np.save('../data/UCSD_Disp/malfunction/{}.npy'.format(test_name), all_faulty_data)


def make_drift_faults(file_path):
    if not os.path.exists('../data/UCSD_Disp/drift'):
        os.mkdir('../data/UCSD_Disp/drift')
    test_names = sorted(os.listdir(file_path), key=extract_number)
    
    for test_name in tqdm(test_names, desc='Making drift faults'):
        data_files = os.listdir(os.path.join(file_path, test_name))
        for data_file in data_files:
            all_normal_data = pd.read_excel(os.path.join(file_path, test_name, data_file), skiprows=[0, 1, 3])
            all_normal_data.columns = all_normal_data.columns.str.strip()
            all_normal_data = all_normal_data[linear_pots]
            all_normal_data = np.array(all_normal_data)
            all_faulty_data = drift_faults(all_normal_data, i_d=[-2, -1, 1, 2], n_d=1)

        #     fig, ax = plt.subplots(figsize=(10, 4))
        #     ax.plot(all_normal_data[:, 0], c='k', ls='-', lw=1.0, label='Normal')
        #     ax.plot(all_faulty_data[:, 0], c='r', ls='--', lw=0.5, label='Faulty')

        #     ax.legend()
        #     fig.savefig('UCSD_Disp_drift_faults.jpg')
        #     break
        # break

        np.save('../data/UCSD_Disp/drift/{}.npy'.format(test_name), all_faulty_data)


def make_bias_faults(file_path):
    if not os.path.exists('../data/UCSD_Disp/bias'):
        os.mkdir('../data/UCSD_Disp/bias')
    test_names = sorted(os.listdir(file_path), key=extract_number)
    
    for test_name in tqdm(test_names, desc='Making bias faults'):
        data_files = os.listdir(os.path.join(file_path, test_name))
        for data_file in data_files:
            all_normal_data = pd.read_excel(os.path.join(file_path, test_name, data_file), skiprows=[0, 1, 3])
            all_normal_data.columns = all_normal_data.columns.str.strip()
            all_normal_data = all_normal_data[linear_pots]
            all_normal_data = np.array(all_normal_data)

            all_faulty_data = bias_faults(all_normal_data, i_b=[1.5], n_b=0)

        #     fig, ax = plt.subplots(figsize=(10, 4))
        #     ax.plot(all_normal_data[:, 0], c='k', ls='-', lw=1.0, label='Normal')
        #     ax.plot(all_faulty_data[:, 0], c='r', ls='--', lw=0.5, label='Faulty')

        #     ax.legend()
        #     fig.savefig('UCSD_Disp_bias_faults.jpg')

        #     break
        # break

        np.save('../data/UCSD_Disp/bias/{}.npy'.format(test_name), all_faulty_data)


def read_data(file_path):
    faulty_types = ['normal', 'random', 'malfunction', 'drift', 'bias']

    all_data = []
    all_labels = []

    for label_idx, faulty_type in enumerate(faulty_types):
        for data_file in sorted(os.listdir(os.path.join(file_path, faulty_type)), key=extract_number):
            data = np.load(os.path.join(file_path, faulty_type, data_file))
            labels = np.full(data.shape[1], label_idx)
            all_data.append(data.T)
            all_labels.append(labels)

    y = np.concatenate(all_labels, axis=0)

    X = []

    for arr in all_data:
        X += arr.tolist()

    # print(len(X))
    # print(len(X[0]))

    # Find the maximum length
    max_length = max(len(seq) for seq in X)

    # Pad sequences
    X_padded = [seq + [0] * (max_length - len(seq)) for seq in X]
    X_padded = np.array(X_padded)

    # print(X_padded.shape)

    # print(X.shape)
    # print(y.shape)
                
    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(y)
    num_classes = np.max(encoded_Y) + 1
    one_hot_label = np.eye(num_classes)[encoded_Y]
    # print(X.shape)
    # print(one_hot_label.shape)

    return X_padded, one_hot_label


if __name__ == '__main__':
    if not os.path.exists('../data/UCSD_Disp'):
        os.mkdir('../data/UCSD_Disp')
    output_dir = '../data/UCSD_Disp'

    file_path = '../data/raw/UCSD'

    rc={'font.family': 'Arial',
            # 'mathtext.fontset': 'stix',
            'figure.constrained_layout.use': True,
            'text.usetex': False,
            'figure.dpi': 300
            }
    matplotlib.rcParams.update(rc)

    # make_normal_data(file_path)

    # make_random_faults(file_path)

    make_malfunction_faults(file_path)

    # make_drift_faults(file_path)
    
    # make_bias_faults(file_path)

    X, y = read_data(output_dir)
    # print(X.shape)
    # print(y.shape)
    data = dict()
    data['samples'] = torch.from_numpy(X).to(torch.float32)
    data['labels'] = torch.from_numpy(y).to(torch.float32)
    torch.save(data, os.path.join(output_dir, 'UCSD_Disp.pt'))
