#!/usr/bin/env python3
# -*- coding:utf-8 -*-
 
'''
@author: Qun Yang
@license: (C) Copyright 2024, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 06/08/24 10:38 am
@description: 
@version: 1.0
'''


import torch
import os


def combine_dataset(output_dir):
    acc_dataset = torch.load('../data/UCSD_Acc/UCSD_Acc.pt')
    disp_dataset = torch.load('../data/UCSD_Disp/UCSD_Disp.pt')

    acc_data = acc_dataset['samples']
    acc_labels = acc_dataset['labels']

    disp_data = disp_dataset['samples']
    disp_labels = disp_dataset['labels']

    all_data = torch.zeros((acc_data.size(0) + disp_data.size(0), acc_data.size(1)))
    all_labels = torch.zeros((acc_data.size(0) + disp_data.size(0), acc_labels.size(1)))

    num_fault_types = 5
    a = acc_data.size(0) // num_fault_types
    b = disp_data.size(0) // num_fault_types

    for i in range(num_fault_types):
        all_data[i * (a + b): i * (a + b) + a, :] = acc_data[i * a: (i + 1) * a, :]
        all_data[i * (a + b) + a: (i + 1) * (a + b), :] = disp_data[i * b: (i + 1) * b, :]
        all_labels[i * (a + b): i * (a + b) + a, :] = acc_labels[i * a: (i + 1) * a, :]
        all_labels[i * (a + b) + a: (i + 1) * (a + b), :] = disp_labels[i * b: (i + 1) * b, :]

    data = dict()
    data['samples'] = all_data
    data['labels'] = all_labels
    torch.save(data, os.path.join(output_dir, 'UCSD.pt'))


if __name__ == '__main__':
    output_dir = '../data/UCSD'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    combine_dataset(output_dir)
