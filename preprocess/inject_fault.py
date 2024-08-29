#!/usr/bin/env python3
# -*- coding:utf-8 -*-
 
'''
@author: Qun Yang
@license: (C) Copyright 2024, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 28/07/24 05:57 pm
@description: 
@version: 1.0
'''


import numpy as np
import copy


def random_faults(S, delta_r, i_r, seed=0):
    np.random.seed(seed)

    data = copy.deepcopy(S)
    N = len(S)
    percentages = np.random.uniform(low=0, high=100, size=N)
    intensities = np.random.uniform(low=1, high=i_r, size=N)

    for i in range(N):
        p = percentages[i]
        r = intensities[i]

        if p <= delta_r * 100:
            data[i] = data[i] * (1 + r)

    return data


def malfunction_faults(S, n_m, seed=0):
    np.random.seed(seed)

    data = copy.deepcopy(S)

    means = np.zeros(S.shape[1])
    vars = np.var(data, axis=0)
    faults = np.random.normal(loc=means, scale=np.sqrt(vars), size=(S.shape[0], S.shape[1]))
    
    faulty_data = data + n_m * faults

    return faulty_data


def drift_faults(S, i_d, n_d, seed=0):
    np.random.seed(seed)

    data = copy.deepcopy(S)

    # Noises
    means = np.zeros(S.shape[1])
    vars = np.var(data, axis=0)
    noises = np.random.normal(loc=means, scale=np.sqrt(vars), size=(S.shape[0], S.shape[1]))

    # Offsets
    r = np.random.randint(0, len(i_d))
    offsets = data[0, :] * i_d[r]
    offsets = np.tile(offsets, (S.shape[0], 1))


    t = np.arange(1, S.shape[0] + 1)
    trend = (np.power(t, 1/9) - 1) * 0.5
    trend = trend.reshape(-1, 1)

    # print(r)
    # print(offsets)
    # print(trend)
    
    # faulty_data = data + n_d * noises + offsets + trend * 0.5
    # faulty_data = data + n_d * noises + offsets + trend * 0.001
    faulty_data = data + n_d * noises + offsets + trend * 0.005
    return faulty_data


def bias_faults(S, i_b, n_b, seed=0):
    np.random.seed(seed)

    data = copy.deepcopy(S)
    seq_len = data.shape[0]
    sensor_num = data.shape[1]

    bias = np.zeros_like(data)

    def sample_step_positions(mean, std, size, min_value, max_value):
        samples = np.random.normal(mean, std, size)
        # Clip the values to be within the specified range
        samples = np.clip(samples, min_value, max_value)
        # Round the values to the nearest integers
        samples = np.round(samples).astype(int)

        return samples

    step_positions = sample_step_positions(mean=seq_len / 2, 
                                           std=seq_len / 10,
                                           size=sensor_num,
                                           min_value=1,
                                           max_value=seq_len)
    
    # i_b = np.random.uniform(low=0.01, high=0.05, size=sensor_num)
    # i_b = np.random.uniform(low=0.0001, high=0.0005, size=sensor_num)
    i_b = np.random.uniform(low=0.01, high=0.05, size=sensor_num)

    for i in range(sensor_num):
        step_idx = step_positions[i]
        bias[step_idx:, i] = i_b[i]

    faulty_data = data + bias

    return faulty_data
