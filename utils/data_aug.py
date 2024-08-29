#!/usr/bin/env python3
# -*- coding:utf-8 -*-
 
'''
@author: Qun Yang
@license: (C) Copyright 2024, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 22/06/24 11:44 am
@description: 
@version: 1.0
'''


import torch
import numpy as np


class DataAug:

    def __init__(self, X, y):
        self.X = X.numpy()
        self.y = y.numpy()
    
    def time_warping(self, series, sigma=0.2):
        orig_steps = np.arange(series.shape[0])
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(series.shape[0],))
        warped_steps = np.cumsum(random_warps)
        warped_steps = warped_steps / warped_steps[-1] * (series.shape[0] - 1)
        return np.interp(orig_steps, warped_steps, series)

    def time_shifting(self, series, shift):
        return np.roll(series, shift)

    def add_noise(self, series, noise_level=0.01):
        noise = np.random.normal(0, noise_level, series.shape)
        return series + noise

    def random_cropping(self, series, crop_size):
        start = np.random.randint(0, len(series) - crop_size)
        return series[start:start + crop_size]
    
    def time_reversing(self, series):
        return series[::-1]
    
    def time_negating(self, series):
        return -series
    
    def transform(self):

        X_aug = []
        y_aug = []

        for i in range(self.X.shape[0]):

            X_aug.append(self.X[i])
            y_aug.append(self.y[i])
            
            if self.y[i][0] != 1:
                
                # X_aug.append(self.time_warping(self.X[i]))
                # y_aug.append(self.y[i])
                
                X_aug.append(self.time_shifting(self.X[i], shift=50))
                y_aug.append(self.y[i])

                # X_aug.append(self.time_reversing(self.X[i]))
                # y_aug.append(self.y[i])

                # X_aug.append(self.time_negating(self.X[i]))
                # y_aug.append(self.y[i])
                
                # X_aug.append(self.add_noise(self.X[i], noise_level=0.05))
                # y_aug.append(self.y[i])
            
        X_aug = np.array(X_aug)
        y_aug = np.array(y_aug)

        return torch.from_numpy(X_aug).to(torch.float32), torch.from_numpy(y_aug).to(torch.float32)
