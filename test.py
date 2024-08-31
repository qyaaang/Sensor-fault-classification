#!/usr/bin/env python3
# -*- coding:utf-8 -*-
 
'''
@author: Qun Yang
@license: (C) Copyright 2024, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 12/06/24 02:12 pm
@description: 
@version: 1.0
'''


import torch
import torch.nn as nn
from models.classifier import Classifier
from torch.utils.data import DataLoader
from dataset.sensor_dataset import SensorDataset
import queue
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import time


def test(model, config, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        total_test_loss = 0.0
        correct_test_predictions = 0
        total_test_samples = 0
        all_test_inputs = []
        all_test_outputs = []
        all_test_labels = []
        all_attn_scores = []
        all_z = []

        start_time = time.time()

        for test_inputs, test_labels in test_dataloader:
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)
            test_outputs = model(test_inputs)
            test_loss = criterion(test_outputs, test_labels)
            total_test_loss += test_loss.item()
            attn_scores = model.extractor.attention.attn
            z = model.z

            _, test_predicted = torch.max(test_outputs.data, 1)
            total_test_samples += test_labels.size(0)
            correct_test_predictions += (test_predicted == torch.argmax(test_labels, 1)).sum().item()
            all_test_inputs.append(test_inputs.cpu())
            all_test_outputs.append(test_outputs.cpu())
            all_test_labels.append(test_labels.cpu())

            all_attn_scores.append(attn_scores.cpu())
            all_z.append(z.cpu())

        all_test_inputs = torch.cat(all_test_inputs, dim=0)
        all_test_outputs = torch.cat(all_test_outputs, dim=0)
        all_test_labels = torch.cat(all_test_labels, dim=0)
        all_attn_scores = torch.cat(all_attn_scores, dim=0)
        all_z = torch.cat(all_z, dim=0)

        predicted_classes = torch.argmax(all_test_outputs, axis=1).cpu().numpy()
        cm = confusion_matrix(all_test_labels.argmax(axis=1), predicted_classes) 
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        plt.figure(figsize=(7, 7))
        sns.set(font_scale=1.2)
        if config['dataset'] == 'IPC_SHM':
            custom_labels = ["normal", "missing", "minor", "outlier", "square", "trend", "drift"]
        else:
            custom_labels = ['normal', 'random', 'malfunction', 'drift', 'bias']
        sns.heatmap(cm_percentage.T, annot=True, fmt='.1f', cmap='Blues', linewidths=.5, square=True,
                    xticklabels=custom_labels,
                    yticklabels=custom_labels)
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title('Confusion Matrix')
        plt.savefig('./results/{}_{}_BATCH_{}_WIN_{}_HIST_{}_HEAD_{}_AUG_{}/Confusion Matrix.jpeg'.format(config['dataset'],
                                                                                                          config['feature'],
                                                                                                          config['batch'],
                                                                                                          config['win_len'],
                                                                                                          config['hist_bin'],
                                                                                                          config['head'],
                                                                                                          config['data_aug']))

        end_time = time.time()  # Record the end time
        test_time = (end_time - start_time) * 1000  # Convert to milliseconds

        average_test_loss = total_test_loss / len(test_dataloader)
        test_accuracy = correct_test_predictions / total_test_samples
        print(f'Test Loss: {average_test_loss}, Test Accuracy: {test_accuracy * 100:.2f}%')
        print(f'Test Time: {test_time:.2f} ms')

        return all_test_inputs, all_test_outputs, all_test_labels, all_attn_scores, all_z


def test_sub_seq(model, config, X_test, y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classifier(config).to(device)
    model_save_path = './checkpoints/{}_{}_BATCH_{}_WIN_{}_HEAD_{}_AUG_{}.pt'.format(config['dataset'],
                                                                                     config['feature'],
                                                                                     config['batch'],
                                                                                     config['win_len'], 
                                                                                     config['head'],
                                                                                     config['data_aug'])
          
    model.load_state_dict(torch.load(model_save_path))
    model = model.to(device)

    X_test = X_test.reshape(X_test.size(0), -1, config['win_len'])

    criterion = nn.CrossEntropyLoss()

    tensor_queue = queue.Queue()

    for i in range(X_test.size(1)):
        X_test_step = X_test[:, i, :]
        tensor_queue.put(X_test_step.unsqueeze(1))

    # results = torch.zeros((X_test.size(0), X_test.size(1)))
    results = []
    while not tensor_queue.empty():
        # 取出一个切片
        X_test_step = tensor_queue.get()

        test_dataset = SensorDataset(X_test_step, y_test, config)
        test_dataloader = DataLoader(test_dataset, 
                                     batch_size=config['batch'],
                                     num_workers=2,
                                     shuffle=False)
        
        with torch.no_grad():
            total_test_loss = 0.0
            correct_test_predictions = 0
            total_test_samples = 0
            all_test_outputs = []
            all_test_labels = []

            for test_inputs, test_labels in test_dataloader:
                test_inputs = test_inputs.to(device)
                test_labels = test_labels.to(device)
                test_outputs = model(test_inputs)
                test_loss = criterion(test_outputs, test_labels)
                total_test_loss += test_loss.item()

                _, test_predicted = torch.max(test_outputs.data, 1)
                total_test_samples += test_labels.size(0)
                correct_test_predictions += (test_predicted == torch.argmax(test_labels, 1)).sum().item()
                all_test_outputs.append(test_outputs.cpu())
                all_test_labels.append(test_labels.cpu())

            all_test_outputs = torch.cat(all_test_outputs, dim=0)
            all_test_labels = torch.cat(all_test_labels, dim=0)

            predicted_classes = torch.argmax(all_test_outputs, axis=1).unsqueeze(1)

        results.append(predicted_classes)

    results = torch.cat(results, dim=1).cpu().numpy()
    y = torch.argmax(y_test, 1).cpu().numpy()

    # print(results)
    # print(y)

    predicted_classes_steps = stats.mode(results[:, :3], axis=1)
    # print(predicted_classes_steps.mode)
    same_elements = y == predicted_classes_steps.mode
    # same_elements_count = np.sum(same_elements)
    # print(same_elements_count / results.shape[0])
    print('Test Accuracy: {:.2f}%'.format(100 * correct_test_predictions / total_test_samples))

    return results
