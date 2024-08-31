#!/usr/bin/env python3
# -*- coding:utf-8 -*-
 
'''
@author: Qun Yang
@license: (C) Copyright 2024, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 29/05/24 11:42 am
@description: 
@version: 1.0
'''


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
# import wandb
import time


def train(model, config, train_dataloader, val_dataloader):
    loss_values = []
    epochs = 500
    best_val_loss = float('inf')
    val_accuracies = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.1)
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # wandb.init(
    # # set the wandb project where this run will be logged
    # project="sensor fault classification",

    # # track hyperparameters and run metadata
    # config={
    # "learning_rate": 0.001,
    # "architecture": "Transformer",
    # "dataset": "OK",
    # "epochs": 100,
    # }
    # )

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        t_0 = time.time()

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == torch.argmax(labels, 1)).sum().item()

        average_loss = total_loss / len(train_dataloader)
        loss_values.append(average_loss)
        accuracy = correct_predictions / total_samples

        # metrics = {"train/train_loss": average_loss, 
        #            "train/train_accuracy": accuracy}
            
        # wandb.log(metrics)

        scheduler.step()

        # Validation
        with torch.no_grad():
            total_val_loss = 0.0
            correct_val_predictions = 0
            total_val_samples = 0
            
            for val_inputs, val_labels in val_dataloader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                total_val_loss += val_loss.item()

                _, val_predicted = torch.max(val_outputs.data, 1)
                total_val_samples += val_labels.size(0)
                correct_val_predictions += (val_predicted == torch.argmax(val_labels, 1)).sum().item()
            average_val_loss = total_val_loss / len(val_dataloader)
            val_accuracy = correct_val_predictions / total_val_samples
            val_accuracies.append(val_accuracy)

            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                torch.save(model.state_dict(), './checkpoints/{}_{}_BATCH_{}_WIN_{}_HIST_{}_HEAD_{}_AUG_{}.pt'.format(config['dataset'], 
                                                                                                                      config['feature'],
                                                                                                                      config['batch'],
                                                                                                                      config['win_len'],
                                                                                                                      config['hist_bin'], 
                                                                                                                      config['head'],
                                                                                                                      config['data_aug']))
        t_1 = time.time()

        print('Epoch: {:4d}/{:4d}, Train Loss: {:.5f}, Train Accuracy: {:.2f}%, Val Accuracy: {:.2f}%, Time Cost: {:.2f}s'.
                format(epoch + 1, epochs, average_loss, accuracy * 100, val_accuracy * 100, t_1 - t_0))
