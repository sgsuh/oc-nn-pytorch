#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 19:07:03 2018

@author: seukgyo
"""

import os
import torch
import numpy as np
import time
import copy

import torch.nn.functional as F

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn


"""
DataLoader
"""
data_fold = 'data'

if not os.path.isdir(data_fold):
    os.makedirs(data_fold)

train_set = MNIST(root=data_fold, train=True, download=True)
test_set = MNIST(root=data_fold, train=False, download=True)

train_data = train_set.train_data.numpy()
train_label = train_set.train_labels.numpy()

normal_train = train_data[np.where(train_label==4), :, :]
normal_train = normal_train.transpose(1, 0, 2, 3)

normal_set = torch.FloatTensor(normal_train/255.)

train_loader = DataLoader(normal_set, shuffle=True, batch_size=128, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_size = len(normal_train)

#%%
"""
Model
"""
cae_model_path = 'model/CAE.pth'

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.dense1 = nn.Linear(392, 32)
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(8)
        
    def forward(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        
        x = x.view(-1, 392)
        x = self.dense1(x)
        x = F.dropout(x, training=self.training)
        x = F.elu(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.deconv3 = nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(8, 16, 3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.dense1 = nn.Linear(32, 392)
        
        self.bn3 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        
    def forward(self, encode):
        x = self.dense1(encode)
        x = F.dropout(x, training=self.training)
        x = F.elu(x)
        
        x = x.view(x.size(0), 8, 7, 7)
        
        x = self.deconv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        
        x = self.upsample2(x)
        
        x = self.deconv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        
        x = self.upsample1(x)
        
        x = self.deconv1(x)
        x = F.sigmoid(x)
        
        return x
        
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, img):
        x = self.encoder(img)
        x = self.decoder(x)
        
        return x

"""
train
"""
if __name__ == '__main__':
    model = CAE()
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000
    
    num_epochs = 100
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
    
        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode
        
        running_loss = 0.0
    
        # Iterate over data.
        for inputs in train_loader:   
            inputs = inputs.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward
            # track history if only in train
            outputs = model(inputs)
            loss = F.mse_loss(inputs, outputs)
    
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
    
            # statistics
            running_loss += loss.item() * inputs.size(0)
    
        epoch_loss = running_loss / dataset_size
    
        print('Loss: {:.4f} '.format(epoch_loss))
    
        # deep copy the model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_model_wts)
            torch.save(model.state_dict(), cae_model_path)
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Loss: {:4f}'.format(best_loss))

