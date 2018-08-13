#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 20:14:43 2018

@author: seukgyo
"""

import os
import numpy as np
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import MNIST
from torch import optim
from torch.utils.data import DataLoader

import cae

from itertools import zip_longest
import csv
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score

#%%
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# normal class - 4
class4 = train_data[np.where(train_label==4), :, :]
class4 = class4.transpose(1, 0, 2, 3)

rand_idx = np.random.choice(len(class4), 220)
class4 = class4[rand_idx, :, :, :]

# anomaly class - 0, 7, 9
class0 = train_data[np.where(train_label==0), :, :]
class0 = class0.transpose(1, 0, 2, 3)

rand_idx = np.random.choice(len(class0), 5)
class0 = class0[rand_idx, :, :, :]

class7 = train_data[np.where(train_label==7), :, :]
class7 = class7.transpose(1, 0, 2, 3)

rand_idx = np.random.choice(len(class7), 3)
class7 = class7[rand_idx, :, :, :]

class9 = train_data[np.where(train_label==9), :, :]
class9 = class9.transpose(1, 0, 2, 3)

rand_idx = np.random.choice(len(class9), 3)
class9 = class9[rand_idx, :, :, :]

normal_class = class4
anomaly_class = np.concatenate((class0, class7, class9), axis=0)

"""
pretrained model
"""
pretrained_model_path = 'model/CAE.pth'

print('loading network...')
model = cae.CAE()
model.load_state_dict(torch.load(pretrained_model_path))
model = model.to(device)
#%%

model.eval()

encoder = model.encoder

"""
forward encoder
"""
# normal encode
normal_encode = []

for normal_img in normal_class:
    normal_img = np.reshape(normal_img, (1, 1, 28, 28))
    normal_img = torch.FloatTensor(normal_img/255.)
    normal_img = normal_img.to(device)
    output = encoder(normal_img)
    
    output = output.cpu()
    output = output.detach().numpy()
    normal_encode.append(output)
    
normal_encode = np.array(normal_encode)
normal_encode = np.reshape(normal_encode, (normal_encode.shape[0], normal_encode.shape[2]))

# anomaly encode
anomaly_encode = []

for anomaly_img in anomaly_class:
    anomaly_img = np.reshape(anomaly_img, (1, 1, 28, 28))
    anomaly_img = torch.FloatTensor(anomaly_img/255.)
    anomaly_img = anomaly_img.to(device)
    output = encoder(anomaly_img)
    
    output = output.cpu()
    output = output.detach().numpy()
    anomaly_encode.append(output)
    
anomaly_encode = np.array(anomaly_encode)
anomaly_encode = np.reshape(anomaly_encode, (anomaly_encode.shape[0], anomaly_encode.shape[2]))

#%%
"""
train oc-nn
"""
"""
oc-nn model
"""
oc_nn_model_path = 'model/oc_nn.pth'

x_size = normal_encode.shape[1]
h_size = 32
y_size = 1

class OC_NN(nn.Module):
    def __init__(self):
        super(OC_NN, self).__init__()
        
        self.dense_out1 = nn.Linear(x_size, h_size)
        self.out2 = nn.Linear(h_size, y_size)
        
    def forward(self, img):
        w1 = self.dense_out1(img)
        w2 = self.out2(w1)
        
        return w1, w2

model = OC_NN()
model.to(device)

theta = np.random.normal(0, 1, h_size + h_size * x_size + 1)
rvalue = np.random.normal(0, 1, (len(normal_encode), y_size))
nu = 0.04

def nnscore(x, w, v):
    return torch.matmul(torch.matmul(x, w), v)

def ocnn_loss(theta, x, nu, w1, w2, r):
    term1 = 0.5 * torch.sum(w1**2)
    term2 = 0.5 * torch.sum(w2**2)
    term3 = 1/nu * torch.mean(F.relu(r - nnscore(x, w1, w2)))
    term4 = -r
    
    return term1 + term2 + term3 + term4

optimizer = optim.SGD(model.parameters(), lr=0.0001)

dataset_size = len(normal_encode)
normal_encode = torch.FloatTensor(normal_encode/255.)

train_loader = DataLoader(normal_encode, batch_size=32, shuffle=True, num_workers=4, drop_last=True)

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
        w1, w2 = model(inputs)
        r = nnscore(inputs, w1, w2)        
        loss = ocnn_loss(theta, inputs, nu, w1, w2, r)
        loss = loss.mean()
        
        # backward + optimize only if in training phase
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)

    r = r.cpu().detach().numpy()
    r = np.percentile(r, q=100*nu)
    epoch_loss = running_loss / dataset_size

    print('Loss: {:.4f} '.format(epoch_loss))
    print('Epoch = %d, r = %f'%(epoch+1, r))

    # deep copy the model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), oc_nn_model_path)
        
    
    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best Loss: {:4f}'.format(best_loss))

normal_encode = normal_encode.to(device)
train_score = nnscore(normal_encode, w1, w2)
train_score = train_score.cpu().detach().numpy() - r

anomaly_encode = torch.FloatTensor(anomaly_encode)
anomaly_encode = anomaly_encode.to(device)

test_score = nnscore(anomaly_encode, w1, w2)
test_score = test_score.cpu().detach().numpy() - r

#%%
"""
Write Decision Scores to CSV
"""
decision_score_path = 'doc/oc-nn_linear.csv'

print ('Writing file to ', decision_score_path)

poslist = train_score.tolist()
neglist = test_score.tolist()

d = [poslist, neglist]
export_data = zip_longest(*d, fillvalue='')
with open(decision_score_path, 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(("Normal", "Anomaly"))
    wr.writerows(export_data)
myfile.close()

#%%
"""
Plot Decision Scores
"""
plt.plot()
plt.title("One Class NN", fontsize="x-large", fontweight='bold');
plt.hist(train_score, bins=25, label='Normal')
plt.hist(test_score, bins=25, label='Anomaly')

#%%
## Obtain the Metrics AUPRC, AUROC, P@10
y_train = np.ones(train_score.shape[0])
y_test = np.zeros(test_score.shape[0])
y_true = np.concatenate((y_train, y_test))

y_score = np.concatenate((train_score, test_score))

average_precision = average_precision_score(y_true, y_score)

print('Average precision-recall score: {0:0.4f}'.format(average_precision))

roc_score = roc_auc_score(y_true, y_score)

print('ROC score: {0:0.4f}'.format(roc_score))

def compute_precAtK(y_true, y_score, K = 10):

    if K is None:
        K = y_true.shape[0]

    # label top K largest predicted scores as + one's've

    idx = np.argsort(y_score)
    predLabel = np.zeros(y_true.shape)

    predLabel[idx[:K]] = 1

    prec = precision_score(y_true, predLabel)

    return prec

prec_atk = compute_precAtK(y_true, y_score)

print('Precision AtK: {0:0.4f}'.format(prec_atk))
