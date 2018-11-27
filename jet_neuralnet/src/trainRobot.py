# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 09:35:32 2018

@author: kmcfall
"""

import numpy as np
import matplotlib.pyplot as plot
import glob
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import os
from skimage import io, transform
from torch.optim import lr_scheduler
import time
import copy
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = 'data'

image_dataset = datasets.ImageFolder(root='data', transform= transforms.Compose([transforms.ToTensor()]))

dataloaders = torch.utils.data.DataLoader(image_dataset, batch_size=4,
                                             shuffle=False, num_workers=4)

dataset_sizes = len(image_dataset) 

f = open('./data/Q2ndFloor.csv','r')
lines = f.readlines()
labels = lines[0].split(',')
target = []
#train = []
for i in range(1,50):
    s = lines[i].split(',')
    s[0] = './data/'+s[0]
    target.append(np.array([float(s[-2]),float(s[-1])]))
    #train.append(np.moveaxis(cv2.imread(s[0]),2,0))

labels = {labels[3], labels[4]}
inputs = next(iter(dataloaders))

class EndToEndNet(nn.Module):
    def __init__(self):
        super(EndToEndNet, self).__init__()
        self.num_outputs = 2
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 24, 5, padding = (0,2))
        self.conv2 = nn.Conv2d(24, 36, 5, padding = (0,2))
        self.conv3 = nn.Conv2d(36, 48, 5, padding = (0,2))
        self.conv4 = nn.Conv2d(48, 64, 5, padding = (0,2))
        self.conv5 = nn.Conv2d(64, 128, 3, padding = (0,1))
        self.fc1   = nn.Linear(1280,100)
        self.fc2   = nn.Linear(100,50)
        self.fc3   = nn.Linear(50,10)
        self.fc4   = nn.Linear(10,self.num_outputs)

    def forward(self, x):
        x = self.pool(x)
        #print(x.shape)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        #print(x.shape)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        #print(x.shape)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        #print(x.shape)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        #print(x.shape)
        x = F.relu(self.conv5(x))
        #print(x.shape)
        x = x.view(x.shape[0],-1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = F.relu(self.fc2(x))
        #print(x.shape)
        x = F.relu(self.fc3(x))
        #print(x.shape)
        x = self.fc4(x)
        #print(x.shape)
        return x


'''
for i in range(5):
    criterion = nn.MSELoss()                              # Define loss function
    optimizer = optim.Adam(net.parameters(),lr=0.00002)   # Initialize optimizer
    optimizer.zero_grad()                                 # Clear gradients
    predict = net(train)                                  # Forward pass
    loss = criterion(predict, target)                     # Calculate loss
    loss.backward()                                       # Backward pass (calculate gradients)
    optimizer.step()                                      # Update tunable parameters                                # Forward pass
    print(loss)
'''
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

                # Iterate over data.
            for inputs in dataloaders:
                #inputs = inputs.to(device)
                #labels = labels.to(device)
    
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, target)
    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == data.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')

def displayImage(inputs,ind):
    # OpenCV loads BGR, matplotlib displays RGB
    im = np.dstack((inputs[ind,2,:,:],inputs[ind,1,:,:],inputs[ind,0,:,:]))
    plot.figure(1)
    plot.clf()
    plot.imshow(im) 
    
# Normalize inputs and outputs to zero mean and max of one
#imMean = np.mean(train)
#train = np.array(train - imMean)
#imScale = np.max(train)
#train = torch.tensor(train/imScale, dtype=torch.float, device=device)
motorMean = np.mean(target)
target = np.array(target - motorMean)
motorScale = np.max(target)
target = torch.tensor(target/motorScale, dtype = torch.float, device=device)
net = EndToEndNet()
net = net.to(device) 

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

net = train_model(net, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)