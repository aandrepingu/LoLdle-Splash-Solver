import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
NUM_CHAMPIONS = 167

class SplashCNN(nn.Module):
    def __init__(self, num_images, num_champs=167):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 37 * 22, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, NUM_CHAMPIONS)


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 37 * 22)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x
    
    def predict(self, x):
        res = self.forward(x)
        # get champ with highest probability out of output from classifier

