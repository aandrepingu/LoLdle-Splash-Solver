import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NUM_CHAMPIONS = 167
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
epochs = 5
batch_size = 4
lr = 0.001


