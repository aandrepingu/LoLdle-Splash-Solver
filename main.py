import torch.nn as nn
import numpy as np

class SplashNN(nn.Module):
    def __init__(num_images, num_champs=167):
        super().__init__()
        
