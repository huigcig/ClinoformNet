import torch
import torch.nn.functional as F
import torch.nn as nn

import unets
from DeepLab import deeplab 


class UNet(nn.Module):
    def __init__(self, param):
        super(UNet, self).__init__()
        self.encoder_decoder = unets.UNet_1(img_ch=param['input_channels'], 
                                         output_ch=param['output_channels'])
    def forward(self, x, filter4, filter8):
        x = self.encoder_decoder(x, filter4, filter8)
#         print("models.py:",x.shape)
        return x   
    
class DeepLab(nn.Module):
    def __init__(self, param):
        super(DeepLab, self).__init__()   
        self.DeepLab = deeplab.DeepLab(img_ch=param['input_channels'],
                                              num_classes=param['output_channels'])
        
    def forward(self, x, filter4, filter16): # add smoothing(gh)
        x = self.DeepLab(x, filter4, filter16) # add smoothing(gh)
        return x     
