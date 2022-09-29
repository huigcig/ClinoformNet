import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from DeepLab.smooth_filter import smooth_conv
from DeepLab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

def smooth_conv(input,conv_kernel):
    w1 = conv_kernel.shape[-2]//2  
    w2 = conv_kernel.shape[-1]//2
    n1 = input.shape[-2]
    n2 = input.shape[-1]
    zeropad = nn.ZeroPad2d(padding=(w1,w1,w2,w2))
    input_pad = zeropad(input)
    input_pad = torch.reshape(input_pad,(input.shape[0],input.shape[1],n1+2*w1,n2+2*w2,1))
    for index1 in range(2*w1+1):                       
        for index2 in range(2*w2+1):
            if index1==0 and index2 ==0:
                input = input_pad[:,:,:n1,:n2,:]
                continue
            input = torch.cat((input,input_pad[:,:, index1:n1+index1, index2:n2+index2,:]),dim=-1)
    conv_kernel = torch.reshape(conv_kernel,(input.shape[0],1,n1,n2,((2*w1+1)*(2*w2+1))))
    input = torch.sum(torch.mul(input,conv_kernel),dim=-1)
    return input


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError
        # initial
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        
        # initial 
#         self.last_conv1 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                        BatchNorm(256),
#                                        nn.ReLU(),
#                                        nn.Dropout(0.5),
#                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                        BatchNorm(256),
#                                        nn.ReLU(),
#                                        nn.Dropout(0.1))


#         no connection
        self.last_conv1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1))
        self.last_conv2 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self._init_weight()
    
    def forward(self, x, low_level_feat, filter4):        
#         low_level_feat = self.conv1(low_level_feat)
#         low_level_feat = self.bn1(low_level_feat)
#         low_level_feat = self.relu(low_level_feat)
#         np.save("/home/huig/gh_test-master/improve-test/d1-s.npy",x.cpu().numpy())
    
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        
#         x = torch.cat((x, low_level_feat), dim=1)
        x = smooth_conv(x, filter4) # 1. add smoothing(1/4)

        x = self.last_conv1(x)
#         np.save("/home/huig/gh_test-master/improve-test/d2-s.npy",x.cpu().numpy())
        x = smooth_conv(x, filter4) # 1. add smoothing(1/4)
        x = self.last_conv2(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()                
                
def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)