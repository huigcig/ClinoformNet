import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch import Tensor

def smooth_conv(input,conv_kernel):
    w1 = conv_kernel.shape[-2]//2  
    w2 = conv_kernel.shape[-1]//2
    n1 = input.shape[-2]
    n2 = input.shape[-1]
#     print(input.shape,conv_kernel.shape)
    zeropad = nn.ZeroPad2d(padding=(w1,w1,w2,w2))
    input_pad = zeropad(input)
    input_pad = torch.reshape(input_pad,(input.shape[0],input.shape[1],n1+2*w1,n2+2*w2,1))
    #构造与卷积核对应的矩阵
    for index1 in range(2*w1+1):                       
        for index2 in range(2*w2+1):
            if index1==0 and index2 ==0:
                input = input_pad[:,:,:n1,:n2,:]
                continue
            input = torch.cat((input,input_pad[:,:, index1:n1+index1, index2:n2+index2,:]),dim=-1)
    conv_kernel = torch.reshape(conv_kernel,(input.shape[0],1,n1,n2,((2*w1+1)*(2*w2+1))))
    #卷积操作
    input = torch.sum(torch.mul(input,conv_kernel),dim=-1)
    return input

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(       
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )
    def forward(self,x,size):
        x = F.interpolate(x,size=size,mode='bilinear',align_corners=True)  
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

# remove the original scale skip connection
class UNet_1(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(UNet_1,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x, filter4,filter8):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)

        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5,tuple(x4.shape[-2:]))
        d5 = torch.cat((x4,d5),dim=1)
        d5 = smooth_conv(d5,filter8) # filter conv filter8
        d5 = smooth_conv(d5,filter8) # filter conv filter8
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5,tuple(x3.shape[-2:]))
        d4 = torch.cat((x3,d4),dim=1)
        d4 = smooth_conv(d4,filter4) # filter conv filter4
        d4 = smooth_conv(d4,filter4) # filter conv filter4
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4,tuple(x2.shape[-2:]))
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3,tuple(x1.shape[-2:]))
#         d2 = torch.cat((x1,d2),dim=1) # remove 1-layer cat
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1    

# remove the original scale skip connection and 1/2 scale skip connenction    
class UNet_2(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(UNet_2,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x, filter4,filter8):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)

        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5,tuple(x4.shape[-2:]))
        d5 = torch.cat((x4,d5),dim=1)
        d5 = smooth_conv(d5,filter8) # filter conv filter8
        d5 = smooth_conv(d5,filter8) # filter conv filter8
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5,tuple(x3.shape[-2:]))
        d4 = torch.cat((x3,d4),dim=1)
        d4 = smooth_conv(d4,filter4) # filter conv filter4
#         d4 = smooth_conv(d4,filter4) # filter conv filter4
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4,tuple(x2.shape[-2:]))
#         d3 = torch.cat((x2,d3),dim=1) # remove the 1/2 scale skip connection
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3,tuple(x1.shape[-2:]))
#         d2 = torch.cat((x1,d2),dim=1) # remove the original scale skip connection
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1    

# remove the original scale skip connection and 1/2 scale skip connenction and 1/4 scale skip connenction       
class UNet_3(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(UNet_3,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x, filter4,filter8):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)

        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5,tuple(x4.shape[-2:]))
        d5 = torch.cat((x4,d5),dim=1)
        d5 = smooth_conv(d5,filter8) # filter conv filter8
        d5 = smooth_conv(d5,filter8) # filter conv filter8
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5,tuple(x3.shape[-2:]))
#         d4 = torch.cat((x3,d4),dim=1) # remove the 1/4 scale skip connection
        d4 = smooth_conv(d4,filter4) # filter conv filter4
#         d4 = smooth_conv(d4,filter4) # filter conv filter4
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4,tuple(x2.shape[-2:]))
#         d3 = torch.cat((x2,d3),dim=1) # remove the 1/2 scale skip connection
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3,tuple(x1.shape[-2:]))
#         d2 = torch.cat((x1,d2),dim=1) # remove the original scale skip connection
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1  
    
class UNet(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(UNet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x, filter4,filter8):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)

        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5,tuple(x4.shape[-2:]))
        d5 = torch.cat((x4,d5),dim=1)
        d5 = smooth_conv(d5,filter8) # filter conv filter8
#         d5 = smooth_conv(d5,filter8) # filter conv filter8
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5,tuple(x3.shape[-2:]))
        d4 = torch.cat((x3,d4),dim=1) # the 1/2 scale skip connection
        d4 = smooth_conv(d4,filter4) # filter conv filter4
#         d4 = smooth_conv(d4,filter4) # filter conv filter4
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4,tuple(x2.shape[-2:]))
        d3 = torch.cat((x2,d3),dim=1) # the 1/2 scale skip connection
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3,tuple(x1.shape[-2:]))
#         d2 = torch.cat((x1,d2),dim=1) # the original scale skip connection
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1