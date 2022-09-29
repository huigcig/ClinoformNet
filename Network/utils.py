import os
import torch
import random
import struct
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def sort_list_IDs(list_IDs):
    list_nums = [int(i.split(".")[0]) for i in list_IDs]
    list_sort = sorted(enumerate(list_nums), key=lambda x:x[1])
    list_index = [i[0] for i in list_sort]
    list_IDs_new = [list_IDs[i] for i in list_index]
    return list_IDs_new

def threshold_predictions(thresholded_preds, thr=0.01):
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds

def write_cube(data, path):
    data = np.transpose(data,[1,2,0]).astype(np.single)
    data.tofile(path)

def read_cube(input_data_path, size):
    input_cube = np.fromfile(input_data_path, dtype=np.single)
    input_cube = np.reshape(input_cube, size)
    input_cube = input_cube.transpose((2,0,1))
    return input_cube

# 读取目录
def read_path_list(path):
    file_list = os.listdir(path)
    file_name_list = [i.split(".")[0] for i in file_list]
    file_name_list = sorted(enumerate(file_name_list), key=lambda x:x[1]) 
    file_list = [file_list[i] for i in [j[0] for j in file_name_list]]
    return file_list

# 归一化
def min_max_norm(x):
    if torch.is_tensor(x) and torch.max(x) != torch.min(x):
            x = x - torch.min(x)
            x = x / torch.max(x)        
    elif np.max(x) != np.min(x):
            x = x - np.min(x)
            x = x / np.max(x)
    return x
    
# 标准化
def mea_std_norm(x):
    if torch.is_tensor(x[x!=0]) and torch.std(x[x!=0]) != 0:
            x[x!=0] = (x[x!=0] - torch.mean(x[x!=0])) / torch.std(x[x!=0])
    elif np.std(x[x!=0]) != 0:
            x[x!=0] = (x[x!=0] - np.mean(x[x!=0])) / np.std(x[x!=0])
    return x

def mea_std_norm2(x):
    x = (x - np.mean(x[x!=0])) / np.std(x[x!=0])
    return x

# 定义数据集    
class build_dataset(Dataset):
    def __init__(self, samples_list, dataset_path, mode, input_attr_list=["data"], 
                 mask_attr_list = ["mask"],
                 filter2_attr_list=["filter2"],filter4_attr_list=["filter4"],
                 filter8_attr_list=["filter8"],filter16_attr_list=["filter16"],
                 output_attr_list=["label"], mask=False,norm=None):
        self.samples_list = samples_list
        self.dataset_path = dataset_path
        self.input_attr_list = input_attr_list
        self.output_attr_list = output_attr_list
        self.mask_attr_list = mask_attr_list
        self.filter2_attr_list = filter2_attr_list
        self.filter4_attr_list = filter4_attr_list
        self.filter8_attr_list = filter8_attr_list
        self.filter16_attr_list = filter16_attr_list
        self.mask = mask
        self.mode = mode
        
    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_file_name = self.samples_list[idx]
        sample_file_path = os.path.join(self.dataset_path, sample_file_name + ".npy")
        sample_dict = np.load(sample_file_path, allow_pickle=True).item()
        
        sample_output = {}
        
        if self.mode in ['Train', 'Valid']:
            
            for i, output_attr in enumerate(self.output_attr_list):
                tmp = sample_dict[output_attr].astype(np.single)
                tmp = tmp[np.newaxis,:,:]
                sample_output[output_attr] = tmp
                
            for i, input_attr in enumerate(self.input_attr_list):
                tmp = np.array(sample_dict[input_attr]).astype(np.single)
                tmp = mea_std_norm(tmp)
                sample_output[input_attr] = tmp
                
#             for i, filter2_attr in enumerate(self.filter2_attr_list):
#                 tmp2 = np.array(sample_dict[filter2_attr]).astype(np.single)
#                 sample_output[filter2_attr] = tmp2
            
            for i, mask_attr in enumerate(self.mask_attr_list):
                tmp_mask = np.array(sample_dict[mask_attr]).astype(np.single)
                sample_output[mask_attr] = tmp_mask
                
            for i, filter4_attr in enumerate(self.filter4_attr_list):
                tmp4 = np.array(sample_dict[filter4_attr]).astype(np.single)
                sample_output[filter4_attr] = tmp4
                
#             for i, filter8_attr in enumerate(self.filter8_attr_list):
#                 tmp8 = np.array(sample_dict[filter8_attr]).astype(np.single)
#                 sample_output[filter8_attr] = tmp8
                
            for i, filter16_attr in enumerate(self.filter16_attr_list):
                tmp16 = np.array(sample_dict[filter16_attr]).astype(np.single)
                sample_output[filter16_attr] = tmp16

        elif self.mode == 'Infer':
            for i, input_attr in enumerate(self.input_attr_list):
                tmp = np.array(sample_dict[input_attr]).astype(np.single)
                tmp = mea_std_norm(tmp)
                sample_output[input_attr] = tmp
         
        if self.mask:    
            sample_output["mask"] = sample_dict["mask"][np.newaxis,:,:].astype(np.single)
            
        sample_output["sample_file_path"] = sample_file_path
        return  sample_output
    
# 曲线光滑函数
def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed    
    

# 训练和验证
def train_valid_netDeepLab(param, model, train_data, valid_data, 
                           input_attrs=["data"], output_attrs=["label"], 
                           mask_attrs=["mask"],filter4_attrs=["filter4"],filter16_attrs=["filter16"],
                           plot=True):

    #初始化参数
    epochs = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    lr_patience = param['lr_patience']
    lr_factor = param['lr_factor']
    optimizer_type = param['optimizer_type']
    gamma = param['gamma']
    step_size = param['step_size']
    momentum = param['momentum']
    weight_decay = param['weight_decay']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_path = param['checkpoint_path']
    rgt_fault_epoch = param['rgt_fault_epoch']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False)

    if optimizer_type == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=lr_factor)

#     num_count = np.array([858883., 626672., 272627., 161818.]).astype(np.single)
#     num_count = 1. / (num_count / num_count.sum())
#     num_count = num_count / num_count.sum()
#     criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(num_count)).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=2).to(device)
#     criterion = nn.BCEWithLogitsLoss().to(device)
#     criterion = nn.BCELoss().to(device)
#     criterion = nn.MSELoss().to(device)

    # 主循环
    epoch_loss_train, epoch_loss_valid, epoch_lr = [], [], []
    
    best_mse = 1e50

    for epoch in range(epochs):
        # 训练阶段
#         model.train()
        
        model.train()
        loss_train_per_epoch = 0
        for batch_idx, batch_samples in enumerate(train_loader):
            
            for i, input_attr in enumerate(input_attrs): # spns
                tmp = batch_samples[input_attr]
                if i  == 0:
                    data = tmp
                else:
                    data = torch.cat((data, tmp), dim=1)
                    
#             for i, mask_attr in enumerate(mask_attrs): # mask
#                 tmp_mask = batch_samples[mask_attr]
#                 if i  == 0:
#                     mask = tmp_mask
#                 else:
#                     mask = torch.cat((mask, tmp_mask), dim=1)
                    
            for j, filter4_attr in enumerate(filter4_attrs): # filter4
                    tmp_f4 = batch_samples[filter4_attr]
                    if j  == 0:
                        filter4 = tmp_f4
                    else:
                        filter4 = torch.cat((filter4, tmp_f4), dim=1)
                        
            for j, filter16_attr in enumerate(filter16_attrs): # filter16
                    tmp_f16 = batch_samples[filter16_attr]
                    if j  == 0:
                        filter16 = tmp_f16
                    else:
                        filter16 = torch.cat((filter16, tmp_f16), dim=1)
            
            target = batch_samples[output_attrs[0]].long().squeeze(1)
            
            data = data.unsqueeze(1)
#             mask = mask.unsqueeze(1) # loss test
#             target = target.unsqueeze(1) # loss test
            filter4 = filter4.unsqueeze(1)
            filter16 = filter16.unsqueeze(1)
            
            data, target = data.to(device), target.to(device)
#             mask = mask.to(device) # loss test
            filter4, filter16 = filter4.to(device), filter16.to(device)
            
            data, target = Variable(data), Variable(target)
#             mask = Variable(mask)
            filter4, filter16 = Variable(filter4), Variable(filter16)
            
            optimizer.zero_grad()
            
            target_i = model(data, filter4,filter16) # 1. add smoothing(gh) 
            
#             target_i = target_i * mask  # loss test
#             target = target * mask # loss test
#             print(target_i.shape,target.shape,mask.shape)# loss test
            
            loss = criterion(target_i, target)
#             print(loss)
            
            loss.backward()
            optimizer.step()

            loss_train_per_epoch += loss.item()

        # 验证阶段
        model.eval()
        loss_valid_per_epoch = 0
        
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                
                for i, input_attr in enumerate(input_attrs):
                    tmp = batch_samples[input_attr]
                    if i  == 0:
                        data = tmp
                    else:
                        data = torch.cat((data, tmp), dim=1)
                        
#                 for i, mask_attr in enumerate(mask_attrs): # mask
#                     tmp_mask = batch_samples[mask_attr]
#                     if i  == 0:
#                         mask = tmp_mask
#                     else:
#                         mask = torch.cat((mask, tmp_mask), dim=1)
                        
                for j, filter4_attr in enumerate(filter4_attrs): # filter4
                    tmp_f4 = batch_samples[filter4_attr]
                    if j  == 0:
                        filter4 = tmp_f4
                    else:
                        filter4 = torch.cat((filter4, tmp_f4), dim=1)
                        
                for j, filter16_attr in enumerate(filter16_attrs): # filter16
                    tmp_f16 = batch_samples[filter16_attr]
                    if j  == 0:
                        filter16 = tmp_f16
                    else:
                        filter16 = torch.cat((filter16, tmp_f16), dim=1)
                
                target = batch_samples[output_attrs[0]].long().squeeze(1)
                
                data = data.unsqueeze(1)
#                 mask = mask.unsqueeze(1) # loss test
#                 target = target.unsqueeze(1) # loss test
                filter4 = filter4.unsqueeze(1)
                filter16 = filter16.unsqueeze(1)

                data, target = data.to(device), target.to(device)
                filter4, filter16 = filter4.to(device), filter16.to(device)
#                 mask = mask.to(device) # loss test
                data, target = Variable(data), Variable(target)
#                 mask = Variable(mask) # loss test
                filter4, filter16 = Variable(filter4), Variable(filter16)

#                 target_i = model(data)
                target_i = model(data, filter4,filter16) # 1. add smoothing(gh)
                
#                 target_i = target_i * mask  # loss test
#                 target = target * mask # loss test
                
#                 print(target_i.shape,target.shape,mask.shape)# loss test
                
                loss = criterion(target_i, target)

                loss_valid_per_epoch += loss.item()

        loss_train_per_epoch = loss_train_per_epoch / len(train_loader)
        loss_valid_per_epoch = loss_valid_per_epoch / len(valid_loader)

        epoch_loss_train.append(loss_train_per_epoch)
        epoch_loss_valid.append(loss_valid_per_epoch)
        
        epoch_lr.append(optimizer.param_groups[0]['lr'])

        # 保存模型
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)

        # 保存最优模型
        if loss_valid_per_epoch < best_mse:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_mse = loss_valid_per_epoch

        scheduler.step(loss_train_per_epoch)

        # 显示loss
        if epoch % disp_inter == 0:
            print('Epoch:{}, Training Loss:{:.8f}  Validation Loss:{:.8f}  Learning rate: {:.8f}'.format(epoch, loss_train_per_epoch, loss_valid_per_epoch, epoch_lr[epoch]))
            
    # 训练loss曲线
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(epoch_loss_train, 0.6), label='Training loss')
        ax.plot(x, smooth(epoch_loss_valid, 0.6), label='Validation loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)
        ax.set_title(f'Training curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title(f'Learning rate curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()
            
    logs = {"epoch_loss_train":epoch_loss_train,
            "epoch_loss_valid":epoch_loss_valid,
            "epoch_lr":epoch_lr}
    np.save(os.path.join(checkpoint_path, 'logs.npy'), logs)
    return model

# 训练和验证 loss test
def train_valid_netDeepLab_loss(param, model, train_data, valid_data, 
                           input_attrs=["data"], output_attrs=["label"], 
                           mask_attrs=["mask"],filter4_attrs=["filter4"],filter16_attrs=["filter16"],
                           plot=True):

    #初始化参数
    epochs = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    lr_patience = param['lr_patience']
    lr_factor = param['lr_factor']
    optimizer_type = param['optimizer_type']
    gamma = param['gamma']
    step_size = param['step_size']
    momentum = param['momentum']
    weight_decay = param['weight_decay']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_path = param['checkpoint_path']
    rgt_fault_epoch = param['rgt_fault_epoch']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False)

    if optimizer_type == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=lr_factor)

#     num_count = np.array([858883., 626672., 272627., 161818.]).astype(np.single)
#     num_count = 1. / (num_count / num_count.sum())
#     num_count = num_count / num_count.sum()
#     criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(num_count)).to(device)
    
#     criterion = nn.CrossEntropyLoss(ignore_index=2).to(device)
#     criterion = nn.BCEWithLogitsLoss().to(device)
    criterion = nn.BCELoss().to(device)
#     criterion = nn.MSELoss().to(device)

    # 主循环
    epoch_loss_train, epoch_loss_valid, epoch_lr = [], [], []
    
    best_mse = 1e50

    for epoch in range(epochs):
        # 训练阶段
#         model.train()
        
        model.train()
        loss_train_per_epoch = 0
        for batch_idx, batch_samples in enumerate(train_loader):
            
            for i, input_attr in enumerate(input_attrs): # spns
                tmp = batch_samples[input_attr]
                if i  == 0:
                    data = tmp
                else:
                    data = torch.cat((data, tmp), dim=1)
                    
            for i, mask_attr in enumerate(mask_attrs): # mask
                tmp_mask = batch_samples[mask_attr]
                if i  == 0:
                    mask = tmp_mask
                else:
                    mask = torch.cat((mask, tmp_mask), dim=1)
                    
            for j, filter4_attr in enumerate(filter4_attrs): # filter4
                    tmp_f4 = batch_samples[filter4_attr]
                    if j  == 0:
                        filter4 = tmp_f4
                    else:
                        filter4 = torch.cat((filter4, tmp_f4), dim=1)
                        
            for j, filter16_attr in enumerate(filter16_attrs): # filter16
                    tmp_f16 = batch_samples[filter16_attr]
                    if j  == 0:
                        filter16 = tmp_f16
                    else:
                        filter16 = torch.cat((filter16, tmp_f16), dim=1)
            
            target = batch_samples[output_attrs[0]].long().squeeze(1)
            
            data = data.unsqueeze(1)
            mask = mask.unsqueeze(1) # loss test
            target = target.unsqueeze(1) # loss test
            filter4 = filter4.unsqueeze(1)
            filter16 = filter16.unsqueeze(1)
            
            data, target = data.to(device), target.to(device)
            mask = mask.to(device) # loss test
            filter4, filter16 = filter4.to(device), filter16.to(device)
            
            data, target = Variable(data), Variable(target)
            mask = Variable(mask)
            filter4, filter16 = Variable(filter4), Variable(filter16)
            
            optimizer.zero_grad()
            
            target_i = model(data, filter4,filter16) # 1. add smoothing(gh) 

            target_i = target_i * mask  # loss test
            target = target * mask # loss test
#             print(target_i.shape,target.shape,mask.shape)# loss test
            sigmoid = nn.Sigmoid() # bce loss addition
            loss = criterion(sigmoid(target_i), target) # bce loss addition
#             loss = criterion(target_i, target)
#             print(loss)
            
            loss.backward()
            optimizer.step()

            loss_train_per_epoch += loss.item()

        # 验证阶段
        model.eval()
        loss_valid_per_epoch = 0
        
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                
                for i, input_attr in enumerate(input_attrs):
                    tmp = batch_samples[input_attr]
                    if i  == 0:
                        data = tmp
                    else:
                        data = torch.cat((data, tmp), dim=1)
                        
                for i, mask_attr in enumerate(mask_attrs): # mask
                    tmp_mask = batch_samples[mask_attr]
                    if i  == 0:
                        mask = tmp_mask
                    else:
                        mask = torch.cat((mask, tmp_mask), dim=1)
                        
                for j, filter4_attr in enumerate(filter4_attrs): # filter4
                    tmp_f4 = batch_samples[filter4_attr]
                    if j  == 0:
                        filter4 = tmp_f4
                    else:
                        filter4 = torch.cat((filter4, tmp_f4), dim=1)
                        
                for j, filter16_attr in enumerate(filter16_attrs): # filter16
                    tmp_f16 = batch_samples[filter16_attr]
                    if j  == 0:
                        filter16 = tmp_f16
                    else:
                        filter16 = torch.cat((filter16, tmp_f16), dim=1)
                
                target = batch_samples[output_attrs[0]].long().squeeze(1)
                
                data = data.unsqueeze(1)
                mask = mask.unsqueeze(1) # loss test
                target = target.unsqueeze(1) # loss test
                filter4 = filter4.unsqueeze(1)
                filter16 = filter16.unsqueeze(1)

                data, target = data.to(device), target.to(device)
                filter4, filter16 = filter4.to(device), filter16.to(device)
                mask = mask.to(device) # loss test
                data, target = Variable(data), Variable(target)
                mask = Variable(mask) # loss test
                filter4, filter16 = Variable(filter4), Variable(filter16)

#                 target_i = model(data)
                target_i = model(data, filter4,filter16) # 1. add smoothing(gh)
                
                target_i = target_i * mask  # loss test
                target = target * mask # loss test
                
#                 print(target_i.shape,target.shape,mask.shape)# loss test
                sigmoid = nn.Sigmoid() # bce loss addition
                loss = criterion(sigmoid(target_i), target) # bce loss addition
#                 loss = criterion(target_i, target)

                loss_valid_per_epoch += loss.item()

        loss_train_per_epoch = loss_train_per_epoch / len(train_loader)
        loss_valid_per_epoch = loss_valid_per_epoch / len(valid_loader)

        epoch_loss_train.append(loss_train_per_epoch)
        epoch_loss_valid.append(loss_valid_per_epoch)
        
        epoch_lr.append(optimizer.param_groups[0]['lr'])

        # 保存模型
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)

        # 保存最优模型
        if loss_valid_per_epoch < best_mse:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_mse = loss_valid_per_epoch

        scheduler.step(loss_train_per_epoch)

        # 显示loss
        if epoch % disp_inter == 0: 
            print('Epoch:{}, Training Loss:{:.8f}  Validation Loss:{:.8f}  Learning rate: {:.8f}'.format(epoch, loss_train_per_epoch, loss_valid_per_epoch, epoch_lr[epoch]))
            
    # 训练loss曲线
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(epoch_loss_train, 0.6), label='Training loss')
        ax.plot(x, smooth(epoch_loss_valid, 0.6), label='Validation loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)
        ax.set_title(f'Training curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title(f'Learning rate curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()
            
    logs = {"epoch_loss_train":epoch_loss_train,
            "epoch_loss_valid":epoch_loss_valid,
            "epoch_lr":epoch_lr}
    np.save(os.path.join(checkpoint_path, 'logs.npy'), logs)
    return model


def train_valid_Unet(param, model, train_data, valid_data, input_attrs=["data"], output_attrs=["label"], 
                     filter2_attrs=["filter2"], filter4_attrs=["filter4"], filter8_attrs=["filter8"],
                     plot=True):

    #初始化参数
    epochs = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    lr_patience = param['lr_patience']
    lr_factor = param['lr_factor']
    optimizer_type = param['optimizer_type']
    gamma = param['gamma']
    step_size = param['step_size']
    momentum = param['momentum']
    weight_decay = param['weight_decay']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_path = param['checkpoint_path']
    rgt_fault_epoch = param['rgt_fault_epoch']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False)

    if optimizer_type == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=lr_factor)

#     num_count = np.array([858883., 626672., 272627., 161818.]).astype(np.single)
#     num_count = 1. / (num_count / num_count.sum())
#     num_count = num_count / num_count.sum()
#     criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(num_count)).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=2).to(device)
#     criterion = nn.BCEWithLogitsLoss().to(device)
#     criterion = nn.MSELoss().to(device)

    # 主循环
    epoch_loss_train, epoch_loss_valid, epoch_lr = [], [], []
    
    best_mse = 1e50

    for epoch in range(epochs):
        # 训练阶段
#         model.train()
        
        model.train()
        loss_train_per_epoch = 0
        for batch_idx, batch_samples in enumerate(train_loader):
            
            for i, input_attr in enumerate(input_attrs): # spns
                tmp = batch_samples[input_attr]
                if i  == 0:
                    data = tmp
                else:
                    data = torch.cat((data, tmp), dim=1)
                    
#             for j, filter2_attr in enumerate(filter2_attrs): # filter
#                 tmp_f2 = batch_samples[filter2_attr]
#                 if j  == 0:
#                     filter2 = tmp_f2
#                 else:
#                     filter2 = torch.cat((filter2, tmp_f2), dim=1)
        
            for j, filter4_attr in enumerate(filter4_attrs): # filter
                tmp_f4 = batch_samples[filter4_attr]
                if j  == 0:
                    filter4 = tmp_f4
                else:
                    filter4 = torch.cat((filter4, tmp_f4), dim=1)
            
            for j, filter8_attr in enumerate(filter8_attrs): # filter
                tmp_f8 = batch_samples[filter8_attr]
                if j  == 0:
                    filter8 = tmp_f8
                else:
                    filter8 = torch.cat((filter8, tmp_f8), dim=1)
                    
            data = data.unsqueeze(1)
            target = batch_samples[output_attrs[0]].long().squeeze(1)
#             filter2 = filter2.unsqueeze(1)
            filter4 = filter4.unsqueeze(1)
            filter8 = filter8.unsqueeze(1)
            
            data, target = data.to(device), target.to(device)
            filter4, filter8 = filter4.to(device), filter8.to(device)
            data, target = Variable(data), Variable(target)
            filter4, filter8 = Variable(filter4), Variable(filter8)
            
            optimizer.zero_grad()
            
#             target_i = model(data,filter2, filter4, filter8) # 1. add smoothing(gh)
            target_i = model(data, filter4, filter8)
            loss = criterion(target_i, target)

            loss.backward()
            optimizer.step()

            loss_train_per_epoch += loss.item()

        # 验证阶段
        model.eval()
        loss_valid_per_epoch = 0
        
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                
                for i, input_attr in enumerate(input_attrs):
                    tmp = batch_samples[input_attr]
                    if i  == 0:
                        data = tmp
                    else:
                        data = torch.cat((data, tmp), dim=1)
                        
#                 for j, filter2_attr in enumerate(filter2_attrs): # filter
#                     tmp_f2 = batch_samples[filter2_attr]
#                     if j  == 0:
#                         filter2 = tmp_f2
#                     else:
#                         filter2 = torch.cat((filter2, tmp_f2), dim=1)

                for j, filter4_attr in enumerate(filter4_attrs): # filter
                    tmp_f4 = batch_samples[filter4_attr]
                    if j  == 0:
                        filter4 = tmp_f4
                    else:
                        filter4 = torch.cat((filter4, tmp_f4), dim=1)

                for j, filter8_attr in enumerate(filter8_attrs): # filter
                    tmp_f8 = batch_samples[filter8_attr]
                    if j  == 0:
                        filter8 = tmp_f8
                    else:
                        filter8 = torch.cat((filter8, tmp_f8), dim=1)
                        
                data=data.unsqueeze(1)
                target = batch_samples[output_attrs[0]].long().squeeze(1)
#                 filter2 = filter2.unsqueeze(1)
                filter4 = filter4.unsqueeze(1)
                filter8 = filter8.unsqueeze(1)
    
                data, target = data.to(device), target.to(device)
                filter4, filter8 = filter4.to(device), filter8.to(device)
                data, target = Variable(data), Variable(target)
                filter4, filter8 = Variable(filter4), Variable(filter8)

#                 target_i = model(data)
#                 target_i = model(data, filter2, filter4, filter8) # 1. add smoothing(gh)
                target_i = model(data, filter4, filter8)
                
                loss = criterion(target_i, target)

                loss_valid_per_epoch += loss.item()

        loss_train_per_epoch = loss_train_per_epoch / len(train_loader)
        loss_valid_per_epoch = loss_valid_per_epoch / len(valid_loader)

        epoch_loss_train.append(loss_train_per_epoch)
        epoch_loss_valid.append(loss_valid_per_epoch)
        
        epoch_lr.append(optimizer.param_groups[0]['lr'])

        # 保存模型
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)

        # 保存最优模型
        if loss_valid_per_epoch < best_mse:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_mse = loss_valid_per_epoch

        scheduler.step(loss_train_per_epoch)

        # 显示loss
        if epoch % disp_inter == 0: 
            print('Epoch:{}, Training Loss:{:.8f}  Validation Loss:{:.8f}  Learning rate: {:.8f}'.format(epoch, loss_train_per_epoch, loss_valid_per_epoch, epoch_lr[epoch]))
            
    # 训练loss曲线
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(epoch_loss_train, 0.6), label='Training loss')
        ax.plot(x, smooth(epoch_loss_valid, 0.6), label='Validation loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)
        ax.set_title(f'Training curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title(f'Learning rate curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()
            
    logs = {"epoch_loss_train":epoch_loss_train,
            "epoch_loss_valid":epoch_loss_valid,
            "epoch_lr":epoch_lr}
    np.save(os.path.join(checkpoint_path, 'logs.npy'), logs)
    return model

# 训练和验证
def train_valid_net(param, model, train_data, valid_data, input_attrs=["data"], 
                     output_attrs=["label"], filter_attrs=["filter"], plot=True):

    #初始化参数
    epochs = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    lr_patience = param['lr_patience']
    lr_factor = param['lr_factor']
    optimizer_type = param['optimizer_type']
    gamma = param['gamma']
    step_size = param['step_size']
    momentum = param['momentum']
    weight_decay = param['weight_decay']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_path = param['checkpoint_path']
    rgt_fault_epoch = param['rgt_fault_epoch']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False)

    if optimizer_type == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=lr_factor)

#     num_count = np.array([858883., 626672., 272627., 161818.]).astype(np.single)
#     num_count = 1. / (num_count / num_count.sum())
#     num_count = num_count / num_count.sum()
#     criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(num_count)).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=2).to(device)
#     criterion = nn.BCEWithLogitsLoss().to(device)
#     criterion = nn.MSELoss().to(device)

    # 主循环
    epoch_loss_train, epoch_loss_valid, epoch_lr = [], [], []
    
    best_mse = 1e50

    for epoch in range(epochs):
        # 训练阶段
#         model.train()
        
        model.train()
        loss_train_per_epoch = 0
        for batch_idx, batch_samples in enumerate(train_loader):
            
            for i, input_attr in enumerate(input_attrs): # spns
                tmp = batch_samples[input_attr]
                if i  == 0:
                    data = tmp
                else:
                    data = torch.cat((data, tmp), dim=1)
                    
            for j, filter_attr in enumerate(filter_attrs): # filter
                tmp_f = batch_samples[filter_attr]
                if j  == 0:
                    smooth_filter = tmp_f
                else:
                    smooth_filter = torch.cat((smooth_filter, tmp_f), dim=1)
        
            data = data.unsqueeze(1)
            target = batch_samples[output_attrs[0]].long().squeeze(1)
            smooth_filter = smooth_filter.unsqueeze(1)
            
            data, target, smooth_filter = data.to(device), target.to(device), smooth_filter.to(device)
            data, target, smooth_filter = Variable(data), Variable(target), Variable(smooth_filter)

            optimizer.zero_grad()
            
            target_i = model(data, smooth_filter) # 1. add smoothing(gh)
            loss = criterion(target_i, target)

            loss.backward()
            optimizer.step()

            loss_train_per_epoch += loss.item()

        # 验证阶段
        model.eval()
        loss_valid_per_epoch = 0
        
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                
                for i, input_attr in enumerate(input_attrs):
                    tmp = batch_samples[input_attr]
                    if i  == 0:
                        data = tmp
                    else:
                        data = torch.cat((data, tmp), dim=1)
                        
                for j, filter_attr in enumerate(filter_attrs): # filter
                    tmp_f = batch_samples[filter_attr]
                    if j  == 0:
                        smooth_filter = tmp_f
                    else:
                        smooth_filter = torch.cat((smooth_filter, tmp_f), dim=1)
                        
                data=data.unsqueeze(1)
                target = batch_samples[output_attrs[0]].long().squeeze(1)
                smooth_filter = smooth_filter.unsqueeze(1)
    
                data, target,smooth_filter = data.to(device), target.to(device), smooth_filter.to(device)
                data, target,smooth_filter = Variable(data), Variable(target),smooth_filter.to(device)

#                 target_i = model(data)
                target_i = model(data, smooth_filter) # 1. add smoothing(gh)
                
                loss = criterion(target_i, target)

                loss_valid_per_epoch += loss.item()

        loss_train_per_epoch = loss_train_per_epoch / len(train_loader)
        loss_valid_per_epoch = loss_valid_per_epoch / len(valid_loader)

        epoch_loss_train.append(loss_train_per_epoch)
        epoch_loss_valid.append(loss_valid_per_epoch)
        
        epoch_lr.append(optimizer.param_groups[0]['lr'])

        # 保存模型
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)

        # 保存最优模型
        if loss_valid_per_epoch < best_mse:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_mse = loss_valid_per_epoch

        scheduler.step(loss_train_per_epoch)

        # 显示loss
        if epoch % disp_inter == 0: 
            print('Epoch:{}, Training Loss:{:.8f}  Validation Loss:{:.8f}  Learning rate: {:.8f}'.format(epoch, loss_train_per_epoch, loss_valid_per_epoch, epoch_lr[epoch]))
            
    # 训练loss曲线
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(epoch_loss_train, 0.6), label='Training loss')
        ax.plot(x, smooth(epoch_loss_valid, 0.6), label='Validation loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)
        ax.set_title(f'Training curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title(f'Learning rate curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()
            
    logs = {"epoch_loss_train":epoch_loss_train,
            "epoch_loss_valid":epoch_loss_valid,
            "epoch_lr":epoch_lr}
    np.save(os.path.join(checkpoint_path, 'logs.npy'), logs)
    return model

# 训练和验证
def train_valid_regression_net(param, model, train_data, valid_data, input_attrs=["data"], output_attrs=["label"], plot=True):

    #初始化参数
    epochs = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    lr_patience = param['lr_patience']
    lr_factor = param['lr_factor']
    optimizer_type = param['optimizer_type']
    gamma = param['gamma']
    step_size = param['step_size']
    momentum = param['momentum']
    weight_decay = param['weight_decay']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_path = param['checkpoint_path']
    rgt_fault_epoch = param['rgt_fault_epoch']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False)

    if optimizer_type == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=lr_factor)

    criterion = nn.MSELoss().to(device)
    
    # 主循环
    epoch_loss_train, epoch_loss_valid, epoch_lr = [], [], []
    
    best_mse = 1e50

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        
        model.train()
        loss_train_per_epoch = 0
        for batch_idx, batch_samples in enumerate(train_loader):
            
            for i, input_attr in enumerate(input_attrs):
                tmp = batch_samples[input_attr]
                if i  == 0:
                    data = tmp
                else:
                    data = torch.cat((data, tmp), dim=1)
             
            target = batch_samples[output_attrs[0]]
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()

            target_i = model(data)
            loss = criterion(target_i, target)

            loss.backward()
            optimizer.step()

            loss_train_per_epoch += loss.item()

        # 验证阶段
        model.eval()
        loss_valid_per_epoch = 0
        
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                
                for i, input_attr in enumerate(input_attrs):
                    tmp = batch_samples[input_attr]
                    if i  == 0:
                        data = tmp
                    else:
                        data = torch.cat((data, tmp), dim=1)

                target = batch_samples[output_attrs[0]]
                data, target = data.to(device), target.to(device)
                data, target = Variable(data), Variable(target)

                target_i = model(data)
                
                loss = criterion(target_i, target)

                loss_valid_per_epoch += loss.item()

        loss_train_per_epoch = loss_train_per_epoch / len(train_loader)
        loss_valid_per_epoch = loss_valid_per_epoch / len(valid_loader)

        epoch_loss_train.append(loss_train_per_epoch)
        epoch_loss_valid.append(loss_valid_per_epoch)
        
        epoch_lr.append(optimizer.param_groups[0]['lr'])

        # 保存模型
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)

        # 保存最优模型
        if loss_valid_per_epoch < best_mse:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_mse = loss_valid_per_epoch

        scheduler.step(loss_train_per_epoch)

        # 显示loss
        if epoch % disp_inter == 0: 
            print('Epoch:{}, Training Loss:{:.8f} Validation Loss:{:.8f} Learning rate: {:.8f}'.format(epoch, loss_train_per_epoch, loss_valid_per_epoch, epoch_lr[epoch]))
            
    # 训练loss曲线
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(epoch_loss_train, 0.6), label='Training loss')
        ax.plot(x, smooth(epoch_loss_valid, 0.6), label='Validation loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)
        ax.set_title(f'Training curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title(f'Learning rate curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()
            
    logs = {"epoch_loss_train":epoch_loss_train,
            "epoch_loss_valid":epoch_loss_valid,
            "epoch_lr":epoch_lr}
    np.save(os.path.join(checkpoint_path, 'logs.npy'), logs)
    return model
                

