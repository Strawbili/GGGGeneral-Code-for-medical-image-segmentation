# ny学长给的刘师兄的数据集


import math
import torch
import numpy as np
import scipy.io as scio
#import matplotlib.pyplot as plt
from torch.utils.data import dataset
from train_drsn import *
import h5py
from DC_SEnet import  config


# path = r'G:\Matlab\bin\data4.mat'
path = r"/home/ny/matlab/data/train_4000_sw{}_ni{}_c{}_chunk{}.mat".format(config.sw,config.ni,config.n,config.chunk)

# matdata = h5py.File(path)
matdata = scio.loadmat(path)

num = 4000
fn = config.fn

FFT = matdata['norm_ideal_sp_real'][:].tolist()
FFTN = matdata['norm_origin_sp_real'][:].tolist()

fidfft = list(FFT)
fidfft_noise = list(FFTN)

train_list = np.zeros((num, 2*fn))
for z in range(num):
    # print(f"fidfft_noise[{z}] shape: {np.array(fidfft_noise[z]).shape}")
    # print(f"fidfft[{z}] shape: {np.array(fidfft[z]).shape}")
    train_list[z] = fidfft_noise[z] + fidfft[z]
# print("train_list shape:", train_list.shape)

train_size = int(0.8 * num)
test_size = (num - train_size)
train_dataset, test_dataset = torch.utils.data.random_split(train_list, [train_size, test_size])


#train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(train_list, [train_size, valid_size, test_size])
# fidfft_test = matdata['FFT_t'].tolist()
# fidfft_noise_test = matdata['FFTN_t'].tolist()
# test_list = fidfft_noise_test+fidfft_test


class MyDataset(dataset.Dataset):
    def __init__(self, data=None):
        self.data = data
        self.data_lengths = len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        src_data = data[:fn]
        trg_data = data[fn:]
        # src_data = np.expand_dims(src_data, axis=0)
        # # ori_data = np.expand_dims(ori_data, axis=2)
        # trg_data = np.expand_dims(trg_data, axis=0)
        # src_data = np.expand_dims(src_data, axis=0)
        # src_data = np.expand_dims(src_data, axis=1)
        # trg_data = np.expand_dims(trg_data, axis=0)
        # trg_data= np.expand_dims(trg_data, axis=1)
        src_data = src_data.reshape(1,fn)
        trg_data = trg_data.reshape(1,fn)
        return src_data, trg_data


    def __len__(self):
        return self.data_lengths
