import numpy as np

# 指定.npy文件的路径
file_path = '/home/project/datasets/nnUNet_raw/nnUNet_preprocessed/Dataset628_BraTS2024/nnUNetPlans_2d/BraTS-GLI-03064-100.npy'

# 读取.npy文件
data = np.load(file_path)

# 打印读取的数据
print(data)
