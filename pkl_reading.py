import pickle

# 指定.pkl文件的路径
file_path = '/home/project/datasets/nnUNet_raw/nnUNet_preprocessed/Dataset628_BraTS2024/nnUNetPlans_2d/BraTS-GLI-03064-100.pkl'

# 读取.pkl文件
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 打印读取的数据
print(data)
