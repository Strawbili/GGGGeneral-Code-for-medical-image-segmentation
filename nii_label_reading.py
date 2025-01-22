import nibabel as nib
import numpy as np

# 读取 NIfTI 标签文件
nii_file = '/home/project/datasets/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-01665-000/BraTS-GLI-01665-000-seg.nii.gz'
img = nib.load(nii_file)

# 获取图像数据（标签值矩阵）
label_data = img.get_fdata()

# 找到所有独特的标签值
unique_labels = np.unique(label_data)

# 输出所有的标签值
print(f"标签值: {unique_labels}")

