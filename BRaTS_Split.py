import os
import shutil

# 原始目录路径
source_dir = "/home/liuyuhan/datasets/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"
# 目标目录路径
target_dir = "/home/liuyuhan/datasets/nnUnet_raw/"

# 创建目标目录结构
os.makedirs(os.path.join(target_dir, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "imagesTs"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "labelsTr"), exist_ok=True)

# 遍历原始目录中的每个子文件夹
for folder in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder)

    if os.path.isdir(folder_path):
        # 找到训练数据和测试数据，训练数据为不包含 'seg' 的文件，测试数据为包含 'seg' 的文件
        train_files = [f for f in os.listdir(folder_path) if 'seg' not in f]
        test_files = [f for f in os.listdir(folder_path) if 'seg' in f]

        # 将训练数据移动到 imagesTr 和 labelsTr
        for i, train_file in enumerate(train_files):
            train_file_path = os.path.join(folder_path, train_file)
            train_id = f"{folder}_{i:03d}"

            if train_file.endswith('.nii.gz'):
                # 将训练数据的不同通道文件（0000, 0001, 0002, 0003）移动到 imagesTr
                shutil.copy(train_file_path, os.path.join(target_dir, "imagesTr", f"{train_id}_{i:04d}.nii.gz"))

            # 假设标签文件格式为 BRATS_xxx.nii.gz
            label_file_name = f"{folder}_{i:03d}.nii.gz"
            shutil.copy(train_file_path, os.path.join(target_dir, "labelsTr", label_file_name))

        # 将测试数据移动到 imagesTs
        for test_file in test_files:
            test_file_path = os.path.join(folder_path, test_file)
            test_id = folder
            shutil.copy(test_file_path, os.path.join(target_dir, "imagesTs", f"{test_id}.nii.gz"))

print("数据重新组织完成。")
