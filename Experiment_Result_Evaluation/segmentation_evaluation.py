import os
import nibabel as nib
import numpy as np
from medpy.metric.binary import dc
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_nii_file(filepath):
    """加载 .nii.gz 文件并返回 numpy 数组"""
    nii_img = nib.load(filepath, mmap=True)
    return nii_img.get_fdata()


def calculate_dice_per_label(gt, pred, label):
    """计算单个标签的 Dice，考虑标签不全的情况"""
    gt_binary = (gt == label).astype(np.uint8)
    pred_binary = (pred == label).astype(np.uint8)

    # 情况1：GT 和预测中都没有这个标签
    if np.sum(gt_binary) == 0 and np.sum(pred_binary) == 0:
        return 1.0  # 完美分割，无误差

    # 情况2：GT 中有这个标签但预测中没有
    if np.sum(gt_binary) != 0 and np.sum(pred_binary) == 0:
        return 0.0  # Dice 为 0

    # 情况3：计算正常的 Dice
    dice_score = dc(pred_binary, gt_binary)

    return dice_score


def process_file(gt_file, pred_file, gt_dir, pred_dir, labels):
    """处理单个文件，计算多个标签的 Dice"""
    gt_path = os.path.join(gt_dir, gt_file)
    pred_path = os.path.join(pred_dir, pred_file)

    gt_data = load_nii_file(gt_path)
    pred_data = load_nii_file(pred_path)

    file_results = {'file': gt_file}

    for label in labels:
        dice = calculate_dice_per_label(gt_data, pred_data, label)
        file_results[f'dice_label_{label}'] = dice

    # 在每个文件处理完成后打印结果
    print(f"处理完成文件: {gt_file}")
    for label in labels:
        print(f"标签: {label}, Dice: {file_results[f'dice_label_{label}']}")

    return file_results


def evaluate_segmentation(gt_dir, pred_dir, labels=[1, 2, 3, 4], num_workers=4):
    """评估分割结果，使用多线程加速，计算每个标签的 Dice"""
    # 获取 .nii.gz 文件
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.nii.gz')])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')])

    # 提取文件名，去除 .nii.gz 扩展名
    gt_filenames = set(os.path.splitext(os.path.splitext(f)[0])[0] for f in gt_files)
    pred_filenames = set(os.path.splitext(os.path.splitext(f)[0])[0] for f in pred_files)

    # 找到同时存在于两个文件夹中的文件名
    common_filenames = gt_filenames.intersection(pred_filenames)

    if not common_filenames:
        raise ValueError("未找到 GT 文件夹和 Prediction 文件夹中同时存在的文件。")

    results = []

    # 使用多线程并行处理文件
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, filename + '.nii.gz', filename + '.nii.gz', gt_dir, pred_dir, labels): filename for filename in common_filenames}

        # 按照任务完成的顺序打印
        for future in as_completed(futures):
            results.append(future.result())

    df_results = pd.DataFrame(results)

    # 计算每个标签的均值和方差
    summary = {}
    for label in labels:
        if f'dice_label_{label}' in df_results.columns:
            summary[f'dice_label_{label}_mean'] = df_results[f'dice_label_{label}'].mean()
            summary[f'dice_label_{label}_std'] = df_results[f'dice_label_{label}'].std()
        else:
            print(f"警告: 未找到 dice_label_{label} 列。")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # 你可以根据需要调节显示宽度和其他参数
    pd.set_option('display.width', 1000)  # 设置显示宽度
    df_summary = pd.DataFrame([summary])

    return df_results, df_summary


# 使用示例
gt_dir = '/home/project/datasets/nnUNet_raw/nnUNet_preprocessed/Dataset629_BraTS2024_region_based/gt_segmentations/'  # 替换为实际路径
pred_dir = '/home/lyh/ExperimentsNResults/Dataset628_BraTS2024/nnUNetTrainer_multichannel3__nnUNetPlans__2d/fold_4/validation/'  # 替换为实际路径
num_workers = 16  # 设定并行的线程数量

df_results, df_summary = evaluate_segmentation(gt_dir, pred_dir, num_workers=num_workers)

# 输出结果
print("每个文件的评估结果：")
print(df_results)

print("\n总体评估均值和方差：")
print(df_summary)
