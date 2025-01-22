import os
import nibabel as nib
import numpy as np
from medpy.metric.binary import dc, hd95
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

def load_nii_file(filepath):
    """加载 .nii.gz 文件并返回 numpy 数组"""
    nii_img = nib.load(filepath, mmap=True)
    return nii_img.get_fdata()


def calculate_metrics_per_label(gt, pred, label):
    """计算单个标签的 Dice 和 HD95，考虑标签不全的情况"""
    gt_binary = (gt == label).astype(np.uint8)
    pred_binary = (pred == label).astype(np.uint8)

    # 情况1：GT 和预测中都没有这个标签
    if np.sum(gt_binary) == 0 and np.sum(pred_binary) == 0:
        return 1.0, 0.0  # 完美分割，无误差

    # 情况2：GT 中有这个标签但预测中没有
    if np.sum(gt_binary) != 0 and np.sum(pred_binary) == 0:
        return 0.0, np.inf  # Dice 为 0，HD95 为无穷大

    # 情况3：计算正常的 Dice 和 HD95
    dice_score = dc(pred_binary, gt_binary)
    try:
        hd95_distance = hd95(pred_binary, gt_binary)
    except RuntimeError:
        hd95_distance = np.inf  # 如果 HD95 计算失败，设为无穷大

    return dice_score, hd95_distance

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

    return dice_score[[]]


def process_file(gt_file, pred_file, gt_dir, pred_dir, labels):
    """处理单个文件，计算多个标签的 Dice"""
    gt_path = os.path.join(gt_dir, gt_file)
    pred_path = os.path.join(pred_dir, pred_file)

    gt_data = load_nii_file(gt_path)
    pred_data = load_nii_file(pred_path)

    file_results = {'file': gt_file}

    for label in labels:
        dice, hd95_value = calculate_metrics_per_label(gt_data, pred_data, label)
        file_results[f'dice_label_{label}'] = dice
        file_results[f'hd95_label_{label}'] = hd95_value
    # 在每个文件处理完成后打印结果
    print(f"处理完成文件: {gt_file}")
    for label in labels:
        print(f"标签: {label}, Dice: {file_results[f'dice_label_{label}']}  HD95: {file_results[f'hd95_label_{label}']}")

    return file_results


def evaluate_segmentation(gt_dir, pred_dir, labels=[1, 2, 3, 4], num_workers=4, output_file_txt="results.txt", output_file_xlsx="results.xlsx", penalty_value=1000):
    """评估分割结果，使用多线程加速，计算每个标签的 Dice，并保存结果到 .txt 文件"""
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

        # 按照任务完成的顺序收集结果
        for future in as_completed(futures):
            results.append(future.result())

    df_results = pd.DataFrame(results)

    for label in labels:
        hd95_column = f'hd95_label_{label}'
        if hd95_column in df_results.columns:
            df_results[hd95_column] = df_results[hd95_column].replace(np.inf, penalty_value)
    # 计算每个标签的均值和方差
    summary = {}
    for label in labels:
        if f'dice_label_{label}' in df_results.columns:
            summary[f'dice_label_{label}_mean'] = df_results[f'dice_label_{label}'].mean()
            summary[f'dice_label_{label}_std'] = df_results[f'dice_label_{label}'].std()
        else:
            print(f"警告: 未找到 dice_label_{label} 列。")
        if f'hd95_label_{label}' in df_results.columns:
            summary[f'hd95_label_{label}_mean'] = df_results[f'hd95_label_{label}'].mean()
            summary[f'hd95_label_{label}_std'] = df_results[f'hd95_label_{label}'].std()
        else:
            print(f"警告: 未找到 hd95_label_{label} 列。")
    df_summary = pd.DataFrame([summary])

    # 保存结果到 .txt 文件
    with open(output_file_txt, "w") as f:
        f.write("每个文件的评估结果：\n")
        f.write(df_results.to_string())
        f.write("\n\n总体评估均值和方差：\n")
        f.write(df_summary.to_string())

    with pd.ExcelWriter(output_file_xlsx, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name="每个文件的评估结果", index=False)
        df_summary.to_excel(writer, sheet_name="总体评估均值和方差", index=False)
    return df_results, df_summary


def evaluate_five_fold(gt_dir, base_pred_dir, plans_name, penalty_value, Dataset_ID, output_base_dir,
                       num_workers=16):
    """五折交叉验证的循环，并将每次的结果保存到指定的 .txt 和 .xlsx 文件中"""

    # 创建保存日志的目录
    os.makedirs(output_base_dir, exist_ok=True)

    # 进行五折交叉验证
    for fold in range(0, 5):  # fold 从 0 到 4
        print(f"正在处理第 {fold} 折...")

        # 设置路径
        pred_dir = f'{base_pred_dir}/{plans_name}/fold_{fold}/validation/'
        output_dir = os.path.join(output_base_dir, Dataset_ID)

        print("Validation Data Comes From:", pred_dir)
        if not os.path.exists(output_dir):
            # 如果路径不存在，创建路径
            os.makedirs(output_dir)
            print(f"路径 {output_dir} 已创建。")
        else:
            print(f"路径 {output_dir} 已存在。")

        output_file_txt = os.path.join(output_base_dir, Dataset_ID, f'{plans_name}_fold_{fold}_results.txt')
        output_file_xlsx = os.path.join(output_base_dir, Dataset_ID, f'{plans_name}_fold_{fold}_results.xlsx')

        # 执行评估
        df_results, df_summary = evaluate_segmentation(
            gt_dir, pred_dir, num_workers=num_workers,
            output_file_txt=output_file_txt, output_file_xlsx=output_file_xlsx, penalty_value=penalty_value
        )

        # 保存结果到文本文件
        with open(output_file_txt, "w") as f:
            f.write(f"第 {fold} 折的评估结果：\n")
            f.write("每个文件的评估结果：\n")
            f.write(df_results.to_string())
            f.write("\n\n总体评估均值和方差：\n")
            f.write(df_summary.to_string())

        # 将评估结果写入 Excel 文件
        with pd.ExcelWriter(output_file_xlsx, engine='openpyxl', mode='w') as writer:
            df_results.to_excel(writer, sheet_name="每文件评估结果", index=False)
            df_summary.to_excel(writer, sheet_name="总体均值和方差", index=False)

        # 打印折的结果到控制台
        print(f"第 {fold} 折的评估结果：")
        print(df_results)
        print("\n总体评估均值和方差：")
        print(df_summary)


    print("Evaluation Finished!")
    print("It has been output to the folder", output_dir)

parser = argparse.ArgumentParser(description="五折交叉验证的分割评估脚本")
parser.add_argument('--plans_name', type=str, required=True, help='训练网络的名称')
parser.add_argument('--id', type=str, required=True, help='nnUNet数据集的ID')
parser.add_argument('--dataset_name', type=str, help='')
parser.add_argument('--gt_dir', type=str, help='GT 目录路径')
parser.add_argument('--base_pred_dir', type=str, default="/home/lyh/ExperimentsNResults/", help='预测结果的基础目录路径')
parser.add_argument('--num_workers', type=int, default=16, help='并行的线程数量')
parser.add_argument('--output_base_dir', type=str, default='/home/lyh/ExperimentsNResults/Evaluation_logs/', help='日志输出目录')
parser.add_argument('--penalty_value', type=int, default=200, help='HD95的惩罚性分数')

# 解析参数
args = parser.parse_args()

args.gt_dir = os.path.join("/home/project/datasets/nnUNet_raw/nnUNet_preprocessed/", args.dataset_name, "gt_segmentations")
args.base_pred_dir = os.path.join("/home/lyh/ExperimentsNResults/", args.dataset_name)
evaluate_five_fold(
        gt_dir=args.gt_dir,
        base_pred_dir=args.base_pred_dir,
        plans_name=args.plans_name,
        penalty_value=args.penalty_value,
        num_workers=args.num_workers,
        output_base_dir=args.output_base_dir,
        Dataset_ID=args.id
    )