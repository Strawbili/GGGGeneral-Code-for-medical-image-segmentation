import os
import numpy as np
import pandas as pd
from medpy.metric.binary import dc, hd95
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from PIL import Image

def load_png_file(filepath):
    """加载 .png 文件并返回 numpy 数组"""
    img = Image.open(filepath)
    return np.array(img)

def calculate_metrics(gt, pred, label):
    """计算单个标签的 Dice, HD95 和 IoU，保留情况3的逻辑"""
    gt_binary = (gt == label).astype(np.uint8)
    pred_binary = (pred == label).astype(np.uint8)

    # 计算 Dice
    dice_score = dc(pred_binary, gt_binary)

    # 计算 HD95
    try:
        hd95_distance = hd95(pred_binary, gt_binary)
    except RuntimeError:
        hd95_distance = np.inf  # 如果 HD95 计算失败，设为无穷大

    # 计算 IoU
    intersection = np.sum(gt_binary * pred_binary)  # 交集
    union = np.sum(gt_binary) + np.sum(pred_binary) - intersection  # 并集
    iou_score = intersection / union if union != 0 else 0  # 避免除以 0

    return dice_score, hd95_distance, iou_score

def process_file(gt_file, pred_file, gt_dir, pred_dir, label):
    """处理单个文件，计算单个标签的 Dice、HD95 和 IoU"""
    gt_path = os.path.join(gt_dir, gt_file)
    pred_path = os.path.join(pred_dir, pred_file)

    gt_data = load_png_file(gt_path)
    pred_data = load_png_file(pred_path)

    file_results = {'file': gt_file}

    dice, hd95_value, iou_value = calculate_metrics(gt_data, pred_data, label)
    file_results[f'dice_label_{label}'] = dice
    file_results[f'hd95_label_{label}'] = hd95_value
    file_results[f'iou_label_{label}'] = iou_value  # 添加 IoU 结果

    # 打印文件处理结果
    print(f"处理完成文件: {gt_file}")
    print(f"标签: {label}, Dice: {dice}, HD95: {hd95_value}, IoU: {iou_value}")

    return file_results

def evaluate_segmentation(gt_dir, pred_dir, label=1, num_workers=4, output_file_txt="results.txt", output_file_xlsx="results.xlsx"):
    """评估分割结果，计算单个标签的 Dice, HD95 和 IoU，并保存结果到 .txt 和 .xlsx 文件"""
    # 获取 .png 文件
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])

    # 提取文件名，去除 .png 扩展名
    gt_filenames = set(os.path.splitext(f)[0] for f in gt_files)
    pred_filenames = set(os.path.splitext(f)[0] for f in pred_files)

    # 找到同时存在于两个文件夹中的文件名
    common_filenames = gt_filenames.intersection(pred_filenames)

    if not common_filenames:
        raise ValueError("未找到 GT 文件夹和 Prediction 文件夹中同时存在的文件。")

    results = []

    # 使用多线程并行处理文件
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, filename + '.png', filename + '.png', gt_dir, pred_dir, label): filename for filename in common_filenames}

        # 按照任务完成的顺序收集结果
        for future in as_completed(futures):
            results.append(future.result())

    df_results = pd.DataFrame(results)

    # 计算标签的均值和方差
    summary = {}
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

    if f'iou_label_{label}' in df_results.columns:
        summary[f'iou_label_{label}_mean'] = df_results[f'iou_label_{label}'].mean()
        summary[f'iou_label_{label}_std'] = df_results[f'iou_label_{label}'].std()
    else:
        print(f"警告: 未找到 iou_label_{label} 列。")

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

def run_cross_validation(base_gt_dir, base_pred_dir, label=1, num_workers=4, output_dir="./", output_file_name="results", dataset_name=""):
    """5折交叉验证"""
    all_fold_results = []
    all_fold_summaries = []

    # 遍历 5 个折
    for fold in range(5):
        print(f"开始评估 fold {fold}...")

        gt_dir = base_gt_dir
        pred_dir = os.path.join(base_pred_dir, f"fold_{fold}", "validation")
        output_file_txt = os.path.join(output_dir, f"{dataset_name}_fold{fold}_{output_file_name}.txt")
        output_file_xlsx = os.path.join(output_dir, f"{dataset_name}_fold{fold}_{output_file_name}.xlsx")

        fold_results, fold_summary = evaluate_segmentation(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            label=label,
            num_workers=num_workers,
            output_file_txt=output_file_txt,
            output_file_xlsx=output_file_xlsx
        )

        # 汇总每一折的结果
        all_fold_results.append(fold_results)
        all_fold_summaries.append(fold_summary)

    # 合并所有折的结果
    df_all_results = pd.concat(all_fold_results, ignore_index=True)
    df_all_summaries = pd.concat(all_fold_summaries, ignore_index=True)

    # 保存所有折的汇总结果
    df_all_results.to_excel(os.path.join(output_dir, f"{dataset_name}_all_folds_{output_file_name}.xlsx"), sheet_name="所有折的评估结果", index=False)
    df_all_summaries.to_excel(os.path.join(output_dir, f"{dataset_name}_all_folds_{output_file_name}_summary.xlsx"), sheet_name="所有折的总体评估均值和方差", index=False)

    return df_all_results, df_all_summaries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5折交叉验证分割评估脚本")
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称')
    parser.add_argument('--trainer_name', type=str, required=True, help='训练网络的名字')
    parser.add_argument('--label', type=int, default=1, help='评估的标签')
    parser.add_argument('--num_workers', type=int, default=24, help='并行的线程数量')
    parser.add_argument('--output_file_name', type=str, default='Evaluation', help='输出结果的文本文件')
    parser.add_argument('--output_dir', type=str, default="/home/lyh/ExperimentsNResults/Evaluation_logs/", help='输出结果的目录路径')

    # 解析参数
    args = parser.parse_args()

    base_gt_dir = os.path.join("/home/SSD1_4T/datasets/nnUNet_raw/nnUNet_preprocessed", args.dataset_name, "gt_segmentations")
    base_pred_dir = os.path.join("/home/lyh/ExperimentsNResults", args.dataset_name, args.trainer_name)
    output_dir = os.path.join(args.output_dir, args.dataset_name, args.trainer_name)
    output_file_name = args.output_file_name+args.trainer_name

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"路径 {output_dir} 已创建。")
    else:
        print(f"路径 {output_dir} 已存在。")

    run_cross_validation(
        base_gt_dir=base_gt_dir,
        base_pred_dir=base_pred_dir,
        label=args.label,
        num_workers=args.num_workers,
        output_dir=output_dir,
        output_file_name=output_file_name,
        dataset_name=args.dataset_name
    )
