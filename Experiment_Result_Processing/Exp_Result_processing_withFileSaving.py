import json
import math  # 用于检测NaN
import os  # 用于文件操作
import pandas as pd  # 用于生成 Excel 表格
from datetime import datetime  # 用于时间戳


# 读取 JSON 文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


# 判断是否为NaN
def is_nan(value):
    return isinstance(value, float) and math.isnan(value)


# 比较两个实验结果中的 metric_per_case
def compare_metrics(file1, file2):
    results = []

    for i, (case1, case2) in enumerate(zip(file1["metric_per_case"], file2["metric_per_case"])):
        diff_case = {
            "prediction_file": os.path.basename(case1["prediction_file"]),
            "reference_file": os.path.basename(case1["reference_file"])
        }

        # 对比 1, 2, 3, 4 标签
        for label in ['1', '2', '3', '4']:
            metrics1 = case1["metrics"].get(label, {})
            metrics2 = case2["metrics"].get(label, {})

            if metrics1 and metrics2:
                # 检查 Dice 值是否为 NaN，如果是 NaN，将其处理为0或忽略
                dice1 = 0 if is_nan(metrics1.get("Dice")) else metrics1.get("Dice", 0)
                dice2 = 0 if is_nan(metrics2.get("Dice")) else metrics2.get("Dice", 0)
                fn1 = metrics1.get("FN", 0)
                fn2 = metrics2.get("FN", 0)
                fp1 = metrics1.get("FP", 0)
                fp2 = metrics2.get("FP", 0)

                # 计算差异，取消 abs() 使用，保留正负差异
                dice_diff = dice1 - dice2
                fn_diff = fn1 - fn2
                fp_diff = fp1 - fp2

                # 如果差异较大，记录下来
                if abs(dice_diff) > 0.05:
                    diff_case[label] = {
                        "Dice_diff": dice_diff,
                        "FN_diff": fn_diff,
                        "FP_diff": fp_diff
                    }

        if len(diff_case) > 2:  # 如果该case有差异
            results.append(diff_case)

    return results


# 显示并保存结果为文件
def save_results(differences, output_dir, file1_name, file2_name):
    txt_filename = os.path.join(output_dir, f"comparison_{file1_name}_vs_{file2_name}.txt")
    excel_filename = os.path.join(output_dir, f"comparison_{file1_name}_vs_{file2_name}.xlsx")

    # 创建文件夹（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化全局统计
    file1_advantages = {"Dice": 0, "FN": 0, "FP": 0}
    file2_advantages = {"Dice": 0, "FN": 0, "FP": 0}

    # 初始化每个标签的统计
    label_stats = {
        "1": {"File1_Dice": 0, "File2_Dice": 0, "File1_FN": 0, "File2_FN": 0, "File1_FP": 0, "File2_FP": 0},
        "2": {"File1_Dice": 0, "File2_Dice": 0, "File1_FN": 0, "File2_FN": 0, "File1_FP": 0, "File2_FP": 0},
        "3": {"File1_Dice": 0, "File2_Dice": 0, "File1_FN": 0, "File2_FN": 0, "File1_FP": 0, "File2_FP": 0},
        "4": {"File1_Dice": 0, "File2_Dice": 0, "File1_FN": 0, "File2_FN": 0, "File1_FP": 0, "File2_FP": 0}
    }

    # 保存文本输出内容
    with open(txt_filename, 'w') as txt_file:
        txt_file.write(f"对比结果：{file1_name} 和 {file2_name}\n\n")

        excel_rows = []  # 用于保存到Excel的行

        for diff in differences:
            txt_file.write(f"Prediction File: {diff['prediction_file']}\n")
            txt_file.write(f"Reference File: {diff['reference_file']}\n")
            for label, values in diff.items():
                if label in ['1', '2', '3', '4']:
                    txt_file.write(f"  Label {label}:\n")

                    # 根据 Dice_diff 的正负号加入提示
                    dice_diff = values['Dice_diff']
                    if dice_diff > 0:
                        dice_hint = "File 1 较高"
                        file1_advantages["Dice"] += 1
                        label_stats[label]["File1_Dice"] += 1
                    elif dice_diff < 0:
                        dice_hint = "File 2 较高"
                        file2_advantages["Dice"] += 1
                        label_stats[label]["File2_Dice"] += 1
                    else:
                        dice_hint = "两者相同"

                    txt_file.write(f"    Dice Difference: {dice_diff} ({dice_hint})\n")

                    # 统计 FN 差异
                    fn_diff = values['FN_diff']
                    if fn_diff > 0:
                        file2_advantages["FN"] += 1  # File 2 FN 较低
                        label_stats[label]["File2_FN"] += 1
                    elif fn_diff < 0:
                        file1_advantages["FN"] += 1  # File 1 FN 较低
                        label_stats[label]["File1_FN"] += 1

                    # 统计 FP 差异
                    fp_diff = values['FP_diff']
                    if fp_diff > 0:
                        file2_advantages["FP"] += 1  # File 2 FP 较低
                        label_stats[label]["File2_FP"] += 1
                    elif fp_diff < 0:
                        file1_advantages["FP"] += 1  # File 1 FP 较低

                    txt_file.write(f"    FN Difference: {fn_diff}\n")
                    txt_file.write(f"    FP Difference: {fp_diff}\n")

                    # 添加到Excel记录
                    excel_rows.append(
                        [diff['prediction_file'], diff['reference_file'], label, dice_diff, fn_diff, fp_diff])

            txt_file.write("\n")

        # 统计结果保存到文本文件
        txt_file.write("\n总体统计结果:\n")
        txt_file.write(f"File 1 在 Dice 上优于 File 2 的次数: {file1_advantages['Dice']}\n")
        txt_file.write(f"File 2 在 Dice 上优于 File 1 的次数: {file2_advantages['Dice']}\n")
        for label in ['1', '2', '3', '4']:
            txt_file.write(f"标签 {label} 统计结果:\n")
            txt_file.write(f"  File 1 在 Dice 上优于 File 2 的次数: {label_stats[label]['File1_Dice']}")
            txt_file.write(f"  File 2 在 Dice 上优于 File 1 的次数: {label_stats[label]['File2_Dice']}")
            txt_file.write("\n")

    # 将对比数据保存为Excel文件
    df = pd.DataFrame(excel_rows,
                      columns=["Prediction File", "Reference File", "Label", "Dice Difference", "FN Difference",
                               "FP Difference"])
    df.to_excel(excel_filename, index=False)

    print(f"结果已保存至 {txt_filename} 和 {excel_filename}")


# 主函数
def main():
    # 定义 nnUNet 的训练器名称
    nnUNet_Trainer_name_1 = "nnUNetTrainer__nnUNetPlans__2d"
    nnUNet_Trainer_name_2 = "nnUNetTrainer_multichannel3__nnUNetPlans__2d"

    # 定义输出结果的基础目录
    output_dir_base = f"/home/lyh/ExperimentsNResults/Result Comparison/"

    # 循环遍历 folder_num 从 0 到 4
    for folder_num in range(5):
        # 将 folder_num 转换为字符串
        folder_num_str = str(folder_num)

        # 根据当前 folder_num 生成文件路径
        file1_json_dir = f"/home/lyh/ExperimentsNResults/Dataset628_BraTS2024/{nnUNet_Trainer_name_1}/fold_{folder_num_str}/validation/summary.json"
        file2_json_dir = f"/home/lyh/ExperimentsNResults/Dataset628_BraTS2024/{nnUNet_Trainer_name_2}/fold_{folder_num_str}/validation/summary.json"

        # 加载两个实验结果的json文件
        file1 = load_json(file1_json_dir)
        file2 = load_json(file2_json_dir)

        # 比较两个文件中的 metric_per_case
        differences = compare_metrics(file1, file2)

        # 创建存储结果的文件夹
        output_dir = os.path.join(output_dir_base,
                                  f"{nnUNet_Trainer_name_1}_vs_{nnUNet_Trainer_name_2}_comparison_results",
                                  f"Folder_{folder_num_str}")

        # 保存比较结果到文本和Excel
        save_results(differences, output_dir, nnUNet_Trainer_name_1, nnUNet_Trainer_name_2)

        print(f"结果已保存到 Folder_{folder_num_str} 文件夹中.")


if __name__ == "__main__":
    main()
