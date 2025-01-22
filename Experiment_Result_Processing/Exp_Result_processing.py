import json
import math  # 用于检测NaN
import os  # 用于提取文件名
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

                # 计算差异
                dice_diff = dice1 - dice2
                fn_diff = fn1 - fn2
                fp_diff = fp1 - fp2

                # 如果差异较大，记录下来
                if abs(dice_diff) > 0.05 :# or fn_diff > 50 or fp_diff > 50:
                    diff_case[label] = {
                        "Dice_diff": dice_diff,
                        "FN_diff": fn_diff,
                        "FP_diff": fp_diff
                    }

        if len(diff_case) > 2:  # 如果该case有差异
            results.append(diff_case)

    return results


# 显示结果
def display_differences(differences):
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

    for diff in differences:
        print(f"Prediction File: {diff['prediction_file']}")
        print(f"Reference File: {diff['reference_file']}")
        for label, values in diff.items():
            if label in ['1', '2', '3', '4']:
                print(f"  Label {label}:")

                # 根据 Dice_diff 的正负号加入提示
                dice_diff = values['Dice_diff']
                if dice_diff > 0:
                    dice_hint = "File 1 较高"
                    file1_advantages["Dice"] += 1
                    label_stats[label]["File1_Dice"] += 1  # 记录每个标签下的File 1优越次数
                elif dice_diff < 0:
                    dice_hint = "File 2 较高"
                    file2_advantages["Dice"] += 1
                    label_stats[label]["File2_Dice"] += 1  # 记录每个标签下的File 2优越次数
                else:
                    dice_hint = "两者相同"

                print(f"    Dice Difference: {dice_diff} ({dice_hint})")

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
                    label_stats[label]["File1_FP"] += 1

                print(f"    FN Difference: {fn_diff}")
                print(f"    FP Difference: {fp_diff}")
        print("\n")

    # 输出总体统计结果
    print("总体统计结果:")
    print(f"File 1 在 Dice 上优于 File 2 的次数: {file1_advantages['Dice']}")
    print(f"File 2 在 Dice 上优于 File 1 的次数: {file2_advantages['Dice']}")
    print(f"File 1 在 FN 上优于 File 2 的次数: {file1_advantages['FN']}")
    print(f"File 2 在 FN 上优于 File 1 的次数: {file2_advantages['FN']}")
    print(f"File 1 在 FP 上优于 File 2 的次数: {file1_advantages['FP']}")
    print(f"File 2 在 FP 上优于 File 1 的次数: {file2_advantages['FP']}")

    # 输出每个标签的统计结果
    print("\n根据标签的统计结果:")
    for label in ['1', '2', '3', '4']:
        print(f"标签 {label} 统计结果:")
        print(f"  File 1 在 Dice 上优于 File 2 的次数: {label_stats[label]['File1_Dice']}")
        print(f"  File 2 在 Dice 上优于 File 1 的次数: {label_stats[label]['File2_Dice']}")
        print(f"  File 1 在 FN 上优于 File 2 的次数: {label_stats[label]['File1_FN']}")
        print(f"  File 2 在 FN 上优于 File 1 的次数: {label_stats[label]['File2_FN']}")
        print(f"  File 1 在 FP 上优于 File 2 的次数: {label_stats[label]['File1_FP']}")
        print(f"  File 2 在 FP 上优于 File 1 的次数: {label_stats[label]['File2_FP']}")
        print("\n")


# 主函数
def main():
    # 加载两个实验结果的json文件
    file1 = load_json("/home/lyh/ExperimentsNResults/Dataset628_BraTS2024/nnUNetTrainer_multichannel2__nnUNetPlans__2d/fold_1/validation/summary.json")
    file2 = load_json("/home/lyh/ExperimentsNResults/Dataset628_BraTS2024/nnUNetTrainer_multichannel3__nnUNetPlans__2d/fold_1/validation/summary.json")  # 替换为第二个.json文件的路径

    # 比较两个文件中的metric_per_case
    differences = compare_metrics(file1, file2)

    # 显示结果
    display_differences(differences)


if __name__ == "__main__":
    main()
