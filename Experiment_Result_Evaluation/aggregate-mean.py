import os
import pandas as pd

def sort_key(file_name):
    # 提取文件名中的前缀和数字
    prefix = '_'.join(file_name.split('_')[:-1])  # 获取前缀部分
    fold_number = int(file_name.split('fold_')[-1].split('_results')[0])  # 获取fold数字部分
    return prefix, fold_number

def collect_data_from_excel_files(directory, output_file):
    # 存储所有数据的列表
    all_data = []

    # 遍历目录中的所有文件
    for file_name in os.listdir(directory):
        if file_name.endswith('.xlsx'):
            # 构建完整的文件路径
            file_path = os.path.join(directory, file_name)
            print(f"Reading file: {file_path}")

            # 读取Excel文件中的特定工作表
            try:
                df = pd.read_excel(file_path, sheet_name="总体均值和方差")
            except ValueError:
                print(f"Sheet '总体均值和方差' not found in {file_path}")
                continue

            # 在DataFrame中添加文件名作为新的一列
            df['File Name'] = file_name

            # 将数据添加到列表中
            all_data.append(df)

    # 如果没有找到任何数据，打印消息并返回
    if not all_data:
        print("No data to concatenate.")
        return

    # 根据文件名中的fold编号和前缀对DataFrame列表进行排序
    all_data_sorted = sorted(all_data, key=lambda x: sort_key(x['File Name'].iloc[0]))

    # 合并所有数据到一个DataFrame
    combined_df = pd.concat(all_data_sorted, ignore_index=True)

    # 保存到新的Excel文件
    combined_df.to_excel(output_file, index=False)
    print(f"数据已保存到 {output_file}")



# 设置文件夹路径和输出文件名
directory = '/home/lyh/models/GGGGeneral/Experiment_Result_Evaluation/logs' # 替换为包含xlsx文件的目录路径
output_file = '/home/lyh/models/GGGGeneral/Experiment_Result_Evaluation/merged_data.xlsx'  # 替换为输出文件的路径

# 收集并保存数据
collect_data_from_excel_files(directory, output_file)