import os
import nibabel as nib
import numpy as np
import cv2
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def load_nifti_file(filepath):
    """加载 .nii.gz 文件并返回图像数据"""
    return nib.load(filepath).get_fdata()

def apply_color_map_to_labels(label_img):
    """
    将标签图像转换为彩色图像
    :param label_img: 标签图像 (假设值为 0, 1, 2, 3)
    :return: 彩色标签图像
    """
    # 创建自定义颜色映射（假设4个标签：0, 1, 2, 3）
    colors = {
        0: [0, 0, 0],        # 黑色 (背景)
        1: [255, 0, 0],      # 红色 (标签 1)
        2: [0, 255, 0],      # 绿色 (标签 2)
        3: [0, 0, 255],      # 蓝色 (标签 3)
        4: [255, 255, 0],
    }

    # 创建彩色图像
    color_img = np.zeros((*label_img.shape, 3), dtype=np.uint8)

    # 将标签映射到颜色
    for label_value, color in colors.items():
        color_img[label_img == label_value] = color

    return color_img


def visualize_and_save_original_slice(original_img, slice_index, dimension, save_path):
    """
    保存原始图像的指定切片。
    """
    # 根据选择的维度获取切片
    if dimension == 0:  # Sagittal
        original_slice = original_img[slice_index, :, :]
    elif dimension == 1:  # Coronal
        original_slice = original_img[:, slice_index, :]
    elif dimension == 2:  # Axial
        original_slice = original_img[:, :, slice_index]

    # 将原始图像归一化到0-255范围，并转换为uint8格式
    original_slice_norm = cv2.normalize(original_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 将原始图像转为三通道灰度图像
    original_slice_colored = cv2.cvtColor(original_slice_norm, cv2.COLOR_GRAY2BGR)

    # 保存原始图像
    cv2.imwrite(save_path, original_slice_colored)
    print(f'Original slice saved at: {save_path}')


def visualize_and_save_slices_masked(original_img, overlay_img, slice_index, dimension, save_path, overlay_type="Ground Truth"):

    if dimension == 0:  # Sagittal
        original_slice = original_img[slice_index, :, :]
        overlay_slice = overlay_img[slice_index, :, :]
    elif dimension == 1:  # Coronal
        original_slice = original_img[:, slice_index, :]
        overlay_slice = overlay_img[:, slice_index, :]
    elif dimension == 2:  # Axial
        original_slice = original_img[:, :, slice_index]
        overlay_slice = overlay_img[:, :, slice_index]

    # 将标签图像转换为彩色
    overlay_color_slice = apply_color_map_to_labels(overlay_slice)

    # 将原始图像归一化到0-255范围，并转换为uint8格式
    original_slice_norm = cv2.normalize(original_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 将原始图像转为三通道灰度图像
    original_slice_colored = cv2.cvtColor(original_slice_norm, cv2.COLOR_GRAY2BGR)

    # 创建一个掩码，只对非背景区域进行混合
    mask = overlay_slice > 0  # 只考虑非0区域
    mask = mask.astype(np.uint8)  # 转换为uint8格式

    # 扩展掩码维度以适用于彩色图像
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # 仅在有标签的区域进行混合
    combined_img = np.where(mask_3d, cv2.addWeighted(original_slice_colored, 0.3, overlay_color_slice, 0.7, 0), original_slice_colored)

    # 保存图片
    cv2.imwrite(save_path, combined_img)
    print(f'{overlay_type} overlay saved at: {save_path}')


def print_dimensions(nifti_img):
    """打印NIfTI图像的维度，并返回维度名称"""
    shape = nifti_img.shape
    print("图像的维度为: ", shape)
    dimensions = ["Sagittal (X-axis)", "Coronal (Y-axis)", "Axial (Z-axis)"]
    for i, dim in enumerate(dimensions):
        print(f"{i}: {dim} (Size: {shape[i]})")
    return shape

def check_prediction_files(prediction_folders):
    """检查所有预测文件夹中的.nii.gz文件数量是否相等，并返回文件名列表。"""
    prediction_files = []
    for folder in prediction_folders:
        files = [f for f in os.listdir(folder) if f.endswith('.nii.gz')]
        prediction_files.append(files)

    # 检查文件数量是否一致
    num_files = len(prediction_files[0])
    for files in prediction_files:
        if len(files) != num_files:
            raise ValueError("各预测文件夹中的 .nii.gz 文件数量不相等。")

    return prediction_files[0]  # 返回第一个文件夹的文件列表

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Medical Image Segmentation Visualization")
    parser.add_argument("--d", type=int, choices=[0, 1, 2], required=True,
                        help="选择切片维度: 0 - Sagittal, 1 - Coronal, 2 - Axial")
    parser.add_argument("--s", type=int, required=True, help="选择切片索引")
    parser.add_argument("--file", type=str, required=True, help="选择的文件名")
    parser.add_argument("--folder", type=int, choices=[0, 1, 2, 3, 4], required=True, help="")
    args = parser.parse_args()
    prediction_folders = [
        f'/home/lyh/ExperimentsNResults/Dataset137_BraTS2023/nnUNetTrainer__nnUNetPlans__2d/fold_{args.folder}/validation',
        f'/home/lyh/ExperimentsNResults/Dataset137_BraTS2023/nnUNetTrainer_I2UNet__nnUNetPlans__2d/fold_{args.folder}/validation',
        f'/home/lyh/ExperimentsNResults/Dataset137_BraTS2023/nnUNetTrainer_multichannel20__nnUNetPlans__2d/fold_{args.folder}/validation',
    ]
    original_image_folder = r"/home/project/datasets/nnUNet_raw/Dataset137_BraTS2023/imagesTr/"  # 替换为原始图像的路径
    ground_truth_folder = r"/home/project/datasets/nnUNet_raw/Dataset137_BraTS2023/labelsTr/"  # 替换为地面真值文件夹的实际路径
    output_folder = r"/home/lyh/ExperimentsNResults/Dataset137_BraTS2023/visualize_output/"  # 输出文件夹的名称

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 检查预测文件夹中的文件数量
    try:
        file_list = check_prediction_files(prediction_folders)
    except ValueError as e:
        print(e)
        return

    # 列出可用文件并选择文件和切片索引
    print("可用文件列表:")
    for i, file_name in enumerate(file_list):
        print(f"{i}: {file_name}")

    # selected_file = file_list[selected_index]
    # selected_file = "BraTS-GLI-00005-100.nii.gz"
    # 加载原始数据和 ground truth
    selected_file = args.file

    ground_truth_img = load_nifti_file(os.path.join(ground_truth_folder, selected_file))
    # shape = print_dimensions(original_img)
    # dimension = int(input("选择切片维度 (0: Sagittal, 1: Coronal, 2: Axial): "))
    # slice_index = int(input(f"输入切片索引 (0 到 {shape[dimension] - 1}): "))
    dimension = args.d
    slice_index = args.s
    # 可视化并保存ground truth叠加图
    save_folder_name = f"Name_{selected_file}_Dim{dimension}_SliceIndex{slice_index}"
    os.makedirs(os.path.join(output_folder, save_folder_name), exist_ok=True)

    modality_num = {0, 1, 2, 3}
    for modality in modality_num:
        original_img = load_nifti_file(os.path.join(original_image_folder, selected_file).replace(".", f"_000{modality}.", 1))
        visualize_and_save_original_slice(
            original_img,
            slice_index,
            dimension,
            save_path=os.path.join(os.path.join(output_folder, save_folder_name), f'original_slice_{modality}.png')
        )

    visualize_and_save_slices_masked(
        original_img,
        ground_truth_img,
        slice_index,
        dimension,
        save_path=os.path.join(output_folder, save_folder_name, 'ground_truth_overlay.png'),
        overlay_type="Ground Truth"
    )

    # 加载和可视化每个预测结果
    for i, prediction_folder in enumerate(prediction_folders):
        prediction_img = load_nifti_file(os.path.join(prediction_folder, selected_file))
        visualize_and_save_slices_masked(
            original_img,
            prediction_img,
            slice_index,
            dimension,
            save_path=os.path.join(output_folder, save_folder_name, f'prediction_{i+1}_overlay.png'),
            overlay_type=f"Prediction {i+1}"
        )

if __name__ == "__main__":
    main()
