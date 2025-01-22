import os

import cv2
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def load_nifti(file_path):
    """加载NIfTI文件并返回图像数据。"""
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata()


def overlay_with_colormap(original_slice, overlay_slice, color_map, alpha=0.5):
    """将叠加图像与原始图像结合，并应用颜色映射。"""
    # 创建一个全黑的图像
    overlay_image = np.zeros((*overlay_slice.shape, 3))

    # 为每个标签分配颜色
    for label, color in color_map.items():
        mask = overlay_slice == label
        overlay_image[mask] = color  # 将对应标签位置填充颜色

    # 将原始图像与颜色叠加
    combined = np.clip(original_slice[..., np.newaxis] + overlay_image * alpha, 0, 1)
    return combined


def visualize_multiple_models(original_slice, ground_truth_slice, prediction_slices, slice_index, output_folder):
    """可视化多个模型的切片并保存图像。"""
    # if original_slice.dtype != np.uint8:
    #     original_slice = (original_slice * 255).astype(np.uint8)
    #
    # if ground_truth_slice.dtype != np.uint8:
    #     ground_truth_slice = (ground_truth_slice * 255).astype(np.uint8)
    #
    # original_slice = cv2.cvtColor(original_slice, cv2.COLOR_GRAY2RGB)
    # ground_truth_slice = cv2.cvtColor(ground_truth_slice, cv2.COLOR_GRAY2RGB)
    # print(original_slice.shape)
    # print(ground_truth_slice.shape)

    color_map = {
        0: [0, 0, 0],  # 背景（黑色）
        1: [255, 0, 0],  # 标签1（红色）
        2: [0, 255, 0],  # 标签2（绿色）
        3: [0, 0, 255],  # 标签3（蓝色）
        4: [255, 255, 0],  # 标签4（黄色）
        # 添加更多标签颜色
    }
    plt.figure(figsize=(15, 5))

    # 原始图像
    plt.subplot(1, len(prediction_slices) + 2, 1)
    plt.title("Original Image")
    plt.imshow(original_slice, cmap='gray')
    plt.axis('off')

    # 地面真值叠加
    gt_overlay = overlay_with_colormap(original_slice, ground_truth_slice, color_map)
    plt.subplot(1, len(prediction_slices) + 2, 2)
    plt.title("Ground Truth Overlay")
    plt.imshow(gt_overlay)
    plt.axis('off')

    # 遍历每个模型的预测结果，叠加
    for i, prediction_slice in enumerate(prediction_slices):
        pred_overlay = overlay_with_colormap(original_slice, prediction_slice, color_map)
        plt.subplot(1, len(prediction_slices) + 2, i + 3)
        plt.title(f"Model {i + 1} Prediction Overlay")
        plt.imshow(pred_overlay)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'slice_{slice_index}.png'))
    plt.close()


def check_prediction_files(prediction_folders):
    """检查所有预测文件夹中的.nii.gz文件数量是否相等，并返回文件名列表。"""
    prediction_files = []
    for folder in prediction_folders:
        files = [f for f in os.listdir(folder) if f.endswith('.nii.gz')]
        prediction_files.append(files)

    # 检查文件数量
    num_files = len(prediction_files[0])
    for files in prediction_files:
        if len(files) != num_files:
            raise ValueError("各预测文件夹中的.nii.gz文件数量不相等。")

    return prediction_files[0]  # 返回第一个文件夹的文件列表


def match_and_visualize(prediction_folders, ground_truth_folder, original_image_folder, output_folder, selected_file,
                        slice_index):
    """匹配多个模型的文件并进行可视化。"""
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载原始图像文件
    original_file_path = os.path.join(original_image_folder, selected_file).replace(".", "_0001.", 1)
    original_data = load_nifti(original_file_path)

    # 加载地面真值文件
    gt_file_path = os.path.join(ground_truth_folder, selected_file)
    ground_truth_data = load_nifti(gt_file_path)

    # 收集所有模型的预测结果
    prediction_slices = []
    for prediction_folder in prediction_folders:
        pred_file_path = os.path.join(prediction_folder, selected_file)
        if os.path.exists(pred_file_path):
            prediction_data = load_nifti(pred_file_path)
            prediction_slices.append(prediction_data[:, :, slice_index])  # 直接获取切片

    # 获取相应切片
    original_slice = original_data[:, :, slice_index]  # 从原始图像获取切片
    ground_truth_slice = ground_truth_data[:, :, slice_index]
    # 可视化
    visualize_multiple_models(original_slice, ground_truth_slice, prediction_slices, slice_index,
                              output_folder)


def main():
    # 设置文件夹路径
    folder_num = 1
    prediction_folders = [
        f'/home/lyh/ExperimentsNResults/Dataset628_BraTS2024/nnUNetTrainer__nnUNetPlans__2d/fold_{folder_num}/validation/',
        f'/home/lyh/ExperimentsNResults/Dataset628_BraTS2024/nnUNetTrainer_multichannel__nnUNetPlans__2d/fold_{folder_num}/validation/',
        f'/home/lyh/ExperimentsNResults/Dataset628_BraTS2024/nnUNetTrainer_multichannel2__nnUNetPlans__2d/fold_{folder_num}/validation/',
        f'/home/lyh/ExperimentsNResults/Dataset628_BraTS2024/nnUNetTrainer_multichannel3__nnUNetPlans__2d/fold_{folder_num}/validation/',
        # f'/home/lyh/ExperimentsNResults/Dataset628_BraTS2024/nnUNetTrainerUMambaBot__nnUNetPlans__2d/fold_{args.folder_num}/validation/',
        # f'/home/lyh/ExperimentsNResults/Dataset628_BraTS2024/nnUNetTrainerUMambaEnc__nnUNetPlans__2d/fold_{args.folder_num}/validation/',
        f'/home/lyh/ExperimentsNResults/Dataset628_BraTS2024/nnUNetTrainerSwinUMambaScratch__nnUNetPlans__2d/fold_{folder_num}/validation/',
    ]
    original_image_folder = r"/home/project/datasets/nnUNet_raw/Dataset628_BraTS2024/imagesTr/"  # 替换为原始图像的路径
    ground_truth_folder = r"/home/project/datasets/nnUNet_raw/Dataset628_BraTS2024/labelsTr/"  # 替换为地面真值文件夹的实际路径
    output_folder = r"/home/lyh/ExperimentsNResults/Dataset628_BraTS2024/visualize_output/"  # 输出文件夹的名称

    # 检查预测文件夹中的文件数量是否相等
    try:
        file_list = check_prediction_files(prediction_folders)
    except ValueError as e:
        print(e)
        return

    # 允许用户选择文件和切片
    print("可用文件列表:")
    for i, file_name in enumerate(file_list):
        print(f"{i}: {file_name}")

    selected_index = int(input("请选择文件索引: "))
    selected_file = file_list[selected_index]

    # 选择切片维度
    dimension = int(input("选择切片维度 (0: Sagittal, 1: Coronal, 2: Axial): "))
    slice_index = int(
        input(f"输入切片索引 (0 到 {load_nifti(os.path.join(ground_truth_folder, selected_file)).shape[2] - 1}): "))

    # 执行匹配和可视化操作
    match_and_visualize(prediction_folders, ground_truth_folder, original_image_folder, output_folder, selected_file,
                        slice_index)


if __name__ == "__main__":
    main()
    # todo add touch bar to switch and choose which slice you would like to save
