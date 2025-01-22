import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
original_image_path = '/home/project/datasets/nnUNet_raw/Dataset628_BraTS2024/imagesTr/BraTS-GLI-00020-100_0000.nii.gz'  # 替换为实际的原始图像路径
ground_truth_path = '/home/project/datasets/nnUNet_raw/nnUNet_preprocessed/Dataset628_BraTS2024/gt_segmentations/BraTS-GLI-00020-100.nii.gz'  # 替换为实际路径
prediction_path = '/home/lyh/ExperimentsNResults/Dataset628_BraTS2024/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation/BraTS-GLI-00020-100.nii.gz'      # 替换为实际路径
def load_nifti(file_path):
    """
    加载NIfTI文件并返回图像数据和仿射矩阵。
    """
    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()
    affine = nifti_img.affine
    return data, affine

def get_slice(data, axis, index):
    """
    获取指定轴和索引的切片。
    """
    if axis == 0:
        slice_img = data[index, :, :]
    elif axis == 1:
        slice_img = data[:, index, :]
    elif axis == 2:
        slice_img = data[:, :, index]
    else:
        raise ValueError("轴必须为0（sagittal）、1（coronal）或2（axial）。")
    return slice_img

def apply_colormap(segmentation, label_colors):
    """
    使用标签颜色映射生成彩色分割图。
    """
    colored = np.zeros(segmentation.shape + (3,), dtype=np.uint8)
    for label, color in label_colors.items():
        colored[segmentation == label] = color
    return colored

def detect_boundaries(ground_truth_slice, label_colors):
    """
    使用Canny算法检测边界，并将不同标签的边界以对应颜色标示。
    """
    boundaries = np.zeros(ground_truth_slice.shape + (3,), dtype=np.uint8)
    for label, color in label_colors.items():
        mask = (ground_truth_slice == label).astype(np.uint8) * 255
        edges = cv2.Canny(mask, 100, 200)
        boundaries[edges > 0] = color
    return boundaries

def overlay_segmentation_on_image(original_slice, boundaries):
    """
    将分割边界叠加到原始图像上。
    """
    # 将原始图像归一化到0-255的范围
    original_slice_normalized = cv2.normalize(original_slice, None, 0, 255, cv2.NORM_MINMAX)
    original_slice_normalized = np.stack([original_slice_normalized] * 3, axis=-1)  # 转换为3通道图像
    overlay = original_slice_normalized.copy()

    # 叠加边界，边界像素将用指定颜色替换
    overlay[boundaries > 0] = boundaries[boundaries > 0]
    return overlay

def visualize_segmentation_with_original(original_slice, ground_truth_slice, prediction_slice, label_colors):
    """
    可视化原始图像、分割结果及其边界叠加效果。
    """
    colored_ground_truth = apply_colormap(ground_truth_slice, label_colors)
    colored_prediction = apply_colormap(prediction_slice, label_colors)
    boundaries = detect_boundaries(ground_truth_slice, label_colors)
    overlay = overlay_segmentation_on_image(boundaries, original_slice)

    fig, ax = plt.subplots(1, 4, figsize=(24, 6))

    ax[0].imshow(original_slice, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(colored_ground_truth)
    ax[1].set_title('Ground Truth')
    ax[1].axis('off')

    ax[2].imshow(colored_prediction)
    ax[2].set_title('Prediction')
    ax[2].axis('off')

    ax[3].imshow(overlay)
    ax[3].set_title('Original with Ground Truth Boundaries')
    ax[3].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    # 定义文件路径

    # 加载NIfTI文件
    original_data, orig_affine = load_nifti(original_image_path)
    ground_truth_data, gt_affine = load_nifti(ground_truth_path)
    prediction_data, pred_affine = load_nifti(prediction_path)

    if ground_truth_data.shape != prediction_data.shape or original_data.shape != ground_truth_data.shape:
        raise ValueError("原始图像、地面真值和预测图像的形状必须一致。")

    # 定义标签颜色映射
    label_colors = {
        1: (255, 0, 0),    # 红色
        2: (0, 255, 0),    # 绿色
        3: (0, 0, 255),    # 蓝色
        4: (255, 255, 0)   # 黄色
    }

    # 用户选择切片轴
    print("选择切片轴：0 - Sagittal, 1 - Coronal, 2 - Axial")
    axis = int(input("输入轴（0, 1, 2）："))
    max_index = ground_truth_data.shape[axis] - 1
    index = int(input(f"输入切片索引（0到{max_index}）："))

    if index < 0 or index > max_index:
        raise ValueError(f"切片索引必须在0到{max_index}之间。")

    # 获取指定切片
    original_slice = get_slice(original_data, axis, index).astype(np.float32)
    ground_truth_slice = get_slice(ground_truth_data, axis, index).astype(np.int32)
    prediction_slice = get_slice(prediction_data, axis, index).astype(np.int32)

    # 可视化原始图像与分割结果
    visualize_segmentation_with_original(original_slice, ground_truth_slice, prediction_slice, label_colors)

if __name__ == "__main__":
    main()
