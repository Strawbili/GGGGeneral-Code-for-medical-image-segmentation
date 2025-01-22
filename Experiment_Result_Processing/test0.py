import os
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def load_nifti(file_path):
    """加载NIfTI文件并返回图像数据和仿射矩阵。"""
    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()
    affine = nifti_img.affine
    return data, affine

def get_slice(data, axis, index):
    """获取指定轴和索引的切片。"""
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
    """使用标签颜色映射生成彩色分割图。"""
    colored = np.zeros(segmentation.shape + (3,), dtype=np.uint8)
    for label, color in label_colors.items():
        colored[segmentation == label] = color
    return colored


def visualize_multiple_predictions(ground_truth_slice, predictions_slices, label_colors):
    """
    可视化多个预测结果，删除边界检测功能。
    """
    num_predictions = len(predictions_slices)
    fig, axs = plt.subplots(1, num_predictions + 1, figsize=(6 * (num_predictions + 1), 6))

    colored_ground_truth = apply_colormap(ground_truth_slice, label_colors)

    # Ground Truth
    axs[0].imshow(colored_ground_truth)
    axs[0].set_title('Ground Truth')
    axs[0].axis('off')

    # Prediction Results
    for i, prediction_slice in enumerate(predictions_slices):
        colored_prediction = apply_colormap(prediction_slice, label_colors)
        axs[i + 1].imshow(colored_prediction)
        axs[i + 1].set_title(f'Prediction {i + 1}')
        axs[i + 1].axis('off')

    plt.tight_layout()
    plt.show()


def update(val, data, axis, label_colors, im_gt, im_preds):
    """
    响应滑动条变化更新显示的切片，不再使用边界检测。
    """
    index = int(val)
    gt_slice = get_slice(data['gt'], axis, index).astype(np.int32)
    pred_slices = [get_slice(pred, axis, index).astype(np.int32) for pred in data['pred']]

    colored_gt = apply_colormap(gt_slice, label_colors)
    im_gt.set_data(colored_gt)

    for i, (pred_slice, im_pred) in enumerate(zip(pred_slices, im_preds)):
        colored_pred = apply_colormap(pred_slice, label_colors)
        im_pred.set_data(colored_pred)

    plt.draw()


def main_interactive():
    # 定义文件路径
    image_name = "BraTS-GLI-00020-001.nii.gz"
    ground_truth_path = '/home/project/datasets/nnUNet_raw/nnUNet_preprocessed/Dataset137_BraTS2023/gt_segmentations/'  # 替换为实际路径
    folder_num = 0

    prediction_folders = [
        f'/home/lyh/ExperimentsNResults/Dataset137_BraTS2023/nnUNetTrainer__nnUNetPlans__2d/fold_{folder_num}/validation',
        f'/home/lyh/ExperimentsNResults/Dataset137_BraTS2023/nnUNetTrainer_I2UNet__nnUNetPlans__2d/fold_{folder_num}/validation',
        f'/home/lyh/ExperimentsNResults/Dataset137_BraTS2023/nnUNetTrainer_multichannel20__nnUNetPlans__2d/fold_{folder_num}/validation',
    ]
    # 加载NIfTI文件
    ground_truth_data, _ = load_nifti(os.path.join(ground_truth_path, image_name))
    prediction_data_list = [load_nifti(os.path.join(pred_path, image_name))[0] for pred_path in prediction_folders]

    if any(pred.shape != ground_truth_data.shape for pred in prediction_data_list):
        raise ValueError("地面真值和某些预测图像的形状不一致。")

    label_colors = {
        0: (0, 0, 0),
        1: (255, 0, 0),    # 红色
        2: (0, 255, 0),    # 绿色
        3: (0, 0, 255),    # 蓝色
        4: (255, 255, 0)   # 黄色
    }

    # 用户选择切片轴
    axis = int(input("选择切片轴：0 - Sagittal, 1 - Coronal, 2 - Axial\n输入轴（0, 1, 2）："))
    max_index = ground_truth_data.shape[axis] - 1
    initial_index = max_index // 2

    fig, axs = plt.subplots(1, len(prediction_data_list) + 1, figsize=(6 * (len(prediction_data_list) + 1), 6))
    plt.subplots_adjust(bottom=0.25)

    # 初始切片
    gt_slice = get_slice(ground_truth_data, axis, initial_index).astype(np.int32)
    pred_slices = [get_slice(pred_data, axis, initial_index).astype(np.int32) for pred_data in prediction_data_list]

    colored_gt = apply_colormap(gt_slice, label_colors)

    # Ground Truth
    im_gt = axs[0].imshow(colored_gt)
    axs[0].set_title('Ground Truth')
    axs[0].axis('off')

    im_preds = []

    for i, pred_slice in enumerate(pred_slices):
        colored_pred = apply_colormap(pred_slice, label_colors)

        # Prediction
        im_pred = axs[i + 1].imshow(colored_pred)
        axs[i + 1].set_title(f'Prediction {i + 1}')
        axs[i + 1].axis('off')
        im_preds.append(im_pred)

    # 添加滑动条
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Slice Index',
        valmin=0,
        valmax=max_index,
        valinit=initial_index,
        valstep=1,
    )

    data = {'gt': ground_truth_data, 'pred': prediction_data_list}
    slider.on_changed(lambda val: update(val, data, axis, label_colors, im_gt, im_preds))

    plt.show()


if __name__ == "__main__":
    main_interactive()
