import os
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt

original_image_path = r"E:\HFUT\BME\BME_competition\OutputNExperiments\Viaualizing\BraTS24\origin\BraTS-GLI-00020-100_0000.nii.gz"  # 替换为实际的原始图像路径
ground_truth_path = r'E:\HFUT\BME\BME_competition\OutputNExperiments\Viaualizing\BraTS24\ground_truth\BraTS-GLI-00020-100.nii.gz'  # 替换为实际路径
prediction_path = r"E:\HFUT\BME\BME_competition\OutputNExperiments\Viaualizing\BraTS24\MambaEnc\BraTS-GLI-00020-100.nii.gz"  # 替换为实际路径
output_dir = 'E:\HFUT\BME\BME_competition\OutputNExperiments\Viaualizing\BraTS24\output/'  # 保存图片的根目录


def load_nifti(file_path):
    """
    加载NIfTI文件并返回图像数据和仿射矩阵。
    """
    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()
    # affine = nifti_img.affine
    return data # , affine

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

def apply_colormap_to_4d(segmentation, label_colors):
    """
    处理四维分割图，将其分成多个二维图并返回彩色图列表。
    """
    colored_images = []
    for i in range(segmentation.shape[0]):
        colored_image = apply_colormap(segmentation[i], label_colors)
        colored_images.append(colored_image)
    return colored_images
def match_and_visualize(prediction_folders, ground_truth_folder, original_image_folder, output_folder, selected_file,
                        slice_index, dimension, selected_index):
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
            prediction_slices.append(prediction_data[:, :, slice_index].astype(np.int32))  # 直接获取切片

    # 获取相应切片
    original_slice = original_data[:, :, slice_index].astype(np.float32)  # 从原始图像获取切片
    ground_truth_slice = ground_truth_data[:, :, slice_index].astype(np.int32)
    # 可视化
    name = f"selected_{selected_index}slice_{slice_index}dimen_{dimension}"
    visualize_multiple_models(original_slice, ground_truth_slice, prediction_slices, slice_index,
                              output_folder, output_folder_name = name)

def visualize_multiple_models(original_slice, ground_truth_slice, prediction_slices, slice_index, output_folder, output_folder_name):
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

    label_colors = {
        1: (255, 0, 0),  # 红色
        2: (0, 255, 0),  # 绿色
        3: (0, 0, 255),  # 蓝色
        4: (255, 255, 0)  # 黄色
    }
    plt.figure(figsize=(15, 5))
    colored_ground_truth = apply_colormap(ground_truth_slice, label_colors)
    colored_ground_truth_overlay_origin = overlay_segmentation_on_image(original_slice, colored_ground_truth)
    colored_ground_truth_overlay_origin_normed = colored_ground_truth_overlay_origin / 255.0
    # 原始图像
    plt.subplot(1, len(prediction_slices) + 2, 1)
    plt.title("Original Image")
    plt.imshow(original_slice, cmap='gray')
    plt.axis('off')

    # 地面真值叠加
    plt.subplot(1, len(prediction_slices) + 2, 2)
    plt.title("Ground Truth Overlay")
    plt.imshow(colored_ground_truth_overlay_origin_normed)
    plt.axis('off')

    # 遍历每个模型的预测结果，叠加
    for i, prediction_slice in enumerate(prediction_slices):
        colored_label = apply_colormap(prediction_slice, label_colors)
        colored_label_overlay_origin = overlay_segmentation_on_image(original_slice, colored_label)
        colored_label_overlay_origin_normed = colored_label_overlay_origin / 255.0
        plt.subplot(1, len(prediction_slices) + 2, i + 3)
        plt.title(f"Model {i + 1} Prediction Overlay")
        plt.imshow(colored_label_overlay_origin_normed)
        plt.axis('off')

    plt.tight_layout()
    if not os.path.exists(os.path.join(output_folder,output_folder_name)):
        os.makedirs(os.path.join(output_folder,output_folder_name))
    plt.savefig(os.path.join(output_folder,output_folder_name,  f'slice_{slice_index}.png'))
    plt.imsave(os.path.join(output_folder, output_folder_name, 'original_with_gt.png'), colored_ground_truth_overlay_origin_normed)
    plt.close()


def detect_boundaries(target_slice, label_colors):
    """
    使用Canny算法检测边界，并将不同标签的边界以对应颜色标示。
    """
    boundaries = np.zeros(target_slice.shape + (3,), dtype=np.uint8)
    for label, color in label_colors.items():
        mask = (target_slice == label).astype(np.uint8) * 255
        edges = cv2.Canny(mask, 100, 200)
        boundaries[edges > 0] = color
    return boundaries


def overlay_segmentation_on_image(original_slice, overlayer):
    """
    将分割边界叠加到原始图像上。
    """
    original_slice_normalized = cv2.normalize(original_slice, None, 0, 255, cv2.NORM_MINMAX)
    original_slice_normalized = np.stack([original_slice_normalized] * 3, axis=-1)  # 转换为3通道图像
    overlay = original_slice_normalized.copy()

    # 叠加边界，边界像素将用指定颜色替换
    overlay[overlayer > 0] = overlayer[overlayer > 0]
    return overlay


def save_slices(output_dir, slice_name, original_slice, ground_truth_slice, prediction_slice, label_colors):
    """
    保存指定切片的分割结果，包括原始图像、地面真值、预测结果和叠加边界的图像。
    """
    # 创建文件夹
    output_path = os.path.join(output_dir, slice_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 创建彩色分割图和边界
    colored_ground_truth = apply_colormap(ground_truth_slice, label_colors)
    colored_prediction = apply_colormap(prediction_slice, label_colors)
    ori_with_boun = overlay_segmentation_on_image(original_slice, colored_prediction)
    boudaries_predict = detect_boundaries(prediction_slice, label_colors)
    predict_with_boun = overlay_segmentation_on_image(original_slice, boudaries_predict)

    # 保存原始图像
    # plt.imsave(os.path.join(output_path, 'original_image.png'), original_slice, cmap='gray')

    # 保存地面真值彩色图
    plt.imsave(os.path.join(output_path, 'ground_truth.png'), colored_ground_truth)

    # 保存预测结果彩色图
    plt.imsave(os.path.join(output_path, 'prediction.png'), colored_prediction)

    # 保存叠加了边界的原始图像
    # 将overlay归一化到0-1范围
    overlay_normalized = ori_with_boun / 255.0
    plt.imsave(os.path.join(output_path, 'original_with_boundaries.png'), overlay_normalized)

    overlay_normalized = predict_with_boun / 255.0
    plt.imsave(os.path.join(output_path, 'prediction_with_boundaries.png'), overlay_normalized)

    print(f"切片 {slice_name} 的结果已保存到: {output_path}")


def main():
    prediction_folders = [
        r'/home/lyh/ExperimentsNResults/Dataset628_BraTS2024/nnUNetTrainerUNETR__nnUNetPlans__2d/fold_0/validation/',
        # 替换为模型1的预测文件夹路径
        r'/home/lyh/ExperimentsNResults/Dataset628_BraTS2024/nnUNetTrainerUMambaEnc__nnUNetPlans__2d/fold_0/validation/',
        # 替换为模型2的预测文件夹路径
        # r'/home/lyh/ExperimentsNResults/Dataset628_BraTS2024/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/'
        # 可以继续添加其他模型的预测文件夹

    ]
    original_image_folder = r"/home/project/datasets/nnUNet_raw/Dataset628_BraTS2024/imagesTr/"  # 替换为原始图像的路径
    ground_truth_folder = r"/home/project/datasets/nnUNet_raw/Dataset628_BraTS2024/labelsTr/"  # 替换为地面真值文件夹的实际路径
    output_folder = r"/home/lyh/ExperimentsNResults/Dataset628_BraTS2024/visualize_output/"  # 输出文件夹的名称
    # 加载NIfTI文件

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
    data = load_nifti(os.path.join(ground_truth_folder, selected_file))
    slice_index = int(
        input(f"输入切片索引 (0 到 {data.shape[dimension]}): "))
    match_and_visualize(prediction_folders, ground_truth_folder, original_image_folder, output_folder, selected_file,
                        slice_index = slice_index, dimension = dimension, selected_index = selected_index)


if __name__ == "__main__":
    main()
