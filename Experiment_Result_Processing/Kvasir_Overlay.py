import os
import cv2
import numpy as np

def load_png_file(filepath):
    """加载 PNG 文件并返回图像数据"""
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像

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
        4: [255, 255, 0],    # 黄色 (标签 4)
    }

    # 创建彩色图像
    color_img = np.zeros((*label_img.shape, 3), dtype=np.uint8)

    # 将标签映射到颜色
    for label_value, color in colors.items():
        color_img[label_img == label_value] = color

    return color_img


def visualize_and_save_original_image(original_img, save_path):
    """
    保存原始图像。
    """
    # 将原始图像归一化到0-255范围，并转换为uint8格式
    original_img_norm = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 保存原始图像
    cv2.imwrite(save_path, original_img_norm)
    print(f'Original image saved at: {save_path}')


def visualize_and_save_overlay(original_img, overlay_img, save_path, overlay_type="Ground Truth"):
    # 将标签图像转换为彩色
    overlay_color_img = apply_color_map_to_labels(overlay_img)

    # 将原始图像归一化到0-255范围，并转换为uint8格式
    original_img_norm = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 如果原始图像是灰度图，转化为三通道彩色图像
    if len(original_img.shape) == 2:  # 检查是否是灰度图像
        original_img_norm = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        original_img_colored = cv2.cvtColor(original_img_norm, cv2.COLOR_GRAY2BGR)
    else:
        # 如果原始图像已经是彩色图像，直接归一化
        original_img_norm = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        original_img_colored = original_img_norm

    # 创建一个掩码，只对非背景区域进行混合
    mask = overlay_img > 0  # 只考虑非0区域
    mask = mask.astype(np.uint8)  # 转换为uint8格式

    # 扩展掩码维度以适用于彩色图像
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # 仅在有标签的区域进行混合
    combined_img = np.where(mask_3d, cv2.addWeighted(original_img_colored, 0.3, overlay_color_img, 0.7, 0), original_img_colored)

    # 保存图片
    cv2.imwrite(save_path, combined_img)
    print(f'{overlay_type} overlay saved at: {save_path}')


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PNG Image Segmentation Visualization")
    parser.add_argument("--file", type=str, 
                        default="cju1b3zgj3d8e0801kpolea6c.png",
                        required=False, help="选择的文件名")
    parser.add_argument("--original_image_folder", 
                        default="/home/SSD1_4T/datasets/nnUNet_raw/Dataset121_Kvasir/imagesTr/",
                        type=str, required=False, help="原始图像文件夹路径")
    parser.add_argument("--output_folder", type=str, 
                        default="/home/lyh/ExperimentsNResults/Predict_OutPut_Folder/121_Kvasir/",
                        required=False, help="输出文件夹")
    parser.add_argument("--ground_truth", 
                        default="/home/SSD1_4T/datasets/nnUNet_raw/Dataset121_Kvasir/labelsTr/",
                        type=str, required=False, help="Ground Truth 标签文件路径")
    parser.add_argument("--fold_num", type=str, default="0", required=False, help="Fold Number")
    args = parser.parse_args()

    file_name_without_ext = os.path.splitext(args.file)[0]
    output_folder = os.path.join(args.output_folder, file_name_without_ext)

    prediction_folders = [
        f'/home/lyh/ExperimentsNResults/Dataset121_Kvasir/nnUNetTrainer__nnUNetPlans__2d/fold_{args.fold_num}/validation',
        f'/home/lyh/ExperimentsNResults/Dataset121_Kvasir/nnUNetTrainer_I2UNet__nnUNetPlans__2d/fold_{args.fold_num}/validation',
        f'/home/lyh/ExperimentsNResults/Dataset121_Kvasir/nnUNetTrainer_multichannel20__nnUNetPlans__2d/fold_{args.fold_num}/validation',
        f'/home/lyh/ExperimentsNResults/Dataset121_Kvasir/nnUNetTrainer_multichannel20_ZI__nnUNetPlans__2d/fold_{args.fold_num}/validation',

    ]
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载原始数据、ground truth 和预测数据
    selected_file = args.file
    
    original_img = cv2.imread(os.path.join(args.original_image_folder, file_name_without_ext + "_0000.png"))
    ground_truth_img = load_png_file(os.path.join(args.ground_truth, selected_file))

    # 可视化并保存原始图像
    save_path = os.path.join(output_folder, f'original_{selected_file}')
    visualize_and_save_original_image(original_img, save_path)

    # 可视化并保存 ground truth 叠加图
    save_path = os.path.join(output_folder, f'ground_truth_overlay_{selected_file}')
    visualize_and_save_overlay(original_img, ground_truth_img, save_path, overlay_type="Ground Truth")

    # 可视化并保存每个预测结果叠加图
    for i, prediction_folder in enumerate(prediction_folders):
        prediction_img = load_png_file(os.path.join(prediction_folder, selected_file))
        save_path = os.path.join(output_folder, f'prediction_{i+1}_overlay_{selected_file}')
        visualize_and_save_overlay(original_img, 
                                   prediction_img, 
                                   save_path, 
                                   overlay_type=f"Prediction {i+1}")
if __name__ == "__main__":
    main()
