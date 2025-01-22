import numpy as np
from scipy.spatial.distance import directed_hausdorff
import nibabel as nib


def dice_coefficient(pred, gt, label):
    """
    计算某一标签的 Dice 系数。

    :param pred: 预测的二值分割标签 (numpy array)
    :param gt: ground truth 的二值标签 (numpy array)
    :param label: 需要计算的标签（int）
    :return: 该标签的 Dice 系数
    """
    # 将指定标签的区域提取出来
    pred_label = (pred == label).astype(np.uint8)
    gt_label = (gt == label).astype(np.uint8)

    intersection = np.sum(pred_label * gt_label)
    return 2 * intersection / (np.sum(pred_label) + np.sum(gt_label))

def convert_labels(input_image_path, output_image_path, label_mapping):
    """
    将医学影像分割的标签转换为新的标签。

    :param input_image_path: 输入的分割结果文件路径
    :param output_image_path: 输出的分割结果文件路径
    :param label_mapping: 标签映射字典 {旧标签: 新标签}
    """
    # 读取输入图像
    img = nib.load(input_image_path)
    data = img.get_fdata()

    # 对标签进行转换
    for old_label, new_label in label_mapping.items():
        data[data == old_label] = new_label

    # 保存新的标签图像
    new_img = nib.Nifti1Image(data, img.affine)
    nib.save(new_img, output_image_path)
    print(f"标签转换完成，保存为 {output_image_path}")


def evaluate_segmentation(pred_path, gt_path):
    """
    评估分割结果，计算 Dice 系数和 HD95。

    :param pred_path: 预测分割结果路径
    :param gt_path: ground truth 标签路径
    """
    # 读取预测结果和 ground truth
    pred_img = nib.load(pred_path)
    gt_img = nib.load(gt_path)

    pred_data = pred_img.get_fdata()
    gt_data = gt_img.get_fdata()

    # 假设标签是二值的，若有多个类别，可选择计算某一类别
    pred_binary = (pred_data > 0).astype(np.uint8)
    gt_binary = (gt_data > 0).astype(np.uint8)

    # 计算 Dice 系数
    dice_score = dice_coefficient(pred_binary, gt_binary)
    print(f"Dice 系数: {dice_score:.4f}")


def evaluate_multiple_labels(pred_path, gt_path, labels):
    """
    评估分割结果，计算多个标签的 Dice 系数。

    :param pred_path: 预测分割结果路径
    :param gt_path: ground truth 标签路径
    :param labels: 要评估的标签列表
    """
    # 读取预测结果和 ground truth
    pred_img = nib.load(pred_path)
    gt_img = nib.load(gt_path)

    pred_data = pred_img.get_fdata()
    gt_data = gt_img.get_fdata()

    dice_scores = {}

    for label in labels:
        dice_scores[label] = dice_coefficient(pred_data, gt_data, label)
        print(f"标签 {label} 的 Dice 系数: {dice_scores[label]:.4f}")

    return dice_scores

label_mapping = {1:5,2:1,5:2}  # 例如将标签 1 转换为 10，标签 2 转换为 20
convert_labels(r"C:\Users\21773\Downloads\BraTS-GLI-00020-100.nii.gz",
                      r"C:\Users\21773\Downloads\BraTS-GLI-00020-100.nii.gz", label_mapping)
label = [1, 2, 3, 4]
# 2. 评估分割结果
evaluate_multiple_labels(r"C:\Users\21773\Downloads\BraTS-GLI-00020-100.nii.gz",
                      r"C:\Users\21773\Downloads\BraTS-GLI-00020-100-seg.nii.gz",
                         label)