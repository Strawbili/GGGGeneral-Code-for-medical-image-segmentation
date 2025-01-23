from PIL import Image
import os
import numpy as np
# 读取图片函数
def load_image(image_path):
    # 打开图像文件
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def search_image_in_folders(image_name, base_dir):
    # 遍历每个 folder_X 文件夹
    for folder_idx in range(5):  # 假设 X 的范围是 0-4
        folder_path = os.path.join(base_dir, f"fold_{folder_idx}", "validation")
        print(folder_path)
        # 判断文件夹是否存在
        if os.path.exists(folder_path):
            print(f"Searching in: {folder_path}")
            
            # 遍历文件夹中的文件
            for root, _, files in os.walk(folder_path):
                if image_name in files:
                    print(f"Found image: {image_name} in {root}")
                    return os.path.join(root, image_name)  # 返回图片的完整路径
    print(f"Image {image_name} not found.")
    return None

image_name = "cju1dnz61vfp40988e78bkjga.png"
gt_path = "/home/SSD1_4T/datasets/nnUNet_raw/Dataset121_Kvasir/labelsTr/"
prediction_path = "/home/lyh/ExperimentsNResults/Dataset121_Kvasir/"
trainer_name = "nnUNetTrainer_multichannel20__nnUNetPlans__2d"
# 示例路径
gt_path = os.path.join(gt_path, image_name)
prediction_path = os.path.join(prediction_path, trainer_name)
print(prediction_path)
prediction_path = search_image_in_folders(image_name, prediction_path)

# 读取ground truth 和 prediction 图片
gt_image = load_image(gt_path)
prediction_image = load_image(prediction_path)

print("gt_image(unique):", np.unique(gt_image))
print("prediction(unique):", np.unique(prediction_image))
# 显示图像（可以跳过这部分，依据实际需求）
if gt_image:
    gt_image.show()  # 显示 ground truth 图像

if prediction_image:
    prediction_image.show()  # 显示预测图像

# 你可以在这里继续进行图像的比较或分析
