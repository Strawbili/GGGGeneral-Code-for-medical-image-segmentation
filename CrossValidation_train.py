import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')
class MyDataset(Dataset):

    def __init__(self, imgs, labels, transform=None, target_transform=None):

        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.imgs[idx]
        target = self.labels[idx]

        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)


def find_classes(dir):
    folder_name = [d.name for d in os.scandir(dir) if d.is_dir()]
    folder_name.sort()
    folder_num = {cls_name: i for i, cls_name in enumerate(folder_name)}
    return folder_name, folder_num


if __name__ == "__main__":
    dir = r"E:\HFUT\BME\BME_competition\Deep_Learning_Datasets\kvasir-seg"
    classes, class_to_idx = find_classes(dir)
    imgs = []
    labels = []
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(dir, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_image_file(path):
                    imgs.append(path)
                    labels.append(class_index)

    skf = StratifiedKFold(n_splits=5)  # 5折
    for i, (train_idx, val_idx) in enumerate(skf.split(imgs, labels)):
        trainset, valset = np.array(imgs)[[train_idx]], np.array(imgs)[[val_idx]]
        traintag, valtag = np.array(labels)[[train_idx]], np.array(labels)[[val_idx]]
        train_dataset = MyDataset(trainset, traintag, data_transforms['train'])
        val_dataset = MyDataset(valset, valtag, data_transforms['val'])
    print("finish")
