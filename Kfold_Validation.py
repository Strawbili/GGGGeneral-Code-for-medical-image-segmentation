import os
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image


# 自定义数据集类
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 假设mask是灰度图

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# 定义五折交叉验证
def five_fold_cross_validation(image_dir, mask_dir, model, criterion, optimizer, num_epochs=25, batch_size=4):
    # 初始化数据增强
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
    kfold = KFold(n_splits=5, shuffle=False, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}")

        # 创建数据集和数据加载器
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

        # 初始化模型、损失函数和优化器
        model = model.cuda()  # 假设使用GPU
        criterion = criterion.cuda()

        # 训练和验证
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # 训练阶段
            model.train()
            running_loss = 0.0
            for images, masks in train_loader:
                images, masks = images.cuda(), masks.cuda()

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_loader.sampler)
            print(f"Training Loss: {epoch_loss:.4f}")

            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.cuda(), masks.cuda()

                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item() * images.size(0)

            epoch_val_loss = val_loss / len(val_loader.sampler)
            print(f"Validation Loss: {epoch_val_loss:.4f}")

        # 每个fold结束后可以保存模型或结果
        torch.save(model.state_dict(), f"model_fold_{fold + 1}.pth")


if __name__ == "__main__":
    # 数据集路径
    image_dir = r"E:\HFUT\BME\BME_competition\Deep_Learning_Datasets\kvasir-seg\images"
    mask_dir = r"E:\HFUT\BME\BME_competition\Deep_Learning_Datasets\kvasir-seg\masks"

    # 假设使用一个简单的UNet模型
    model = UNet()  # 替换为您的模型
    criterion = nn.CrossEntropyLoss()  # 根据任务选择合适的损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 开始五折交叉验证
    five_fold_cross_validation(image_dir, mask_dir, model, criterion, optimizer)
