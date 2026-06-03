import glob
import os

import torch
from monai.data.dataset import Dataset
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import Resized
from monai.transforms.utility.dictionary import EnsureChannelFirstd, ToTensord
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_lossfunc, get_model, get_optimizer


def main():
    # ==========================================
    # 1. 基础配置
    # ==========================================
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")

    # 你的解压路径（注意截图里的俄罗斯套娃层级）
    data_dir = "./dataset"
    images_dir = os.path.join(data_dir, "images", "images")
    masks_dir = os.path.join(data_dir, "masks", "masks")

    # 获取所有图片和掩码的路径 (假设格式为 png 或 tif)
    # 请根据实际文件后缀修改 "*.png"
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.*")))

    # 构建 MONAI 需要的字典格式
    data_dicts = [
        {"image": img, "label": mask} for img, mask in zip(image_files, mask_files)
    ]

    # 划分训练集和验证集 (简单 8:2 划分)
    split_idx = int(len(data_dicts) * 0.8)
    train_files, val_files = data_dicts[:split_idx], data_dicts[split_idx:]

    # ==========================================
    # 2. 数据增强与预处理 (Transforms)
    # ==========================================
    # Swin-UNETR 对输入尺寸有要求，通常需要是 32 的整数倍，这里统一 resize 到 256x256
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),  # 归一化 CT 图像
            Resized(
                keys=["image", "label"],
                spatial_size=(256, 256),
                mode=("bilinear", "nearest"),
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    # ==========================================
    # 3. 构建 DataLoader
    # ==========================================
    train_ds = Dataset(data=train_files, transform=train_transforms)
    # A100 显存极大 (40G/80G)，batch_size 可以放心开到 32 或 64 加快训练
    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )

    # ==========================================
    # 4. 定义模型 (Swin-UNETR 2D版本)
    # ==========================================
    # 胸部 CT 任务包含：背景(0)、肺、心脏、气管 -> 共 4 个类别
    model = get_model().to(device)

    # ==========================================
    # 5. 损失函数与优化器
    # ==========================================
    # DiceFocalLoss 是刷高分的神器，能很好地处理小器官（如气管）的类别不平衡问题
    loss_function = get_lossfunc().to(device)
    optimizer = get_optimizer(model)

    # ==========================================
    # 6. 开始训练
    # ==========================================
    max_epochs = 10
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}")
        for batch_data in progress_bar:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = (batch_data["label"] > 127).float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # 计算损失
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        epoch_loss /= step
        print(f"Epoch {epoch + 1} Average Loss: {epoch_loss:.4f}")

        # 建议每隔 10 个 Epoch 保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"swin_unetr_epoch_{epoch + 1}.pth")
            print(f"✅ 模型 swin_unetr_epoch_{epoch + 1}.pth 已保存")


if __name__ == "__main__":
    main()
