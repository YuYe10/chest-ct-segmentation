import os
import random

import numpy as np
import pandas as pd
import torch
from monai.data.dataset import Dataset
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import RandFlipd, Resized
from monai.transforms.utility.dictionary import EnsureChannelFirstd, ToTensord
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_lossfunc, get_model, get_optimizer


def main():
    # ==========================================
    # 1. 基础配置
    # ==========================================
    random.seed(520)
    np.random.seed(520)
    torch.manual_seed(520)
    torch.cuda.manual_seed(520)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")

    # 你的解压路径（注意截图里的俄罗斯套娃层级）
    data_dir = "./dataset"
    images_dir = os.path.join(data_dir, "images", "images")
    masks_dir = os.path.join(data_dir, "masks", "masks")

    # === 按 exmaple.py 标准：通过 CSV 读取配对关系 ===
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    print(f"CSV 共 {len(df)} 对图像-掩码")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=520)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"训练集: {len(train_df)} | 验证集: {len(val_df)}")

    # 构建 MONAI 需要的字典格式（拼接完整路径）
    def build_data_dicts(df: pd.DataFrame):
        return [
            {
                "image": os.path.join(images_dir, row["ImageId"]),
                "label": os.path.join(masks_dir, row["MaskId"]),
            }
            for _, row in df.iterrows()
        ]

    train_files = build_data_dicts(train_df)
    val_files = build_data_dicts(val_df)

    # ==========================================
    # 2. 数据增强与预处理 (Transforms)
    # ==========================================
    # Swin-UNETR 对输入尺寸有要求，通常需要是 32 的整数倍，这里统一 resize 到 256x256
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
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
    max_epochs = 200
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}")
        for batch_data in progress_bar:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = (batch_data["label"] >= 240).float().to(device)
            labels = labels[
                :, [2, 1, 0], :, :
            ]  # 添加通道顺序修正（PIL RGB → OpenCV BGR），将通道0和2互换

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
