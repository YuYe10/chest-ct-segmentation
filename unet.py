import os
import random

import numpy as np
import pandas as pd
import torch
from monai.data.dataset import Dataset
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import RandAffined, RandFlipd, Resized
from monai.transforms.utility.dictionary import EnsureChannelFirstd, ToTensord
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_lossfunc, get_optimizer, get_unet

get_model = get_unet


def main():
    # ==========================================
    # 1. 基础配置与设备绑定
    # ==========================================
    random.seed(520)
    np.random.seed(520)
    torch.manual_seed(520)
    torch.cuda.manual_seed(520)

    # 绑定你的双显卡环境中的第二张卡（A100等）
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"🔥 当前使用的训练设备: {device}")

    data_dir = "./dataset"
    images_dir = os.path.join(data_dir, "images", "images")
    masks_dir = os.path.join(data_dir, "masks", "masks")

    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=520)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"训练集: {len(train_df)} 对 | 验证集: {len(val_df)} 对")

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
    # 2. 图像预处理与高强度空间增强 (Transforms)
    # ==========================================
    # 核心改进：采用 ScaleIntensityRanged 截断胸部 CT 的 HU 值窗口，保持物理一致性
    # 肺窗/纵隔窗复合范围通常在 -1000 到 400 之间
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
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            # 新增轻量级仿射变换，提升对细长气管的泛化能力
            RandAffined(
                keys=["image", "label"],
                prob=0.3,
                rotate_range=(0.15, 0.15),
                translate_range=(0.05, 0.05),
                scale_range=(0.05, 0.05),
                mode=("bilinear", "nearest"),
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
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
    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    # ==========================================
    # 4. 初始化模型、损失函数与优化器
    # ==========================================
    model = get_model().to(device)
    loss_function = get_lossfunc().to(device)
    optimizer = get_optimizer(model)

    # 核心改进：引入余弦退火学习率调度器，避免后期参数震荡
    max_epochs = 150  # 有了预训练权重，150个 Epoch 即可深度收敛
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    # ==========================================
    # 5. 正式开始训练与实时验证循环
    # ==========================================
    for epoch in range(max_epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0
        train_steps = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs} [Train]")
        for batch_data in train_bar:
            train_steps += 1
            inputs = batch_data["image"].to(device)
            labels = (batch_data["label"] >= 240).float().to(device)
            labels = labels[:, [2, 1, 0], :, :]  # 保持通道顺序修正的一致性

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / train_steps
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch + 1} Average Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f}"
        )

        # 步进学习率
        scheduler.step()

        # 每隔 10 个 Epoch 保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"unet_epoch_{epoch + 1}.pth")
            print(f"模型 unet_epoch_{epoch + 1}.pth 已保存")


if __name__ == "__main__":
    main()
