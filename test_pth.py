import glob
import os

import pandas as pd
import torch
from monai.data.dataset import Dataset
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import Resized
from monai.transforms.utility.dictionary import EnsureChannelFirstd, ToTensord
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_lossfunc, get_unet, get_optimizer

get_model = get_unet

def compute_metrics(preds, targets, smooth=1e-5):
    """
    手动计算 Dice 和 IoU
    preds, targets 形状均为: [Batch_size, Channels, H, W]
    返回每个通道的 Dice 和 IoU
    """
    # 按照空间维度 (H, W) 进行求和
    intersection = (preds * targets).sum(dim=(2, 3))
    preds_sum = preds.sum(dim=(2, 3))
    targets_sum = targets.sum(dim=(2, 3))
    union = preds_sum + targets_sum - intersection

    # 计算 Dice 和 IoU
    dice = (2.0 * intersection + smooth) / (preds_sum + targets_sum + smooth)
    iou = (intersection + smooth) / (union + smooth)

    # 对 Batch 维度求均值，返回每个通道的平均值 [Channels]
    return dice.mean(dim=0), iou.mean(dim=0)


def main():
    # ==========================================
    # 1. 基础环境与配置
    # ==========================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始测试流程，使用设备: {device}")

    # 权重文件路径 (支持用户交互选择)
    pth_files = sorted(glob.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError("当前目录下没有找到 .pth 模型文件")
    print("可用的模型文件:")
    for i, f in enumerate(pth_files):
        size = os.path.getsize(f) / 1024 / 1024
        print(f"  [{i}] {f} ({size:.1f} MB)")
    while True:
        try:
            choice = int(input("\n请输入编号选择模型文件: "))
            if 0 <= choice < len(pth_files):
                weight_path = pth_files[choice]
                break
            else:
                print(f"编号超出范围，请输入 0 ~ {len(pth_files)-1}")
        except ValueError:
            print("请输入有效数字")
    print(f"已选择: {weight_path}")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到权重文件: {weight_path}")

    # ==========================================
    # 2. 数据集加载 (与训练集严格保持一致)
    # ==========================================
    data_dir = "./dataset"
    images_dir = os.path.join(data_dir, "images", "images")
    masks_dir = os.path.join(data_dir, "masks", "masks")

    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    _, test_df = train_test_split(df, test_size=0.2, random_state=520)
    test_df = test_df.reset_index(drop=True)

    test_files = [
        {
            "image": os.path.join(images_dir, row["ImageId"]),
            "label": os.path.join(masks_dir, row["MaskId"]),
        }
        for _, row in test_df.iterrows()
    ]
    print(f"📦 载入测试集图片数量: {len(test_files)} 张")

    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            Resized(
                keys=["image", "label"],
                spatial_size=(256, 256),
                mode=("bilinear", "nearest"),
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    test_ds = Dataset(data=test_files, transform=test_transforms)
    # 测试时不需打乱 (shuffle=False)，Batch Size 可以适当开大
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    # ==========================================
    # 3. 初始化模型并加载参数
    # ==========================================
    model = get_model().to(device)

    # 加载权重
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()  # 切换到推断模式，锁定 Dropout 和 BatchNorm
    print("✅ 模型参数加载完毕！")

    # ==========================================
    # 4. 开始推理与评估
    # ==========================================
    total_dice = torch.zeros(3).to(device)
    total_iou = torch.zeros(3).to(device)
    steps = 0

    # 器官名称映射 (假设原数据集通道 0,1,2 分别对应这三个，具体请根据数据集说明确认)
    class_names = ["肺 (Lung)", "心脏 (Heart)", "气管 (Trachea)"]

    with torch.no_grad():  # 测试阶段禁用梯度计算，大幅节省显存
        progress_bar = tqdm(test_loader, desc="Testing")
        for batch_data in progress_bar:
            inputs = batch_data["image"].to(device)
            # 将标签二值化，确保其值为 0 或 1，防止掩码有 255 的灰度值
            labels = (batch_data["label"].to(device) >= 240).float()
            labels = labels[:, [2, 1, 0], :, :]

            # 模型前向传播
            logits = model(inputs)

            # 使用 Sigmoid 将输出映射到 0~1 的概率，并用 0.5 作为阈值二值化
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # 计算当前 Batch 的指标
            batch_dice, batch_iou = compute_metrics(preds, labels)

            total_dice += batch_dice
            total_iou += batch_iou
            steps += 1

    # ==========================================
    # 5. 计算最终平均指标并美化输出
    # ==========================================
    mean_dice = (total_dice / steps).cpu().numpy()
    mean_iou = (total_iou / steps).cpu().numpy()

    print("\n" + "=" * 50)
    print("🎯 测试集最终评估结果 (Test Set Metrics)")
    print("=" * 50)

    overall_dice = 0
    overall_iou = 0

    for i in range(3):
        print(f"🫀 {class_names[i]}:")
        print(f"   - Dice 系数: {mean_dice[i]:.4f}")
        print(f"   - IoU (交并比):  {mean_iou[i]:.4f}")
        print("-" * 30)
        overall_dice += mean_dice[i]
        overall_iou += mean_iou[i]

    print(f"🏆 全局平均 Dice 系数 (mDice): {overall_dice / 3:.4f}")
    print(f"🏆 全局平均 IoU (mIoU):       {overall_iou / 3:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
