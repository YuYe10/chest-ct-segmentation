import glob
import os
import shutil
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager

font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
font_manager.fontManager.addfont(
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
)
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "WenQuanYi Zen Hei",
    "Droid Sans Fallback",
    "DejaVu Sans",
]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import torch
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import Resized
from monai.transforms.utility.dictionary import EnsureChannelFirstd, ToTensord
from sklearn.model_selection import train_test_split

from model import get_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 选择 pth 文件（同 test_pth.py 的方式）
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
                print(f"编号超出范围，请输入 0 ~ {len(pth_files) - 1}")
        except ValueError:
            print("请输入有效数字")
    print(f"已选择: {weight_path}")

    # 加载模型
    model = get_model().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # 从训练集中随机挑选一张图片
    data_dir = "./dataset"
    images_dir = os.path.join(data_dir, "images", "images")
    masks_dir = os.path.join(data_dir, "masks", "masks")
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    train_df, _ = train_test_split(df, test_size=0.2, random_state=520)
    train_df = train_df.reset_index(drop=True)

    row = train_df.sample(n=1, random_state=None).iloc[0]
    img_name = row["ImageId"]
    mask_name = row["MaskId"]
    img_path = os.path.join(images_dir, img_name)
    mask_path = os.path.join(masks_dir, mask_name)
    print(f"\n随机选中: {img_name}")

    # 创建 views 输出目录
    out_dir = f"views-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    # 复制原始图片和 mask
    shutil.copy2(img_path, os.path.join(out_dir, img_name))
    shutil.copy2(mask_path, os.path.join(out_dir, mask_name))
    print(f"原始图片和 mask 已复制到 {out_dir}/")

    # 做和训练集一致的预处理
    transforms = Compose(
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
    data = transforms({"image": img_path, "label": mask_path})
    image = data["image"].unsqueeze(0).to(device)
    label = (data["label"].unsqueeze(0).to(device) >= 240).float()
    label = label[:, [2, 1, 0], :, :]

    with torch.no_grad():
        logits = model(image)
        preds = (torch.sigmoid(logits) > 0.5).float()

    img_np = image.squeeze().cpu().numpy()
    gt_np = label.squeeze().cpu().numpy()
    pred_np = preds.squeeze().cpu().numpy()

    # 计算并打印各器官 IoU
    smooth = 1e-5
    ious = []
    class_names = ["肺 (Lung)", "心脏 (Heart)", "气管 (Trachea)"]
    for i, name in enumerate(class_names):
        intersection = (pred_np[i] * gt_np[i]).sum()
        union = pred_np[i].sum() + gt_np[i].sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)
        print(f"  {name} IoU: {iou:.4f}")

    # 渲染并保存
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    # 总体对比图：原始CT + GT + 预测
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Original CT")
    axes[0].axis("off")

    gt_overlay = np.zeros((*img_np.shape, 3))
    for i in range(3):
        gt_overlay[gt_np[i] > 0] = colors[i]
    axes[1].imshow(img_np, cmap="gray", alpha=0.6)
    axes[1].imshow(gt_overlay, alpha=0.4)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    pred_overlay = np.zeros((*img_np.shape, 3))
    for i in range(3):
        pred_overlay[pred_np[i] > 0] = colors[i]
    axes[2].imshow(img_np, cmap="gray", alpha=0.6)
    axes[2].imshow(pred_overlay, alpha=0.4)
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    iou_text = "\n".join(
        [f"{n.split('(')[0].strip()}: {v:.4f}" for n, v in zip(class_names, ious)]
    )
    axes[2].text(
        5,
        20,
        iou_text,
        fontsize=9,
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        va="top",
    )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overview.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("overview.png 已保存")

    # 分别保存 3 个部件的独立对比图
    for i, name in enumerate(class_names):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")

        gt_disp = np.zeros((*img_np.shape, 3))
        gt_disp[gt_np[i] > 0] = colors[i]
        axes[1].imshow(img_np, cmap="gray", alpha=0.6)
        axes[1].imshow(gt_disp, alpha=0.4)
        axes[1].set_title(f"GT: {name}")
        axes[1].axis("off")

        pred_disp = np.zeros((*img_np.shape, 3))
        pred_disp[pred_np[i] > 0] = colors[i]
        axes[2].imshow(img_np, cmap="gray", alpha=0.6)
        axes[2].imshow(pred_disp, alpha=0.4)
        axes[2].set_title(f"Pred: {name}")
        axes[2].axis("off")
        axes[2].text(
            5,
            20,
            f"IoU: {ious[i]:.4f}",
            fontsize=11,
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
            va="top",
        )

        plt.tight_layout()
        fname = f"organ_{i}_{name.split('(')[0].strip()}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"{fname} 已保存")

    print(f"\n所有结果已保存至: {out_dir}/")


if __name__ == "__main__":
    main()
