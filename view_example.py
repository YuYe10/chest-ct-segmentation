import glob
import os
import shutil
from datetime import datetime

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import Unet
from sklearn.model_selection import train_test_split


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    best_path = "best_model.pth"
    if not os.path.exists(best_path):
        pth_files = sorted(glob.glob("*best*.pth"))
        if not pth_files:
            raise FileNotFoundError("no best_model.pth found")
        best_path = pth_files[0]
    print(f"loading: {best_path}")

    model = Unet(
        "efficientnet-b2", encoder_weights=None, classes=3, activation=None
    ).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    data_dir = "./dataset"
    images_dir = os.path.join(data_dir, "images", "images")
    masks_dir = os.path.join(data_dir, "masks", "masks")
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    train_df, _ = train_test_split(df, test_size=0.2, random_state=69)
    train_df = train_df.reset_index(drop=True)

    row = train_df.sample(n=1, random_state=None).iloc[0]
    img_name = row["ImageId"]
    mask_name = row["MaskId"]
    img_path = os.path.join(images_dir, img_name)
    mask_path = os.path.join(masks_dir, mask_name)
    print(f"selected: {img_name}")

    img_bgr = cv2.imread(img_path)
    mask_bgr = cv2.imread(mask_path)

    mask = mask_bgr.copy()
    mask[mask < 240] = 0
    mask[mask > 0] = 1
    mask_tensor = (
        torch.from_numpy(mask.astype(np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    transform = Compose(
        [
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    augmented = transform(image=img_bgr)
    image_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        preds = (torch.sigmoid(logits) > 0.5).float()

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gt_np = mask_tensor.squeeze().cpu().numpy()
    pred_np = preds.squeeze().cpu().numpy()

    smooth = 1e-5
    ious = []
    class_names = ["Lung", "Heart", "Trachea"]
    for i, name in enumerate(class_names):
        intersection = (pred_np[i] * gt_np[i]).sum()
        union = pred_np[i].sum() + gt_np[i].sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)
        print(f"  {name} IoU: {iou:.4f}")

    out_dir = f"views-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    shutil.copy2(img_path, os.path.join(out_dir, img_name))
    shutil.copy2(mask_path, os.path.join(out_dir, mask_name))
    print(f"original image and mask copied to {out_dir}/")

    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_gray, cmap="gray")
    axes[0].set_title("Original CT")
    axes[0].axis("off")

    gt_overlay = np.zeros((*img_gray.shape, 3))
    for i in range(3):
        gt_overlay[gt_np[i] > 0] = colors[i]
    axes[1].imshow(img_gray, cmap="gray", alpha=0.6)
    axes[1].imshow(gt_overlay, alpha=0.4)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    pred_overlay = np.zeros((*img_gray.shape, 3))
    for i in range(3):
        pred_overlay[pred_np[i] > 0] = colors[i]
    axes[2].imshow(img_gray, cmap="gray", alpha=0.6)
    axes[2].imshow(pred_overlay, alpha=0.4)
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    iou_text = "\n".join([f"{n}: {v:.4f}" for n, v in zip(class_names, ious)])
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
    print("overview.png saved")

    for i, name in enumerate(class_names):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_gray, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")

        gt_disp = np.zeros((*img_gray.shape, 3))
        gt_disp[gt_np[i] > 0] = colors[i]
        axes[1].imshow(img_gray, cmap="gray", alpha=0.6)
        axes[1].imshow(gt_disp, alpha=0.4)
        axes[1].set_title(f"GT: {name}")
        axes[1].axis("off")

        pred_disp = np.zeros((*img_gray.shape, 3))
        pred_disp[pred_np[i] > 0] = colors[i]
        axes[2].imshow(img_gray, cmap="gray", alpha=0.6)
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
        fname = f"organ_{i}_{name}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"{fname} saved")

    print(f"\nall results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
