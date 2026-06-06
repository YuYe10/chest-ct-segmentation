import glob
import os
import re

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

from model import get_unet as get_model


def compute_metrics(preds, targets, smooth=1e-5):
    intersection = (preds * targets).sum(dim=(2, 3))
    preds_sum = preds.sum(dim=(2, 3))
    targets_sum = targets.sum(dim=(2, 3))
    union = preds_sum + targets_sum - intersection

    dice = (2.0 * intersection + smooth) / (preds_sum + targets_sum + smooth)
    iou = (intersection + smooth) / (union + smooth)

    return dice.mean(dim=0), iou.mean(dim=0)


def extract_epoch(filename):
    match = re.search(r"epoch_(\d+)", filename)
    return int(match.group(1)) if match else 0


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

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
    print(f"测试集图片数量: {len(test_files)} 张")

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
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    class_names = ["Lung", "Heart", "Trachea"]

    pth_files = sorted(glob.glob("*epoch_*.pth"), key=extract_epoch)
    print(f"找到 {len(pth_files)} 个模型文件")

    results = []

    for pth_file in pth_files:
        epoch = extract_epoch(pth_file)
        print(f"\n{'='*50}")
        print(f"处理模型: {pth_file} (epoch {epoch})")

        model = get_model().to(device)
        model.load_state_dict(torch.load(pth_file, map_location=device))
        model.eval()

        total_dice = torch.zeros(3).to(device)
        total_iou = torch.zeros(3).to(device)
        steps = 0

        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc=f"Epoch {epoch}")
            for batch_data in progress_bar:
                inputs = batch_data["image"].to(device)
                labels = (batch_data["label"].to(device) >= 240).float()
                labels = labels[:, [2, 1, 0], :, :]

                logits = model(inputs)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                batch_dice, batch_iou = compute_metrics(preds, labels)

                total_dice += batch_dice
                total_iou += batch_iou
                steps += 1

        mean_dice = (total_dice / steps).cpu().numpy()
        mean_iou = (total_iou / steps).cpu().numpy()

        overall_dice = mean_dice.mean()
        overall_iou = mean_iou.mean()

        row = {
            "Epoch": epoch,
        }
        for i, name in enumerate(class_names):
            row[f"Dice_{name}"] = round(mean_dice[i], 4)
            row[f"IoU_{name}"] = round(mean_iou[i], 4)
        row["mDice"] = round(overall_dice, 4)
        row["mIoU"] = round(overall_iou, 4)

        results.append(row)
        print(f"  Lung:   Dice={mean_dice[0]:.4f}, IoU={mean_iou[0]:.4f}")
        print(f"  Heart:  Dice={mean_dice[1]:.4f}, IoU={mean_iou[1]:.4f}")
        print(f"  Trachea:Dice={mean_dice[2]:.4f}, IoU={mean_iou[2]:.4f}")
        print(f"  mDice={overall_dice:.4f}, mIoU={overall_iou:.4f}")

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("Epoch").reset_index(drop=True)
    result_df.to_csv("test_results.csv", index=False)
    print(f"\n结果已保存至 test_results.csv")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
