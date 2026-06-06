import glob
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import Unet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class LungsDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, df, phase):
        self.root_imgs_dir = imgs_dir
        self.root_masks_dir = masks_dir
        self.df = df
        self.augmentations = get_augmentations(phase)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "ImageId"]
        mask_name = self.df.loc[idx, "MaskId"]
        img_path = os.path.join(self.root_imgs_dir, img_name)
        mask_path = os.path.join(self.root_masks_dir, mask_name)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        mask[mask < 240] = 0
        mask[mask > 0] = 1
        augmented = self.augmentations(image=img, mask=mask.astype(np.float32))
        img = augmented["image"]
        mask = augmented["mask"].permute(2, 0, 1)
        return img, mask


def get_augmentations(phase, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    list_transforms = [Normalize(mean=mean, std=std, p=1), ToTensorV2()]
    return Compose(list_transforms)


def compute_metrics(preds, targets, smooth=1e-5):
    intersection = (preds * targets).sum(dim=(2, 3))
    preds_sum = preds.sum(dim=(2, 3))
    targets_sum = targets.sum(dim=(2, 3))
    union = preds_sum + targets_sum - intersection
    dice = (2.0 * intersection + smooth) / (preds_sum + targets_sum + smooth)
    iou = (intersection + smooth) / (union + smooth)
    return dice.mean(dim=0), iou.mean(dim=0)


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

    data_dir = "./dataset"
    images_dir = os.path.join(data_dir, "images", "images")
    masks_dir = os.path.join(data_dir, "masks", "masks")
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    _, test_df = train_test_split(df, test_size=0.2, random_state=69)
    test_df = test_df.reset_index(drop=True)
    print(f"test samples: {len(test_df)}")

    test_ds = LungsDataset(images_dir, masks_dir, test_df, "val")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = Unet("efficientnet-b2", encoder_weights=None, classes=3, activation=None).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    total_dice = torch.zeros(3).to(device)
    total_iou = torch.zeros(3).to(device)
    steps = 0
    class_names = ["lung", "heart", "trachea"]

    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="testing"):
            imgs, masks = imgs.to(device), masks.to(device)
            masks = (masks >= 0.5).float()
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            batch_dice, batch_iou = compute_metrics(preds, masks)
            total_dice += batch_dice
            total_iou += batch_iou
            steps += 1

    mean_dice = (total_dice / steps).cpu().numpy()
    mean_iou = (total_iou / steps).cpu().numpy()

    print("\n" + "=" * 50)
    print("test set results")
    print("=" * 50)
    for i in range(3):
        print(f"{class_names[i]}: dice={mean_dice[i]:.4f}, iou={mean_iou[i]:.4f}")
    print(f"mean dice: {mean_dice.mean():.4f}")
    print(f"mean iou:  {mean_iou.mean():.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
