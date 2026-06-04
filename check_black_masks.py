import os

import numpy as np
from PIL import Image
from tqdm import tqdm

masks_dir = "./dataset/masks/masks"
mask_files = sorted(os.listdir(masks_dir))

black_count = 0
total = len(mask_files)

for fname in tqdm(mask_files):
    path = os.path.join(masks_dir, fname)
    img = np.array(Image.open(path))
    if img.sum() == 0:
        black_count += 1
        print(f"  纯黑: {fname} (shape={img.shape})")

print(f"\n总计: {total} 张, 纯黑: {black_count} 张 ({100*black_count/total:.1f}%)")
