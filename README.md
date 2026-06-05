# Chest CT Segmentation

基于 **Swin-UNETR v2** 的胸部CT图像多器官分割项目，实现对肺（Lung）、心脏（Heart）、气管（Trachea）三个关键器官的自动分割。

## 项目概述

本项目利用深度学习技术对胸部CT图像进行多器官语义分割，采用 MONAI 框架构建训练与推理流程。模型基于 Swin Transformer 编码器与 U-Net 解码器的混合架构（Swin-UNETR v2），结合 DiceFocalLoss 损失函数与余弦退火学习率调度，在胸部CT数据集上取得了优异的分割性能。

### 最佳实验结果

| 器官 | Dice 系数 | IoU |
|------|-----------|-----|
| 肺 (Lung) | 0.9664 | 0.9480 |
| 心脏 (Heart) | 0.8995 | 0.8860 |
| 气管 (Trachea) | 0.9455 | 0.9242 |
| **平均 (mDice/mIoU)** | **0.9371** | **0.9194** |

## 核心功能

- **模型训练**：支持 Swin-UNETR v2 的完整训练流程，含数据增强、梯度检查点、余弦退火调度
- **模型评估**：批量评估多个 checkpoint 的 Dice/IoU 指标，自动生成 CSV 报告
- **单模型测试**：交互式选择 checkpoint，输出各器官分割指标
- **可视化推理**：生成原图、真实掩码与预测结果的对比图，支持逐器官独立可视化
- **实验报告**：基于 LaTeX 的完整学术实验报告（见 `docs/main.tex`）

## 项目结构

```
chest-ct-segmentation/
├── model.py              # 模型定义（Swin-UNETR v2）、损失函数、优化器
├── train.py              # 训练入口脚本
├── test_all.py           # 批量评估所有 checkpoint
├── test_pth.py           # 单模型交互式测试
├── views.py              # 可视化推理结果
├── check_black_masks.py  # 检查数据集中纯黑掩码
├── exmaple.py            # 基线示例（U-Net + EfficientNet-B2）
├── download.sh           # 数据集下载脚本
├── dataset/              # 数据目录（需下载）
│   ├── train.csv         # 图像-掩码配对表
│   └── images/ & masks/  # CT图像与分割掩码
├── train_1/              # 第一轮训练结果
│   ├── loss_per_epoch.csv
│   ├── test_results.csv
│   ├── test_all.py
│   ├── test_pth.py
│   └── views.py
├── train_2/              # 第二轮训练结果
│   └── ...
├── docs/
│   └── main.tex          # LaTeX 实验报告
└── pyproject.toml        # 项目依赖配置
```

## 环境要求

- **Python** >= 3.12
- **CUDA**（推荐，GPU 训练必需）
- **系统依赖**（可视化功能需要）：`fonts-wqy-zenhei`、`fonts-droid-fallback`

### Python 依赖

| 包 | 版本要求 |
|----|----------|
| monai | >= 1.5.2 |
| kagglehub | >= 1.0.1 |
| torch | >= 2.0 |
| pandas | - |
| scikit-learn | - |
| matplotlib | - |
| numpy | - |
| tqdm | - |
| Pillow | - |

## 安装步骤

1. **克隆项目**

```bash
git clone <repository-url>
cd chest-ct-segmentation
```

2. **安装依赖**（推荐使用 [uv](https://github.com/astral-sh/uv)）

```bash
uv sync
```

或使用 pip：

```bash
pip install monai>=1.5.2 kagglehub>=1.0.1
```

3. **下载数据集**

```bash
bash download.sh
```

该脚本会从 Kaggle 下载 [Chest CT Segmentation](https://www.kaggle.com/datasets/polomarco/chest-ct-segmentation) 数据集并解压到 `./dataset` 目录。

## 使用指南

### 训练

```bash
python train.py
```

训练配置：

| 参数 | 值 |
|------|----|
| 模型 | Swin-UNETR v2 (2D) |
| 特征维度 | 48 |
| 输入尺寸 | 256 × 256 |
| Batch Size | 16 |
| 训练轮数 | 200 |
| 优化器 | AdamW (lr=1e-4, weight_decay=1e-5) |
| 学习率调度 | CosineAnnealingLR (eta_min=1e-6) |
| 损失函数 | DiceFocalLoss |
| 数据划分 | 80% 训练 / 20% 验证 |
| 随机种子 | 520 |

模型每 10 个 epoch 保存一次 checkpoint，文件格式为 `swin_unetr_epoch_{N}.pth`。

### 批量评估

评估当前目录下所有 checkpoint 并生成 `test_results.csv`：

```bash
python test_all.py
```

### 单模型测试

交互式选择 checkpoint 进行评估：

```bash
python test_pth.py
```

程序会列出当前目录下所有 `.pth` 文件，输入编号即可选择。

### 可视化推理

生成原图、真实掩码与预测结果的对比图：

```bash
python views.py
```

输出目录 `views-{timestamp}/` 包含：
- `overview.png`：总体对比图（原图 / GT / 预测）
- `organ_0_Lung.png`：肺部分割对比
- `organ_1_Heart.png`：心脏分割对比
- `organ_2_Trachea.png`：气管分割对比

### 编译实验报告

```bash
cd docs
xelatex main.tex
xelatex main.tex  # 第二次运行以解析交叉引用
```

## 配置说明

### 模型配置

模型定义在 `model.py` 中，可修改以下参数：

```python
SwinUNETR(
    in_channels=1,       # 输入通道数（1=灰度CT）
    out_channels=3,      # 输出类别数（肺、心脏、气管）
    feature_size=48,     # 基础特征维度，可调为 24/48/96
    spatial_dims=2,      # 2D 分割
    use_checkpoint=True, # 梯度检查点，节省显存
    use_v2=True,         # 使用 v2 版本
)
```

### 数据增强

训练时的数据增强在 `train.py` 中配置：

| 增强方法 | 参数 | 概率 |
|----------|------|------|
| RandFlipd | 水平/垂直翻转 | 0.5 |
| RandAffined | 旋转±0.2rad, 平移±10%, 缩放±10% | 0.5 |
| RandRotate90d | 随机90°旋转 | 0.5 |
| RandGaussianNoised | std=0.02 | 0.15 |
| RandAdjustContrastd | gamma=[0.8, 1.2] | 0.15 |

### 掩码通道顺序

数据集掩码为 RGB 三通道图像，通道含义为 BGR 顺序（OpenCV 格式）。训练与测试代码中已做通道交换 `labels[:, [2, 1, 0], :, :]`，将 BGR 转换为 RGB 对应 `[肺, 心脏, 气管]`。

## 数据集

本项目使用 Kaggle 上的 [Chest CT Segmentation](https://www.kaggle.com/datasets/polomarco/chest-ct-segmentation) 数据集，包含胸部CT横断面图像及对应的三器官分割掩码。

- **图像格式**：JPG，原始尺寸 512 × 512
- **掩码格式**：JPG（RGB），像素值 240 以上为有效区域
- **分割类别**：肺（R通道）、心脏（G通道）、气管（B通道）
- **配对关系**：通过 `train.csv` 中的 `ImageId` 和 `MaskId` 关联

## 贡献规范

欢迎贡献代码！请遵循以下流程：

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m "Add your feature"`
4. 推送分支：`git push origin feature/your-feature`
5. 提交 Pull Request

### 代码规范

- 使用 Python 3.12+ 语法
- 遵循 PEP 8 代码风格
- 添加必要的中文注释说明关键逻辑

## 许可证

本项目仅供学习与研究使用。数据集版权归原作者所有，请遵循其对应的使用条款。

## 联系方式

如有问题或建议，请通过 GitHub Issues 提交。
