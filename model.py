import segmentation_models_pytorch as smp
from monai.losses.dice import DiceFocalLoss
from monai.networks.nets.swin_unetr import SwinUNETR
from torch.optim import AdamW


def get_model():
    return SwinUNETR(
        in_channels=1,  # 单通道灰度图
        out_channels=3,  # 输出 3 个分类
        feature_size=48,  # 基础特征维度大小
        spatial_dims=2,  # 必须指定为2，因为默认是3D
        use_checkpoint=True,  # 梯度检查点，用一点点计算时间换取极大的显存节约
        use_v2=True,  # 使用SwinUNETR v2版本，性能更好
    )


def get_optimizer(model):
    return AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)


def get_lossfunc():
    return DiceFocalLoss(to_onehot_y=False, sigmoid=True, include_background=True)


def get_unet():
    return smp.UnetPlusPlus(
        encoder_name="efficientnet-b2",  # 强力且轻量级的骨干网络
        encoder_weights="imagenet",  # 关键：加载 ImageNet 预训练权重，拒绝从零盲训
        in_channels=1,  # 胸部 CT 单通道灰度图
        classes=3,  # 同时预测 3 个前景类别（肺、心脏、气管）
        activation=None,  # 损失函数内部自带 Sigmoid，此处保持 None
    )
