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
