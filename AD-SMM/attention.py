
import torch
import torch.nn as nn


class SpatialAttentionModule(nn.Module):
    """
    生成注意力图 F_att(X), 输出与输入同尺寸.
    输入: (B, C, H, W) 的图像矩阵
    输出: (B, 1, H, W) 的注意力权重图, 值域 [0,1]
    """
    def __init__(self, in_channels=1):
        super().__init__()
        mid_channels = 32
        self.att = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.att(x)
