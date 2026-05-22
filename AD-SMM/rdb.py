
import torch
import torch.nn as nn


class RDB_Conv(nn.Module):
    def __init__(self, in_channels, grow_rate, ksize=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, grow_rate, ksize, padding=(ksize - 1) // 2, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    """
    Residual Dense Block: 用于学习近端映射.
    输入输出通道数相同 (matrix_channels).
    """
    def __init__(self, matrix_channels=1, grow_rate0=64, grow_rate=32, n_conv_layers=6):
        super().__init__()
        G0 = grow_rate0
        G = grow_rate
        C = n_conv_layers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)
        self.in_conv = nn.Conv2d(matrix_channels, G0, 3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(G0, matrix_channels, 3, stride=1, padding=1)

    def forward(self, x):
        f = self.in_conv(x)
        f = self.LFF(self.convs(f)) + f
        return self.out_conv(f)
