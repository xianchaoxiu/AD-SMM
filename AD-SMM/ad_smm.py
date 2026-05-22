
import torch
import torch.nn as nn
from .attention import SpatialAttentionModule
#from attention import SpatialAttentionModule
from .rdb import RDB
#from rdb import RDB


class FeatureBlock(nn.Module):
    """特征提取展开块"""
    def __init__(self, use_rdb=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.use_rdb = use_rdb
        if use_rdb:
            self.rdb = RDB(matrix_channels=1, grow_rate0=32, grow_rate=16, n_conv_layers=4)

    def forward(self, F, X, A):
        """
        F: (B,1,H,W) 当前特征
        X: (B,1,H,W) 输入
        A: (B,1,H,W) 注意力图
        """
        alpha = torch.sigmoid(self.alpha)
        residual = A * (X - F)  # 注意力加权残差
        if self.use_rdb:
            F_new = self.rdb(F) + alpha * residual
        else:
            F_new = F + alpha * residual
        return F_new


class AGSMM_Net(nn.Module):
   
    def __init__(self, img_size=(64, 64), in_channels=1, K=6, use_rdb=True):
        super().__init__()
        self.K = K
        H, W = img_size

        # 注意力模块
        self.attention = SpatialAttentionModule(in_channels=in_channels)

        # K层特征提取展开块
        self.blocks = nn.ModuleList([
            FeatureBlock(use_rdb=use_rdb) for _ in range(K)
        ])

        # 全局判别矩阵W (SMM的核心)
        self.W = nn.Parameter(torch.randn(1, 1, H, W) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, X_batch, y_batch=None, return_attention=False):
        B = X_batch.shape[0]

        # 注意力图
        att_map = self.attention(X_batch)  # (B, 1, H, W)

        # 特征提取
        F = X_batch
        for k in range(self.K):
            F = self.blocks[k](F, X_batch, att_map)

        # 决策: <W, A * F> + b
        weighted_feat = att_map * F  # (B, 1, H, W)
        decision = (self.W * weighted_feat).sum(dim=(-3, -2, -1)) + self.b  # (B,)

        if return_attention:
            return decision, att_map, self.W
        return decision




class AGSVM_Net(nn.Module):

    def __init__(self, img_size=(64, 64), in_channels=1, K=6, use_rdb=True):
        super().__init__()
        self.K = K
        H, W = img_size
    
        self.feat_dim = in_channels * H * W

        self.attention = SpatialAttentionModule(in_channels=in_channels)

        self.blocks = nn.ModuleList([
            FeatureBlock(use_rdb=use_rdb) for _ in range(K)
        ])

     
        self.w = nn.Parameter(torch.randn(self.feat_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))
        
     

    def forward(self, X_batch, y_batch=None, return_attention=False):
        B = X_batch.shape[0]
        att_map = self.attention(X_batch)  # (B, 1, H, W)
        F = X_batch
        for k in range(self.K):
            F = self.blocks[k](F, X_batch, att_map)
        weighted_feat = att_map * F  # (B, 1, H, W)
        flat_feat = weighted_feat.view(B, -1) 
    
        decision = torch.matmul(flat_feat, self.w) + self.b  # 输出形状: (B,)

        if return_attention:
            return decision, att_map, self.w.view(1, 1, X_batch.shape[-2], X_batch.shape[-1])
        return decision