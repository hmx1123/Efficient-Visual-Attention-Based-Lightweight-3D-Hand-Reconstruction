import torch
import torch.nn as nn
import torch.nn.functional as F

def heatmap_to_coords_expectation(heatmaps):
    """
    通过加权平均法从高斯热图提取坐标点集合（归一化到[-1,1]坐标系）
    Args:
        heatmaps (torch.Tensor): 高斯热图，形状为 (B, C, H, W)
    Returns:
        torch.Tensor: 二维点坐标集合，形状为 (B, C, 2)，数值范围为[-1, 1]
    """
    device = heatmaps.device
    B, C, H, W = heatmaps.shape

    # 生成坐标网格 (H, W)
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    xx = xx.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    yy = yy.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # 计算加权坐标
    eps = 1e-6
    heatmap_sum = heatmaps.sum(dim=(2, 3), keepdim=True) + eps
    weighted_x = (heatmaps * xx).sum(dim=(2, 3)) / heatmap_sum.squeeze(-1).squeeze(-1)
    weighted_y = (heatmaps * yy).sum(dim=(2, 3)) / heatmap_sum.squeeze(-1).squeeze(-1)

    # 坐标归一化到[-1, 1]范围
    normalized_x = (weighted_x / (W - 1)) * 2 - 1  # 映射到[-1,1]
    normalized_y = (weighted_y / (H - 1)) * 2 - 1

    return torch.stack([normalized_x, normalized_y], dim=2)


def sample_features(coords: torch.Tensor, feat: torch.Tensor, align_corners: bool = False) -> torch.Tensor:
    """
    通过二维坐标从视觉特征中采样，生成指定形状的特征张量。

    参数:
        coords: Tensor(b, n, 2)    - 归一化到[-1,1]的坐标网格
        feat: Tensor(b, c, h, w)   - 视觉特征图
        align_corners: bool        - 坐标对齐方式（需与归一化方式一致）

    返回:
        sampled_feat: Tensor(b, n, c) - 采样后的特征
    """
    # 调整坐标维度并采样
    grid = coords.unsqueeze(1)  # (b, 1, n, 2)
    sampled = F.grid_sample(feat, grid, align_corners=align_corners)  # 输出(b, c, 1, n)

    # 调整形状得到(b, n, c)
    sampled_feat = sampled.squeeze(2).permute(0, 2, 1)
    return sampled_feat

if __name__ == "__main__":
    map = torch.randn(64, 128, 64, 64)
    coord = heatmap_to_coords_expectation(map)
    print(coord.shape)
    map=torch.randn(64,256,64,64)
    feat=sample_features(coord,map)
    print(feat.shape)