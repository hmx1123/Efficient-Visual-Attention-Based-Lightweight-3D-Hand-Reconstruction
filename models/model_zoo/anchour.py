import torch
import torch.nn as nn
from einops import rearrange, repeat

def get_relative_depth_anchour(k, map_size=64):
    range_arr = torch.arange(map_size, dtype=torch.float32, device=k.device) / map_size  # (0, 1)
    Y_map = range_arr.reshape(1, 1, 1, map_size, 1).repeat(1, 1, map_size, 1, map_size)
    X_map = range_arr.reshape(1, 1, 1, 1, map_size).repeat(1, 1, map_size, map_size, 1)
    Z_map = torch.pow(range_arr, k)
    Z_map = Z_map.reshape(1, 1, map_size, 1, 1).repeat(1, 1, 1, map_size, map_size)
    return torch.cat([Z_map, Y_map, X_map], dim=1)  # 1, 3, 8, 8, 8

def get_2d_anchors(k=1.0, map_size=8, scales=[1.0], ratios=[1.0]):
    """
    生成符合锚点原理的多尺度+多比例2D锚点网格（形状为 [batch=1, num_anchors*4, H, W]）
    
    Args:
        k (float): 控制坐标分布的非线性参数（k=1为线性）
        map_size (int): 网格分辨率（默认8x8）
        scales (list): 尺度列表（如[0.5, 1.0, 2.0]）
        ratios (list): 长宽比列表（如[0.5, 1.0, 2.0]）
    
    Returns:
        anchors (torch.Tensor): 形状 (1, num_anchors*4, H, W)
    """
    # 生成基础网格中心坐标 [0.5/map_size, 1.5/map_size, ..., (map_size-0.5)/map_size]
    range_arr = (torch.arange(map_size, dtype=torch.float32) + 0.5) / map_size
    
    # 创建Y和X坐标（形状统一为 [1,1,H,W]）
    Y = torch.pow(range_arr, k).reshape(1, 1, map_size, 1).repeat(1, 1, 1, map_size)  # (1,1,H,W)
    X = torch.pow(range_arr, k).reshape(1, 1, 1, map_size).repeat(1, 1, map_size, 1)  # (1,1,H,W)
    
    anchors = []
    for scale in scales:
        for ratio in ratios:
            # 计算当前尺度和比例的宽高
            w = scale * torch.sqrt(torch.tensor(ratio))
            h = scale / torch.sqrt(torch.tensor(ratio))
            
            # 生成当前锚点参数（形状必须与Y/X一致）
            cx = X  # 中心点X坐标 (1,1,H,W)
            cy = Y  # 中心点Y坐标 (1,1,H,W)
            w_tensor = torch.full_like(X, w)  # 宽度 (1,1,H,W)
            h_tensor = torch.full_like(Y, h)  # 高度 (1,1,H,W)
            
            # 合并当前锚点的(cx, cy, w, h)
            anchor = torch.cat([cx, cy, w_tensor, h_tensor], dim=1)  # (1,4,H,W)
            anchors.append(anchor)
    
    # 合并所有锚点
    return torch.cat(anchors, dim=1)  # (1, num_anchors*4, H, W)

if __name__ =="__main__":
    # 测试
    exponent = nn.Parameter(torch.tensor(3, dtype=torch.float32))
    exponent = torch.clamp(exponent, 1, 20)
    relative_depth_anchour = get_relative_depth_anchour(exponent)
    cam_anchour_maps = repeat(relative_depth_anchour, 'n c d h w -> (b n) c d h w', b=1)
    print(cam_anchour_maps.shape)
          
    anchors = get_2d_anchors(k=1.0, map_size=8, scales=[0.5, 1.0, 2.0], ratios=[0.5, 1.0, 2.0])
    print(anchors.shape)  # 输出: torch.Size([1, 3*3*4, 8, 8]) -> 3 scales * 3 ratios * 4 params
