import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, in_dims=512, num_heads=2, dropout=0.):
        super().__init__()

        token_dim = in_dims // num_heads
        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        # self.final = nn.Linear(token_dim * num_heads, in_dims)
        self.drop_path = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.position_emb = nn.Parameter(torch.zeros(in_dims))

    def forward(self, x):
        x = x + self.position_emb
        
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1)  # BxNxD
        key = torch.nn.functional.normalize(key, dim=-1)  # BxNxD

        query_weight = query @ self.w_g  # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor  # BxNx1

        A = torch.nn.functional.normalize(A, dim=1)  # BxNx1

        G = torch.sum(A * query, dim=1)  # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        )  # BxNxD

        G = self.drop_path(G)

        out = self.Proj(G * key) + query  # BxNxD

        # out = self.final(out)  # BxNxD

        return out

class EfficientSeparableAttention(nn.Module):
    """
    更高效的Separable Self-attention实现
    通过参数共享减少计算量
    """
    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        assert self.head_dim * num_heads == out_dim, "out_dim必须能被num_heads整除"
        
        # 共享的QKV投影
        self.qkv_proj = nn.Linear(in_dim, 3 * out_dim)
        self.output_proj = nn.Linear(out_dim, out_dim)
        
        # 单独的通道注意力参数
        self.channel_attn = nn.Sequential(
            nn.Linear(out_dim, out_dim // 4),
            nn.ReLU(),
            nn.Linear(out_dim // 4, out_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 生成QKV
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # 每个都是 (b, n, h, d_h)
        
        # 转置用于空间注意力
        q = q.transpose(1, 2)  # (b, h, n, d_h)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 空间注意力
        spatial_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        spatial_attn = F.softmax(spatial_scores, dim=-1)
        spatial_attn = self.dropout(spatial_attn)
        
        spatial_out = torch.matmul(spatial_attn, v)  # (b, h, n, d_h)
        spatial_out = spatial_out.transpose(1, 2).reshape(batch_size, seq_len, self.out_dim)
        
        # 通道注意力
        channel_weights = self.channel_attn(x)  # (b, n, out_dim)
        channel_out = v.transpose(1, 2).reshape(batch_size, seq_len, self.out_dim) * channel_weights
        
        # 融合
        combined = spatial_out + channel_out
        
        # 输出投影
        output = self.output_proj(combined)
        output = self.dropout(output)
        
        if self.in_dim == self.out_dim:
            output = output + x
            
        output = self.layer_norm(output)
        
        return output
    
if __name__=="__main__":
    net=EfficientSeparableAttention(in_dim=128,out_dim=128,num_heads=4,dropout=0.07)
    x=torch.randn(1,64,128)
    y=net(x)
        