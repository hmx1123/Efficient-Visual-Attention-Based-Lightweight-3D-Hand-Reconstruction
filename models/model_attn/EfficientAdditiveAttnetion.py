import torch
import torch.nn as nn
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
