from models.position_embedding import build_position_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F

from .EfficientAdditiveAttention import (
    EfficientAdditiveAttnetion,
    EfficientSeparableAttention,
)
from .self_attn import SelfAttn
from models.modules.InvertedResidual import InvertedResidual, DepthWiseSeparable

from torch import Tensor


def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(layer.weight.data)
    elif classname.find("Linear") != -1:
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)


class MLP_res_block(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, in_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        x = self.fc2(self.dropout1(F.relu(self.fc1(x))))
        return self.dropout2(x)

    def forward(self, x):
        x = x + self._ff_block(self.layer_norm(x))
        return x


def remap_uv(feat: Tensor, uv_coord: Tensor) -> Tensor:
    """
    args:
        feat; [N, C, H, W]
        uv_coord: [N, J, 2] range ~ [0, 1]
    return:
        select_feat: [N, J, C]
    """
    uv_coord = torch.clamp((uv_coord), -1, 1)  # [N, J, 2], range ~ [-1, 1]
    uv_coord = uv_coord.unsqueeze(2)  # [N, J, 1, 2]
    select_feat = F.grid_sample(feat, uv_coord, align_corners=True)  # [N, C, J, 1]
    select_feat = select_feat.permute((0, 2, 1, 3))
    select_feat = select_feat[:, :, :, 0]
    return select_feat


class img_feat_to_grid(nn.Module):
    def __init__(
        self, img_size, img_f_dim, grid_size, grid_f_dim, n_heads=4, dropout=0.01
    ):
        super().__init__()
        self.img_f_dim = img_f_dim
        self.img_size = img_size
        self.grid_f_dim = grid_f_dim
        self.grid_size = grid_size
        self.position_embeddings = nn.Embedding(grid_size * grid_size, grid_f_dim)

        patch_size = img_size // grid_size
        self.proj = DepthWiseSeparable(
            img_f_dim,
            grid_f_dim,
            kernel=patch_size,
            stride=patch_size,
            padding=0,
            e=0.25,
        )

    def forward(self, img):
        bs = img.shape[0]
        assert img.shape[1] == self.img_f_dim
        assert img.shape[2] == self.img_size
        assert img.shape[3] == self.img_size

        grid_feat = F.relu(self.proj(img))
        grid_feat = grid_feat.view(bs, self.grid_f_dim, -1).transpose(-1, -2)

        return grid_feat


class img_attn(nn.Module):
    def __init__(
        self,
        verts_f_dim,
        img_f_dim,
        n_heads=4,
        d_q=None,
        d_v=None,
        dropout=0.1,
        atten_tpye="eaa",
    ):
        super().__init__()
        self.img_f_dim = img_f_dim
        self.verts_f_dim = verts_f_dim

        self.fc = nn.Linear(img_f_dim, verts_f_dim)
        if atten_tpye == "eaa":
            self.Attn = EfficientAdditiveAttnetion(
                in_dims=verts_f_dim, num_heads=n_heads, dropout=dropout
            )
        elif atten_tpye == "esa":
            self.Attn = EfficientSeparableAttention(
                dim=f_dim, num_heads=n_heads, dropout=dropout
            )
        elif atten_tpye == "atten":
            self.Attn = SelfAttn(
                verts_f_dim, n_heads=n_heads, hid_dim=verts_f_dim, dropout=dropout
            )

    def forward(self, verts_f, img_f):
        assert verts_f.shape[2] == self.verts_f_dim
        # assert img_f.shape[2] == self.img_f_dim
        # assert verts_f.shape[0] == img_f.shape[0]

        img_f = self.fc(img_f)

        x = torch.cat([verts_f, img_f], dim=1)
        x = self.Attn(x)
        verts_f = verts_f + x[:, : verts_f.shape[1]]

        return verts_f


class img_ex(nn.Module):
    def __init__(
        self,
        img_size,
        img_f_dim,
        grid_size,
        grid_f_dim,
        verts_f_dim,
        n_heads=4,
        dropout=0.01,
        atten_tpye="eaa",
    ):
        super().__init__()
        self.verts_f_dim = verts_f_dim
        # self.encoder = img_feat_to_grid(img_size, img_f_dim, grid_size, grid_f_dim, n_heads, dropout)
        self.attn = img_attn(
            verts_f_dim,
            grid_f_dim,
            n_heads=n_heads,
            dropout=dropout,
            atten_tpye=atten_tpye,
        )

        for m in self.modules():
            weights_init(m)

    def forward(self, grid_feat, verts_f):
        assert verts_f.shape[2] == self.verts_f_dim
        # grid_feat = self.encoder(img)
        verts_f = self.attn(verts_f, grid_feat)
        return verts_f
