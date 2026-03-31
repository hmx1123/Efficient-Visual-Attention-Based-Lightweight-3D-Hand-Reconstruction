import torch
import torch.nn as nn
import torch.nn.functional as F

from .EfficientAdditiveAttnetion import EfficientAdditiveAttnetion
from .self_attn import SelfAttn


def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
    elif classname.find('Linear') != -1:
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


class inter_attn(nn.Module):
    def __init__(self, f_dim, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        super().__init__()
        self.f_dim = f_dim

        self.L_self_attn_layer = EfficientAdditiveAttnetion(in_dims=f_dim, num_heads=n_heads, dropout=dropout)
        self.R_self_attn_layer = EfficientAdditiveAttnetion(in_dims=f_dim, num_heads=n_heads, dropout=dropout)
        self.inter_attn_layer = EfficientAdditiveAttnetion(in_dims=f_dim, num_heads=n_heads, dropout=dropout)

        # self.L_self_attn_layer = SelfAttn(
        #     f_dim, n_heads=n_heads, hid_dim=f_dim, dropout=dropout)
        # self.R_self_attn_layer = SelfAttn(
        #     f_dim, n_heads=n_heads, hid_dim=f_dim, dropout=dropout)
        # self.inter_attn_layer = nn.Sequential(
        #     SelfAttn(f_dim, n_heads=n_heads, hid_dim=f_dim, dropout=dropout)
        # )
        
        # self.L_self_attn_layer = nn.Sequential(
        #     EfficientAdditiveAttnetion(in_dims=f_dim, num_heads=n_heads, dropout=dropout),
        #     EfficientAdditiveAttnetion(in_dims=f_dim, num_heads=n_heads, dropout=dropout),
        # )
        # self.R_self_attn_layer = nn.Sequential(
        #     EfficientAdditiveAttnetion(in_dims=f_dim, num_heads=n_heads, dropout=dropout),
        #     EfficientAdditiveAttnetion(in_dims=f_dim, num_heads=n_heads, dropout=dropout),
        # )
        # self.inter_attn_layer = nn.Sequential(
        #     EfficientAdditiveAttnetion(in_dims=f_dim, num_heads=n_heads, dropout=dropout),
        #     EfficientAdditiveAttnetion(in_dims=f_dim, num_heads=n_heads, dropout=dropout),
        # )

        for m in self.modules():
            weights_init(m)

    def forward(self, Lf, Rf):
        BS, V, fdim = Lf.shape
        assert fdim == self.f_dim
        BS, V, fdim = Rf.shape
        assert fdim == self.f_dim

        Lf_res = Lf
        Rf_res = Rf

        feat = torch.cat((Lf, Rf), dim=1)
        feat = self.inter_attn_layer(feat)
        Lf = feat[:, :V]
        Rf = feat[:, V:]

        Lf = self.L_self_attn_layer(Lf) + Lf_res
        Rf = self.R_self_attn_layer(Rf) + Rf_res

        return Lf, Rf
