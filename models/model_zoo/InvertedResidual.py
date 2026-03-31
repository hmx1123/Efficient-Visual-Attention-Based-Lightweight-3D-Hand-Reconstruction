import torch
import torch.nn as nn


class DepthWiseSeparable(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=1, padding=1, e=1):
        super().__init__()

        hid_dim = int(in_dim * e)
        self.pw1 = nn.Conv2d(in_dim, hid_dim, 1)  # kernel size = 1
        self.norm1 = nn.BatchNorm2d(hid_dim)
        self.act1 = nn.GELU()

        self.dw = nn.Conv2d(
            hid_dim,
            hid_dim,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=hid_dim,
        )  # kernel size = 3
        self.norm2 = nn.BatchNorm2d(hid_dim)
        self.act2 = nn.GELU()

        self.pw2 = nn.Conv2d(hid_dim, out_dim, 1)
        self.norm3 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.pw1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.dw(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = self.pw2(x)
        x = self.norm3(x)
        return x


class DepthWiseSeparableRes(nn.Module):
    def __init__(self, in_dim, out_dim, hid_layer=1, kernel=3, e=1):
        super().__init__()
        padding = kernel // 2
        stride = 1

        hid_dim = int(in_dim * e)
        self.pw1 = nn.Conv2d(in_dim, hid_dim, 1)  # kernel size = 1
        self.norm1 = nn.BatchNorm2d(hid_dim)
        self.act1 = nn.GELU()

        self.resconv = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.resnorm = nn.BatchNorm2d(out_dim)

        self.dw = nn.ModuleList()
        for i in range(hid_layer):
            self.dw.append(nn.Sequential(
                nn.Conv2d(
                    hid_dim,
                    hid_dim,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    groups=hid_dim,
                ),  # kernel size = 3
                nn.BatchNorm2d(hid_dim),
                nn.GELU()
            ))

            self.pw2 = nn.Conv2d(hid_dim, out_dim, 1)
            self.norm3 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        res_x = self.resconv(x)
        res_x = self.resnorm(res_x)
        res_x = self.act1(res_x)
        x = self.pw1(x)
        x = self.norm1(x)
        x = self.act1(x)

        for layer in self.dw:
            x = x + layer(x)

        x = self.pw2(x)
        x = self.norm3(x)
        x = x + res_x
        return x


class InvertedResidual(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            kernel,
            stride,
            padding,
            expansion_ratio=1,
            drop_path=0.0,
            use_layer_scale=False,
            layer_scale_init_value=1e-5,
    ):
        super().__init__()

        self.dws = DepthWiseSeparable(
            in_dim=in_dim,
            out_dim=out_dim,
            kernel=kernel,
            stride=stride,
            padding=padding,
            e=expansion_ratio,
        )
        self.conv = nn.Conv2d(in_dim, out_dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.dws(x))
        else:
            x = self.conv(x) + self.drop_path(self.dws(x))
        return x
