import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from einops.layers.torch import Rearrange

from dataset.dataset_utils import IMG_SIZE
from utils.utils import projection_batch
from models.manolayer import ManoLayer
from models.modules import (
    weights_init,
    sample_features,
    heatmap_to_coords_expectation,
)
from models.modules.InvertedResidual import DepthWiseSeparable, DepthWiseSeparableRes

from utils.config import load_cfg

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import torchvision.models as models


class ResNetSimple_decoder(nn.Module):
    def __init__(
        self,
        expansion=4,
        fDim=[256, 256, 256, 256],
        direction=["flat", "up", "up", "up"],
        out_dim=3,
        conv_type="hpds",
        hid_layer=[2, 3, 4, 5],
    ):
        super(ResNetSimple_decoder, self).__init__()
        self.models = nn.ModuleList()
        fDim = [512 * expansion] + fDim
        for i in range(len(direction)):
            self.models.append(
                self.make_layer(
                    fDim[i],
                    fDim[i + 1],
                    direction[i],
                    kernel_size=3,
                    hid_layer=hid_layer[i],
                    padding=1,
                    conv_type=conv_type,
                )
            )

        self.final_layer = nn.Sequential(
            nn.Conv2d(fDim[-1], out_dim, 1), nn.BatchNorm2d(out_dim)
        )

    def make_layer(
        self,
        in_dim,
        out_dim,
        direction,
        kernel_size=3,
        hid_layer=2,
        padding=1,
        conv_type="hpds",
    ):
        assert direction in ["flat", "up"]

        layers = []
        if direction == "up":
            layers.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            )
        if conv_type == "hpds":
            layers.append(
                DepthWiseSeparableRes(
                    in_dim, out_dim, hid_layer=hid_layer, kernel=kernel_size, e=0.25
                )
            )
        elif conv_type == "conv":
            layers.append(
                nn.Conv2d(
                    in_dim,
                    out_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm2d(out_dim))

        return nn.Sequential(*layers)

    def forward(self, x):
        fmaps = []
        for i in range(len(self.models)):
            x = self.models[i](x)
            fmaps.append(x)
        output = self.final_layer(x)
        return output, fmaps


class ResNetSimple(nn.Module):
    def __init__(
        self,
        model_type="resnet50",
        pretrained=False,
        fmapDim=[256, 256, 256, 256],
        handNum=2,
        heatmapDim=21,
        conv_type="eaa",
    ):
        super(ResNetSimple, self).__init__()
        assert model_type in [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ]
        if model_type == "resnet18":
            self.resnet = resnet18(weights=models.ResNet518_Weights.IMAGENET1K_V1)
            self.expansion = 1
        elif model_type == "resnet34":
            self.resnet = resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self.expansion = 1
        elif model_type == "resnet50":
            self.resnet = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.expansion = 4
        elif model_type == "resnet101":
            self.resnet = resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            self.expansion = 4
        elif model_type == "resnet152":
            self.resnet = resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
            self.expansion = 4

        self.hms_decoder = ResNetSimple_decoder(
            expansion=self.expansion,
            fDim=fmapDim,
            direction=["flat", "up", "up", "up"],
            out_dim=heatmapDim * handNum,
            conv_type=conv_type,
            hid_layer=[2, 3, 4, 5],
        )

        self.dp_decoder = ResNetSimple_decoder(
            expansion=self.expansion,
            fDim=fmapDim,
            direction=["flat", "up", "up", "up"],
            out_dim=3 * handNum,
            conv_type=conv_type,
            hid_layer=[2, 3, 4, 5],
        )

        self.mask_decoder = ResNetSimple_decoder(
            expansion=self.expansion,
            fDim=fmapDim,
            direction=["flat", "up", "up", "up"],
            out_dim=handNum,
            conv_type=conv_type,
            hid_layer=[2, 3, 4, 5],
        )
        self.handNum = handNum

        for m in self.modules():
            weights_init(m)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x4 = self.resnet.layer1(x)
        x3 = self.resnet.layer2(x4)
        x2 = self.resnet.layer3(x3)
        x1 = self.resnet.layer4(x2)

        img_fmaps = [x1, x2, x3, x4]

        hms, hms_fmaps = self.hms_decoder(x1)
        dp, dp_fmaps = self.dp_decoder(x1)
        mask, mask_fmaps = self.mask_decoder(x1)

        return hms, mask, dp, img_fmaps, hms_fmaps, mask_fmaps, dp_fmaps


class resnet_mid(nn.Module):
    def __init__(
        self,
        model_type="resnet50",
        img_size=[8, 16, 32, 64],
        in_fmapDim=[256, 256, 256, 256],
        out_fmapDim=[256, 256, 256, 256],
    ):
        super(resnet_mid, self).__init__()
        assert model_type in [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ]
        if model_type == "resnet18" or model_type == "resnet34":
            self.expansion = 1
        elif (
            model_type == "resnet50"
            or model_type == "resnet101"
            or model_type == "resnet152"
        ):
            self.expansion = 4

        self.dp_fmaps_dim = in_fmapDim
        self.hms_fmaps_dim = in_fmapDim
        self.mask_fmaps_dim = in_fmapDim

        self.img_to_patch_l = nn.ModuleList()
        self.img_to_patch_r = nn.ModuleList()

        # for i in range(len(out_fmapDim)):
        #     inDim = (
        #         self.dp_fmaps_dim[i] + self.hms_fmaps_dim[i] + self.mask_fmaps_dim[i]
        #     ) // 2
        #     self.img_to_patch_l.append(img_to_patch(
        #         image_size=img_size[i],
        #         grid_size=8,
        #         in_fmapDim=inDim,
        #         out_fmapDim=out_fmapDim[i],
        #     ))
        #     self.img_to_patch_r.append(img_to_patch(
        #         image_size=img_size[i],
        #         grid_size=8,
        #         in_fmapDim=inDim,
        #         out_fmapDim=out_fmapDim[i],
        #     ))


        self.to_patch_embedding = nn.ModuleList()
        
        self.convs_l = nn.ModuleList()
        self.convs_r = nn.ModuleList()
        for i in range(len(out_fmapDim)):
            inDim = (
                self.dp_fmaps_dim[i] + self.hms_fmaps_dim[i] + self.mask_fmaps_dim[i]
            ) // 2

            patch_height = img_size[i] // 8
            patch_width = img_size[i] // 8
            patch_dim = inDim * patch_height * patch_width
            # self.convs_l.append(
            #     nn.Sequential(
            #         nn.Conv2d(inDim, out_fmapDim[i], 1), nn.BatchNorm2d(out_fmapDim[i])
            #     )
            # )
            # self.convs_r.append(
            #     nn.Sequential(
            #         nn.Conv2d(inDim, out_fmapDim[i], 1), nn.BatchNorm2d(out_fmapDim[i])
            #     )
            # )
            self.to_patch_embedding.append(
                nn.Sequential(
                    Rearrange(
                        "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                        p1=patch_height,
                        p2=patch_width,
                    ),
                    nn.LayerNorm(patch_dim),
                    nn.Linear(patch_dim, out_fmapDim[i]),
                    nn.LayerNorm(out_fmapDim[i]),
                )
            )

        self.inDim = inDim

        self.global_feature_dim = 512 * self.expansion
        self.fmaps_dim = out_fmapDim

        for m in self.modules():
            weights_init(m)

    def get_info(self):
        return {
            "global_feature_dim": self.global_feature_dim,
            "fmaps_dim": self.fmaps_dim,
        }

    def forward(self, img_fmaps, hms_fmaps, mask_fmaps, dp_fmaps, mask):
        global_feature = img_fmaps[0]
        # coord = heatmap_to_coords_expectation(mask_fmaps[-1])

        fmaps_l, fmaps_r = [], []
        grid_fmaps_l, grid_fmaps_r = [], []

        for i, to_patch_embedding in enumerate(self.to_patch_embedding):
            
            t = F.interpolate(mask.detach().clone(), size=(hms_fmaps[i].size(2), hms_fmaps[i].size(3)), mode='bilinear')

            # 定义切片辅助函数
            def get_slice(dim, is_left):
                return slice(0, dim // 2) if is_left else slice(dim // 2, None)

            # 处理左分支
            slices = (
                get_slice(self.hms_fmaps_dim[i], True),
                get_slice(self.dp_fmaps_dim[i], True),
                get_slice(self.mask_fmaps_dim[i], True),
            )
            x_l = torch.cat(
                [
                    hms_fmaps[i][:, slices[0]],
                    dp_fmaps[i][:, slices[1]],
                    mask_fmaps[i][:, slices[2]],
                ],
                dim=1,
            )
            x_l = x_l * t[:, :1]  # 使用掩码对左分支特征进行加权
            # grid_fmaps_l.append(self.img_to_patch_l[i](x_l))
            fmaps_l.append(x_l)
            grid_fmaps_l.append(to_patch_embedding(x_l))
            # grid_fmaps_l.append(sample_features(coord, fmaps_l[-1]))

            # 处理右分支
            slices = (
                get_slice(self.hms_fmaps_dim[i], False),
                get_slice(self.dp_fmaps_dim[i], False),
                get_slice(self.mask_fmaps_dim[i], False),
            )
            x_r = torch.cat(
                [
                    hms_fmaps[i][:, slices[0]],
                    dp_fmaps[i][:, slices[1]],
                    mask_fmaps[i][:, slices[2]],
                ],
                dim=1,
            )
            x_r = x_r * t[:, 1:]  # 使用掩码对右分支特征进行加权
            # grid_fmaps_r.append(self.img_to_patch_r[i](x_r))
            fmaps_r.append(x_r)
            grid_fmaps_r.append(to_patch_embedding(x_r))
            # grid_fmaps_r.append(sample_features(coord, fmaps_r[-1]))

        return global_feature, fmaps_l, fmaps_r, grid_fmaps_l, grid_fmaps_r


class img_to_patch(nn.Module):
    def __init__(
        self,
        image_size=64,
        grid_size=8,
        in_fmapDim=256,
        out_fmapDim=256,
    ):
        super(img_to_patch, self).__init__()

        self.in_fmapDim = in_fmapDim
        patch_size = image_size // grid_size

        patch_dim = in_fmapDim * patch_size * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, out_fmapDim),
            nn.LayerNorm(out_fmapDim),
        )

    def forward(self, img):
        return self.to_patch_embedding(img)


def load_encoder(cfg):
    if cfg.MODEL.ENCODER_TYPE.find("resnet") != -1:
        encoder = ResNetSimple(
            model_type=cfg.MODEL.ENCODER_TYPE,
            pretrained=True,
            fmapDim=[128, 128, 128, 128],
            handNum=2,
            heatmapDim=21,
            conv_type=cfg.MODEL.CONV_TYPE,
        )
        mid_model = resnet_mid(
            model_type=cfg.MODEL.ENCODER_TYPE,
            img_size=[8, 16, 32, 64],
            in_fmapDim=[128, 128, 128, 128],
            out_fmapDim=cfg.MODEL.DECONV_DIMS,
        )

    return encoder, mid_model
