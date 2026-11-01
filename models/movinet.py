"""
MoViNet: Mobile Video Networks for Efficient Video Recognition

Code inspired and modified from:
- https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html
- https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv3.html
  (BSD 3-Clause License, Copyright (c) Soumith Chintala 2016)
- https://github.com/Atze00/MoViNet-pytorch/blob/main/movinets/models.py
  (MIT License, Copyright (c) 2021 Atze00)

Reference:
@inproceedings{kondratyuk2021movinets,
  title={MoViNets: Mobile Video Networks for Efficient Video Recognition},
  author={Kondratyuk, Dan and Yuan, Liangzhe and Li, Yandong and Zhang, Li and Tan, Mingxing and Brown, Matthew and Gong, Boqing},
  booktitle={CVPR},
  year={2021}
}
"""
from collections import OrderedDict
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Any, Callable, Optional

from .movinet_config import get_movinet_cfg
from .movinet_modules import (
    ConvBlock3D, BasicBneck, Swish, CausalModule, TemporalCGAvgPool3D
)


class ModifiedMoViNet(nn.Module):
    """
    MoViNet-A2 with the Pixel-Wise Temporal Encoder (PTE) as its stem, for sleep video classification.

    The stem has two temporal-only convolution branches (kernels 3 and 5) that are
    concatenated with the input clip, followed by a strided temporal projection.

    Args:
        cfg: Model configuration from movinet_config
        causal: Whether to use causal mode (for streaming)
        pretrained: Whether to load pretrained weights
        num_classes: Number of output classes
        conv_type: Type of convolution ("3d" or "2plus1d")
        tf_like: Whether to use TensorFlow-like same padding
        double_loss: Whether to train with ISDL (triplet loss) in addition to classification
        make_features: Whether to output features for Stage 2
    """
    def __init__(self,
                cfg: "CfgNode",
                causal: bool = True,
                pretrained: bool = False,
                num_classes: int = 600,
                conv_type: str = "3d",
                tf_like: bool = False,
                double_loss: bool = False,
                make_features: bool = False
                ) -> None:
        super().__init__()

        self.original_num_classes = None
        self.double_loss = double_loss
        self.make_features = make_features

        if pretrained:
            self.original_num_classes = num_classes
            tf_like = True
            num_classes = 600
            conv_type = "2plus1d" if causal else "3d"

        blocks_dic = OrderedDict()
        norm_layer = nn.BatchNorm3d if conv_type == "3d" else nn.BatchNorm2d
        activation_layer = Swish if conv_type == "3d" else nn.Hardswish

        # PTE stem: two temporal-only branches (kernel 3 and 5, three convs each) + strided temporal projection
        conv_names = ["conv000", "conv001", "conv002", "conv010", "conv011", "conv012", "convpool"]
        self.stem_conv = nn.ModuleDict()
        for name in conv_names:
            block_cfg = getattr(cfg, name)
            self.stem_conv[name] = ConvBlock3D(
                in_planes=block_cfg.input_channels,
                out_planes=block_cfg.out_channels,
                kernel_size=block_cfg.kernel_size,
                stride=block_cfg.stride,
                padding=block_cfg.padding,
                causal=causal,
                conv_type=conv_type,
                tf_like=False,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )

        # conv1
        self.conv1 = ConvBlock3D(
            in_planes=cfg.conv1.input_channels,
            out_planes=cfg.conv1.out_channels,
            kernel_size=cfg.conv1.kernel_size,
            stride=cfg.conv1.stride,
            padding=cfg.conv1.padding,
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=activation_layer
        )

        # blocks
        for i, block in enumerate(cfg.blocks):
            for j, basicblock in enumerate(block):
                blocks_dic[f"b{i}_l{j}"] = BasicBneck(
                    basicblock,
                    causal=causal,
                    conv_type=conv_type,
                    tf_like=tf_like,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer
                )
        self.blocks = nn.Sequential(blocks_dic)

        # conv7
        self.conv7 = ConvBlock3D(
            in_planes=cfg.conv7.input_channels,
            out_planes=cfg.conv7.out_channels,
            kernel_size=cfg.conv7.kernel_size,
            stride=cfg.conv7.stride,
            padding=cfg.conv7.padding,
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=activation_layer
        )

        # classifier
        self.classifier = nn.Sequential(
            ConvBlock3D(cfg.conv7.out_channels,
                        cfg.dense9.hidden_dim,
                        kernel_size=(1, 1, 1),
                        tf_like=tf_like,
                        causal=causal,
                        conv_type=conv_type,
                        bias=True),
            Swish(),
            nn.Dropout(p=0.2, inplace=True),
            ConvBlock3D(cfg.dense9.hidden_dim,
                        num_classes,
                        kernel_size=(1, 1, 1),
                        tf_like=tf_like,
                        causal=causal,
                        conv_type=conv_type,
                        bias=True),
        )

        if double_loss:
            self.projection_head = nn.Sequential(
                nn.Linear(cfg.conv7.out_channels, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256)
            )

        if causal:
            self.cgap = TemporalCGAvgPool3D()

        if pretrained:
            if causal:
                state_dict = torch.hub.load_state_dict_from_url(cfg.stream_weights)
            else:
                state_dict = torch.hub.load_state_dict_from_url(cfg.weights)

            if self.original_num_classes:
                self.classifier = nn.Sequential(
                    ConvBlock3D(cfg.conv7.out_channels,
                                cfg.dense9.hidden_dim,
                                kernel_size=(1, 1, 1),
                                tf_like=tf_like,
                                causal=causal,
                                conv_type=conv_type,
                                bias=True),
                    Swish(),
                    nn.Dropout(p=0.2, inplace=True),
                    ConvBlock3D(cfg.dense9.hidden_dim,
                                self.original_num_classes,
                                kernel_size=(1, 1, 1),
                                tf_like=tf_like,
                                causal=causal,
                                conv_type=conv_type,
                                bias=True),
                )

            self.apply(self._weight_init)

            # Remove incompatible keys for modified stem
            for key in list(state_dict.keys()):
                if "conv1" in key:
                    del state_dict[key]
                if "b0_" in key:
                    del state_dict[key]
                if "classifier" in key:
                    del state_dict[key]
                if "blocks.b1_l0.res.1.conv_1.conv3d.weight" in key:
                    del state_dict[key]
                if "blocks.b1_l0.expand.conv_1.conv3d.weight" in key:
                    del state_dict[key]

            self.load_state_dict(state_dict, strict=False)
        else:
            self.apply(self._weight_init)

        self.causal = causal

    def avg(self, x: Tensor) -> Tensor:
        if self.causal:
            avg = F.adaptive_avg_pool3d(x, (x.shape[2], 1, 1))
            avg = self.cgap(avg)[:, :, -1:]
        else:
            avg = F.adaptive_avg_pool3d(x, 1)
        return avg

    def avg_feature(self, x: Tensor) -> Tensor:
        if self.causal:
            avg = F.adaptive_avg_pool3d(x, (x.shape[2], 1, 1))
            avg = self.cgap(avg)[:, :, -1:]
        else:
            avg = F.adaptive_avg_pool3d(x, (50, 1, 1))
        return avg

    @staticmethod
    def _weight_init(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # PTE stem
        x0 = self.stem_conv.conv000(x)
        x0 = self.stem_conv.conv001(x0)
        x0 = self.stem_conv.conv002(x0)

        x1 = self.stem_conv.conv010(x)
        x1 = self.stem_conv.conv011(x1)
        x1 = self.stem_conv.conv012(x1)

        x = torch.cat((x, x0, x1), dim=1)
        x = self.stem_conv.convpool(x)

        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv7(x)  # [B, 640, 50, 7, 7]

        x_c = self.avg(x)
        x_c = self.classifier(x_c)
        x_c = x_c.flatten(1)

        if self.make_features:
            x_f = self.avg_feature(x)
            return x_c, x_f

        if self.double_loss:
            x_f = self.avg(x)
            x_f = x_f.view(x_f.size(0), -1)
            x_f = self.projection_head(x_f)
            return x_c, x_f
        else:
            x_f = self.avg_feature(x)
            return x_c, x_f

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    @staticmethod
    def _clean_activation_buffers(m):
        if issubclass(type(m), CausalModule):
            m.reset_activation()

    def clean_activation_buffers(self) -> None:
        self.apply(self._clean_activation_buffers)


def build_movinet(
    causal: bool = False,
    pretrained: bool = False,
    num_classes: int = 5,
    make_features: bool = False,
    double_loss: bool = False,
    tf_like: bool = False
) -> nn.Module:
    """
    Build the Stage 1 model (PTE + MoViNet-A2).

    Args:
        causal: Whether to use causal mode (for streaming inference)
        pretrained: Whether to load pretrained weights
        num_classes: Number of output classes
        make_features: Whether to output features for Stage 2
        double_loss: Whether to train with ISDL (triplet loss) in addition to classification
        tf_like: TensorFlow-style same padding (True when pretrained=True)

    Returns:
        MoViNet model
    """
    cfg = get_movinet_cfg().MODEL.PTE_MoViNetA2
    model = ModifiedMoViNet(
        cfg,
        causal=causal,
        pretrained=pretrained,
        num_classes=num_classes,
        tf_like=tf_like,
        double_loss=double_loss,
        make_features=make_features
    )

    return model
