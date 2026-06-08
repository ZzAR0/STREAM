"""Model definitions for CoViAR baselines and STREAM.

The ``stream`` representation implements the paper-style compressed-domain
fusion:

    F_align(t) = Warp(F_iframe, MV(t))
    G(t)       = sigmoid(Conv1x1([F_res(t), F_align(t)]))
    F_fused(t) = G(t) * F_res(t) + (1 - G(t)) * F_align(t)

Motion vectors are used as a displacement field only; they are not classified
by a separate branch.
"""

import os

os.environ.setdefault("TORCH_HOME", os.path.join(os.path.dirname(__file__), ".torch"))

import torch
import torch.nn.functional as F
from torch import nn
import torchvision

from transforms import (
    GroupMultiScaleCrop,
    GroupRandomHorizontalFlip,
    StreamGroupMultiScaleCrop,
    StreamRandomHorizontalFlip,
)


def _make_resnet(name, pretrained=True):
    if not pretrained:
        return getattr(torchvision.models, name)(weights=None)
    try:
        return getattr(torchvision.models, name)(weights="IMAGENET1K_V1")
    except Exception as exc:
        print(f"Warning: failed to load pretrained weights for {name}: {exc}")
        print(f"Warning: falling back to randomly initialized {name}.")
        return getattr(torchvision.models, name)(weights=None)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ResNetFeatureExtractor(nn.Module):
    """ResNet up to layer4, returning a feature map instead of logits."""

    def __init__(self, arch, in_channels=3, pretrained=True):
        super().__init__()
        base = _make_resnet(arch, pretrained=pretrained)
        if in_channels != 3:
            old = base.conv1
            base.conv1 = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )
            if pretrained and in_channels == 2:
                with torch.no_grad():
                    base.conv1.weight.copy_(old.weight[:, :2])
        self.out_channels = base.fc.in_features
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class MultiScaleTemporalConv(nn.Module):
    """Dilated temporal convolution stack from STREAM Sec. 3.2.2."""

    def __init__(self, channels, layers=3, kernel_size=3, dropout=0.1):
        super().__init__()
        blocks = []
        for layer_idx in range(layers):
            dilation = 2 ** layer_idx
            padding = dilation * (kernel_size - 1) // 2
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        bias=True,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2).contiguous()
        for block in self.blocks:
            x = block(x)
        return x.transpose(1, 2).contiguous()


class StreamFusionModel(nn.Module):
    """Single-branch STREAM model with motion-guided feature fusion."""

    def __init__(
        self,
        num_class,
        num_segments,
        iframe_arch="resnet18",
        residual_arch="resnet18",
        fusion_channels=512,
        temporal_layers=3,
        temporal_kernel_size=3,
        dropout=0.1,
        mv_clip=20.0,
        warp_direction=1.0,
    ):
        super().__init__()
        self.num_class = num_class
        self.num_segments = num_segments
        self.mv_clip = float(mv_clip)
        self.warp_direction = float(warp_direction)

        self.iframe_encoder = ResNetFeatureExtractor(iframe_arch, in_channels=3)
        self.residual_encoder = ResNetFeatureExtractor(residual_arch, in_channels=3)
        self.iframe_proj = nn.Conv2d(self.iframe_encoder.out_channels, fusion_channels, 1, bias=False)
        self.residual_proj = nn.Conv2d(self.residual_encoder.out_channels, fusion_channels, 1, bias=False)
        self.iframe_bn = nn.BatchNorm2d(fusion_channels)
        self.residual_bn = nn.BatchNorm2d(fusion_channels)
        self.residual_data_bn = nn.BatchNorm2d(3)

        self.gate = nn.Conv2d(fusion_channels * 2, 1, kernel_size=1)
        self.temporal_model = MultiScaleTemporalConv(
            fusion_channels,
            layers=temporal_layers,
            kernel_size=temporal_kernel_size,
            dropout=dropout,
        )
        self.classifier = nn.Linear(fusion_channels, num_class)
        self._input_size = 224

    @staticmethod
    def _flatten_segments(x):
        return x.view((-1,) + x.size()[-3:]).contiguous()

    def _decode_mv_to_pixels(self, mv):
        """Convert normalized dataset MV tensor back to approximate pixels.

        Dataset MV normalization is:
            raw_mv -> clip_and_scale(raw_mv, mv_clip) + 128 -> /255 - 0.5
        so the inverse is:
            raw_mv ~= (((mv + 0.5) * 255) - 128) * mv_clip / 127.5
        """
        return (((mv + 0.5) * 255.0) - 128.0) * (self.mv_clip / 127.5)

    def warp_features(self, features, mv, input_hw):
        """Warp iframe feature maps with motion vectors.

        Args:
            features: [N, C, Hf, Wf] iframe feature map.
            mv: [N, 2, Hi, Wi] normalized MV tensor.
            input_hw: spatial size of the original MV/image tensor.

        Returns:
            [N, C, Hf, Wf] motion-aligned iframe features.
        """
        n, _, hf, wf = features.shape
        hi, wi = input_hw
        flow = self._decode_mv_to_pixels(mv)
        flow = F.interpolate(flow, size=(hf, wf), mode="bilinear", align_corners=True)

        scale_x = float(wf) / max(float(wi), 1.0)
        scale_y = float(hf) / max(float(hi), 1.0)
        flow_x = flow[:, 0] * scale_x
        flow_y = flow[:, 1] * scale_y

        if wf > 1:
            flow_x = flow_x * (2.0 / (wf - 1))
        else:
            flow_x = flow_x * 0.0
        if hf > 1:
            flow_y = flow_y * (2.0 / (hf - 1))
        else:
            flow_y = flow_y * 0.0

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, hf, device=features.device, dtype=features.dtype),
            torch.linspace(-1.0, 1.0, wf, device=features.device, dtype=features.dtype),
            indexing="ij",
        )
        base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(n, hf, wf, 2)
        flow_grid = torch.stack((flow_x, flow_y), dim=-1)
        grid = base_grid + self.warp_direction * flow_grid
        return F.grid_sample(features, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, inputs):
        batch_size = inputs["iframe"].size(0)
        num_segments = inputs["iframe"].size(1)
        iframe = self._flatten_segments(inputs["iframe"])
        residual = self._flatten_segments(inputs["residual"])
        mv = self._flatten_segments(inputs["mv"])

        input_hw = mv.shape[-2:]
        residual = self.residual_data_bn(residual)

        iframe_feat = self.iframe_encoder(iframe)
        residual_feat = self.residual_encoder(residual)

        iframe_feat = self.iframe_bn(self.iframe_proj(iframe_feat))
        residual_feat = self.residual_bn(self.residual_proj(residual_feat))

        aligned = self.warp_features(iframe_feat, mv, input_hw)
        gate = torch.sigmoid(self.gate(torch.cat([residual_feat, aligned], dim=1)))
        fused = gate * residual_feat + (1.0 - gate) * aligned

        pooled = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        temporal_sequence = pooled.view(batch_size, num_segments, -1).contiguous()
        temporal_features = self.temporal_model(temporal_sequence)
        video_feature = temporal_features.mean(dim=1)
        return self.classifier(video_feature)


class Model(nn.Module):
    def __init__(
        self,
        num_class,
        num_segments,
        representation,
        base_model="resnet18",
        stream_dropout=0.1,
        warp_direction=1.0,
    ):
        super().__init__()
        self._representation = representation
        self.num_segments = num_segments
        self.num_class = num_class

        print(
            """
Initializing model:
    base model:              {}.
    input_representation:    {}.
    num_class:               {}.
    num_segments:            {}.
            """.format(base_model, self._representation, num_class, self.num_segments)
        )

        if self._representation == "stream":
            self.stream_model = StreamFusionModel(
                num_class=num_class,
                num_segments=num_segments,
                iframe_arch=base_model,
                residual_arch=base_model,
                dropout=stream_dropout,
                warp_direction=warp_direction,
            )
            self._input_size = self.stream_model._input_size
        else:
            self._prepare_base_model(base_model)
            self._prepare_tsn(num_class)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, "fc").in_features
        setattr(self.base_model, "fc", nn.Linear(feature_dim, num_class))

        if self._representation == "mv":
            setattr(
                self.base_model,
                "conv1",
                nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            )
            self.data_bn = nn.BatchNorm2d(2)
        if self._representation == "residual":
            self.data_bn = nn.BatchNorm2d(3)

    def _prepare_base_model(self, base_model):
        if "resnet" not in base_model:
            raise ValueError("Unknown base model: {}".format(base_model))
        self.base_model = _make_resnet(base_model, pretrained=True)
        self._input_size = 224

    def forward(self, input):
        if self._representation == "stream":
            return self.stream_model(input)

        input = input.view((-1,) + input.size()[-3:]).contiguous()
        if self._representation in ["mv", "residual"]:
            input = self.data_bn(input)
        return self.base_model(input)

    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 224

    def get_augmentation(self):
        if self._representation == "stream":
            scales = [1, 0.875, 0.75]
            print("STREAM augmentation scales:", scales)
            return torchvision.transforms.Compose(
                [StreamGroupMultiScaleCrop(self._input_size, scales), StreamRandomHorizontalFlip()]
            )

        if self._representation in ["mv", "residual"]:
            scales = [1, 0.875, 0.75]
        else:
            scales = [1, 0.875, 0.75, 0.66]

        print("Augmentation scales:", scales)
        return torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, scales), GroupRandomHorizontalFlip(is_mv=(self._representation == "mv"))]
        )
