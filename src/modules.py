from typing import Literal

import torch
from torch.nn import (
    BatchNorm2d, Conv2d, ConvTranspose2d, Module, ModuleList, ReLU, Sequential
)
from torch.nn.functional import softmax

EPSILON: float = 1e-6


class ResidualConv(Module):
    """
    Implements a residual convolutional block for deep learning models.

    This class defines a residual convolutional block that combines the outputs of a primary convolutional
    path and a shortcut connection. It is useful for creating residual learning architectures, allowing
    for better gradient flow and efficient training of deeper networks.

    :ivar conv: The main path of the residual block consists of a sequence of batch normalization,
        activation, and convolutional layers.
    :type conv: torch.nn.Sequential
    :ivar shortcut: The shortcut path of the residual block, used for bypassing the main computation
        path and adding the input to the output.
    :type shortcut: torch.nn.Sequential
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, padding: int = 1
    ) -> None:
        super().__init__()

        self.conv = Sequential(
            BatchNorm2d(in_channels),
            ReLU(inplace=True),
            Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.shortcut = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class Upsample(Module):
    """
    Performs upsampling of the input tensor using a transposed convolution.

    This class implements an upsampling layer using the PyTorch `ConvTranspose2d`
    module. It allows for the upscaling of input tensors by employing a transposed
    convolution operation with configurable parameters such as input channels, output
    channels, kernel size, and stride.

    :ivar upsample: Transposed convolution layer for performing upsampling.
    :type upsample: torch.nn.ConvTranspose2d
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel: int = 2, stride: int = 2
    ) -> None:
        super().__init__()

        self.upsample = ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class ResUNet(Module):
    """
    ResUNet is a residual U-Net architecture designed for tasks such as segmentation and feature extraction.

    This class implements a deep convolutional neural network with an encoder-decoder structure. The encoder
    downsamples the input, extracting features at multiple levels using residual connections. The decoder upsamples
    the features, reconstructing the output while incorporating skip connections to preserve spatial resolution.
    It is highly customizable in the number of channels and filter configurations.

    :ivar DEFAULT_FILTERS: Default filter sizes are used for each layer of the network.
    :type DEFAULT_FILTERS: tuple[int, int, int, int, int]
    :ivar depth: Number of encoder layers, calculated based on the length of the filter configuration minus one.
    :type depth: int
    :ivar input_layer: Initial input processing block for the network, including convolution, batch normalization,
                       and activation layers.
    :type input_layer: torch.nn.Sequential
    :ivar input_skip: Skip connection for the input layer, providing additional spatial information to the network.
    :type input_skip: torch.nn.Conv2d
    :ivar encoder_blocks: Module list containing residual convolution blocks for the encoder.
    :type encoder_blocks: torch.nn.ModuleList
    :ivar upsample_blocks: Module list containing upsampling layers for the decoder.
    :type upsample_blocks: torch.nn.ModuleList
    :ivar decoder_blocks: Module list containing residual convolution blocks for the decoder.
    :type decoder_blocks: torch.nn.ModuleList
    """

    DEFAULT_FILTERS: tuple[int, ...] = (32, 64, 128, 256, 512)

    def __init__(
        self,
        channel: int = 3,
        filters: tuple[int, ...] = DEFAULT_FILTERS,
        feature_source: Literal["encoder", "decoder"] = "encoder",
        internal_index: int = 4,
    ) -> None:
        super().__init__()

        if len(filters) < 2:
            raise ValueError("filters must have at least two elements (encoder + bridge).")

        if feature_source not in {"encoder", "decoder"}:
            raise ValueError(f"feature_source must be one of 'encoder' or 'decoder', got {feature_source}.")

        self.filters = filters
        self.feature_source = feature_source
        self.internal_index = internal_index
        self.depth = len(filters) - 1  # Number of encoder layers

        # Encoder input block
        self.input_layer = Sequential(
            Conv2d(channel, filters[0], kernel_size=3, padding=1),
            BatchNorm2d(filters[0]),
            ReLU(inplace=True),
            Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = Conv2d(channel, filters[0], kernel_size=3, padding=1)

        # Encoder residual blocks
        self.encoder_blocks = ModuleList()
        for i in range(self.depth):
            self.encoder_blocks.append(
                ResidualConv(filters[i], filters[i + 1], stride=2, padding=1)
            )

        # Decoder blocks
        self.upsample_blocks = ModuleList()
        self.decoder_blocks = ModuleList()
        for i in reversed(range(self.depth)):
            self.upsample_blocks.append(Upsample(filters[i + 1], filters[i + 1], 2, 2))
            self.decoder_blocks.append(
                ResidualConv(filters[i + 1] + filters[i], filters[i], stride=1)
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Encode
        skips: list[torch.Tensor] = []
        x1 = self.input_layer(x) + self.input_skip(x)
        skips.append(x1)
        x = x1

        encoder_outputs: list[torch.Tensor] = [x1]
        for encoder in self.encoder_blocks:
            x = encoder(x)
            skips.append(x)
            encoder_outputs.append(x)

        # Decode
        decoder_outputs: list[torch.Tensor] = []
        for i in range(self.depth):
            x = self.upsample_blocks[i](x)
            x = torch.cat([x, skips[-i - 2]], dim=1)  # Skip connections
            x = self.decoder_blocks[i](x)
            decoder_outputs.append(x)

        # Select internal features
        if self.feature_source == "decoder":
            if 0 <= self.internal_index < len(decoder_outputs):
                internal_features = decoder_outputs[self.internal_index]
            else:
                raise ValueError(f"Invalid internal_index={self.internal_index}, must be in [0,{len(decoder_outputs)-1}]")
        else:  # encoder
            if 0 <= self.internal_index < len(encoder_outputs):
                internal_features = encoder_outputs[self.internal_index]
            else:
                raise ValueError(f"Invalid internal_index={self.internal_index}, must be in [0,{len(encoder_outputs)-1}]")

        return {"final": x, "internal": internal_features}


class OCAModule(Module):
    """
    OCAModule is designed for spatial context aggregation using soft object regions.

    It uses their weighted aggregation to enhance feature representations. This module projects
    pixel-level features into a lower-dimensional key space, computes soft object regions
    using a learned region logits convolution, and fuses pixel and contextual feature
    representations for improved semantic understanding.

    The module is particularly useful in segmentation or dense prediction tasks where
    accurate spatial context is required to differentiate between similar regions.

    :ivar key_channels: Number of channels in the intermediate key space used for projection
        and region feature processing.
    :type key_channels: int
    :ivar num_classes: The number of soft object regions to compute. Typically corresponds to
        the semantic classes in the task.
    :type num_classes: int
    """

    def __init__(
        self, in_channels: int, key_channels: int = 512, num_classes: int = 2
    ) -> None:
        super().__init__()
        self.key_channels = key_channels
        self.num_classes = num_classes

        # Produce soft region logits
        self.region_logits_conv = Conv2d(in_channels, num_classes, kernel_size=1)

        # Project feature -> key space
        self.pixel_proj = Sequential(
            Conv2d(in_channels, key_channels, kernel_size=1),
            BatchNorm2d(key_channels),
            ReLU(inplace=True),
        )

        # Transform region representations
        self.region_transform = Sequential(
            Conv2d(key_channels, key_channels, kernel_size=1),
            BatchNorm2d(key_channels),
            ReLU(inplace=True),
        )

        # Final fuse conv: concat(pixel_rep, contextual_rep) -> out channels = in_channels (or smaller)
        self.fusion = Sequential(
            Conv2d(key_channels * 2, in_channels, kernel_size=1, bias=False),
            BatchNorm2d(in_channels),
            ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        b, _, h, w = features.shape

        # 1) Soft object regions
        region_logits = self.region_logits_conv(features)
        soft_regions = softmax(region_logits.view(b, self.num_classes, -1), dim=-1)
        soft_regions = soft_regions.view(b, self.num_classes, h, w)

        # 2) Pixel features projected
        pixel_feats = self.pixel_proj(features)
        pixel_feats_flat = pixel_feats.view(b, self.key_channels, -1)
        pixel_feats_t = pixel_feats_flat.permute(0, 2, 1)

        # 3) Compute region representations: weighted aggregate of pixel features
        soft_regions_flat = soft_regions.view(b, self.num_classes, -1)
        region_repr = torch.bmm(soft_regions_flat, pixel_feats_t)
        region_sum = soft_regions_flat.sum(dim=2, keepdim=True)
        region_repr /= region_sum + EPSILON

        # Transform of region_repr (as 1x1 conv style)
        region_repr_4d = region_repr.view(b * self.num_classes, self.key_channels, 1, 1)
        region_repr_4d = self.region_transform(region_repr_4d)
        region_repr = region_repr_4d.view(b, self.num_classes, self.key_channels)

        # 4) Compute pixel region similarity -> attention weights
        region_repr_t = region_repr.permute(0, 2, 1)
        similarity = torch.bmm(pixel_feats_t, region_repr_t)
        similarity = softmax(similarity, dim=-1)
        contextual_flat = torch.bmm(similarity, region_repr)
        contextual = (
            contextual_flat.permute(0, 2, 1)
            .contiguous()
            .view(b, self.key_channels, h, w)
        )

        # 5) Fuse pixel and contextual
        fused = torch.cat([pixel_feats, contextual], dim=1)
        augmented = self.fusion(fused)

        return augmented, soft_regions, region_logits


if __name__ == "__main__":
    print("Nothing to see here, lol")
