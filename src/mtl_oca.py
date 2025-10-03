import torch
from torch.nn import BatchNorm1d, Conv2d, Linear, Module, ReLU, Sequential
from torch.nn.functional import adaptive_avg_pool2d, interpolate

from src.modules import OCAModule, ResUNet


class ClassifierHead(Module):
    """
    ClassifierHead is a neural network module designed for dual-stage classification tasks.

    This class implements a neural network for detecting tumor presence and determining the tumor type
    in medical imaging data. The model is divided into two distinct stages: tumor presence prediction
    and tumor type prediction. Each stage is configured with fully connected layers, activation functions,
    batch normalization, and dropout layers to enforce regularization and robustness.

    This class takes two separate inputs: one for tumor presence classification (Stage 1) and another
    for tumor type classification (Stage 2), outputting predictions for both tasks.

    :ivar tumor_presence: A sequential neural network responsible for the tumor presence classification task.
        It consists of multiple linear layers, activation functions, normalization, and dropout layers,
        producing a binary output indicating tumor presence probability.
    :type tumor_presence: torch.nn.Sequential

    :ivar tumor_type: A sequential neural network responsible for the tumor type classification task.
        It consists of similar layers and architecture as the tumor presence network but outputs the tumor
        type prediction.
    :type tumor_type: torch.nn.Sequential
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()

        # Stage 1: Tumor presence
        self.tumor_presence = Sequential(
            Linear(in_channels, 64),
            ReLU(inplace=True),
            BatchNorm1d(64),
            Linear(64, 1),  # Binary output for presence (sigmoid later)
        )

        # Stage 2: Tumor type
        self.tumor_type = Sequential(
            Linear(in_channels, 256),
            ReLU(inplace=True),
            BatchNorm1d(256),
            Linear(256, 128),
            ReLU(inplace=True),
            BatchNorm1d(128),
            Linear(128, 1),  # Binary output for type (sigmoid later)
        )

    def forward(self, presence_input: torch.Tensor, type_input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        presence = self.tumor_presence(presence_input)
        tumor_type = self.tumor_type(type_input)

        return presence, tumor_type


class MultiTaskOCAModel(Module):
    """
    A multi-task model for simultaneous segmentation and classification.

    This class implements a multi-task model capable of performing both segmentation and
    classification. It combines a backbone network, a region attention module, and separate
    heads for segmentation and classification. The segmentation head predicts the segmentation
    map, while the classification heads predict lesion presence and type.

    :ivar backbone: The backbone network that is used to extract high-level features from the input.
    :type backbone: BackBone
    :ivar oca: A module implementing region attention for enhanced feature representation.
    :type oca: OCAModule
    :ivar seg_head: The final convolution layer for segmentation.
    :type seg_head: torch.nn.Conv2d
    :ivar classifier: A two-stage classifier head for tumor presence and type.
    :type classifier: ClassifierHead
    """

    def __init__(self, backbone: ResUNet, seg_classes: int = 2, oca_regions: int = 2) -> None:
        super().__init__()
        self.backbone = backbone

        if not hasattr(self.backbone, "filters") or len(self.backbone.filters) < 2:
            raise AttributeError(
                "The provided backbone must have a 'filters' attribute (tuple of channel sizes) "
                "to determine output channels for internal layers."
            )

        # The segmentation head uses the channel count of the final DECODER of the backbone.
        self.seg_backbone_channels: int = self.backbone.filters[0]  # 32 channels

        # The classification head uses the channel count of the final ENCODER of the backbone.
        self.cls_backbone_channels: int = self.backbone.filters[self.backbone.internal_index]

        self.oca = OCAModule(
            self.seg_backbone_channels,
            key_channels=min(512, self.seg_backbone_channels),
            num_classes=oca_regions,
        )

        # Segmentation final conv
        self.seg_head = Conv2d(self.seg_backbone_channels, seg_classes, kernel_size=1)

        # Classification head
        cls_input_dim = self.cls_backbone_channels + oca_regions  # cls_backbone_channels + 2
        self.classifier = ClassifierHead(cls_input_dim)

    def forward(self, input_images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Extract high-level features
        features = self.backbone(input_images)
        seg_features = features["final"]  # High-level features for segmentation
        cls_features = features["internal"]  # Lower-level, semantic features for classification

        if 4 not in (seg_features.dim(), cls_features.dim()):
            raise RuntimeError(
                "Backbone must return a 4D tensor (B,C,H,W), got shape "
                f"{tuple(seg_features.shape)} and {tuple(cls_features.shape)} instead."
            )

        # Segmentation
        augmented, soft_regions, region_logits = self.oca(seg_features)
        seg_logits = self.seg_head(augmented)

        # Classification
        seg_probs = torch.sigmoid(seg_logits[:, 1:2])
        _, _, h, w = cls_features.shape  # (B, C, H, W)
        # Downsample seg_probs to match cls_features spatial dimensions
        seg_probs_downsampled = interpolate(seg_probs, size=(h, w), mode="bilinear", align_corners=False)

        lesion_features: torch.Tensor = (cls_features * seg_probs_downsampled).sum(dim=(2, 3)) / (
                seg_probs_downsampled.sum(dim=(2, 3)) + 1e-6
        )
        global_features = adaptive_avg_pool2d(cls_features, (1, 1)).view(cls_features.size(0), -1)
        region_pool = adaptive_avg_pool2d(soft_regions, (1, 1)).view(soft_regions.size(0), -1)

        presence_cls_input = torch.cat((global_features, region_pool), dim=1)
        type_cls_input = torch.cat((lesion_features, region_pool), dim=1)

        presence_logits, type_logits = self.classifier(presence_cls_input, type_cls_input)

        return seg_logits, region_logits, presence_logits, type_logits


if __name__ == "__main__":
    # Sanity check
    model = MultiTaskOCAModel(backbone=ResUNet(channel=3))
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    y = model(x)

    print(y[0].shape)  # Should be (1, 2, 128, 128)
    print(y[1].shape)  # Should be (1, 2, 128, 128)
    print(y[2].shape)  # Should be (1, 1)
    print(y[3].shape)  # Should be (1, 1)

    # Total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
