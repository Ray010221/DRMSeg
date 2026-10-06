import torch.nn as nn
import timm

from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class ConvNeXtV2PyramidEncoder(nn.Module):
    """Expose ConvNeXtV2-Tiny features in the six-level U-Net convention.

    The first two levels preserve the input and add a shallow stem. The four
    remaining levels are the feature maps returned by timm's ConvNeXtV2.
    """

    def __init__(self, in_channels, image_size=128, pretrained=False,
                 model_name="convnextv2_tiny", stem_channels=64):
        super().__init__()
        del image_size  # Kept in the signature for reproducible train scripts.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            out_indices=(0, 1, 2, 3),
        )
        feature_channels = tuple(self.backbone.feature_info.channels())
        self.out_channels = (in_channels, stem_channels, *feature_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
        )

    @staticmethod
    def _channels_first(feature, channels):
        if feature.ndim == 4 and feature.shape[-1] == channels and feature.shape[1] != channels:
            return feature.permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, x):
        features = self.backbone(x)
        features = [
            self._channels_first(feature, channels)
            for feature, channels in zip(features, self.out_channels[2:])
        ]
        return [x, self.stem(x), *features]


class ConvNeXtV2UNetBranch(nn.Module):
    """One modality-specific ConvNeXtV2 encoder and U-Net decoder."""

    def __init__(self, in_channels, decoder_channels, num_classes,
                 image_size=128, pretrained=False, encoder=None):
        super().__init__()
        self.encoder = encoder or ConvNeXtV2PyramidEncoder(
            in_channels=in_channels, image_size=image_size, pretrained=pretrained
        )
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
        )
