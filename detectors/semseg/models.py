"""
Semantic segmentation models (semseg task), all trained from scratch under the common protocol.

U-Net               Ronneberger, Fischer & Brox (MICCAI 2015): 4 down / 4 up, 64-1024 channels, skip concatenation.
                    [Unified] padded 3x3 convs + BN (the original Caffe net uses unpadded convs without BN and
                    output cropping), bilinear-free transposed-conv upsampling as in the paper.

[Unified] no ImageNet weights (the papers all start from ImageNet), common init, no mean / std, 608 input with the
YOLO-paper augmentation, Adam 1e-4 constant, cross-entropy with ignore 255 on the logits upsampled to the input,
PreciseBN for BN models (micro batch 2), K = C + 1 classes.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors.semseg.common import SemSegBase


def cbr(cin, cout, k=3, d=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=d * (k // 2), dilation=d, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class _PreciseBN:
    precise_bn, precise_bn_images, precise_bn_batch_size = True, 1600, 16


class UNet(_PreciseBN, SemSegBase):
    def __init__(self, num_classes, c=64):
        super().__init__(num_classes)
        chs = [c, 2 * c, 4 * c, 8 * c, 16 * c]
        self.down = nn.ModuleList()
        cin = 3
        for ch in chs:
            self.down.append(nn.Sequential(cbr(cin, ch), cbr(ch, ch)))
            cin = ch
        self.up = nn.ModuleList(nn.ConvTranspose2d(chs[k + 1], chs[k], 2, stride=2) for k in range(4))
        self.dec = nn.ModuleList(nn.Sequential(cbr(2 * chs[k], chs[k]), cbr(chs[k], chs[k])) for k in range(4))
        self.head = nn.Conv2d(c, self.K, 1)

    precise_bn_batch_size = 2    # [Unified] 2026-10-14: batch 16 ran out of memory (full-resolution decoder)

    def logits(self, x):
        skips = []
        for k, d in enumerate(self.down):
            x = d(x if k == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for k in reversed(range(4)):
            x = self.dec[k](torch.cat([skips[k], self.up[k](x)], 1))
        return self.head(x), None


# ---------------------------------------------------------------- SegFormer (NVlabs mix_transformer.py)


MODELS = {"unet": UNet}   # [Release] this repository's model only


def make_build(name):
    def build_model(task, num_classes):
        return MODELS[name](num_classes)
    return build_model
