import torch.nn as nn
import torchvision.models as models
from torchvision.ops import FrozenBatchNorm2d

from models.stage_utils import split_sequential


def create_malaria_stages(num_stages: int = 5, num_classes: int = 2, pretrained: bool = True):
    """
    MobileNetV2 (ImageNet-pretrained) for parasitized/uninfected blood-smear cells,
    sliced into `num_stages` sequential stages. The 19 `features` blocks are split
    evenly; the last stage also holds pooling + the classifier head.

    BatchNorm is frozen (FrozenBatchNorm2d): the engine uses tiny micro-batches
    (4), where batch statistics are unreliable, so pretrained statistics are kept.
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    base = models.mobilenet_v2(weights=weights, norm_layer=FrozenBatchNorm2d)

    stages = split_sequential(base.features, num_stages)
    head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(1280, num_classes),
    )
    stages[-1] = nn.Sequential(*stages[-1], *head)
    return stages
