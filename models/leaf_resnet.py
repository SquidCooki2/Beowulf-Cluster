import torch
import torch.nn as nn
import torchvision.models as models

def create_resnet18_stages(num_classes: int = 38):
    """
    Slices a 38-class ResNet-18 model into 5 sequential stage modules:
    - Stage 0: Initial Conv1, BatchNorm, ReLU, MaxPool
    - Stage 1: Residual Layer 1 (2 residual blocks)
    - Stage 2: Residual Layer 2 (2 residual blocks)
    - Stage 3: Residual Layer 3 (2 residual blocks)
    - Stage 4: Residual Layer 4, AvgPool, Flatten, Linear Classifier
    """
    base = models.resnet18(weights=None)

    stage0 = nn.Sequential(
        base.conv1,
        base.bn1,
        base.relu,
        base.maxpool
    )
    stage1 = nn.Sequential(base.layer1)
    stage2 = nn.Sequential(base.layer2)
    stage3 = nn.Sequential(base.layer3)
    stage4 = nn.Sequential(
        base.layer4,
        base.avgpool,
        nn.Flatten(),
        nn.Linear(512, num_classes)
    )

    return [stage0, stage1, stage2, stage3, stage4]
