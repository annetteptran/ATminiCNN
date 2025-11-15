import torch.nn as nn

class BasicBlock(nn.Module):
    """
    residual block:
    input -> conv-bn-relu -> conv-bn -> +shortcut -> relu
    if channels or stride change, we use a 1x1 conv on the shortcut.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class ResNetTiny(nn.Module):
    """
    small ResNet-ish model for 64x64 images.
    stages:
      - stem: 3x3 conv -> BN -> ReLU  (64x64)
      - stage 1: 2 blocks at 64 ch    (64x64)
      - stage 2: 2 blocks at 128 ch   (32x32)
      - stage 3: 2 blocks at 256 ch   (16x16)
      - GAP -> Linear(num_classes)
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # keep spatial size, 64 -> 64 channels
        self.stage1 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1),
        )

        # downsample 64x64 -> 32x32, 64 -> 128 channels
        self.stage2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1),
        )

        # downsample 32x32 -> 16x16, 128 -> 256 channels
        self.stage3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)  # (B, 256, 1, 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)     # (B, 64, 64, 64)
        x = self.stage1(x)   # (B, 64, 64, 64)
        x = self.stage2(x)   # (B, 128, 32, 32)
        x = self.stage3(x)   # (B, 256, 16, 16)
        x = self.pool(x)     # (B, 256, 1, 1)
        x = self.head(x)     # (B, num_classes)
        return x


def build_model(num_classes: int):
    return ResNetTiny(num_classes)
