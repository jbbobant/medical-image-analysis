import torch
import torch.nn as nn
from typing import Optional

class DoubleConv(nn.Module):
    """
    Standard double convolution block: 
    (Conv2d => ReLU) * 2
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.step = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.step(x)


class UNet(nn.Module):
    """
    Classic U-Net architecture for segmentation.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        
        # Downward path (Encoder)
        self.layer1 = DoubleConv(in_channels, 64)
        self.layer2 = DoubleConv(64, 128)
        self.layer3 = DoubleConv(128, 256)
        self.layer4 = DoubleConv(256, 512)

        # Upward path (Decoder)
        self.layer5 = DoubleConv(512 + 256, 256)
        self.layer6 = DoubleConv(256 + 128, 128)
        self.layer7 = DoubleConv(128 + 64, 64)
        
        # Final output
        self.layer8 = nn.Conv2d(64, out_channels, kernel_size=1)

        self.maxpool = nn.MaxPool2d(2)
        # Using Upsample instead of ConvTranspose2d to reduce checkerboard artifacts
        # Bilinear is usually sufficient for segmentation masks
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.layer1(x)
        x1m = self.maxpool(x1)
        
        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)

        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)
        
        # Bottleneck
        x4 = self.layer4(x3m)

        # Decoder with Skip Connections
        x5 = self.upsample(x4)
        # Crop or Padding might be needed if input size isn't divisible by 16.
        # Assuming 256x256 input, straight concat works.
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.layer5(x5)

        x6 = self.upsample(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.layer6(x6)
        
        x7 = self.upsample(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.layer7(x7)
        
        # Final output
        out = self.layer8(x7)
        return out