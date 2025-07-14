import torch
import torch.nn as nn

# Convolution Block For Simple U-net which predicts noise at each timestep t
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.convs(x)

# Simple Unet without sinusoidal positional encoding
class UNet(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        # Convolution Modules For Each Stage
        self.down1 = ConvBlock(in_ch, 64)
        self.down2 = ConvBlock(64, 128)
        self.bot = ConvBlock(128, 256)
        self.up1 = ConvBlock(128+256, 128)
        self.up2 = ConvBlock(64+128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

        # Downsampling & Upsampling
        self.downsample = nn.MaxPool2d(kernel_size=2) # strde default value is same as kernel_size
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    
    def forward(self, x):
        x1 = self.down1(x)
        x = self.downsample(x1)

        x2 = self.down2(x)
        x = self.downsample(x2)

        x = self.bot(x)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1) # Concat Channelwisely (Skip connection)
        x = self.up1(x)

        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1) # Concat Channelwisely (Skip connection)
        x = self.up2(x)

        out = self.out(x) # Channel Reduction with 1 Conv

        return out

if __name__ == "__main__":
    model = UNet(in_ch=1)
    x = torch.randn(10, 1, 28, 28)
    y = model(x)
    breakpoint() 