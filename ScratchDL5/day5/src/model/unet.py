import torch
import torch.nn as nn
from src.model.pos_enc import pos_encoding

# ConvBlock with Timestep Encoding
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        # Convolution
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        # MLP that transforms timestep encoding vector
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    # Gets Input image and encoded timestep vector
    def forward(self, x, v):
        N, C, _, _ = x.shape
        # Timestep to MLP
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        # Add v to x and self.convs
        y = self.convs(x+v)

        return y

class UNet(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=100):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(256+128, 128, time_embed_dim)
        self.up1 = ConvBlock(64+128, 64, time_embed_dim)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    
    def forward(self, x, ts):
        v = pos_encoding(ts, self.time_embed_dim, x.device)

        x1 = self.down1(x, v)
        x = self.downsample(x1)
        x2 = self.down2(x, v)
        x = self.downsample(x2)

        x = self.bot(x, v)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)

        x = self.out(x)

        return x

if __name__ == '__main__':
    pass
