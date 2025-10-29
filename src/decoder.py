import torch
import torch.nn as nn
import torch.nn.functional as F

class MidBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        layers = []
        for _ in range(6):
            layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super().__init__()
        convs = []
        for _ in range(6):
            convs += [nn.Conv2d(in_c, mid_c, 3, padding=1), nn.ReLU(inplace=True)]
            in_c = mid_c
        self.convs = nn.Sequential(*convs)
        self.t1 = nn.ConvTranspose2d(mid_c, mid_c, kernel_size=4, stride=2, padding=1)
        self.t2 = nn.ConvTranspose2d(mid_c, out_c, kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        x = self.convs(x)
        x = F.relu(self.t1(x))
        x = F.relu(self.t2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=256, out_ch=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 32 * 8 * 28)
        self.mid = MidBlock(32)
        self.up1 = UpBlock(32, 32, 16)  # (8→32)
        self.up2 = UpBlock(16, 16, 8)   # (32→128)
        self.final_t = nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1)  # (128→256)
        self.final_conv = nn.Conv2d(8, out_ch, 3, padding=1)

    def forward(self, z):
        b = z.size(0)
        x = self.fc(z).view(b, 32, 8, 28)
        x = self.mid(x)
        x = self.up1(x)          # (16, 32, 112)
        x = self.up2(x)          # (8, 128, 448)
        x = F.relu(self.final_t(x))  # (8, 256, 896)
        x = torch.tanh(self.final_conv(x))  # (3, 256, 896)
        # Fix width from 896→900 via padding (no interpolate)
        pad = (2, 2, 0, 0)  # pad 4 columns evenly (L,R)
        x = F.pad(x, pad, mode='reflect')
        return x