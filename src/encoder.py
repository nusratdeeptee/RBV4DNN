import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Building Blocks ----------

def conv_block(in_c, out_c, stride=1, norm=True):
    layers = [nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1)]
    if norm:
        layers.append(nn.GroupNorm(8, out_c))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c)
        )
    def forward(self, x):
        return F.relu(x + self.block(x))

# ---------- Encoder (Complex) ----------

class Encoder(nn.Module):
    def __init__(self, in_ch=3, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_ch, 16, stride=2),   # 256→128, 900→450
            conv_block(16, 32, stride=2),      # 128→64, 450→225
            conv_block(32, 64, stride=2),      # 64→32, 225→112
            conv_block(64, 64, stride=2),      # 32→16, 112→56
            conv_block(64, 64, stride=2),      # 16→8, 56→28
            ResidualBlock(64),
            ResidualBlock(64)
        )
        flat_dim = 64 * 8 * 28  # 14336
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar