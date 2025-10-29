import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder


# ---------- Full VAE ----------

class VAE(nn.Module):
    def __init__(self, latent_dim=256, in_ch=3):
        super().__init__()
        self.enc = Encoder(in_ch, latent_dim)
        self.dec = Decoder(latent_dim, in_ch)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparam(mu, logvar)
        recon = self.dec(z)
        return recon, mu, logvar, z

# ---------- Loss Function ----------

def vae_loss(x, recon, mu, logvar, z, beta=1.0, gamma=0.05, rec_weight=1.0):
    # Reconstruction (L1 preferred for images)
    rec_loss = F.l1_loss(recon, x)

    # KL divergence (analytic for Gaussian)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    # Covariance regularizer: encourage isotropic latent space
    z_centered = z - z.mean(0, keepdim=True)
    cov = (z_centered.T @ z_centered) / (z_centered.size(0) - 1)
    cov_reg = torch.mean((cov - torch.eye(cov.size(0), device=z.device))**2)

    total_loss = rec_weight * rec_loss + beta * kl_loss + gamma * cov_reg
    return total_loss, rec_loss, kl_loss, cov_reg