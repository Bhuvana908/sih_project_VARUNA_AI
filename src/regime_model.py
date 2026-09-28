"""
Step 1: The "Weather Mood Ring" — Probabilistic Regime Identification.

A convolutional Variational Autoencoder (VAE) compresses the multi-channel
atmospheric predictor grid (winds, moisture, pressure, CAPE) into a low-dimensional
latent space. A Gaussian Mixture Model (GMM) is then fit on that latent space so
that, for any day, we get a *soft* probability vector over weather regimes instead
of a hard label (e.g. 70% Break Monsoon / 20% Active / 10% Depression).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture


class ConvVAE(nn.Module):
    def __init__(self, in_channels, grid_size, latent_dim=8):
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1), nn.ReLU(),  # grid/2
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),          # grid/4
        )
        reduced = grid_size // 4
        self.flat_dim = 32 * reduced * reduced
        self.reduced = reduced

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1),
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 32, self.reduced, self.reduced)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta=0.5):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss.item(), kld.item()


def normalize_predictors(predictors):
    """Per-channel standardization. Returns normalized array + (mean, std) for reuse."""
    mean = predictors.mean(axis=(0, 2, 3), keepdims=True)
    std = predictors.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    return (predictors - mean) / std, mean, std


def train_vae(predictors_norm, latent_dim=8, epochs=60, batch_size=32, lr=1e-3, beta=0.5, device="cpu"):
    n, c, h, w = predictors_norm.shape
    model = ConvVAE(c, h, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.tensor(predictors_norm, dtype=torch.float32)

    n_train = n
    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        total = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch = x[idx].to(device)
            opt.zero_grad()
            recon, mu, logvar = model(batch)
            loss, _, _ = vae_loss(recon, batch, mu, logvar, beta=beta)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [VAE] epoch {epoch+1}/{epochs}  loss={total/n_train:.4f}")
    return model


def get_latents(model, predictors_norm, device="cpu", batch_size=64):
    model.eval()
    x = torch.tensor(predictors_norm, dtype=torch.float32)
    mus = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i:i + batch_size].to(device)
            mu, _ = model.encode(batch)
            mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0)


def fit_gmm(latents, n_components=5, covariance_type="full", seed=42):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=seed,
        n_init=5,
        reg_covar=1e-4,
    )
    gmm.fit(latents)
    return gmm


def regime_probabilities(gmm, latents):
    """Soft (probabilistic) regime assignment — the 'dimmer switch', not a hard label."""
    return gmm.predict_proba(latents)


def align_gmm_clusters_to_names(gmm, latents, true_regimes, n_regimes):
    """Utility used only for validation/reporting: finds the best ONE-TO-ONE
    permutation mapping arbitrary GMM cluster indices -> known synthetic regime
    indices, via optimal (Hungarian) assignment on the cluster/regime overlap
    counts, purely so printed reports and dashboard labels read naturally and
    unambiguously. A naive per-cluster majority vote can map two clusters to the
    same name (leaving another name unused) whenever clusters are imbalanced;
    Hungarian assignment guarantees a valid bijection. The pipeline itself never
    uses `true_regimes` for training — only for this readability step."""
    from scipy.optimize import linear_sum_assignment

    hard = gmm.predict(latents)
    overlap = np.zeros((n_regimes, n_regimes), dtype=np.int64)
    for cluster in range(n_regimes):
        mask = hard == cluster
        if mask.sum() == 0:
            continue
        overlap[cluster] = np.bincount(true_regimes[mask], minlength=n_regimes)

    cluster_idx, name_idx = linear_sum_assignment(-overlap)  # maximize overlap
    mapping = {int(c): int(n) for c, n in zip(cluster_idx, name_idx)}
    return mapping
