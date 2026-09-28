"""
Step 2: The "Smart Editor" — Regime-Conditioned Bias Correction.

A U-Net takes the raw (biased) NWP rainfall grid and corrects it. The regime
probability vector from the VAE+GMM ("Weather Mood Ring") is injected into the
U-Net's internal feature maps via FiLM (Feature-wise Linear Modulation): a small
MLP maps the regime-probability vector to a per-channel (gamma, beta) pair that
rescales and shifts the feature maps at each block, so the correction applied
smoothly follows the probabilistic regime mix rather than switching abruptly
between regimes.

Dropout layers are kept active at inference time (MC-Dropout) to support the
uncertainty module.
"""
import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Maps a condition vector -> per-channel scale (gamma) and shift (beta)."""
    def __init__(self, cond_dim, n_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 32), nn.ReLU(),
            nn.Linear(32, 2 * n_channels),
        )
        self.n_channels = n_channels

    def forward(self, feat, cond):
        gb = self.net(cond)  # (B, 2C)
        gamma, beta = gb[:, :self.n_channels], gb[:, self.n_channels:]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feat * (1 + gamma) + beta


class FiLMConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, dropout_p):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.film = FiLM(cond_dim, out_ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout2d(dropout_p)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x, cond):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.norm(x)
        x = self.film(x, cond)
        x = self.act(x)
        x = self.drop(x)
        return x


class FiLMUNet(nn.Module):
    """Small U-Net (2 down / 2 up) with FiLM conditioning on the regime
    probability vector, plus quantile regression output heads."""
    def __init__(self, in_ch=1, cond_dim=5, base=16, dropout_p=0.15, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles
        n_q = len(quantiles)

        self.enc1 = FiLMConvBlock(in_ch, base, cond_dim, dropout_p)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = FiLMConvBlock(base, base * 2, cond_dim, dropout_p)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = FiLMConvBlock(base * 2, base * 4, cond_dim, dropout_p)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = FiLMConvBlock(base * 4, base * 2, cond_dim, dropout_p)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = FiLMConvBlock(base * 2, base, cond_dim, dropout_p)

        # output heads: quantiles of corrected rainfall
        self.out_head = nn.Conv2d(base, n_q, 1)

    def forward(self, x, cond):
        e1 = self.enc1(x, cond)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1, cond)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2, cond)

        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1), cond)
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1), cond)

        out = self.out_head(d1)          # (B, n_quantiles, H, W)
        out = torch.relu(out)             # rainfall cannot be negative
        return out


def pinball_loss(preds, target, quantiles):
    """Quantile (pinball) loss across the quantile output heads."""
    losses = []
    for i, q in enumerate(quantiles):
        err = target - preds[:, i:i + 1]
        losses.append(torch.max((q - 1) * err, q * err))
    return torch.mean(torch.cat(losses, dim=1))


def heavy_rain_weighted_pinball(preds, target, quantiles, heavy_threshold=65.0, heavy_weight=4.0):
    """Pinball loss with extra weight where observed rainfall exceeds a
    heavy-rain threshold, so the model is pushed to get flood-relevant
    events right (directly answers the brief's 'improve heavy/very heavy
    rainfall forecasts' requirement)."""
    weight = torch.where(target >= heavy_threshold, heavy_weight, 1.0)
    losses = []
    for i, q in enumerate(quantiles):
        err = target - preds[:, i:i + 1]
        pl = torch.max((q - 1) * err, q * err)
        losses.append(pl * weight)
    return torch.mean(torch.cat(losses, dim=1))


def train_unet(model, raw_nwp, regime_probs, truth, epochs=80, batch_size=16, lr=5e-4,
               quantiles=(0.1, 0.5, 0.9), heavy_threshold=65.0, heavy_weight=4.0, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    x = torch.tensor(raw_nwp, dtype=torch.float32).unsqueeze(1)     # (N,1,H,W)
    cond = torch.tensor(regime_probs, dtype=torch.float32)          # (N,n_regimes)
    y = torch.tensor(truth, dtype=torch.float32).unsqueeze(1)       # (N,1,H,W)

    n = x.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, cb, yb = x[idx].to(device), cond[idx].to(device), y[idx].to(device)
            opt.zero_grad()
            preds = model(xb, cb)
            loss = heavy_rain_weighted_pinball(preds, yb, quantiles, heavy_threshold, heavy_weight)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [U-Net] epoch {epoch+1}/{epochs}  weighted pinball loss={total/n:.4f}")
    return model


def mc_dropout_predict(model, raw_nwp_single, regime_prob_single, n_passes=50, device="cpu"):
    """Run the U-Net n_passes times with dropout active (train() mode) to get an
    ensemble of corrected-rainfall samples for uncertainty quantification."""
    model.to(device)
    model.train()  # keep dropout ON
    x = torch.tensor(raw_nwp_single, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    cond = torch.tensor(regime_prob_single, dtype=torch.float32).unsqueeze(0).to(device)
    samples = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = model(x, cond)          # (1, n_q, H, W)
            median_idx = out.shape[1] // 2
            samples.append(out[0, median_idx].cpu().numpy())
    return samples  # list of (H,W) arrays
