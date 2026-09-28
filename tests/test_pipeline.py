"""
Lightweight smoke tests — not a full test suite, but enough to catch broken
imports/shape mismatches after you modify the pipeline.

Run with:  python -m pytest tests/ -q     (or just: python tests/test_pipeline.py)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src import data_generation as dg
from src import regime_model as rm
from src import bias_correction as bc
from src import verification as ver


def test_data_generation_shapes():
    ds = dg.generate_dataset(n_seasons=1, season_length=20, grid=16, seed=1, n_subdaily_steps=8)
    assert ds["predictors"].shape == (20, dg.N_PREDICTORS, 16, 16)
    assert ds["raw_nwp"].shape == (20, 16, 16)
    assert ds["truth"].shape == (20, 16, 16)
    assert (ds["truth"] >= 0).all()
    assert ds["subdaily_predictors"].shape == (20, 8, dg.N_PREDICTORS, 16, 16)
    assert ds["subdaily_regime"].shape == (20, 8)
    assert ds["subdaily_hours"].shape == (8,)


def test_vae_forward_pass():
    x = np.random.randn(4, dg.N_PREDICTORS, 16, 16).astype(np.float32)
    model = rm.ConvVAE(dg.N_PREDICTORS, 16, latent_dim=4)
    recon, mu, logvar = model(torch.tensor(x))
    assert recon.shape == x.shape
    assert mu.shape == (4, 4)


def test_filmunet_forward_pass():
    model = bc.FiLMUNet(in_ch=1, cond_dim=5, base=8, dropout_p=0.1, quantiles=(0.1, 0.5, 0.9))
    x = torch.randn(2, 1, 16, 16)
    cond = torch.softmax(torch.randn(2, 5), dim=1)
    out = model(x, cond)
    assert out.shape == (2, 3, 16, 16)
    assert (out >= 0).all()


def test_verification_metrics_perfect_forecast():
    obs = np.random.rand(5, 16, 16) * 100
    report = ver.full_report(obs, obs, thresholds=[15, 65], neighborhood=3)
    assert report["RMSE"] == 0.0
    assert abs(report["CSI_15mm"] - 1.0) < 1e-6


if __name__ == "__main__":
    test_data_generation_shapes()
    test_vae_forward_pass()
    test_filmunet_forward_pass()
    test_verification_metrics_perfect_forecast()
    print("All smoke tests passed.")
