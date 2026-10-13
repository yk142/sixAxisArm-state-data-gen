"""モデル A/B/C の構造・損失・統計の単体テスト（合成データ）。"""
import math

import numpy as np
import pytest
import torch

from sa_nsm.model import LIM_UNIT, MODELS, limit_features
from sa_nsm.train_lib import make_stats

N_AX = 3


def fake(n=512, seed=0):
    rs = np.random.RandomState(seed)
    xs = rs.randn(n, 12).astype(np.float32)
    ana = (0.01 * rs.randn(n, 6)).astype(np.float32)
    lim = (rs.rand(n, 6) * 8).astype(np.float32)
    at = rs.rand(n) < 0.2
    y = ana + (0.001 * rs.randn(n, 6) + at[:, None] * 0.3 * rs.randn(n, 6)).astype(np.float32)
    return {"xs": xs, "y": y, "ana": ana, "lim": lim, "at": at}


def build(kind, d):
    return MODELS[kind](N_AX, make_stats(kind, d), width=32, depth=2)


def tens(d):
    return tuple(torch.from_numpy(d[k]) for k in ("xs", "ana", "lim", "y", "at"))


@pytest.mark.parametrize("kind", ["A", "B", "C"])
def test_forward_shape_and_loss_finite(kind):
    d = fake()
    m = build(kind, d)
    xs, ana, lim, y, at = tens(d)
    p = m(xs, ana, lim)
    assert p.shape == (512, 6)
    assert torch.isfinite(m.loss(p, y, at))


@pytest.mark.parametrize("kind", ["B", "C"])
def test_residual_models_start_as_analytic(kind):
    d = fake()
    m = build(kind, d)
    xs, ana, lim, _, _ = tens(d)
    assert torch.allclose(m(xs, ana, lim), ana, atol=1e-7)            # 最終層ゼロ初期化 → 開始時点は解析モデルそのもの


def test_limit_features_units_and_clamp():
    qmin, qmax = torch.tensor([-1.0, -1.0, -1.0]), torch.tensor([1.0, 1.0, 1.0])
    q = torch.tensor([[1.0 - LIM_UNIT, -1.0 + 2 * LIM_UNIT, 1.0 + LIM_UNIT]])
    f = limit_features(q, qmin, qmax)
    assert torch.allclose(f[0, 0], torch.tensor(1.0), atol=1e-4)      # 軸 1 は上限まで 1°
    assert torch.allclose(f[0, 4], torch.tensor(2.0), atol=1e-4)      # 軸 2 は下限から 2°
    assert torch.allclose(f[0, 2], torch.tensor(-1.0), atol=1e-5)    # 上限の外側は負
    assert f.max() <= 10.0 and f.min() >= -3.0


def test_gate_opens_near_limit_only():
    d = fake()
    m = build("C", d)
    near = torch.zeros(1, 6)
    near[0, :3] = 0.1                                                   # 上限までの距離 0.1°
    near[0, 3:] = 9.0
    far = torch.full((1, 6), 9.0)
    assert m.gate(near).item() > 0.8 and m.gate(far).item() < 0.01


def test_group_normalized_loss_weights_groups_equally_in_scale():
    d = fake()
    m = build("C", d)
    xs, ana, lim, y, at = tens(d)
    p = ana.clone()
    r = y - p
    sig = torch.where(at.unsqueeze(-1), m.sig_contact, m.sig_free)
    assert torch.isclose(m.loss(p, y, at), ((r / sig) ** 2).mean())
    assert m.sig_contact.mean() > 10 * m.sig_free.mean()                # 区分で σ が大きく違う前提（合成データでも）


def test_pure_stats_standardize_train():
    d = fake()
    st = make_stats("A", d)
    assert torch.allclose(torch.as_tensor(st["y_std"]), torch.as_tensor(d["y"].std(0)), atol=1e-6)


def test_training_step_reduces_loss():
    d = fake(2048)
    d["y"] = d["ana"] + 0.01 * d["xs"][:, :6] * d["xs"][:, 6:12]          # 残差が入力の関数（学習できる）
    torch.manual_seed(0)
    m = build("B", d)
    xs, ana, lim, y, at = tens(d)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    l0 = m.loss(m(xs, ana, lim), y, at).item()
    for _ in range(60):
        opt.zero_grad(); m.loss(m(xs, ana, lim), y, at).backward(); opt.step()
    assert m.loss(m(xs, ana, lim), y, at).item() < 0.5 * l0
