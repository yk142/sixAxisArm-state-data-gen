"""高速版解析モデル（FastArm）が元の Arm と同じ結果を返すこと。issue #41。"""
import json
import os

import pytest
import torch

from sa_nsm.physics import Arm
from sa_nsm.physics_fast import FastArm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARAMS = os.path.join(REPO, "data", "export", "c8_params.json")
pytestmark = pytest.mark.skipif(not os.path.exists(PARAMS), reason="data/export/c8_params.json が無い")


@pytest.fixture(scope="module")
def P():
    with open(PARAMS) as f:
        return json.load(f)


@pytest.mark.parametrize("n", [3, 6])
def test_matches_arm(P, n):
    torch.manual_seed(0)
    a, f = Arm(P, n), FastArm(P, n)
    B = 50
    q = (torch.rand(B, n, dtype=torch.float64) - 0.5) * 2.5
    v = torch.randn(B, n, dtype=torch.float64)
    t = 5 * torch.randn(B, n, dtype=torch.float64)
    for x, y in zip(a.matrices(q, v), f.matrices(q, v)):
        assert (x - y).abs().max() < 1e-9 * max(1.0, x.abs().max().item())
    assert (a.accel(q, v, t) - f.accel(q, v, t)).abs().max() < 1e-9
    for sub in (1, 2):
        for x, y in zip(a.step(q, v, t, 1e-3, sub), f.step(q, v, t, 1e-3, sub)):
            assert (x - y).abs().max() < 1e-12


def test_merged_wrist_link_conserves_mass(P):
    f = FastArm(P, 3)
    assert abs(f.mass_n[2].item() - sum(l["mass"] for l in P["link"][2:])) < 1e-12


def test_batch_independent(P):
    torch.manual_seed(1)
    f = FastArm(P, 3)
    q = torch.rand(5, 3, dtype=torch.float64) - 0.5
    v, t = torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64)
    full = f.accel(q, v, t)
    for i in range(5):
        assert (f.accel(q[i:i + 1], v[i:i + 1], t[i:i + 1]) - full[i:i + 1]).abs().max() < 1e-12
