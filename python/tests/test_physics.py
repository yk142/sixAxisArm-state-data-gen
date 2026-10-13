"""解析モデルの Python 移植の検証。MATLAB の saFwdDyn（`tools/sa_export_params.m` の出力）と一致すること。"""
import json
import os

import numpy as np
import pytest
import torch

from sa_nsm.baseline import accel_rmse, apply_stopper
from sa_nsm.physics import Arm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARAMS = os.path.join(REPO, "data", "export", "c8_params.json")
pytestmark = pytest.mark.skipif(not os.path.exists(PARAMS), reason="data/export/c8_params.json が無い（tools/sa_export_params.m で作る）")


@pytest.fixture(scope="module")
def P():
    with open(PARAMS) as f:
        return json.load(f)


@pytest.mark.parametrize("k", [0, 1])
def test_accel_matches_matlab(P, k):
    T = P["test"][k]
    n = T["nAx"]
    arm = Arm(P, n)
    q, dq, tau, ddq = (torch.tensor(T[x], dtype=torch.float64).reshape(-1, n) for x in ("q", "dq", "tau", "ddq"))
    a = arm.accel(q, dq, tau)
    assert (a - ddq).abs().max() < 1e-8 * max(1.0, ddq.abs().max().item())


def test_mass_matrix_symmetric_positive_definite(P):
    arm = Arm(P, 3)
    q = torch.tensor([[0.3, -0.5, 0.8], [-1.0, 0.2, 1.2]], dtype=torch.float64)
    _, _, M = arm.matrices(q, torch.zeros_like(q))
    assert torch.allclose(M, M.transpose(-1, -2), atol=1e-9)
    assert (torch.linalg.eigvalsh(M) > 0).all()


def test_gravity_hold_is_equilibrium(P):
    arm = Arm(P, 3)
    q = torch.tensor([[0.3, -0.5, 0.8]], dtype=torch.float64)
    g, _, _ = arm.matrices(q, torch.zeros_like(q))
    a = arm.accel(q, torch.zeros_like(q), g)                  # 重力保持トルクで加速度 0（速度 0 なので摩擦も 0）
    assert a.abs().max() < 1e-9


def test_step_is_consistent_with_small_dt(P):
    arm = Arm(P, 3)
    q = torch.tensor([[0.3, -0.5, 0.8]], dtype=torch.float64)
    dq = torch.tensor([[0.4, -0.2, 0.3]], dtype=torch.float64)
    tau = torch.tensor([[5.0, 20.0, -3.0]], dtype=torch.float64)
    _, dv1 = arm.step(q, dq, tau, dt=1e-3, substeps=2)
    _, dv2 = arm.step(q, dq, tau, dt=1e-3, substeps=16)
    assert (dv1 - dv2).abs().max() < 1e-9                    # 1 ms・2 分割の RK4 は十分に収束している


def test_stopper_clamps_and_zeroes_outward_velocity(P):
    arm = Arm(P, 3)
    qmax = arm.q_max.numpy()
    q = (qmax - 1e-4)[None]
    dq = np.array([[0.5, 0.0, 0.0]])
    d_q, d_dq = np.array([[2e-3, 0.0, 0.0]]), np.array([[0.01, 0.0, 0.0]])
    dq_, dv_ = apply_stopper(arm, q, dq, d_q, d_dq)
    assert np.isclose(q[0, 0] + dq_[0, 0], qmax[0]) and np.isclose(dq[0, 0] + dv_[0, 0], 0.0)
    assert accel_rmse(np.zeros((1, 3)), np.zeros((1, 3))).max() == 0
