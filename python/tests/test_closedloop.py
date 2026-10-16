"""閉ループ評価の制御器の移植（issue #45）。Simscape の実印加トルクを、真値の状態から再現できること。"""
import os

import h5py
import numpy as np
import pytest

from sa_nsm.closedloop import DT, R, load_scenarios

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASES = [(n, os.path.join(REPO, "data", "export", f"closed_eval{n}.h5")) for n in (3, 6)]       # 6 軸は issue #55
CASES = [pytest.param(n, p, marks=pytest.mark.skipif(not os.path.exists(p), reason=f"{p} が無い（tools/sa_export_closed_eval.m）")) for n, p in CASES]


@pytest.mark.parametrize("n,PATH", CASES)
def test_controller_reproduces_applied_torque(n, PATH):
    sc, ctl = load_scenarios(PATH)
    with h5py.File(PATH, "r") as f:
        for s in [x for x in sc if x.kind == "ptp"][:3] + [x for x in sc if x.split == "test"]:
            tau8 = f[s.name]["truth_tau"][:].T
            K = s.K
            tau = (0.5 * (tau8[:-1] + tau8[1:]))[:R * K].reshape(K, R, -1).mean(1)
            a_prev = np.vstack([np.zeros((1, n)), (s.dq[1:K] - s.dq[:K - 1]) / DT])
            qm, vm = s.q[:-1] + s.dq[:-1] * DT / 2 + a_prev * (DT / 2) ** 2 / 2, s.dq[:-1] + a_prev * DT / 2
            c = np.clip(s.u - ctl["Kp"] * qm - ctl["Kd"] * vm, -ctl["peak"], ctl["peak"])
            assert np.sqrt(((c - tau) ** 2).mean(0)).max() < 0.2, s.name


@pytest.mark.parametrize("n,PATH", CASES)
def test_scenario_counts(n, PATH):
    sc, _ = load_scenarios(PATH)
    assert sum(s.kind == "ptp" for s in sc) == 12
    n_new = {3: 12, 6: 24}[n]                                  # 新規の保持シナリオ（nearlimit は 3 軸 12 本、6 軸 24 本）
    assert sum(s.split == "eval" for s in sc) == n_new
    assert sum(s.split == "test" and s.kind == "hold" for s in sc) == {3: 2, 6: 4}[n]
