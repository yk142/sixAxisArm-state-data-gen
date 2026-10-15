"""閉ループ評価の制御器の移植（issue #45）。Simscape の実印加トルクを、真値の状態から再現できること。"""
import os

import h5py
import numpy as np
import pytest

from sa_nsm.closedloop import DT, R, load_scenarios

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = os.path.join(REPO, "data", "export", "closed_eval3.h5")
pytestmark = pytest.mark.skipif(not os.path.exists(PATH), reason="data/export/closed_eval3.h5 が無い（tools/sa_export_closed_eval.m）")


def test_controller_reproduces_applied_torque():
    sc, ctl = load_scenarios(PATH)
    with h5py.File(PATH, "r") as f:
        for s in [x for x in sc if x.kind == "ptp"][:3] + [x for x in sc if x.split == "test"]:
            tau8 = f[s.name]["truth_tau"][:].T
            K = s.K
            tau = (0.5 * (tau8[:-1] + tau8[1:]))[:R * K].reshape(K, R, -1).mean(1)
            a_prev = np.vstack([np.zeros((1, 3)), (s.dq[1:K] - s.dq[:K - 1]) / DT])
            qm, vm = s.q[:-1] + s.dq[:-1] * DT / 2 + a_prev * (DT / 2) ** 2 / 2, s.dq[:-1] + a_prev * DT / 2
            c = np.clip(s.u - ctl["Kp"] * qm - ctl["Kd"] * vm, -ctl["peak"], ctl["peak"])
            assert np.sqrt(((c - tau) ** 2).mean(0)).max() < 0.2, s.name


def test_scenario_counts():
    sc, _ = load_scenarios(PATH)
    assert sum(s.kind == "ptp" for s in sc) == 12
    assert sum(s.split == "eval" for s in sc) == 12 and sum(s.split == "test" and s.kind == "hold" for s in sc) == 2
