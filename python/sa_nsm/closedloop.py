"""閉ループ評価: PD + RNEA フィードフォワード（MATLAB の buildSaClosedLoop）を 1 kHz に移植し、各モデルをプラントにする。issue #45。

    τ = sat( τ_ff + τ_dist + Kp·(q_ref − q) + Kd·(dq_ref − dq) )、Kp = Mmean·wn²、Kd = 2ζ·Mmean·wn、軸ごとに ±tauPeak

Simscape の PD は連続形（8 kHz のソルバ刻みで評価）。ここでは 1 ms の区間ごとに、
  τ_k = sat( ū_ref,k − Kp·q_mid − Kd·dq_mid )、ū_ref,k = 区間 [t_k, t_k+1 ms] の参照項 (τ_ff + τ_dist + Kp·q_ref + Kd·dq_ref) の台形平均
とする。NSM の τ は区間の台形平均なので、状態項は区間の中点の状態で評価する:
  q_mid = q_k + dq_k·h/2 + a_prev·(h/2)²/2、dq_mid = dq_k + a_prev·h/2（h = 1 ms、a_prev = 直前の区間の平均加速度）。
先頭値（ZOH）のままでは、Simscape の実印加トルクとの差が RMS で最大 6 N·m（PTP の J1）あり、中点で評価すると 0.1 N·m 以下（接触を含む保持でも 0.12 N·m 以下）になる。
"""
import h5py
import numpy as np
import torch

R = 8       # 8 kHz → 1 kHz
DT = 1e-3


class Scenario:
    """HDF5（tools/sa_export_closed_eval.m の出力）の 1 シナリオを 1 kHz に変換したもの。"""

    def __init__(self, g, ctl):
        ref, tau_d = g["ref"][:].T, g["tau_d"][:].T          # MATLAB は列優先で書くので h5py では転置して見える
        self.name = g.name.strip("/")
        self.kind, self.pattern, self.split = (g.attrs[k] for k in ("type", "pattern", "split"))
        self.kind, self.pattern, self.split = (x.decode() if isinstance(x, bytes) else str(x) for x in (self.kind, self.pattern, self.split))
        Kp, Kd = ctl["Kp"], ctl["Kd"]
        n = tau_d.shape[1]
        qr, dqr, ff = ref[:, 1:1 + n], ref[:, 1 + n:1 + 2 * n], ref[:, 1 + 2 * n:1 + 3 * n]
        u8 = ff + tau_d + Kp * qr + Kd * dqr
        K = (len(ref) - 1) // R
        pair = 0.5 * (u8[:-1] + u8[1:])
        self.u = pair[:R * K].reshape(K, R, n).mean(1)                           # [K, n]
        idx = R * np.arange(K + 1)
        self.q_ref, self.dq_ref = qr[idx], dqr[idx]
        self.q, self.dq = g["truth_q"][:].T[idx], g["truth_dq"][:].T[idx]            # Simscape の真値（1 kHz）
        self.K = K


def load_controller(f):
    c = f["controller"]
    Mmean, wn, zeta, peak = c["Mmean"][:].ravel(), float(c["wn"][:].ravel()[0]), float(c["zeta"][:].ravel()[0]), c["tauPeak"][:].ravel()
    return {"Kp": Mmean * wn ** 2, "Kd": 2 * zeta * Mmean * wn, "peak": peak}


def load_scenarios(path, splits=None):
    with h5py.File(path, "r") as f:
        ctl = load_controller(f)
        sc = [Scenario(f[k], ctl) for k in f if k != "controller"]
    return [s for s in sc if splits is None or s.split in splits or s.kind in splits], ctl


@torch.no_grad()
def closed_loop(pred, scen, ctl):
    """シナリオ（同じ種類の複数本をまとめて）を、プラント `pred.step` で閉ループ実行する。
    返り値: q, dq [B, K+1, n]（K は最長のシナリオ。短いものは末尾を参照の最後の値で延長する）。"""
    B, n = len(scen), scen[0].q.shape[1]
    K = max(s.K for s in scen)
    pad = lambda a, k: np.concatenate([a, np.repeat(a[-1:], k - len(a), 0)]) if len(a) < k else a
    U = torch.from_numpy(np.stack([pad(s.u, K) for s in scen]))
    QR = torch.from_numpy(np.stack([pad(s.q_ref, K + 1) for s in scen]))
    q = torch.from_numpy(np.stack([s.q[0] for s in scen])); dq = torch.from_numpy(np.stack([s.dq[0] for s in scen]))
    Kp, Kd, peak = (torch.as_tensor(ctl[k]) for k in ("Kp", "Kd", "peak"))
    qs, dqs = [q], [dq]
    h = DT
    a_prev = torch.zeros_like(dq)
    for k in range(K):
        q_mid = q + dq * h / 2 + a_prev * (h / 2) ** 2 / 2
        dq_mid = dq + a_prev * h / 2
        tau = torch.minimum(torch.maximum(U[:, k] - Kp * q_mid - Kd * dq_mid, -peak), peak)
        q1, dq1 = pred.step(q, dq, tau)
        a_prev = (dq1 - dq) / h
        q, dq = q1, dq1
        qs.append(q); dqs.append(dq)
    return torch.stack(qs, 1).numpy(), torch.stack(dqs, 1).numpy()
