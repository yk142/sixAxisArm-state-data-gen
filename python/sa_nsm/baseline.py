"""学習前の基準線（解析モデル、状態を変えない）の 1 ステップ予測と、加速度換算の誤差。"""
import numpy as np
import torch

from .physics import Arm


def analytic_delta(arm, q, dq, tau, dt=1e-3, substeps=2, chunk=50000):
    """解析モデルの 1 ステップ予測 (Δq, Δdq)（numpy、[N,n]）。"""
    dqs, dvs = [], []
    with torch.no_grad():
        for s in range(0, q.shape[0], chunk):
            a, b = arm.step(torch.from_numpy(q[s:s + chunk]), torch.from_numpy(dq[s:s + chunk]),
                            torch.from_numpy(tau[s:s + chunk]), dt, substeps)
            dqs.append(a.numpy()); dvs.append(b.numpy())
    return np.concatenate(dqs), np.concatenate(dvs)


def apply_stopper(arm, q, dq, d_q, d_dq):
    """解析モデルの予測に、可動範囲の端で位置を止め外向きの速度を 0 にする単純な非弾性ストッパを付ける（接触の公平な基準線）。"""
    q1, v1 = q + d_q, dq + d_dq
    qmax, qmin = arm.q_max.numpy(), arm.q_min.numpy()
    hi, lo = q1 > qmax, q1 < qmin
    q1 = np.where(hi, qmax, np.where(lo, qmin, q1))
    v1 = np.where(hi & (v1 > 0), 0.0, np.where(lo & (v1 < 0), 0.0, v1))
    return q1 - q, v1 - dq


def accel_rmse(d_pred, d_true, dt=1e-3, mask=None):
    """Δdq の誤差を加速度換算 [rad/s²] の RMSE にする（軸別）。mask: 対象の遷移。"""
    if mask is not None:
        d_pred, d_true = d_pred[mask], d_true[mask]
    return np.sqrt(np.mean((d_pred - d_true) ** 2, axis=0)) / dt


def accel_rms(d_true, dt=1e-3, mask=None):
    """目標（真の加速度）自体の RMS。誤差の大きさの目安に並べる。"""
    if mask is not None:
        d_true = d_true[mask]
    return np.sqrt(np.mean(d_true ** 2, axis=0)) / dt


def groups(ds, split, dt=1e-3):
    """評価区分のマスク: all / free（非接触）/ contact / dynamic（動的接触）/ static（静止押し付け）と、パターン別。"""
    n = ds.n_ax
    q, dq, tau = ds.q_dq_tau(split)
    d_q, d_dq = ds.delta(split)
    at = ds.at_limit(split)
    moving = (np.abs(dq).max(1) > 0.02) | (np.abs(d_dq).max(1) > 0.005)
    g = {"all": np.ones(len(at), bool), "free": ~at, "contact": at, "dynamic": at & moving, "static": at & ~moving}
    pat = ds.pattern(split)
    for p in np.unique(pat):
        g["pat:" + p] = pat == p
    return {k: v for k, v in g.items() if v.any()}


def evaluate_baselines(ds, arm, splits=("val", "test", "benchmark"), dt=1e-3, substeps=2, limit=None, seed=0):
    """{split: {区分: {"n", "target", "analytic", "persist"}}}。軸別の配列。limit: split ごとの最大遷移数（等間隔抽出）。"""
    out = {}
    for sp in splits:
        if sp not in ds.part:
            continue
        q, dq, tau = ds.q_dq_tau(sp)
        d_q, d_dq = ds.delta(sp)
        gm = groups(ds, sp, dt)
        idx = np.arange(len(q))
        if limit and len(q) > limit:
            idx = np.linspace(0, len(q) - 1, limit).astype(int)
        pq, pv = analytic_delta(arm, q[idx], dq[idx], tau[idx], dt, substeps)
        _, sv = apply_stopper(arm, q[idx], dq[idx], pq, pv)
        res = {}
        for k, m in gm.items():
            mi = m[idx]
            if not mi.any():
                continue
            res[k] = {"n": int(mi.sum()), "target": accel_rms(d_dq[idx], dt, mi),
                      "analytic": accel_rmse(pv, d_dq[idx], dt, mi),
                      "analytic_stop": accel_rmse(sv, d_dq[idx], dt, mi),
                      "persist": accel_rmse(np.zeros_like(pv), d_dq[idx], dt, mi)}
        out[sp] = res
    return out
