"""短い窓の開ループ評価: 窓の先頭の実測状態から、実測のトルク列で L ステップ予測し、真の状態列との誤差を測る。
全長の開ループ再生はしない（docs/handover.md §3）。方策: persist（状態維持）/ analytic（解析モデル）/ hybrid（リミット拘束中は状態維持、他は解析）/ NN（A・B・C）。"""
import math

import numpy as np
import torch

from .model import limit_features

TOL_LIMIT = math.radians(0.1)       # at_limit の判定幅（saAssembleDataset と同じ 0.1°）


def make_windows(ds, split, L, stride):
    """同一シナリオ内で連続する L 遷移の窓の先頭行を返す。"""
    d = ds.part[split]
    sid, st = d["scenario_id"], d["step"]
    n = len(sid)
    ok = np.zeros(n, bool)
    if n > L:
        ok[: n - L + 1] = (sid[: n - L + 1] == sid[L - 1:]) & (st[L - 1:] - st[: n - L + 1] == L - 1)
    starts = np.flatnonzero(ok)
    keep, last = [], -10 ** 9
    for s in starts:                # シナリオごとに stride 間隔で間引く
        if s - last >= stride or sid[s] != sid[last]:
            keep.append(s); last = s
    return np.array(keep, dtype=int)


class Predictor:
    """状態 (q, dq) とトルクから次状態を返す。kind: persist / analytic / hybrid / model。"""

    def __init__(self, arm, ds, kind, model=None, d_switch_deg=None):
        self.arm, self.kind, self.model, self.sc = arm, kind, model, ds.scale
        self.d_switch = math.radians(d_switch_deg) if d_switch_deg is not None else None
        self.n_model = self.n_all = 0           # switch: モデルを使った（窓×ステップ）数

    @torch.no_grad()
    def step(self, q, dq, tau):
        arm, sc = self.arm, self.sc
        if self.kind == "persist":
            return q, dq
        dqa, dva = arm.step(q, dq, tau, 1e-3, 2)
        if self.kind == "analytic":
            return q + dqa, dq + dva
        if self.kind == "hybrid":
            lim = ((q <= arm.q_min + TOL_LIMIT) | (q >= arm.q_max - TOL_LIMIT)).any(-1, keepdim=True)
            return torch.where(lim, q, q + dqa), torch.where(lim, dq, dq + dva)
        D_Q, D_DQ = (torch.as_tensor(sc[k]) for k in ("D_Q_MAX", "D_DQ_MAX"))
        near = None
        if self.kind == "switch":       # いずれかの軸がリミットから d_switch 以内 → NN、他は解析
            near = (torch.minimum(arm.q_max - q, q - arm.q_min) < self.d_switch).any(-1, keepdim=True)
            self.n_model += int(near.sum()); self.n_all += len(q)
        xs = torch.cat([torch.sin(q), torch.cos(q), dq / torch.as_tensor(sc["DQ_MAX"]), tau / torch.as_tensor(sc["TAU_MAX"])], -1).float()
        ana = torch.cat([dqa / D_Q, dva / D_DQ], -1).float()
        lim = limit_features(q, arm.q_min, arm.q_max).float()
        p = self.model(xs, ana, lim).double()
        n = q.shape[-1]
        if near is not None:
            return torch.where(near, q + p[:, :n] * D_Q, q + dqa), torch.where(near, dq + p[:, n:] * D_DQ, dq + dva)
        return q + p[:, :n] * D_Q, dq + p[:, n:] * D_DQ


@torch.no_grad()
def rollout(pred, q0, dq0, tau_seq):
    """q0,dq0: [B,n]、tau_seq: [B,L,n] → 予測状態列 q,dq: [B,L+1,n]。"""
    q, dq = q0, dq0
    qs, dqs = [q], [dq]
    for k in range(tau_seq.shape[1]):
        q, dq = pred.step(q, dq, tau_seq[:, k])
        qs.append(q); dqs.append(dq)
    return torch.stack(qs, 1), torch.stack(dqs, 1)


def true_states(ds, split, starts, L):
    """窓ごとの真の状態列 [B,L+1,n]（q, dq）とトルク列 [B,L,n]。末尾の状態は最後の遷移の真の Δ で作る。"""
    q, dq, tau = ds.q_dq_tau(split)
    dQ, dV = ds.delta(split)
    idx = starts[:, None] + np.arange(L)[None]
    qt = np.concatenate([q[idx], (q[idx[:, -1]] + dQ[idx[:, -1]])[:, None]], 1)
    vt = np.concatenate([dq[idx], (dq[idx[:, -1]] + dV[idx[:, -1]])[:, None]], 1)
    return qt, vt, tau[idx]
