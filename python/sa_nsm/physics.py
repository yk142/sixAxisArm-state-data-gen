"""解析モデル（RNEA ＋ 摩擦）の torch 移植。MATLAB の `c8RNEA` と `saFwdDyn` と同じ式（バッチ対応、倍精度）。

    M(q)·ddq = τ − g(q) − C(q,dq)·dq − (Bv·dq + Fc·tanh(dq/ε))

c8RNEA は armature（反映慣性）を含み摩擦は含まない。可動でない軸（nAx+1..6）は 0 rad で固定する（手首固定の 3 軸）。
パラメータは `tools/sa_export_params.m` が書く JSON（MATLAB の c8_params 由来）から読む。
"""
import json

import torch

DTYPE = torch.float64


def _rpy2rotm(r):
    cr, sr, cp, sp, cy, sy = (torch.cos(r[0]), torch.sin(r[0]), torch.cos(r[1]), torch.sin(r[1]),
                              torch.cos(r[2]), torch.sin(r[2]))
    rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=DTYPE)
    ry = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=DTYPE)
    rx = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=DTYPE)
    return rz @ ry @ rx


class Arm:
    """C8-A901S の解析モデル。`n_ax`: 可動軸数（3 または 6）。"""

    def __init__(self, params, n_ax=3):
        P = params
        t = lambda x: torch.as_tensor(x, dtype=DTYPE)
        self.n_ax = int(n_ax)
        self.gravity = t(P["gravity"]).reshape(3)
        self.r_base = _rpy2rotm(t(P["baseRpy"]).reshape(3))
        self.R0 = torch.stack([_rpy2rotm(t(j["rpy"]).reshape(3)) for j in P["joint"]])      # (6,3,3)
        self.pj = torch.stack([t(j["xyz"]).reshape(3) for j in P["joint"]])                 # (6,3)
        self.mass = t([l["mass"] for l in P["link"]])
        self.com = torch.stack([t(l["com"]).reshape(3) for l in P["link"]])
        self.Ic = torch.stack([t(l["inertiaC"]).reshape(3, 3) for l in P["link"]])
        self.armature = t(P["armature"]).reshape(6)
        self.Fc, self.Bv, self.eps = (t(P[k]).reshape(6)[:n_ax] for k in ("frictionFc", "frictionBv", "frictionEps"))
        self.q_min, self.q_max, self.dq_max, self.tau_peak, self.tau_rated = (
            t(P[k]).reshape(6)[:n_ax] for k in ("qMin", "qMax", "qdMax", "tauPeak", "tauRated"))

    @classmethod
    def from_json(cls, path, n_ax=3):
        with open(path) as f:
            return cls(json.load(f), n_ax)

    # ---- 逆動力学（再帰ニュートン・オイラー法。バッチ [B,6]）----
    def rnea(self, q, dq, ddq):
        B = q.shape[0]
        z = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE).expand(B, 3)
        cq, sq = torch.cos(q), torch.sin(q)
        zero, one = torch.zeros_like(cq), torch.ones_like(cq)
        A = []
        for i in range(6):
            rz = torch.stack([torch.stack([cq[:, i], -sq[:, i], zero[:, i]], -1),
                              torch.stack([sq[:, i], cq[:, i], zero[:, i]], -1),
                              torch.stack([zero[:, i], zero[:, i], one[:, i]], -1)], -2)
            A.append(self.R0[i] @ rz)                                                      # フレーム i → i-1
        om = torch.zeros(B, 3, dtype=DTYPE)
        al = torch.zeros(B, 3, dtype=DTYPE)
        ac = (self.r_base.T @ (-self.gravity)).expand(B, 3)
        oms, als, acs = [], [], []
        for i in range(6):
            RT = A[i].transpose(-1, -2)
            omR = torch.einsum("bij,bj->bi", RT, om)
            zd = z * dq[:, i:i + 1]
            pj = self.pj[i]
            om_n = omR + zd
            al_n = torch.einsum("bij,bj->bi", RT, al) + z * ddq[:, i:i + 1] + torch.linalg.cross(omR, zd)
            tmp = ac + torch.linalg.cross(al, pj.expand(B, 3)) + torch.linalg.cross(om, torch.linalg.cross(om, pj.expand(B, 3)))
            ac = torch.einsum("bij,bj->bi", RT, tmp)
            om, al = om_n, al_n
            oms.append(om); als.append(al); acs.append(ac)
        tau = [None] * 6
        f_next = torch.zeros(B, 3, dtype=DTYPE)
        n_next = torch.zeros(B, 3, dtype=DTYPE)
        for i in range(5, -1, -1):
            c = self.com[i].expand(B, 3)
            aG = acs[i] + torch.linalg.cross(als[i], c) + torch.linalg.cross(oms[i], torch.linalg.cross(oms[i], c))
            F = self.mass[i] * aG
            Iw = torch.einsum("ij,bj->bi", self.Ic[i], oms[i])
            N = torch.einsum("ij,bj->bi", self.Ic[i], als[i]) + torch.linalg.cross(oms[i], Iw)
            if i < 5:
                Rn = A[i + 1]
                fN = torch.einsum("bij,bj->bi", Rn, f_next)
                nN = torch.einsum("bij,bj->bi", Rn, n_next)
                pN = self.pj[i + 1].expand(B, 3)
            else:
                fN = torch.zeros(B, 3, dtype=DTYPE); nN = fN; pN = fN
            f_this = fN + F
            n_this = nN + N + torch.linalg.cross(c, F) + torch.linalg.cross(pN, fN)
            tau[i] = n_this[:, 2] + self.armature[i] * ddq[:, i]
            f_next, n_next = f_this, n_this
        return torch.stack(tau, -1)

    def _pad(self, x):
        x = x.to(DTYPE)
        return torch.cat([x, torch.zeros(x.shape[0], 6 - self.n_ax, dtype=DTYPE)], -1) if self.n_ax < 6 else x

    def matrices(self, q, dq):
        """重力 g(q)、コリオリ・遠心 b(q,dq)、質量行列 M(q)（可動軸のみ）を返す。q, dq: [B,n]。"""
        n, B = self.n_ax, q.shape[0]
        qq, dd = self._pad(q), self._pad(dq)
        z6 = torch.zeros(B, 6, dtype=DTYPE)
        E = torch.eye(6, dtype=DTYPE)[:n]
        # 1 回の呼び出しにまとめる: [g, b, M の n 列]
        Q = torch.cat([qq] * (n + 2), 0)
        D = torch.cat([z6, dd] + [z6] * n, 0)
        DD = torch.cat([z6, z6] + [E[j].expand(B, 6) for j in range(n)], 0)
        T = self.rnea(Q, D, DD)
        g = T[:B]
        b = T[B:2 * B] - g
        M = torch.stack([T[(2 + j) * B:(3 + j) * B] - g for j in range(n)], -1)           # [B,6,n]
        return g[:, :n], b[:, :n], M[:, :n, :]

    def friction(self, dq):
        return self.Bv * dq + self.Fc * torch.tanh(dq / self.eps)

    def accel(self, q, dq, tau):
        """前進動力学（MATLAB saFwdDyn と同じ）。q, dq, tau: [B,n] → ddq [B,n]。"""
        q, dq, tau = q.to(DTYPE), dq.to(DTYPE), tau.to(DTYPE)
        g, b, M = self.matrices(q, dq)
        rhs = tau - g - b - self.friction(dq)
        return torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)

    def step(self, q, dq, tau, dt=1e-3, substeps=2):
        """トルクを 1 ステップ一定として RK4 で dt 進める。返り値: (Δq, Δdq)。"""
        q0, v0 = q.to(DTYPE), dq.to(DTYPE)
        q, v, h = q0, v0, dt / substeps
        for _ in range(substeps):
            k1q, k1v = v, self.accel(q, v, tau)
            k2q, k2v = v + 0.5 * h * k1v, self.accel(q + 0.5 * h * k1q, v + 0.5 * h * k1v, tau)
            k3q, k3v = v + 0.5 * h * k2v, self.accel(q + 0.5 * h * k2q, v + 0.5 * h * k2v, tau)
            k4q, k4v = v + h * k3v, self.accel(q + h * k3q, v + h * k3v, tau)
            q = q + h / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
            v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        return q - q0, v - v0
