"""2 慣性の解析モデル（RNEA ＋ 摩擦 ＋ 減速機のねじりばね・ダンパ ＋ モータ慣性、issue #77）。弾性真値（#71）の解析版。

    リンク側:  M_link(q)·ddq = τs − g(q) − b(q,dq) − (Bv·dq + Fc·tanh(dq/ε))        （M_link は armature を除いた質量行列）
    モータ側:  Jm·ddqm = τ − τs
    τs = k·(qm − q) + c·(dqm − dq)

k, c, Jm は `saElasticParams`（tools で JSON に書き出したもの）。`step(q, dq, qm, dqm, tau)` は ElPlant と同じ形で、`payload` [kg] を属性で与える。
トルクは 1 ステップ一定（ZOH）として RK4 で `substeps` 分割して進める（手首の固有振動数 ≈ 86 Hz、1 ms では ω·dt ≈ 0.5 なので分割が要る）。"""
import copy
import json

import torch

from .physics import Arm, DTYPE


class TwoMass:
    def __init__(self, params_json, elastic_json, n_ax=6, dt=1e-3, substeps=4):
        self.params_json, self.n_ax, self.dt, self.substeps = params_json, n_ax, dt, substeps
        e = json.load(open(elastic_json))
        t = lambda x: torch.as_tensor(x, dtype=DTYPE).reshape(-1)[:n_ax]
        self.k, self.c, self.Jm = t(e["k"]), t(e["c"]), t(e["Jm"])
        self._arms, self._payload = {}, 0.0

    @property
    def payload(self):
        return self._payload

    @payload.setter
    def payload(self, m):
        self._payload = float(m)

    def arm(self):
        m = self._payload
        if m not in self._arms:
            a = Arm.from_json(self.params_json, self.n_ax, m)
            a.armature = torch.zeros_like(a.armature)          # 反映慣性はモータ側へ移す
            self._arms[m] = a
        return self._arms[m]

    def accel(self, q, dq, qm, dqm, tau):
        """(ddq, ddqm)。"""
        a = self.arm()
        ts = self.k * (qm - q) + self.c * (dqm - dq)
        g, b, M = a.matrices(q, dq)
        rhs = ts - g - b - a.friction(dq)
        return torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1), (tau - ts) / self.Jm

    def _f(self, s, tau):
        a, am = self.accel(*s, tau)
        return torch.cat([s[1], a, s[3], am], -1)

    def _jac(self, s):
        """ROS2 用の近似ヤコビアン（ばね・ダンパ・摩擦の傾き。重力・コリオリの微分は無視）。[B, 4n, 4n]。"""
        q, dq, qm, dqm = s
        a = self.arm(); n, B = self.n_ax, q.shape[0]
        _, _, M = a.matrices(q, dq)
        Mi = torch.linalg.inv(M)
        Dfr = a.Bv + a.Fc / a.eps / torch.cosh(dq / a.eps) ** 2           # 摩擦の傾き [B, n]
        K, C = torch.diag(self.k), torch.diag(self.c)
        Z, I = torch.zeros(B, n, n, dtype=DTYPE), torch.eye(n, dtype=DTYPE).expand(B, n, n)
        Dm = torch.diag_embed(Dfr)
        MK, MC = Mi @ K, Mi @ C
        Kj, Cj = (K / self.Jm[:, None]).expand(B, n, n), (C / self.Jm[:, None]).expand(B, n, n)
        rows = [torch.cat([Z, I, Z, Z], -1),
                torch.cat([-MK, -(MC + Mi @ Dm), MK, MC], -1),
                torch.cat([Z, Z, Z, I], -1),
                torch.cat([Kj, Cj, -Kj, -Cj], -1)]
        return torch.cat(rows, -2)

    @torch.no_grad()
    def step(self, q, dq, qm, dqm, tau):
        """線形陰解法 ROS2（Verwer、2 次精度・L 安定）。手首はリンク側慣性が極小（1e-3〜1e-4）で、摩擦の剛性 Fc/(ε·J) が 1e4〜1e5 /s になり、
        明示法（RK4）は 1 ms 刻みで不安定になる。ヤコビアンは近似でよい（W 法）。"""
        s = tuple(x.to(DTYPE) for x in (q, dq, qm, dqm)); tau = tau.to(DTYPE)
        n, B = self.n_ax, s[0].shape[0]
        h = self.dt / self.substeps; g = 1.0 + 1.0 / 2 ** 0.5
        Id = torch.eye(4 * n, dtype=DTYPE)
        split = lambda y: tuple(y[:, i * n:(i + 1) * n] for i in range(4))
        y = torch.cat(s, -1)
        for _ in range(self.substeps):
            W = Id - g * h * self._jac(split(y))
            f0 = self._f(split(y), tau)
            k1 = torch.linalg.solve(W, f0.unsqueeze(-1)).squeeze(-1)
            f1 = self._f(split(y + h * k1), tau)
            k2 = torch.linalg.solve(W, (f1 - 2 * k1).unsqueeze(-1)).squeeze(-1)
            y = y + 1.5 * h * k1 + 0.5 * h * k2
        return split(y)
