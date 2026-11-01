"""③ 純 NN（連続時間）の numba 版（issue #111）。fit_pure_el.py の pure.npz を読み、twomass_nb と同じ ROS2 で積分する。
f(y, τ, payload) = [dq, NN_ddq, dqm, NN_ddqm]。ヤコビアンは NN の微分（tanh の 2 層 MLP を連鎖律で）から作り、1 ms の始点で 1 回だけ計算する
（jac_sub=1 ならサブステップごと）。アンサンブル（第 0 軸に積んだ複数の MLP）は出力と微分の平均。物理の式は使わない。"""
import numpy as np
from numba import njit

from .twomass_nb import _lu, _lu_solve

NIN = 37


@njit(cache=True)
def _feat(y, tau, pl):
    n = 6
    x = np.empty(NIN)
    for i in range(n):
        q, dq, qm, dqm = y[i], y[n + i], y[2 * n + i], y[3 * n + i]
        x[i] = np.sin(q); x[6 + i] = np.cos(q); x[12 + i] = dq; x[18 + i] = qm - q; x[24 + i] = dqm - dq; x[30 + i] = tau[i]
    x[36] = pl
    return x


@njit(cache=True)
def _nn(y, tau, pl, P, jac):
    """返り値 (acc [12], G [12, 37] = ∂acc/∂特徴量（jac=0 なら 0）)。"""
    W0, b0, W1, b1, W2, b2, xm, xs, ym, ys = P
    nm, H = b0.shape[0], b0.shape[1]
    f = _feat(y, tau, pl)
    acc = np.zeros(12); G = np.zeros((12, NIN))
    for m in range(nm):
        x = (f - xm[m]) / xs[m]
        h1 = np.empty(H); s1 = np.empty(H)
        for a in range(H):
            v = b0[m, a]
            for j in range(NIN):
                v += W0[m, a, j] * x[j]
            h1[a] = np.tanh(v); s1[a] = 1.0 - h1[a] ** 2
        h2 = np.empty(H); s2 = np.empty(H)
        for a in range(H):
            v = b1[m, a]
            for b in range(H):
                v += W1[m, a, b] * h1[b]
            h2[a] = np.tanh(v); s2[a] = 1.0 - h2[a] ** 2
        for o in range(12):
            v = b2[m, o]
            for b in range(H):
                v += W2[m, o, b] * h2[b]
            acc[o] += (v * ys[m, o] + ym[m, o]) / nm
        if jac:
            # C = W2·diag(s2)·W1·diag(s1)·W0（12 × 37）。先に W2·diag(s2)（12 × H）、次に ·W1（12 × H）、·diag(s1)·W0
            A = np.empty((12, H))
            for o in range(12):
                for b in range(H):
                    A[o, b] = W2[m, o, b] * s2[b]
            B = np.zeros((12, H))
            for o in range(12):
                for b in range(H):
                    v = A[o, b]
                    if v != 0.0:
                        for c in range(H):
                            B[o, c] += v * W1[m, b, c]
            for o in range(12):
                for c in range(H):
                    v = B[o, c] * s1[c]
                    for j in range(NIN):
                        G[o, j] += v * W0[m, c, j] * ys[m, o] / xs[m, j] / nm
    return acc, G


@njit(cache=True)
def _f(y, tau, pl, P):
    n = 6
    acc, _ = _nn(y, tau, pl, P, 0)
    out = np.empty(4 * n)
    out[:n] = y[n:2 * n]; out[n:2 * n] = acc[:n]; out[2 * n:3 * n] = y[3 * n:]; out[3 * n:] = acc[n:]
    return out


@njit(cache=True)
def _jac(y, tau, pl, P):
    """J = ∂f/∂y [24, 24]。特徴量の微分: sin q → cos q、cos q → −sin q、θ = qm − q、dθ = dqm − dq。"""
    n = 6
    _, G = _nn(y, tau, pl, P, 1)
    J = np.zeros((4 * n, 4 * n))
    for i in range(n):
        J[i, n + i] = 1.0; J[2 * n + i, 3 * n + i] = 1.0
    for o in range(12):
        r = n + o if o < n else 3 * n + (o - n)
        for i in range(n):
            q = y[i]
            J[r, i] += G[o, i] * np.cos(q) - G[o, 6 + i] * np.sin(q) - G[o, 18 + i]
            J[r, n + i] += G[o, 12 + i] - G[o, 24 + i]
            J[r, 2 * n + i] += G[o, 18 + i]
            J[r, 3 * n + i] += G[o, 24 + i]
    return J


@njit(cache=True)
def _step1(y, tau, pl, h, nsub, jac_sub, P):
    gam = 1.0 + 1.0 / np.sqrt(2.0)
    I24 = np.eye(24)
    WLU = np.empty((24, 24)); Wpiv = np.empty(24, np.int64)
    for s in range(nsub):
        if s == 0 or jac_sub:
            J = _jac(y, tau, pl, P)
            WLU, Wpiv = _lu(I24 - gam * h * J)
        k1 = _lu_solve(WLU, Wpiv, _f(y, tau, pl, P))
        y1 = y + h * k1
        k2 = _lu_solve(WLU, Wpiv, _f(y1, tau, pl, P) - 2.0 * k1)
        y = y + 1.5 * h * k1 + 0.5 * h * k2
    return y


def _load(path):
    d = np.load(path)
    W0 = d["W0"]; nm = W0.shape[0]
    st = lambda k: np.asarray(d[k], float).reshape(nm, -1)
    return (np.asarray(W0, float), st("b0"), np.asarray(d["W1"], float), st("b1"), np.asarray(d["W2"], float), st("b2"),
            st("xm"), st("xs"), st("ym"), st("ys")), np.asarray(d["jm"], float)


class PureNB:
    """step(q, dq, qm, dqm, tau) は twomass_nb.TwoMassNB と同じ形（closed_loop_el から呼ぶ）。"""
    def __init__(self, path, dt=1e-3, substeps=2, jac_sub=0):
        self.P, jm = _load(path)
        self.dt, self.substeps, self.jac_sub = dt, substeps, jac_sub
        import torch
        self.Jm = torch.from_numpy(jm.copy())          # 学習したモータ慣性（closed_loop_el のトルク反復の緩和）
        self.payload = 0.0

    def step_np(self, y, tau):
        return _step1(y, tau, float(self.payload), self.dt / self.substeps, self.substeps, self.jac_sub, self.P)

    def step(self, q, dq, qm, dqm, tau):
        is_t = hasattr(q, "numpy")
        cv = (lambda x: x.detach().numpy().astype(float)) if is_t else (lambda x: np.asarray(x, float))
        Y = np.concatenate([cv(q), cv(dq), cv(qm), cv(dqm)], -1); T = cv(tau)
        out = np.empty_like(Y)
        for b in range(Y.shape[0]):
            out[b] = self.step_np(Y[b], T[b])
        if is_t:
            import torch
            out = torch.from_numpy(out)
        return tuple(out[:, i * 6:(i + 1) * 6] for i in range(4))
