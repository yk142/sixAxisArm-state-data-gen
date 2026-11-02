"""リンク側の状態の再構成（issue #115）。測ったモータ側の角度 qm（1 kHz）を入力にして、リンク側（剛体＋ばね＋摩擦）だけを模擬し、q̂, dq̂ を作る。
リンク側はばねで qm につながっているので、模擬は発散しない（誤差は積み上がらない）。
式: M(q)·ddq = k·θ + k3·θ³ + c·dθ − (g + b) − Bv·dq − Fc·tanh(dq/ε) − Σ_j a_j·exp(−(dq/(m_j·ε))²)·tanh(dq/ε)、θ = qm − q。
積分は ROS2（twomass_nb と同じ）、M と g+b は 1 ms の始点で 1 回だけ計算する。qm, dqm は 1 ms の中で 3 次エルミート補間（dqm は中心差分）。"""
import numpy as np
from numba import njit

from .twomass_nb import _lu, _lu_solve, _mass_bias


@njit(cache=True)
def _fric(dq, Bv, Fc, eps, A, mb):
    """摩擦トルクと、その dq による微分の正の部分（ヤコビアン用）。A: [6, nb] 基底の係数、mb: [nb] 基底の幅の倍率。"""
    n = 6
    f = np.empty(n); df = np.empty(n)
    for i in range(n):
        t = np.tanh(dq[i] / eps[i]); s2 = 1.0 - t * t
        f[i] = Bv[i] * dq[i] + Fc[i] * t
        d = Bv[i] + Fc[i] * s2 / eps[i]
        for j in range(mb.shape[0]):
            w = mb[j] * eps[i]; e = np.exp(-(dq[i] / w) ** 2)
            f[i] += A[i, j] * e * t
            dj = A[i, j] * e * (s2 / eps[i] - 2.0 * dq[i] / w ** 2 * t)
            if dj > 0.0:
                d += dj
        df[i] = d
    return f, df


@njit(cache=True)
def _herm(p0, p1, v0, v1, u, dt):
    """3 次エルミート補間（位置と速度、u ∈ [0, 1]）。"""
    u2 = u * u; u3 = u2 * u
    p = (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * dt * v0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * dt * v1
    v = ((6 * u2 - 6 * u) * p0 + (3 * u2 - 4 * u + 1) * dt * v0 + (-6 * u2 + 6 * u) * p1 + (3 * u2 - 2 * u) * dt * v1) / dt
    return p, v


@njit(cache=True)
def _acc(q, dq, qm, dqm, MLU, Mpiv, gb, k, k3, c, Bv, Fc, eps, A, mb):
    th = qm - q
    ts = k * th + k3 * th ** 3 + c * (dqm - dq)
    f, _ = _fric(dq, Bv, Fc, eps, A, mb)
    return _lu_solve(MLU, Mpiv, ts - gb - f)


@njit(cache=True)
def simulate(qm, dqm, q0, dq0, nsub, R0, pj, mass, com, Ic, g0, k, k3, c, Bv, Fc, eps, A, mb, dt):
    """qm, dqm: [T, 6]（1 kHz の測定）→ q̂, dq̂ [T, 6]（同じ時刻）。q0, dq0: 初期状態。"""
    n = 6; T = qm.shape[0]
    Q = np.empty((T, n)); DQ = np.empty((T, n))
    q = q0.copy(); dq = dq0.copy()
    Q[0] = q; DQ[0] = dq
    h = dt / nsub; gam = 1.0 + 1.0 / np.sqrt(2.0)
    I12 = np.eye(2 * n)
    for t in range(T - 1):
        M, gb = _mass_bias(q, dq, R0, pj, mass, com, Ic, g0)
        MLU, Mpiv = _lu(M)
        Mi = np.empty((n, n))
        for j in range(n):
            e = np.zeros(n); e[j] = 1.0
            Mi[:, j] = _lu_solve(MLU, Mpiv, e)
        for s in range(nsub):
            a0 = s / nsub; a1 = (s + 1) / nsub
            qa, dqa = _herm(qm[t], qm[t + 1], dqm[t], dqm[t + 1], a0, dt)
            qb, dqb = _herm(qm[t], qm[t + 1], dqm[t], dqm[t + 1], a1, dt)
            _, Dfr = _fric(dq, Bv, Fc, eps, A, mb)
            ke = k + 3.0 * k3 * (qa - q) ** 2
            J = np.zeros((2 * n, 2 * n))
            for i in range(n):
                J[i, n + i] = 1.0
                for j in range(n):
                    J[n + i, j] = -Mi[i, j] * ke[j]
                    J[n + i, n + j] = -Mi[i, j] * (c[j] + Dfr[j])
            W = I12 - gam * h * J
            WLU, Wpiv = _lu(W)
            f0 = np.empty(2 * n); f0[:n] = dq; f0[n:] = _acc(q, dq, qa, dqa, MLU, Mpiv, gb, k, k3, c, Bv, Fc, eps, A, mb)
            k1 = _lu_solve(WLU, Wpiv, f0)
            q1 = q + h * k1[:n]; dq1 = dq + h * k1[n:]
            f1 = np.empty(2 * n); f1[:n] = dq1; f1[n:] = _acc(q1, dq1, qb, dqb, MLU, Mpiv, gb, k, k3, c, Bv, Fc, eps, A, mb)
            k2 = _lu_solve(WLU, Wpiv, f1 - 2.0 * k1)
            q = q + 1.5 * h * k1[:n] + 0.5 * h * k2[:n]
            dq = dq + 1.5 * h * k1[n:] + 0.5 * h * k2[n:]
        Q[t + 1] = q; DQ[t + 1] = dq
    return Q, DQ
