"""2 慣性解析モデルの numba 版（issue #91）。`twomass.TwoMass` と同じ式・同じ積分（ROS2、近似ヤコビアン）で、逐次（1 本ずつ）の速度を測るための強い基準線。

torch 版との違いは実装だけ: RNEA・質量行列・ヤコビアン・線形解法を 1 行ずつ numba（倍精度）で計算する。小さい行列なので BLAS/LAPACK は使わず、3×3 積と LU 分解を手で書く（numba の linalg は scipy を要する）。
`step(q, dq, qm, dqm, tau)` は torch のテンソルまたは numpy 配列 [B, n] を受け、同じ型で返す（B 行は順に計算）。
6 軸専用（n_ax=6）。payload [kg] を属性で与える（ペイロードごとにリンク定数を作り直す）。"""
import json

import numpy as np
from numba import njit

from .physics import Arm


@njit(cache=True)
def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


@njit(cache=True)
def _mm3(A, B):
    C = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            C[i, j] = A[i, 0] * B[0, j] + A[i, 1] * B[1, j] + A[i, 2] * B[2, j]
    return C


@njit(cache=True)
def _mv3(A, v):
    return np.array([A[0, 0] * v[0] + A[0, 1] * v[1] + A[0, 2] * v[2], A[1, 0] * v[0] + A[1, 1] * v[1] + A[1, 2] * v[2],
                     A[2, 0] * v[0] + A[2, 1] * v[1] + A[2, 2] * v[2]])


@njit(cache=True)
def _mtv3(A, v):
    """A^T v。"""
    return np.array([A[0, 0] * v[0] + A[1, 0] * v[1] + A[2, 0] * v[2], A[0, 1] * v[0] + A[1, 1] * v[1] + A[2, 1] * v[2],
                     A[0, 2] * v[0] + A[1, 2] * v[1] + A[2, 2] * v[2]])


@njit(cache=True)
def _lu(A):
    """部分ピボット付き LU 分解（小さい行列用。BLAS/LAPACK を使わない）。返り値 (LU, piv)。"""
    n = A.shape[0]
    LU = A.copy(); piv = np.arange(n)
    for k in range(n):
        p = k; mx = abs(LU[k, k])
        for i in range(k + 1, n):
            if abs(LU[i, k]) > mx:
                mx = abs(LU[i, k]); p = i
        if p != k:
            for j in range(n):
                t = LU[k, j]; LU[k, j] = LU[p, j]; LU[p, j] = t
            t2 = piv[k]; piv[k] = piv[p]; piv[p] = t2
        for i in range(k + 1, n):
            LU[i, k] /= LU[k, k]
            f = LU[i, k]
            if f != 0.0:
                for j in range(k + 1, n):
                    LU[i, j] -= f * LU[k, j]
    return LU, piv


@njit(cache=True)
def _lu_solve(LU, piv, b):
    n = LU.shape[0]
    x = np.empty(n)
    for i in range(n):
        x[i] = b[piv[i]]
    for i in range(n):
        for j in range(i):
            x[i] -= LU[i, j] * x[j]
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            x[i] -= LU[i, j] * x[j]
        x[i] /= LU[i, i]
    return x


@njit(cache=True)
def _rnea(q, dq, ddq, R0, pj, mass, com, Ic, g0):
    """逆動力学（armature なし）。q, dq, ddq: [6]。"""
    n = 6
    A = np.empty((n, 3, 3))
    for i in range(n):
        c, s = np.cos(q[i]), np.sin(q[i])
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        A[i] = _mm3(R0[i], Rz)
    om = np.zeros(3); al = np.zeros(3); ac = g0.copy()
    oms = np.empty((n, 3)); als = np.empty((n, 3)); acs = np.empty((n, 3))
    for i in range(n):
        omR = _mtv3(A[i], om)
        zd = np.array([0.0, 0.0, dq[i]])
        tmp = ac + _cross(al, pj[i]) + _cross(om, _cross(om, pj[i]))
        al = _mtv3(A[i], al) + np.array([0.0, 0.0, ddq[i]]) + _cross(omR, zd)
        om = omR + zd
        ac = _mtv3(A[i], tmp)
        oms[i] = om; als[i] = al; acs[i] = ac
    tau = np.empty(n)
    f = np.zeros(3); nn = np.zeros(3)
    for i in range(n - 1, -1, -1):
        cm = com[i]
        aG = acs[i] + _cross(als[i], cm) + _cross(oms[i], _cross(oms[i], cm))
        F = mass[i] * aG
        N = _mv3(Ic[i], als[i]) + _cross(oms[i], _mv3(Ic[i], oms[i]))
        if i == n - 1:
            f_this = F
            n_this = N + _cross(cm, F)
        else:
            fN = _mv3(A[i + 1], f)
            nN = _mv3(A[i + 1], nn)
            f_this = fN + F
            n_this = nN + N + _cross(cm, F) + _cross(pj[i + 1], fN)
        tau[i] = n_this[2]
        f = f_this; nn = n_this
    return tau


@njit(cache=True)
def _mass_bias(q, dq, R0, pj, mass, com, Ic, g0):
    """M(q) [6,6] と g+b = rnea(q, dq, 0) [6]。"""
    n = 6
    z = np.zeros(n)
    g = _rnea(q, z, z, R0, pj, mass, com, Ic, g0)
    gb = _rnea(q, dq, z, R0, pj, mass, com, Ic, g0)
    M = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = 1.0
        M[:, j] = _rnea(q, z, e, R0, pj, mass, com, Ic, g0) - g
    return M, gb


@njit(cache=True)
def _f(y, tau, MLU, Mpiv, gb, k, c, Jm, Bv, Fc, eps):
    n = 6
    q, dq, qm, dqm = y[:n], y[n:2 * n], y[2 * n:3 * n], y[3 * n:]
    ts = k * (qm - q) + c * (dqm - dq)
    rhs = ts - gb - (Bv * dq + Fc * np.tanh(dq / eps))
    out = np.empty(4 * n)
    out[:n] = dq
    out[n:2 * n] = _lu_solve(MLU, Mpiv, rhs)
    out[2 * n:3 * n] = dqm
    out[3 * n:] = (tau - ts) / Jm
    return out


@njit(cache=True)
def _step1(y, tau, h, nsub, frozen, R0, pj, mass, com, Ic, g0, k, c, Jm, Bv, Fc, eps):
    """1 行の ROS2（frozen=0 は twomass.TwoMass._step と同じ）。y: [24]。
    frozen=1: M(q) と g+b を 1 ms の始点で 1 回だけ計算し、サブステップ内で使い回す（RNEA 8 回/ms。q の変化は 1 ms で小さい）。"""
    n = 6
    gam = 1.0 + 1.0 / np.sqrt(2.0)
    I24 = np.eye(4 * n)
    M, gb = _mass_bias(y[:n], y[n:2 * n], R0, pj, mass, com, Ic, g0)
    for _ in range(nsub):
        q, dq = y[:n], y[n:2 * n]
        if frozen == 0:
            M, gb = _mass_bias(q, dq, R0, pj, mass, com, Ic, g0)
        MLU, Mpiv = _lu(M)
        Mi = np.empty((n, n))
        for j in range(n):
            e = np.zeros(n); e[j] = 1.0
            Mi[:, j] = _lu_solve(MLU, Mpiv, e)
        Dfr = Bv + Fc / eps / np.cosh(dq / eps) ** 2
        J = np.zeros((4 * n, 4 * n))
        for i in range(n):
            J[i, n + i] = 1.0
            J[2 * n + i, 3 * n + i] = 1.0
            for j in range(n):
                J[n + i, j] = -Mi[i, j] * k[j]
                J[n + i, n + j] = -(Mi[i, j] * c[j] + Mi[i, j] * Dfr[j])
                J[n + i, 2 * n + j] = Mi[i, j] * k[j]
                J[n + i, 3 * n + j] = Mi[i, j] * c[j]
            J[3 * n + i, i] = k[i] / Jm[i]
            J[3 * n + i, n + i] = c[i] / Jm[i]
            J[3 * n + i, 2 * n + i] = -k[i] / Jm[i]
            J[3 * n + i, 3 * n + i] = -c[i] / Jm[i]
        W = I24 - gam * h * J
        WLU, Wpiv = _lu(W)
        f0 = _f(y, tau, MLU, Mpiv, gb, k, c, Jm, Bv, Fc, eps)
        k1 = _lu_solve(WLU, Wpiv, f0)
        y1 = y + h * k1
        if frozen == 0:
            M1, gb1 = _mass_bias(y1[:n], y1[n:2 * n], R0, pj, mass, com, Ic, g0)
            M1LU, M1piv = _lu(M1)
        else:
            gb1 = gb; M1LU = MLU; M1piv = Mpiv
        f1 = _f(y1, tau, M1LU, M1piv, gb1, k, c, Jm, Bv, Fc, eps)
        k2 = _lu_solve(WLU, Wpiv, f1 - 2.0 * k1)
        y = y + 1.5 * h * k1 + 0.5 * h * k2
    return y


class TwoMassNB:
    def __init__(self, params_json, elastic_json, dt=1e-3, substeps=8, frozen=0):
        self.params_json, self.dt, self.substeps, self.frozen = params_json, dt, substeps, frozen
        e = json.load(open(elastic_json))
        f = lambda x: np.asarray(x, float).reshape(-1)[:6].copy()
        self.k, self.c, self.Jm_np = f(e["k"]), f(e["c"]), f(e["Jm"])
        import torch
        self.Jm = torch.from_numpy(self.Jm_np.copy())          # closed_loop_el が参照する（torch）
        self._consts, self._payload = {}, 0.0

    @property
    def payload(self):
        return self._payload

    @payload.setter
    def payload(self, m):
        self._payload = float(m)

    def consts(self):
        m = self._payload
        if m not in self._consts:
            a = Arm.from_json(self.params_json, 6, m)
            n = lambda t: t.detach().numpy().astype(float).copy()
            g0 = n(a.r_base.T @ (-a.gravity))
            self._consts[m] = (n(a.R0), n(a.pj), n(a.mass), n(a.com), n(a.Ic), g0, self.k, self.c, self.Jm_np, n(a.Bv), n(a.Fc), n(a.eps))
        return self._consts[m]

    def step_np(self, y, tau):
        """y: [24]、tau: [6]（numpy）→ [24]。逐次の計測用（変換なし）。"""
        return _step1(y, tau, self.dt / self.substeps, self.substeps, self.frozen, *self.consts())

    def step(self, q, dq, qm, dqm, tau):
        is_t = hasattr(q, "numpy")
        cv = (lambda x: x.detach().numpy().astype(float)) if is_t else (lambda x: np.asarray(x, float))
        Y = np.concatenate([cv(q), cv(dq), cv(qm), cv(dqm)], -1); T = cv(tau)
        cs = self.consts(); h = self.dt / self.substeps
        out = np.empty_like(Y)
        for b in range(Y.shape[0]):
            out[b] = _step1(Y[b], T[b], h, self.substeps, self.frozen, *cs)
        if is_t:
            import torch
            out = torch.from_numpy(out)
        return tuple(out[:, i * 6:(i + 1) * 6] for i in range(4))
