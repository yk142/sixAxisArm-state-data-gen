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
def _resid(th, dth, dq, R):
    """残差トルク NN（issue #101）: 軸ごとの MLP [θ, dθ, dq(, tanh(dq/ε))] → (Δs ばねの非線形分, Δf 摩擦の非線形分)。R = (use, W0, b0, W1, b1, W2, b2, xs, ys, eps, [blk])。"""
    n = 6
    ds = np.zeros(n); df = np.zeros(n)
    if R[0] == 0:
        return ds, df
    W0, b0, W1, b1, W2, b2, xs, ys, ep = R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8], R[9]
    H = b0.shape[1]; nin = W0.shape[2]; blk = int(R[10][0])          # blk: 第 2 層のブロック幅（アンサンブル、issue #105）。ブロックの外は 0 なので飛ばす
    for i in range(n):
        x0 = th[i] / xs[i, 0]; x1 = dth[i] / xs[i, 1]; x2 = dq[i] / xs[i, 2]
        h1 = np.empty(H)
        for a in range(H):
            acc = W0[i, a, 0] * x0 + W0[i, a, 1] * x1 + W0[i, a, 2] * x2 + b0[i, a]
            if nin == 4:                                                # issue #105: 4 番目の入力 tanh(dq/ε)
                acc += W0[i, a, 3] * np.tanh(dq[i] / ep[i]) / xs[i, 3]
            h1[a] = np.tanh(acc)
        h2 = np.empty(H)
        for a in range(H):
            acc = b1[i, a]
            b_lo = (a // blk) * blk
            for b in range(b_lo, b_lo + blk):
                acc += W1[i, a, b] * h1[b]
            h2[a] = np.tanh(acc)
        o0 = b2[i, 0]; o1 = b2[i, 1]
        for b in range(H):
            o0 += W2[i, 0, b] * h2[b]; o1 += W2[i, 1, b] * h2[b]
        ds[i] = o0 * ys[i, 0]; df[i] = o1 * ys[i, 1]
    return ds, df


@njit(cache=True)
def _oracle(th, dq, U):
    """正解の式（issue #103）: 真値に入れたモデル外の効果そのもの（addSaUnmodeled と同じ式）。U = (use, a, θr, Fs[6], vs[6], eps[6])。
    返り値 (Δs, Δf, ∂Δs/∂θ, ∂Δf/∂dq)。Δs = k·a·θ³/θr² は k を掛ける前の値（呼び出し側で k 倍）。"""
    n = 6
    ds = np.zeros(n); df = np.zeros(n); dds = np.zeros(n); ddf = np.zeros(n)
    if U[0] == 0:
        return ds, df, dds, ddf
    a, thr, Fs, vs, ep = U[1][0], U[1][1], U[2], U[3], U[4]
    for i in range(n):
        ds[i] = a * th[i] ** 3 / thr ** 2; dds[i] = 3.0 * a * th[i] ** 2 / thr ** 2
        ex = np.exp(-(dq[i] / vs[i]) ** 2); th_ = np.tanh(dq[i] / ep[i])
        df[i] = Fs[i] * ex * th_
        ddf[i] = Fs[i] * ex * ((1.0 - th_ ** 2) / ep[i] - 2.0 * dq[i] / vs[i] ** 2 * th_)
    return ds, df, dds, ddf


@njit(cache=True)
def _f(y, tau, MLU, Mpiv, gb, k, c, Jm, Bv, Fc, eps, k3, R, U):
    n = 6
    q, dq, qm, dqm = y[:n], y[n:2 * n], y[2 * n:3 * n], y[3 * n:]
    ds, df = _resid(qm - q, dqm - dq, dq, R)
    os_, of, _, _ = _oracle(qm - q, dq, U)
    ds = ds + k * os_; df = df + of
    th = qm - q
    ts = k * th + k3 * th ** 3 + c * (dqm - dq) + ds
    rhs = ts - gb - (Bv * dq + Fc * np.tanh(dq / eps) + df)
    out = np.empty(4 * n)
    out[:n] = dq
    out[n:2 * n] = _lu_solve(MLU, Mpiv, rhs)
    out[2 * n:3 * n] = dqm
    out[3 * n:] = (tau - ts) / Jm
    return out


@njit(cache=True)
def _step1(y, tau, h, nsub, frozen, R0, pj, mass, com, Ic, g0, k, c, Jm, Bv, Fc, eps, k3, R, U):
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
        _, _, dds, ddf = _oracle(y[2 * n:3 * n] - q, dq, U)       # 正解の式の微分（issue #103。U 無効なら 0）
        if U[0] != 0 and U[1][2] == 0.0:                             # 微分を J に入れない（④ の残差 NN と同じ扱い）
            dds = np.zeros(n); ddf = np.zeros(n)
        ke = k * (1.0 + dds) + 3.0 * k3 * (y[2 * n:3 * n] - q) ** 2; Dfr = Dfr + np.maximum(ddf, 0.0)
        # Stribeck の負の傾きはヤコビアンに入れない: J に正の固有値 λ ができ、γ·h·λ ≈ 1 で W = I − γhJ が特異になって発散した（sub8 で 48 回中 24 回 NaN）
        J = np.zeros((4 * n, 4 * n))
        for i in range(n):
            J[i, n + i] = 1.0
            J[2 * n + i, 3 * n + i] = 1.0
            for j in range(n):
                J[n + i, j] = -Mi[i, j] * ke[j]
                J[n + i, n + j] = -(Mi[i, j] * c[j] + Mi[i, j] * Dfr[j])
                J[n + i, 2 * n + j] = Mi[i, j] * ke[j]
                J[n + i, 3 * n + j] = Mi[i, j] * c[j]
            J[3 * n + i, i] = ke[i] / Jm[i]
            J[3 * n + i, n + i] = c[i] / Jm[i]
            J[3 * n + i, 2 * n + i] = -ke[i] / Jm[i]
            J[3 * n + i, 3 * n + i] = -c[i] / Jm[i]
        W = I24 - gam * h * J
        WLU, Wpiv = _lu(W)
        f0 = _f(y, tau, MLU, Mpiv, gb, k, c, Jm, Bv, Fc, eps, k3, R, U)
        k1 = _lu_solve(WLU, Wpiv, f0)
        y1 = y + h * k1
        if frozen == 0:
            M1, gb1 = _mass_bias(y1[:n], y1[n:2 * n], R0, pj, mass, com, Ic, g0)
            M1LU, M1piv = _lu(M1)
        else:
            gb1 = gb; M1LU = MLU; M1piv = Mpiv
        f1 = _f(y1, tau, M1LU, M1piv, gb1, k, c, Jm, Bv, Fc, eps, k3, R, U)
        k2 = _lu_solve(WLU, Wpiv, f1 - 2.0 * k1)
        y = y + 1.5 * h * k1 + 0.5 * h * k2
    return y


def _load_residual(path):
    z1 = np.zeros((6, 1, 3)); z = np.zeros((6, 1))
    if path is None:
        return (0, z1, z, np.zeros((6, 1, 1)), z, np.zeros((6, 2, 1)), np.zeros((6, 2)), np.ones((6, 3)), np.ones((6, 2)), np.ones(6), np.ones(1))
    d = np.load(path)
    st = lambda key: np.stack([d[f"{key}_{i}"] for i in range(6)]).astype(float)
    ep = st("eps").reshape(6) if "eps_0" in d else np.ones(6)               # issue #105 より前の npz には無い
    W1 = st("W1"); blk = float(d["blk"]) if "blk" in d else float(W1.shape[1])
    return (1, st("W0"), st("b0"), W1, st("b1"), st("W2"), st("b2"), st("xs"), st("ys"), ep, np.array([blk]))


class TwoMassNB:
    def __init__(self, params_json, elastic_json, dt=1e-3, substeps=8, frozen=0, scale=None, residual=None, oracle=None):
        """scale: パラメータ倍率 {k, c, Jm, fric（Fc と Bv）, Fc, Bv, mass（全リンクの質量・慣性）}（twomass.TwoMass と同じ。issue #95）。
        oracle: 真値に入れたモデル外の効果の式をそのまま足す（issue #103）。{stiffA, stiffThetaRef, stribeck, vsMul}（saUnmodeledDefaults と同じ意味）。"""
        self.scale = dict(scale or {})
        self.params_json, self.dt, self.substeps, self.frozen = params_json, dt, substeps, frozen
        e = json.load(open(elastic_json))
        f = lambda x: np.asarray(x, float).reshape(-1)[:6].copy()
        g = lambda key: self.scale.get(key, 1.0)
        self.k, self.c, self.Jm_np = f(e["k"]) * g("k"), f(e["c"]) * g("c"), f(e["Jm"]) * g("Jm")
        import torch
        self.Jm = torch.from_numpy(self.Jm_np.copy())          # closed_loop_el が参照する（torch）
        self._consts, self._payload = {}, 0.0
        self.R = _load_residual(residual)                         # 残差トルク NN（issue #101、fit_residual_el.py の residual.npz）
        self.oracle = oracle

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
            g = lambda key: self.scale.get(key, 1.0)
            ms = g("mass")
            self._consts[m] = (n(a.R0), n(a.pj), n(a.mass) * ms, n(a.com), n(a.Ic) * ms, g0, self.k, self.c, self.Jm_np,
                               n(a.Bv) * g("fric") * g("Bv"), n(a.Fc) * g("fric") * g("Fc"), n(a.eps),
                               np.asarray(self.scale.get("k3", np.zeros(6)), float) * np.ones(6))      # θ³ の係数（issue #109、同定の scale.k3。無ければ 0）
        return self._consts[m]

    def _U(self):
        if not hasattr(self, "_Uc"):
            o = self.oracle; z = np.zeros(6)
            if not o:
                self._Uc = (0, np.zeros(3), z, np.ones(6), np.ones(6))
            else:
                P = json.load(open(self.params_json)); f = lambda x: np.asarray(x, float).reshape(-1)[:6].copy()
                ep, Fc = f(P["frictionEps"]), f(P["frictionFc"])          # 真値の Stribeck は公称の Fc・ε が基準
                self._Uc = (1, np.array([o.get("stiffA", 0.3), o.get("stiffThetaRef", 0.005), float(o.get("jac", 1))]), o.get("stribeck", 0.3) * Fc, o.get("vsMul", 2.0) * ep, ep)
        return self._Uc

    def step_np(self, y, tau):
        """y: [24]、tau: [6]（numpy）→ [24]。逐次の計測用（変換なし）。"""
        return _step1(y, tau, self.dt / self.substeps, self.substeps, self.frozen, *self.consts(), self.R, self._U())

    def step(self, q, dq, qm, dqm, tau):
        is_t = hasattr(q, "numpy")
        cv = (lambda x: x.detach().numpy().astype(float)) if is_t else (lambda x: np.asarray(x, float))
        Y = np.concatenate([cv(q), cv(dq), cv(qm), cv(dqm)], -1); T = cv(tau)
        cs = self.consts(); h = self.dt / self.substeps
        out = np.empty_like(Y)
        for b in range(Y.shape[0]):
            out[b] = _step1(Y[b], T[b], h, self.substeps, self.frozen, *cs, self.R, self._U())
        if is_t:
            import torch
            out = torch.from_numpy(out)
        return tuple(out[:, i * 6:(i + 1) * 6] for i in range(4))
