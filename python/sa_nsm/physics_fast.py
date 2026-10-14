"""解析モデルの高速版（issue #41）。`physics.Arm` と同じ式・同じ結果（倍精度で 1e-10 程度の一致）。

高速化の要点（演算数＝呼び出しオーバーヘッドの削減。バッチ 1 のレイテンシが支配的だったため）:
  1. 手首固定（n_ax=3）では 4〜6 軸のリンクを 3 軸リンクに合成（質量・重心・慣性を平行軸定理でまとめる）。
     RNEA の再帰が 6 リンクから 3 リンクになる。
  2. 回転行列を全軸まとめて 1 回で作る（軸ごとの stack をやめる）。
  3. 可動軸の列だけを扱い、0 パディングを持たない。
"""
import torch

from .physics import Arm, DTYPE


class FastArm(Arm):
    def __init__(self, params, n_ax=3):
        super().__init__(params, n_ax)
        n = self.n_ax
        self.R0n, self.pjn = self.R0[:n].clone(), self.pj[:n].clone()
        self.mass_n, self.com_n, self.Ic_n = self._merged_links()
        self.armature_n = self.armature[:n].clone()
        self.zvec = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
        self.g0 = self.r_base.T @ (-self.gravity)

    def _merged_links(self):
        """可動軸 n 本の各リンク（最後のリンクに固定軸以降を合成）の質量・重心・重心まわりの慣性（自フレーム）。"""
        n = self.n_ax
        mass, com, Ic = list(self.mass[:n]), list(self.com[:n]), list(self.Ic[:n])
        if n < 6:
            Rk, pk = torch.eye(3, dtype=DTYPE), torch.zeros(3, dtype=DTYPE)       # フレーム k → フレーム n-1
            ms, cs, Is = [mass[-1]], [com[-1]], [Ic[-1]]
            for k in range(n, 6):                                                  # 固定軸は 0 rad
                pk = pk + Rk @ self.pj[k]
                Rk = Rk @ self.R0[k]
                ms.append(self.mass[k]); cs.append(pk + Rk @ self.com[k]); Is.append(Rk @ self.Ic[k] @ Rk.T)
            m = sum(ms)
            c = sum(mi * ci for mi, ci in zip(ms, cs)) / m
            I = torch.zeros(3, 3, dtype=DTYPE)
            for mi, ci, Ii in zip(ms, cs, Is):
                d = ci - c
                I = I + Ii + mi * ((d @ d) * torch.eye(3, dtype=DTYPE) - torch.outer(d, d))
            mass[-1], com[-1], Ic[-1] = m, c, I
        return torch.stack(mass), torch.stack(com), torch.stack(Ic)

    # ---- 逆動力学（[B,n]）----
    def rnea_n(self, q, dq, ddq):
        n, B = self.n_ax, q.shape[0]
        c, s = torch.cos(q), torch.sin(q)
        zero, one = torch.zeros_like(c), torch.ones_like(c)
        Rz = torch.stack([c, -s, zero, s, c, zero, zero, zero, one], -1).reshape(B, n, 3, 3)
        A = self.R0n @ Rz                                                          # [B,n,3,3]  フレーム i → i-1
        RT = A.transpose(-1, -2)
        mv = lambda M, v: torch.matmul(M, v.unsqueeze(-1)).squeeze(-1)
        cross = torch.linalg.cross
        z = self.zvec
        om = al = torch.zeros(B, 3, dtype=DTYPE)
        ac = self.g0.expand(B, 3)
        oms, als, acs = [], [], []
        for i in range(n):
            RTi, pj = RT[:, i], self.pjn[i].expand(B, 3)
            omR = mv(RTi, om)
            zd = z * dq[:, i:i + 1]
            tmp = ac + cross(al, pj) + cross(om, cross(om, pj))
            al = mv(RTi, al) + z * ddq[:, i:i + 1] + cross(omR, zd)
            om = omR + zd
            ac = mv(RTi, tmp)
            oms.append(om); als.append(al); acs.append(ac)
        taus = [None] * n
        f = nn_ = None
        for i in range(n - 1, -1, -1):
            cm = self.com_n[i].expand(B, 3)
            aG = acs[i] + cross(als[i], cm) + cross(oms[i], cross(oms[i], cm))
            F = self.mass_n[i] * aG
            N = mv(self.Ic_n[i], als[i]) + cross(oms[i], mv(self.Ic_n[i], oms[i]))
            if f is None:
                f_this, n_this = F, N + cross(cm, F)
            else:
                Rn, pN = A[:, i + 1], self.pjn[i + 1].expand(B, 3)
                fN, nN = mv(Rn, f), mv(Rn, nn_)
                f_this = fN + F
                n_this = nN + N + cross(cm, F) + cross(pN, fN)
            taus[i] = n_this[:, 2] + self.armature_n[i] * ddq[:, i]
            f, nn_ = f_this, n_this
        return torch.stack(taus, -1)

    def matrices(self, q, dq):
        n, B = self.n_ax, q.shape[0]
        q, dq = q.to(DTYPE), dq.to(DTYPE)
        zn = torch.zeros(B, n, dtype=DTYPE)
        E = torch.eye(n, dtype=DTYPE)
        Q = q.repeat(n + 2, 1)
        D = torch.cat([zn, dq] + [zn] * n, 0)
        DD = torch.cat([zn, zn] + [E[j].expand(B, n) for j in range(n)], 0)
        T = self.rnea_n(Q, D, DD)
        g = T[:B]
        b = T[B:2 * B] - g
        M = T[2 * B:].reshape(n, B, n).permute(1, 2, 0) - g.unsqueeze(-1)          # M[b,i,j] = T_j[b,i] - g[b,i]
        return g, b, M
