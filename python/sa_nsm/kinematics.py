"""順運動学: 関節角 q → 先端位置（issue #93）。physics.Arm と同じフレームの定義（A_i = R0_i·Rz(q_i) がフレーム i → i-1、pj_i はフレーム i-1 で見た関節 i の原点）。
先端 = arm6 フレームの z 方向 `tip` [m]（既定 0.1 m。docs/phase4_8 の換算と同じ）。ベースの位置は差を取るので含めない。"""

import numpy as np

from .physics import Arm


class Kinematics:
    def __init__(self, params_json, tip=0.1):
        a = Arm.from_json(params_json, 6)
        n = lambda t: t.detach().numpy().astype(float)
        self.R0, self.pj, self.r_base = n(a.R0), n(a.pj), n(a.r_base)
        self.tip = np.array([0.0, 0.0, float(tip)])

    def tip_pos(self, q):
        """q: [..., 6] → 先端位置 [..., 3]（ベース座標、m）。"""
        q = np.asarray(q, float)
        sh = q.shape[:-1]; q = q.reshape(-1, 6); N = len(q)
        R = np.broadcast_to(self.r_base, (N, 3, 3)).copy(); p = np.zeros((N, 3))
        for i in range(6):
            p = p + np.einsum("nij,j->ni", R, self.pj[i])
            c, s = np.cos(q[:, i]), np.sin(q[:, i])
            Rz = np.zeros((N, 3, 3)); Rz[:, 0, 0] = c; Rz[:, 0, 1] = -s; Rz[:, 1, 0] = s; Rz[:, 1, 1] = c; Rz[:, 2, 2] = 1.0
            R = R @ self.R0[i] @ Rz
        p = p + np.einsum("nij,j->ni", R, self.tip)
        return p.reshape(*sh, 3)
