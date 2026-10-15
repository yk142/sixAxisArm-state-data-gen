"""issue #49: B(4-2) が cp_nearlimit_j3m_002 で J2 を外した原因の調査。
 1) 同シナリオの B 閉ループを再現し、トルク・B の加速度・解析の加速度を記録（J2 のずれが J3 の接触後に起きるか）
 2) 真値の状態・トルクから B / 解析の 1 ステップ加速度を比べる（B の J2 加速度に偏りがあるか）
使い方: ~/.venvs/sa-train/bin/python python/diag_b_contact.py [scenario] （リポジトリ直下で実行）"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, "python")
import eval_window
from sa_nsm.closedloop import load_scenarios
from sa_nsm.data import Dataset
from sa_nsm.physics_fast import FastArm
from sa_nsm.rollout import Predictor
from sa_nsm.train_lib import prepare

name = sys.argv[1] if len(sys.argv) > 1 else "cp_nearlimit_j3m_002"
torch.set_num_threads(4)
ds = Dataset("data/export/sa_flat_full3.h5", splits=("train",)); n = ds.n_ax
arm = FastArm.from_json("data/export/c8_params.json", n)
data = prepare(ds, arm, "data/export/sa_flat_full3_analytic.npz", splits=("train",))
model = eval_window.load_model("data/models/full3", "B_s0", data["train"])
pb = Predictor(arm, ds, "model", model)
scen, ctl = load_scenarios("data/export/closed_contact3.h5")
s = next(x for x in scen if x.name == name)
DEG = 180 / np.pi
Kp, Kd, peak = (torch.as_tensor(ctl[k]) for k in ("Kp", "Kd", "peak"))
U = torch.from_numpy(s.u)

# 1) B の閉ループ（closed_loop と同じ式）＋ログ
q = torch.from_numpy(s.q[0])[None]; dq = torch.from_numpy(s.dq[0])[None]
a_prev = torch.zeros_like(dq); h = 1e-3
log = {k: [] for k in ("q", "dq", "tau", "aB", "aA")}
for k in range(s.K):
    q_mid = q + dq * h / 2 + a_prev * (h / 2) ** 2 / 2; dq_mid = dq + a_prev * h / 2
    tau = torch.minimum(torch.maximum(U[k][None] - Kp * q_mid - Kd * dq_mid, -peak), peak)
    q1, dq1 = pb.step(q, dq, tau)
    dqa, dva = arm.step(q, dq, tau, 1e-3, 2)
    for key, v in zip(log, (q, dq, tau, (dq1 - dq) / h, dva / h)): log[key].append(v[0].numpy())
    a_prev = (dq1 - dq) / h; q, dq = q1, dq1
L = {k: np.array(v) for k, v in log.items()}
np.savez_compressed("data/eval/closed_contact3/diag_b.npz", **L)
t = s.q                                                # 真値（K+1）
lo = np.degrees(arm.q_min.numpy())
print("J3 下限", lo[2], "deg。B の J3 が下限 0.5° 以内に初めて入るステップ:", int(np.argmax(np.degrees(L["q"][:, 2]) < lo[2] + 0.5)))
for k in (1000, 1400, 1500, 1550, 1600, 1700, 2000, 3000, 5000):
    print(k, "q", np.round(L["q"][k] * DEG, 1), "dq[deg/s]", np.round(L["dq"][k] * DEG, 1), "tau", np.round(L["tau"][k], 1),
          "aB", np.round(L["aB"][k], 2), "aA", np.round(L["aA"][k], 2))

# 2) 真値の状態・トルクでの 1 ステップ: B と解析の加速度差（J2 に偏りがあるか）
Tq, Tdq = torch.from_numpy(s.q[:-1]), torch.from_numpy(s.dq[:-1])
Tu = torch.minimum(torch.maximum(U - Kp * Tq - Kd * Tdq, -peak), peak)          # 近似トルク（ZOH 評価）
q1, dq1 = pb.step(Tq, Tdq, Tu); dqa, dva = arm.step(Tq, Tdq, Tu, 1e-3, 2)
aB = ((dq1 - Tdq) / h).numpy(); aA = (dva / h).numpy(); aT = ((torch.from_numpy(s.dq[1:]) - Tdq) / h).numpy()
for lab, sl in (("接触前 0-1400", slice(0, 1400)), ("接触後 1800-6000", slice(1800, 6000))):
    print(f"{lab}: J2 加速度 平均 B-解析 {np.mean(aB[sl, 1] - aA[sl, 1]):+.3f}, 真値-解析 {np.mean(aT[sl, 1] - aA[sl, 1]):+.3f} rad/s²; "
          f"RMS B-解析 {np.sqrt(np.mean((aB[sl, 1] - aA[sl, 1]) ** 2)):.3f}, 真値-解析 {np.sqrt(np.mean((aT[sl, 1] - aA[sl, 1]) ** 2)):.3f}; "
          f"J3 B-解析 平均 {np.mean(aB[sl, 2] - aA[sl, 2]):+.2f}")

# 3) B の応答の探索: J3 の下限からの距離 × トルク（速度 0、J1=0, J2=-29°）。B の J3 / J2 加速度を解析と比べる
qmin3 = arm.q_min[2].item()
print("\n[探索] J3 下限からの距離 g [deg] × トルク → 加速度 [rad/s²]（B / 解析）。J2 トルク 40 N·m")
for tau3 in (-381., -100., -20.):
    for g in (20, 10, 6, 4, 2, 1, 0.1):
        qq = torch.tensor([[0., -0.51, qmin3 + np.radians(g)]], dtype=torch.float64); tt = torch.tensor([[0., 40., tau3]], dtype=torch.float64)
        z0 = torch.zeros(1, 3, dtype=torch.float64); _, v1 = pb.step(qq, z0, tt); _, va = arm.step(qq, z0, tt, 1e-3, 2)
        print(f"  tau3 {tau3:6.0f} g {g:5}°: J3 B {v1[0, 2] / 1e-3:7.1f} / 解析 {va[0, 2] / 1e-3:7.1f}")
print("\n[探索] J3 トルク -381（飽和）で J2 トルクを変えたときの J2 加速度（B / 解析）")
for t2 in (860., 100., 0.):
    for g in (20, 6, 0.1):
        qq = torch.tensor([[-0.01, -0.51, qmin3 + np.radians(g)]], dtype=torch.float64); tt = torch.tensor([[-57., t2, -381.]], dtype=torch.float64)
        z0 = torch.zeros(1, 3, dtype=torch.float64); _, v1 = pb.step(qq, z0, tt); _, va = arm.step(qq, z0, tt, 1e-3, 2)
        print(f"  tau2 {t2:5.0f} g {g:5}°: J2 B {v1[0, 1] / 1e-3:7.1f} / 解析 {va[0, 1] / 1e-3:7.1f}")
