"""PD の離散化の影響の切り分け（issue #77）。2 慣性解析モデルを 8 kHz（真値の記録刻み）で、PD をモータ側の現在値から毎ステップ計算して閉ループにする。
1 kHz の評価（PD を区間の中点の推定で計算）との差が、評価側の近似の大きさ。使い方:
  ~/.venvs/sa-train/bin/python python/diag_pd8k.py --case pl8_wn150_z200 --axes 1 2"""
import argparse, os, sys
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_payload import step_metrics, DEG
from sa_nsm.closedloop import Scenario, closed_loop_el, R
from sa_nsm.twomass import TwoMass

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/export/payload_eval6_el20.h5")
ap.add_argument("--case", default="pl8_wn150_z200")
ap.add_argument("--iters", nargs="*", type=int, default=[1, 2])
ap.add_argument("--axes", nargs="+", type=int, default=[1, 2])
a = ap.parse_args()
torch.set_num_threads(2)
with h5py.File(a.data, "r") as f:
    peak = f["controller/tauPeak"][:].ravel()
    ctl = {"Kp": f[a.case + "_j1/Kp"][:].ravel(), "Kd": f[a.case + "_j1/Kd"][:].ravel(), "peak": peak}
    at = f[a.case + "_j1"].attrs; m = int(np.ravel(at["payload"])[0]); step, t_step = float(np.ravel(at["step"])[0]), float(np.ravel(at["tStep"])[0])
    for ax in a.axes:
        g = f[f"{a.case}_j{ax}"]
        sc = Scenario(g, ctl)
        ref, tau_d = g["ref"][:].T, g["tau_d"][:].T
        n = tau_d.shape[1]
        u8 = ref[:, 1 + 2 * n:1 + 3 * n] + tau_d + ctl["Kp"] * ref[:, 1:1 + n] + ctl["Kd"] * ref[:, 1 + n:1 + 2 * n]
        pair = 0.5 * (u8[:-1] + u8[1:])                    # 8 kHz の区間ごとの参照項（台形平均）
        K8 = len(pair)
        tm1 = TwoMass("data/export/c8_params.json", "data/export/elastic20.json", dt=1e-3, substeps=8); tm1.payload = m
        q1 = closed_loop_el(tm1, [sc], ctl)[0][0][:, ax - 1]
        q1i = {it: closed_loop_el(tm1, [sc], ctl, iters=it)[0][0][:, ax - 1] for it in a.iters}
        tm8 = TwoMass("data/export/c8_params.json", "data/export/elastic20.json", dt=1e-3 / R, substeps=1); tm8.payload = m
        T = lambda x: torch.from_numpy(x[None]).double()
        q, dq, qm, dqm = T(sc.q[0]), T(sc.dq[0]), T(sc.qm[0]), T(sc.dqm[0])
        Kp, Kd, pk = (torch.as_tensor(ctl[k]) for k in ("Kp", "Kd", "peak"))
        U = torch.from_numpy(pair)
        qs = [q[0, ax - 1].item()]
        for k in range(K8):
            tau = torch.minimum(torch.maximum(U[k] - Kp * qm - Kd * dqm, -pk), pk)
            q, dq, qm, dqm = tm8.step(q, dq, qm, dqm, tau)
            if (k + 1) % R == 0: qs.append(q[0, ax - 1].item())
        q8 = np.array(qs)
        t = np.arange(len(sc.q)) * 1e-3; i_pre = int(round((t_step - 0.01) / 1e-3))
        tr = sc.q[:, ax - 1]
        for nm, qq in [("真値", tr), ("2慣性 1kHz（中点推定 PD）", q1)] + [(f"2慣性 1kHz（平均トルク整合 {it} 反復）", v) for it, v in q1i.items()] + [("2慣性 8kHz（PD を毎ステップ）", q8[:len(tr)])]:
            mt = step_metrics(t[:len(qq)], qq, qq[i_pre], step, t_step)
            print(f"軸{ax} {a.case} {nm}: OS {mt['overshoot_pct']:.2f}% 整定 {mt['settle_s']*1000:.0f} ms 最大差(対真値) {np.abs(qq - tr[:len(qq)]).max() * DEG:.3f}°", flush=True)
