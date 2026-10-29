"""A_el と 2 慣性解析モデルの誤差の切り分け（issue #77）。使い方:
  ~/.venvs/sa-train/bin/python python/diag_el.py --out reports/issue-77
(1) 1 ステップ誤差: 学習データの val（自由運動・|θ|<0.06）から無作為に抽出した遷移で、状態別・軸別の相対誤差（RMSE / 目標の RMS）。
(2) 窓のロールアウト: 弾性真値のステップ応答（payload 0/8 kg、wn 100/ζ 1 など）の真値状態から、閉ループ（PD はモータ側）で
    K = 10, 100, 1000 ステップ進め、リンク角 q とねじれ角 θ の誤差（軸別の最大）を見る。"""
import argparse, copy, json, os, sys
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.data import Dataset
from sa_nsm.closedloop import Scenario, closed_loop_el, DT
from sa_nsm.twomass import TwoMass
from eval_payload_el import ElPlant
from train_a_el import features, targets, PAYLOAD_SCALE


def one_step(ds, plants, nsamp, rng):
    n = ds.n_ax
    q, dq, tau = ds.q_dq_tau("val"); qm, dqm = ds.motor("val"); pl = ds.payload("val")
    th = np.abs(qm - q).max(1)
    idx = np.where(~ds.at_limit("val") & (th < 0.06))[0]
    idx = np.sort(rng.choice(idx, nsamp, replace=False))
    T = targets(ds, "val")[idx]
    res = {}
    for name, pf in plants.items():
        P = np.zeros_like(T)
        for m in np.unique(pl[idx]):
            k = pl[idx] == m; s = idx[k]; pf.payload = float(m)
            a = [torch.from_numpy(x[s]) for x in (q, dq, qm, dqm, tau)]
            r = [x.numpy() for x in pf.step(*a)]
            P[k] = np.concatenate([(r[0] - q[s] - dq[s] * DT) / DT ** 2, (r[1] - dq[s]) / DT, (r[2] - qm[s] - dqm[s] * DT) / DT ** 2, (r[3] - dqm[s]) / DT], 1)
        e = np.sqrt(((P - T) ** 2).mean(0)); r = np.sqrt((T ** 2).mean(0))
        res[name] = {nm: (e[i * n:(i + 1) * n] / r[i * n:(i + 1) * n]).tolist() for i, nm in enumerate(["q", "dq", "qm", "dqm"])}
        # 状態別（θ の大きさ・トルク）の dq 誤差
        th_ax = np.abs(qm - q)[idx]; ed = np.abs(P[:, n:2 * n] - T[:, n:2 * n])
        res[name]["by_theta_dq_abs"] = {}
        for ax in range(n):
            qs = np.quantile(th_ax[:, ax], [0, .5, .9, 1.0])
            res[name]["by_theta_dq_abs"][str(ax + 1)] = [float(ed[(th_ax[:, ax] >= qs[i]) & (th_ax[:, ax] <= qs[i + 1]), ax].mean()) for i in range(3)]
    return res


def windows(plants, data, cases, Ks):
    out = {}
    with h5py.File(data, "r") as f:
        peak = f["controller/tauPeak"][:].ravel()
        for gk in cases:
            ctl = {"Kp": f[gk + "_j1/Kp"][:].ravel(), "Kd": f[gk + "_j1/Kd"][:].ravel(), "peak": peak}
            sc = [Scenario(f[f"{gk}_j{ax}"], ctl) for ax in range(1, len(peak) + 1)]
            at = f[gk + "_j1"].attrs; m = int(np.ravel(at["payload"])[0]); t_step = float(np.ravel(at["tStep"])[0])
            i0 = int(round((t_step - 0.01) / DT))
            for K in Ks:
                ws = []
                for s in sc:
                    w = copy.copy(s)
                    w.u, w.q_ref, w.dq_ref = s.u[i0:i0 + K], s.q_ref[i0:i0 + K + 1], s.dq_ref[i0:i0 + K + 1]
                    w.q, w.dq, w.qm, w.dqm = (x[i0:i0 + K + 1] for x in (s.q, s.dq, s.qm, s.dqm)); w.K = K
                    ws.append(w)
                for name, pf in plants.items():
                    pf.payload = float(m)
                    q, dq, qm, dqm = closed_loop_el(pf, ws, ctl)
                    # q[b]: シナリオ b（= 軸 b+1 のステップ）の全軸の軌跡。対象は軸 b の q とねじれ角
                    eq = np.array([np.abs(q[b][:, b] - ws[b].q[:K + 1, b]).max() for b in range(len(ws))])
                    th = np.array([np.abs((qm[b] - q[b])[:, b] - (ws[b].qm - ws[b].q)[:K + 1, b]).max() for b in range(len(ws))])
                    out.setdefault(name, {}).setdefault(str(K), []).append({"case": gk, "payload": m, "q_err_deg": (eq * 180 / np.pi).tolist(), "theta_err_mrad": (th * 1e3).tolist()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/export/sa_flat_fullplpu6el.h5")
    ap.add_argument("--data", default="data/export/payload_eval6_el20.h5")
    ap.add_argument("--ael", default="data/models/a_el/el_e30_s0")
    ap.add_argument("--out", default="reports/issue-77")
    ap.add_argument("--nsamp", type=int, default=4000)
    ap.add_argument("--substeps", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(2)
    os.makedirs(a.out, exist_ok=True)
    ds = Dataset(a.h5, splits=("val",))
    with h5py.File(a.h5, "r") as f:
        dq_max, tau_max = f["scale/DQ_MAX"][:].ravel(), f["scale/TAU_MAX"][:].ravel()
    plants = {"A_el": ElPlant(a.ael, dq_max, tau_max), "2慣性解析": TwoMass("data/export/c8_params.json", "data/export/elastic20.json", substeps=a.substeps)}
    rng = np.random.RandomState(0)
    R = {"one_step": one_step(ds, plants, a.nsamp, rng)}
    cases = ["pl0_wn100_z100", "pl8_wn100_z100", "pl0_wn60_z50", "pl8_wn150_z50"]
    with h5py.File(a.data, "r") as f:
        have = {k.rsplit("_j", 1)[0] for k in f if k != "controller"}
    cases = [c for c in cases if c in have] or sorted(have)[:4]
    print("窓の評価ケース", cases, flush=True)
    R["windows"] = windows(plants, a.data, cases, (10, 100, 1000))
    json.dump(R, open(f"{a.out}/diag.json", "w"), indent=1, ensure_ascii=False)
    n = 6
    L = ["## 1 ステップ誤差（相対 RMSE、val 自由運動・|θ|<0.06、%d 遷移）\n" % a.nsamp, "| モデル | 状態 | " + " | ".join(f"軸{i+1}" for i in range(n)) + " |", "|---|---|" + "---|" * n]
    for nm, d in R["one_step"].items():
        for st in ("q", "dq", "qm", "dqm"):
            L.append(f"| {nm} | {st} | " + " | ".join(f"{v * 100:.1f}%" for v in d[st]) + " |")
    L += ["\n## 窓のロールアウト（真値状態から閉ループ。軸 b のステップ応答シナリオの軸 b の最大誤差、ケース=%s）\n" % ",".join(cases)]
    for K in (10, 100, 1000):
        L += [f"\n### K = {K} ステップ\n", "| モデル | 指標 | " + " | ".join(f"軸{i+1}" for i in range(n)) + " |", "|---|---|" + "---|" * n]
        for nm, d in R["windows"].items():
            for key, lab in (("q_err_deg", "q 誤差 最大[°]"), ("theta_err_mrad", "θ 誤差 最大[mrad]")):
                v = np.max([c[key] for c in d[str(K)]], 0)
                L.append(f"| {nm} | {lab} | " + " | ".join(f"{x:.3g}" for x in v) + " |")
    open(f"{a.out}/diag.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
