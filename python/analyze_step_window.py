"""Phase 4-10: ステップ応答の区間（Simscape の真値軌道上）での A_pl の加速度誤差（issue #67）。
真値の状態・トルクを各モデルに与えた 1 ステップ予測（教師強制）の Δdq/dt を、真値（次状態との差）と比べる。
軸ごとに、誤差・トルク・速度の範囲を集計する。使い方: python python/analyze_step_window.py --models <A_pl のディレクトリ> --out data/eval/stepwin"""
import argparse, json, os, sys
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.data import Dataset
from sa_nsm.physics_fast import FastArm
from sa_nsm.closedloop import Scenario, DT
from eval_a_window import AccelNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/export/sa_flat_fullpl6.h5")
    ap.add_argument("--data", default="data/export/payload_eval6.h5")
    ap.add_argument("--params-fmt", default="data/export/c8_params_pl{m}.json")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    ds = Dataset(a.h5, splits=("train",)); n = ds.n_ax
    nets = {os.path.basename(p): AccelNet(ds, p) for p in a.models}
    rows = []
    with h5py.File(a.data, "r") as f:
        peak = f["controller/tauPeak"][:].ravel()
        for gk in sorted({k.rsplit("_j", 1)[0] for k in f if k != "controller"}):
            m = int(f[gk + "_j1"].attrs["payload"]); arm = FastArm.from_json(a.params_fmt.format(m=m), n)
            ctl = {"Kp": f[gk + "_j1/Kp"][:].ravel(), "Kd": f[gk + "_j1/Kd"][:].ravel(), "peak": peak}
            at = f[gk + "_j1"].attrs; wn, zeta, t_step = float(at["wn"]), float(at["zeta"]), float(at["tStep"])
            for ax in range(n):
                s = Scenario(f[f"{gk}_j{ax + 1}"], ctl)
                q, dq = s.q, s.dq
                qm, dqm = 0.5 * (q[:-1] + q[1:]), 0.5 * (dq[:-1] + dq[1:])
                tau = np.clip(s.u - ctl["Kp"] * qm - ctl["Kd"] * dqm, -peak, peak)
                k0 = int(round((t_step - 0.01) / DT)); sl = slice(k0, len(tau))
                acc_true = (dq[1:] - dq[:-1]) / DT
                tq, tdq, ttau = (torch.as_tensor(x[sl]) for x in (q[:-1], dq[:-1], tau))
                _, dv = arm.step(tq, tdq, ttau, DT, 1); e_an = dv.numpy() / DT - acc_true[sl]
                row = {"payload": m, "wn": wn, "zeta": zeta, "axis": ax + 1, "tau_abs_max": float(np.abs(tau[sl, ax]).max()), "tau_frac": float((np.abs(tau[sl, ax]) / peak[ax]).max()),
                       "dq_abs_max": float(np.abs(dq[sl, ax]).max()), "an_rms": float(np.sqrt((e_an[:, ax] ** 2).mean()))}
                for nm, net in nets.items():
                    net.payload = float(m); _, dv = net.step(tq, tdq, ttau); e = (dv - tdq).numpy() / DT - acc_true[sl]
                    row[nm] = {"rms": float(np.sqrt((e[:, ax] ** 2).mean())), "max": float(np.abs(e[:, ax]).max())}
                rows.append(row)
            print(gk, flush=True)
    os.makedirs(a.out, exist_ok=True); json.dump(rows, open(f"{a.out}/stepwin.json", "w"), indent=1)


if __name__ == "__main__":
    main()
