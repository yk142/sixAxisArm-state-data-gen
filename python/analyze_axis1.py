"""Phase 4-10: A_pl の軸別誤差の分解（issue #67）。test の自由運動行で、A_pl の加速度誤差（Δdq/dt）を
ペイロード・速度・トルク・姿勢・シナリオ種別ごとに集計し、軸 1 が悪い理由を探る。使い方:
  ~/.venvs/sa-train/bin/python python/analyze_axis1.py --models data/models/a_pl6/pl_wd0.2_e100_s0_s0 --out data/eval/axis1
解析（ペイロード反映）との差（= 摩擦・数値の差）も併記する。"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.data import Dataset
from sa_nsm.physics_fast import FastArm
from eval_a_window import AccelNet
from train_a import DT


def rmse(e): return np.sqrt((e ** 2).mean(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/export/sa_flat_fullpl6.h5")
    ap.add_argument("--params", default="data/export/c8_params_pl0.json")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=150000)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    ds = Dataset(a.h5, splits=tuple(dict.fromkeys(("train", a.split)))); n = ds.n_ax
    sp = a.split
    free = np.where(~ds.at_limit(sp))[0]
    idx = np.sort(np.random.RandomState(0).choice(free, min(a.n, len(free)), replace=False))
    q, dq, tau = (x[idx] for x in ds.q_dq_tau(sp)); pl = ds.payload(sp)[idx]
    d_q, d_dq = ds.delta(sp); acc_true = d_dq[idx] / DT
    # 解析（ペイロード反映）の加速度
    acc_an = np.zeros_like(acc_true)
    for m in np.unique(pl):
        arm = FastArm.from_json(a.params, n, payload=float(m)); r = np.where(pl == m)[0]
        _, dv = arm.step(torch.as_tensor(q[r]), torch.as_tensor(dq[r]), torch.as_tensor(tau[r]), DT, 1)
        acc_an[r] = dv.numpy() / DT
    res = {"n": int(len(idx)), "analytic_rmse": rmse(acc_an - acc_true).tolist(), "true_rms": rmse(acc_true).tolist()}
    tq = torch.as_tensor(q); tdq = torch.as_tensor(dq); ttau = torch.as_tensor(tau)
    for mp in a.models:
        net = AccelNet(ds, mp); acc = np.zeros_like(acc_true)
        for m in np.unique(pl):
            r = np.where(pl == m)[0]; net.payload = float(m)
            _, dv = net.step(tq[r], tdq[r], ttau[r]); acc[r] = (dv - tdq[r]).numpy() / DT
        e = acc - acc_true; name = os.path.basename(mp); out = {"rmse": rmse(e).tolist(), "bias": e.mean(0).tolist()}
        # 軸ごとに、因子別の RMSE（軸 1 と、比較用に軸 2）
        for ax in (0, 1):
            tab = {}
            tab["payload"] = {f"{m:g}": float(np.sqrt((e[pl == m, ax] ** 2).mean())) for m in np.unique(pl)}
            sp_ = np.abs(dq[:, ax]); edges = [0, 0.005, 0.02, 0.1, 0.3, 1, 10]
            tab["|dq|[rad/s]"] = {f"{edges[i]:g}-{edges[i+1]:g}": [int(((sp_ >= edges[i]) & (sp_ < edges[i+1])).sum()), float(np.sqrt((e[(sp_ >= edges[i]) & (sp_ < edges[i+1]), ax] ** 2).mean()))]
                                  for i in range(len(edges) - 1) if ((sp_ >= edges[i]) & (sp_ < edges[i+1])).any()}
            tmax = np.abs(tau[:, ax]); qe = [0, 0.1, 0.3, 0.6, 1.01]
            tab["|tau|/max"] = {}
            tn = tmax / max(tmax.max(), 1e-9)
            for i in range(len(qe) - 1):
                s = (tn >= qe[i]) & (tn < qe[i + 1])
                if s.any(): tab["|tau|/max"][f"{qe[i]:g}-{qe[i+1]:g}"] = [int(s.sum()), float(np.sqrt((e[s, ax] ** 2).mean()))]
            # 解析との差の同じ集計（摩擦・数値の寄与）
            ea = acc_an - acc_true
            tab["解析-真値 payload"] = {f"{m:g}": float(np.sqrt((ea[pl == m, ax] ** 2).mean())) for m in np.unique(pl)}
            # 重力成分: 姿勢（q2: 肩）・ペイロード別に誤差の符号（バイアス）
            tab["bias payload"] = {f"{m:g}": float(e[pl == m, ax].mean()) for m in np.unique(pl)}
            out[f"axis{ax + 1}"] = tab
        res[name] = out
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/axis1.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
