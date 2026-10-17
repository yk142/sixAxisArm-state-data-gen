"""閉ループ PTP の時系列比較（6 軸）。使い方:
  ~/.venvs/sa-train/bin/python python/plot_closed_ptp.py --out data/eval/closed6_plots --names ptp_v40_001 ptp_v80_003
各シナリオについて、左: 関節角（Simscape の真値・参照・A）、右: 真値との差（A 調整後・A 調整前・解析）を軸別に描く。"""
import argparse, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_a_window import AccelNet
from sa_nsm.closedloop import closed_loop, load_scenarios
from sa_nsm.data import Dataset
from sa_nsm.physics_fast import FastArm
from sa_nsm.rollout import Predictor

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
ap = argparse.ArgumentParser()
ap.add_argument("--h5", default="data/export/sa_flat_full6.h5")
ap.add_argument("--cl", default="data/export/closed_eval6.h5")
ap.add_argument("--params", default="data/export/c8_params.json")
ap.add_argument("--a-tuned", default="data/models/a_exp6/r10_wd0.2_e100_s0")
ap.add_argument("--a-orig", default="data/models/a_exp6/r6_w512d4_e60_s0")
ap.add_argument("--names", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(4)
ds = Dataset(a.h5, splits=("train",)); arm = FastArm.from_json(a.params, ds.n_ax)
scen, ctl = load_scenarios(a.cl)
os.makedirs(a.out, exist_ok=True)
models = {"解析": Predictor(arm, ds, "analytic"), "A 調整前": AccelNet(ds, a.a_orig), "A 調整後（seed 0）": AccelNet(ds, a.a_tuned)}
col = {"解析": "tab:gray", "A 調整前": "tab:red", "A 調整後（seed 0）": "tab:orange"}
D = 180 / np.pi
for name in a.names:
    s = next(x for x in scen if x.name == name)
    res = {m: closed_loop(p, [s], ctl)[0][0] for m, p in models.items()}
    t = np.arange(s.K + 1) * 1e-3
    fig, ax = plt.subplots(6, 2, figsize=(14, 13), sharex=True)
    print(name, "K =", s.K)
    for j in range(6):
        ax[j, 0].plot(t, s.q[:, j] * D, "k", lw=1.6, label="Simscape（真値）")
        ax[j, 0].plot(t, s.q_ref[:, j] * D, "c--", lw=.8, label="参照")
        ax[j, 0].plot(t, res["A 調整後（seed 0）"][:, j] * D, color="tab:orange", lw=.9, label="A 調整後")
        for m in models:
            e = (res[m][:, j] - s.q[:, j]) * D
            ax[j, 1].plot(t, e, color=col[m], lw=.9, label=m)
        ea = {m: (res[m][:, j] - s.q[:, j]) * D for m in models}
        print(f"  J{j + 1}: 角度範囲 {np.ptp(s.q[:, j]) * D:6.1f}°  最大誤差 解析 {np.abs(ea['解析']).max():.5f}  A 調整前 {np.abs(ea['A 調整前']).max():.3f}  A 調整後 {np.abs(ea['A 調整後（seed 0）']).max():.3f}  [deg]  RMS 調整後 {np.sqrt((ea['A 調整後（seed 0）'] ** 2).mean()):.4f}")
        ax[j, 0].set_ylabel(f"J{j + 1} [deg]"); ax[j, 1].set_ylabel(f"J{j + 1} 誤差 [deg]")
        for c in (0, 1): ax[j, c].grid(alpha=.3)
    ax[0, 0].legend(fontsize=7, ncol=3); ax[0, 1].legend(fontsize=7, ncol=3)
    ax[0, 0].set_title(f"{name}: 関節角（閉ループ）"); ax[0, 1].set_title("真値との差（モデル − Simscape）")
    ax[-1, 0].set_xlabel("時刻 [s]"); ax[-1, 1].set_xlabel("時刻 [s]")
    fig.tight_layout(); fig.savefig(f"{a.out}/{name}.png", dpi=110); plt.close(fig)
