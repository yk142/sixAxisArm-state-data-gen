"""issue #45: 閉ループ評価の集計表と図。使い方: python python/summarize_closed.py [data/eval/closed3] [reports/issue-45]"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
src = sys.argv[1] if len(sys.argv) > 1 else "data/eval/closed3"
out = sys.argv[2] if len(sys.argv) > 2 else "reports/issue-45"
d = json.load(open(f"{src}/closed.json")); z = np.load(f"{src}/traj.npz")
models = [k for k in next(iter(d["ptp"].values())) if k != "truth"]
cols = ["tab:gray", "tab:green", "tab:red", "tab:blue", "tab:orange"]

lines = ["| 群 | モデル | 位置RMS 中央値 [°] | 位置RMS 最大 [°] | 位置 最大誤差 [°] | 速度RMS 最大 [rad/s] | 終端位置誤差 最大 [°] | 追従RMS 最大 [°] | 発散 |", "|---|---|---|---|---|---|---|---|---|"]
for g in ("ptp", "nearlimit"):
    sc = list(d[g])
    for m in models:
        r = [d[g][s][m] for s in sc]
        lines.append(f"| {g} | {m} | {np.median([np.mean(x['q_err_rms_deg']) for x in r]):.4f} | {max(np.max(x['q_err_rms_deg']) for x in r):.4f} | "
                     f"{max(np.max(x['q_err_max_deg']) for x in r):.3f} | {max(np.max(x['v_err_rms']) for x in r):.4f} | "
                     f"{max(np.max(np.abs(x['q_err_end_deg'])) for x in r):.3f} | {max(np.max(x['track_rms_deg']) for x in r):.4f} | {sum(not x['ok'] for x in r)} |")
    t = [d[g][s]["truth"] for s in sc]
    lines.append(f"| {g} | Simscape 真値 | - | - | - | - | - | {max(np.max(x['track_rms_deg']) for x in t):.4f} | - |")
open(f"{out}/summary.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
for a, g in zip(ax[:2], ("ptp", "nearlimit")):
    sc = list(d[g]); w = 0.16
    for i, m in enumerate(models):
        a.bar(np.arange(len(sc)) + (i - 2) * w, [max(np.max(d[g][s][m]["q_err_rms_deg"]), 1e-5) for s in sc], w, color=cols[i], label=m)
    a.set_yscale("log"); a.set_xticks(range(len(sc))); a.set_xticklabels([s.replace("cl_nearlimit_", "cl_").replace("ptp_", "") for s in sc], rotation=60, fontsize=7)
    a.set_ylabel("位置誤差 RMS（軸最大）[deg]"); a.set_title(f"{g}: Simscape 閉ループとの差"); a.grid(alpha=.3, axis="y")
ax[0].legend(fontsize=7)
# 軌道を保存してある PTP（先頭の 1 本）の誤差時間履歴
w_ = "ptp_v20_001"
tr = z[f"ptp|{w_}|truth"]
for i in (0, 2, 3):
    e = np.degrees(np.abs(z[f"ptp|{w_}|{models[i]}"] - tr)).max(1)
    ax[2].plot(np.arange(len(e)) * 1e-3, np.maximum(e, 1e-5), color=cols[i], label=models[i], lw=.8)
ax[2].set_yscale("log"); ax[2].set_xlabel("時刻 [s]"); ax[2].set_ylabel("位置誤差（軸最大）[deg]"); ax[2].set_title(f"{w_}（軌道を保存した PTP）"); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{out}/closed_loop.png", dpi=130)
