"""窓評価（eval_window.py）の集計: 表（Markdown）と誤差曲線の図。"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK TC", "DejaVu Sans"]

DIRS = {50: "data/eval/window3", 200: "data/eval/window3_L200"}
OUT = sys.argv[1] if len(sys.argv) > 1 else "reports/issue-37"
NAMES = {"persist": "状態維持", "analytic": "解析", "hybrid": "hybrid（拘束中は状態維持、他は解析）"}
lines = []
for L, d in DIRS.items():
    r = json.load(open(f"{d}/window_metrics.json"))
    for key, v in r.items():
        if not key.startswith(f"L{L}/") or key.endswith("/all"):
            continue
        M = v["models"]
        lines.append(f"\n### {key}（窓 {v['n']} 本）終端の速度誤差 RMSE [rad/s] / 位置誤差 RMSE [rad]、J1/J2/J3\n")
        lines.append("| モデル | 終端速度誤差 | 終端位置誤差 |\n|---|---|---|")
        f = lambda a: " / ".join(f"{x:.3g}" for x in a)
        for k in ("persist", "analytic", "hybrid"):
            lines.append(f"| {NAMES[k]} | {f(M[k]['v_end_rmse'])} | {f(M[k]['q_end_rmse'])} |")
        for k in "ABC":
            a = np.array([M[f"{k}_s{s}"]["v_end_rmse"] for s in range(3)]); b = np.array([M[f"{k}_s{s}"]["q_end_rmse"] for s in range(3)])
            lines.append(f"| {k}（3 シード平均 ± 標準偏差） | " + " / ".join(f"{m:.3g}±{s:.2g}" for m, s in zip(a.mean(0), a.std(0))) + " | "
                         + " / ".join(f"{m:.3g}±{s:.2g}" for m, s in zip(b.mean(0), b.std(0))) + " |")
open(f"{OUT}/window_tables.md", "w").write("\n".join(lines) + "\n")

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
col = {"persist": "gray", "analytic": "k", "hybrid": "tab:purple", "A_s0": "tab:blue", "B_s0": "tab:orange", "C_s0": "tab:green"}
for i, g in enumerate(["free", "contact"]):
    for j, L in enumerate([50, 200]):
        for k, c in col.items():
            cv = np.load(f"{DIRS[L]}/curve_L{L}_{g}_{k}_v.npy")      # [L+1, n]
            ax[i, j].plot(np.sqrt((cv ** 2).mean(1)), color=c, label=NAMES.get(k, k).split("（")[0])
        ax[i, j].set_yscale("log"); ax[i, j].set_title(f"test {'自由運動' if g == 'free' else '接触を含む'}の窓 L={L}（3 軸の RMS、シード 0）")
        ax[i, j].set_xlabel("予測ステップ [ms]"); ax[i, j].set_ylabel("速度誤差 RMSE [rad/s]"); ax[i, j].grid(alpha=.3, which="both")
ax[0, 0].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/window_error_curves.png", dpi=130)
