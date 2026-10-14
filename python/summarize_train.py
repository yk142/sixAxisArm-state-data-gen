"""4-2 の 9 本の学習結果を集計し、表(Markdown)と図を出力する。"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/models/full3"
OUT = sys.argv[2] if len(sys.argv) > 2 else "reports/issue-35"
GROUPS = [("free", "自由運動"), ("dynamic", "動的接触"), ("static", "静的接触")]
BASE = {"free": [0.0105, 0.0064, 0.0122], "dynamic": [63, 98.6, 222], "static": [0.145, 0.173, 0.177]}
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK TC", "DejaVu Sans"]
COL = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}

def load(k, s, f):
    return json.load(open(f"{ROOT}/{k}_s{s}/{f}"))

lines = []
for sp in ["val", "test", "benchmark"]:
    lines.append(f"\n### {sp}（加速度換算 RMSE rad/s²、J1/J2/J3、3 シードの平均 ± 標準偏差）\n")
    lines.append("| 区分 | モデル | J1 | J2 | J3 |\n|---|---|---|---|---|")
    for g, gn in GROUPS + [("ptp", "PTP")]:
        for k in "ABC":
            r = [load(k, s, "metrics.json")[sp].get(g) for s in range(3)]
            if r[0] is None:
                continue
            a = np.array([x["acc_rmse"] for x in r])
            lines.append(f"| {gn} | {k} | " + " | ".join(f"{m:.4g} ± {d:.2g}" for m, d in zip(a.mean(0), a.std(0))) + " |")
open(f"{OUT}/summary_tables.md", "w").write("\n".join(lines) + "\n")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for i, (g, gn) in enumerate(GROUPS):
    w = 0.25
    for j, k in enumerate("ABC"):
        a = np.array([load(k, s, "metrics.json")["test"][g]["acc_rmse"] for s in range(3)])
        ax[i].bar(np.arange(3) + (j - 1) * w, a.mean(0), w, yerr=a.std(0), color=COL[k], label=f"{k}", capsize=2)
    ax[i].scatter(np.arange(3), BASE[g], marker="_", s=600, color="k", zorder=5, label="4-1 基準線")
    ax[i].set_yscale("log"); ax[i].set_xticks(range(3)); ax[i].set_xticklabels(["J1", "J2", "J3"])
    ax[i].set_title(f"test {gn}"); ax[i].set_ylabel("加速度換算 RMSE [rad/s²]")
    ax[i].grid(alpha=.3, which="both")
ax[0].legend(); fig.tight_layout(); fig.savefig(f"{OUT}/region_error.png", dpi=130)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k in "ABC":
    for s in range(3):
        h = load(k, s, "history.json")
        ax[0].plot([x["train"] for x in h], color=COL[k], alpha=.7, label=k if s == 0 else None)
        ax[1].plot([x["val"] for x in h], color=COL[k], alpha=.7, label=k if s == 0 else None)
for a_, t in zip(ax, ["train 損失", "val 損失"]):
    a_.set_title(t + "（標準化の基準がモデルごとに違うため値は比較不可）"); a_.set_xlabel("epoch"); a_.set_yscale("log"); a_.grid(alpha=.3)
ax[0].legend(); fig.tight_layout(); fig.savefig(f"{OUT}/training_curves.png", dpi=130)
print("ok")
