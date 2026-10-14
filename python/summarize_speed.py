"""bench_speed.py の結果から表と図を作る（issue #39）。"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
OUT = sys.argv[1] if len(sys.argv) > 1 else "reports/issue-39"
R = {t: json.load(open(f"data/eval/speed3/bench_t{t}.json")) for t in (4, 1)}
B = R[4]["batches"]
lines = []
for t in (4, 1):
    lines.append(f"\n### スレッド {t}: 1 ステップ（1 ms）当たりの時間 [ms]（バッチ 1 はレイテンシ、大バッチはスループットの逆数）\n")
    lines.append("| 方式 | " + " | ".join(f"B={b}" for b in B) + " |\n|---|" + "---|" * len(B))
    for k, v in R[t]["time_per_call_s"].items():
        lines.append(f"| {k} | " + " | ".join(f"{x / b * 1e3:.4g}" for x, b in zip(v, B)) + " |")
open(f"{OUT}/speed_tables.md", "w").write("\n".join(lines) + "\n")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
col = {"解析 RK4 サブステップ1": "gray", "解析 RK4 サブステップ2": "k", "解析 RK4 サブステップ4": "dimgray", "NN A": "tab:blue", "NN B": "tab:orange", "NN C": "tab:green"}
for k, v in R[4]["time_per_call_s"].items():
    ax[0].plot(B, [x / b * 1e3 for x, b in zip(v, B)], "o-", color=col[k], label=k)
ax[0].axhline(7688.138 / 1289294 * 1e3, color="tab:red", ls="--", label="Simscape（生成の実時間 / 遷移数）")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("バッチサイズ"); ax[0].set_ylabel("1 ステップ当たりの時間 [ms]（4 スレッド）")
ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=8)
acc = {"解析 RK4 サブステップ1": 0.02, "解析 RK4 サブステップ2": 0.0105, "NN A": 7.2}   # test 自由運動の 1 ステップ加速度 RMSE（3 軸の概算、rad/s²）
for k, a in acc.items():
    ax[1].scatter(R[4]["time_per_call_s"][k][0] * 1e3, a, color=col[k], s=80, label=k)
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("1 ステップの時間 [ms]（バッチ 1、4 スレッド）"); ax[1].set_ylabel("1 ステップ誤差（test 自由運動、加速度換算 RMSE の概算 [rad/s²]）")
ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/speed.png", dpi=130)
