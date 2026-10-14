"""NN A 改善実験の表と図（issue #42）。"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
OUT = "reports/issue-42"
RUNS = [("4-2 の A（再掲）", "data/models/full3/A_s0"),
        ("r0 標準化出力・全データ", "data/models/a_exp/r0_std_all_s0"), ("r1 加速度出力・全データ", "data/models/a_exp/r1_acc_all_s0"),
        ("r2 加速度出力・自由のみ", "data/models/a_exp/r2_acc_free_s0"), ("r4 同・40 エポック", "data/models/a_exp/r4_acc_free_e40_s0"),
        ("r5 同・wd 0.05", "data/models/a_exp/r5_acc_free_wd_s0"), ("r3 幅512 深さ4", "data/models/a_exp/r3_acc_free_w512d4_s0"),
        ("r6 幅512 深さ4 60エポック", "data/models/a_exp/r6_w512d4_e60_s0")]
base = json.load(open("data/eval/baseline3/metrics.json")) if os.path.exists("data/eval/baseline3/metrics.json") else None
lines = ["### 1 ステップ加速度誤差 RMSE [rad/s²]（test 自由運動、J1/J2/J3、モデル選択は val で行った）\n", "| モデル | test J1 | J2 | J3 | 平均 |", "|---|---|---|---|---|"]
means = []
for lab, p in RUNS:
    if not os.path.exists(f"{p}/metrics.json"):
        continue
    m = json.load(open(f"{p}/metrics.json"))["test"]["free"]["acc_rmse"]
    lines.append(f"| {lab} | " + " | ".join(f"{x:.3f}" for x in m) + f" | {np.mean(m):.3f} |"); means.append((lab, np.mean(m)))
lines.append("| 解析サブステップ 1（4-3b） | 0.014 | 0.024 | 0.030 | 0.023 |\n| 解析サブステップ 2（4-3b） | 0.013 | 0.007 | 0.013 | 0.011 |")
W = json.load(open("data/eval/a_window/window_free.json"))
for L in ("L50", "L200"):
    lines += [f"\n### 自由窓（test、開ループ、{L[1:]} ステップ、窓 {W[L]['n']} 本）の終端誤差", "", "| 方式 | 速度 J1 | J2 | J3 [rad/s] | 位置 J1 | J2 | J3 [rad] |", "|---|---|---|---|---|---|---|"]
    for k, v in W[L]["models"].items():
        lines.append(f"| {k} | " + " | ".join(f"{x:.2g}" for x in v["v_end_rmse"]) + " | " + " | ".join(f"{x:.2g}" for x in v["q_end_rmse"]) + " |")
open(f"{OUT}/a_tables.md", "w").write("\n".join(lines) + "\n")
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
ax[0].barh([l for l, _ in means][::-1], [m for _, m in means][::-1], color="tab:blue")
ax[0].axvline(0.023, color="tab:purple", ls="--", label="解析サブステップ1（0.023）"); ax[0].axvline(0.011, color="tab:brown", ls="--", label="解析サブステップ2（0.011）")
ax[0].set_xscale("log"); ax[0].set_xlabel("1 ステップ加速度誤差 RMSE の 3 軸平均 [rad/s²]（test 自由運動）"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which="both", axis="x")
for k, c in (("persist", "k"), ("A 4-2", "tab:gray"), ("r2_acc_free_s0", "tab:cyan"), ("r6_w512d4_e60_s0", "tab:blue"), ("解析 サブステップ1", "tab:purple"), ("解析 サブステップ2", "tab:brown")):
    v = np.array(W["L200"]["models"][k]["v_curve"]); ax[1].plot(np.arange(len(v)), np.maximum(v, 1e-6), color=c, label=k)
ax[1].set_yscale("log"); ax[1].set_xlabel("予測ステップ数（1 ステップ = 1 ms）"); ax[1].set_ylabel("速度誤差 RMSE（3 軸平均）[rad/s]"); ax[1].set_title("自由窓 L=200（test、開ループ）"); ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/a_progress.png", dpi=130)
