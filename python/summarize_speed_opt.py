"""bench_speed_opt.py の結果から表と図を作る（issue #41）。"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
OUT = sys.argv[1] if len(sys.argv) > 1 else "reports/issue-41"
R = {t: json.load(open(f"data/eval/speed3_opt/bench_t{t}.json")) for t in (4, 1)}
B = R[4]["batches"]
lines = []
for t in (4, 1):
    lines.append(f"\n### スレッド {t}: 1 ステップ（1 ms）当たりの時間 [ms]（各点は 5 回測定の最良値。B=1 はレイテンシ、大バッチはスループットの逆数）\n")
    lines.append("| 方式 | " + " | ".join(f"B={b}" for b in B) + " |\n|---|" + "---|" * len(B))
    for k, v in R[t]["time_per_call_s"].items():
        lines.append(f"| {k} | " + " | ".join(f"{x / b * 1e3:.4g}" for x, b in zip(v, B)) + " |")
open(f"{OUT}/speed_tables.md", "w").write("\n".join(lines) + "\n")

T = R[4]["time_per_call_s"]
show = {"解析 旧 サブステップ1": ("gray", "--"), "解析 旧 サブステップ2": ("k", "--"),
        "解析 高速+compile サブステップ1": ("tab:purple", "-"), "解析 高速+compile サブステップ2": ("tab:brown", "-"),
        "NN A": ("tab:blue", "-"), "NN B (解析 高速+compile)": ("tab:orange", "-"), "NN C (解析 高速+compile)": ("tab:green", "-")}
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
for k, (c, ls) in show.items():
    ax[0].plot(B, [x / b * 1e3 for x, b in zip(T[k], B)], "o" + ls, color=c, label=k)
ax[0].axhline(7688.138 / 1289294 * 1e3, color="tab:red", ls=":", label="Simscape（生成の実時間 / 遷移数）")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("バッチサイズ"); ax[0].set_ylabel("1 ステップ当たりの時間 [ms]（4 スレッド）")
ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=7)
# 1 ステップ加速度 RMSE（test 自由運動、3 軸の平均 [rad/s²]。4-3b の値。最適化は数値を変えない）
acc = {"解析 サブステップ1": 0.0227, "解析 サブステップ2": 0.011, "NN A": 7.2}
for lab, kk, a, c, m in (("解析 サブステップ1（旧）", "解析 旧 サブステップ1", acc["解析 サブステップ1"], "gray", "s"),
                         ("解析 サブステップ2（旧）", "解析 旧 サブステップ2", acc["解析 サブステップ2"], "k", "s"),
                         ("解析 サブステップ1（高速+compile）", "解析 高速+compile サブステップ1", acc["解析 サブステップ1"], "tab:purple", "o"),
                         ("解析 サブステップ2（高速+compile）", "解析 高速+compile サブステップ2", acc["解析 サブステップ2"], "tab:brown", "o"),
                         ("NN A", "NN A", acc["NN A"], "tab:blue", "o")):
    ax[1].scatter(T[kk][0] * 1e3, a, color=c, marker=m, s=70, label=lab)
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("1 ステップの時間 [ms]（B=1、4 スレッド）")
ax[1].set_ylabel("1 ステップ誤差（test 自由運動、加速度換算 RMSE の 3 軸平均 [rad/s²]）")
ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/speed_opt.png", dpi=130)
