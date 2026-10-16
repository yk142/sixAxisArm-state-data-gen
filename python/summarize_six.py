"""issue #53: 6 軸の評価の集計表と図。使い方: python python/summarize_six.py [reports/issue-53]"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
out = sys.argv[1] if len(sys.argv) > 1 else "reports/issue-53"
J = lambda p: json.load(open(p))
ana1 = np.array(J("data/eval/baseline6/baseline.json")["test"]["free"]["analytic"])           # 解析（サブステップ 2）の 1 ステップ加速度 RMSE [rad/s²]
b1 = np.array(J("data/models/full6/B_s0/metrics.json")["test"]["free"]["acc_rmse"])
a1 = np.array(J("data/models/a_exp6/r6_w512d4_e60_s0/metrics.json")["test"]["free"]["acc_rmse"])
a1_3 = np.array(J("data/models/a_exp/r6_w512d4_e60_s0/metrics.json")["test"]["free"]["acc_rmse"])
w6 = J("data/eval/window6/window_metrics.json"); aw6 = J("data/eval/a_window6/window_free.json"); aw3 = J("data/eval/a_window/window_free.json")
f = lambda x, n=4: " / ".join(f"{v:.{n}g}" for v in x)
lines = ["## 1 ステップ加速度 RMSE（test、自由運動、[rad/s²]。軸 J1〜J6）", "", "| モデル | RMSE |", "|---|---|",
         f"| 解析（サブステップ 2） | {f(ana1)} |", f"| B（seed 0） | {f(b1)} |", f"| A（r6、seed 0） | {f(a1)} |",
         f"| （参考）3 軸の A（r6）J1〜J3 | {f(a1_3)} |", ""]
for Lw in ("L50", "L200"):
    m6 = aw6[Lw]["models"]; n6 = aw6[Lw]["n"]
    wk = w6[f"{Lw}/test/free"]["models"]
    lines += [f"## 自由窓の終端誤差（test、{Lw.replace('L', 'L=')}、窓 {n6} 本、開ループ。軸 J1〜J6）", "", "| モデル | 終端速度 [rad/s] | 終端位置 [deg] |", "|---|---|---|"]
    for lab, v, q in (("persist（状態維持）", m6["persist"]["v_end_rmse"], m6["persist"]["q_end_rmse"]),
                      ("解析 サブステップ 1", m6["解析 サブステップ1"]["v_end_rmse"], m6["解析 サブステップ1"]["q_end_rmse"]),
                      ("解析 サブステップ 2", m6["解析 サブステップ2"]["v_end_rmse"], m6["解析 サブステップ2"]["q_end_rmse"]),
                      ("B（seed 0）", wk["B_s0"]["v_end_rmse"], wk["B_s0"]["q_end_rmse"]),
                      ("A（r6、seed 0）", m6["r6_w512d4_e60_s0"]["v_end_rmse"], m6["r6_w512d4_e60_s0"]["q_end_rmse"])):
        lines.append(f"| {lab} | {f(v, 3)} | {f(np.degrees(q), 3)} |")
    m3 = aw3[Lw]["models"]
    lines.append(f"| （参考）3 軸の A（r6）J1〜J3 | {f(m3['r6_w512d4_e60_s0']['v_end_rmse'], 3)} | {f(np.degrees(m3['r6_w512d4_e60_s0']['q_end_rmse']), 3)} |")
    lines.append("")
open(f"{out}/summary.md", "w").write("\n".join(lines)); print("\n".join(lines))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4)); x = np.arange(6); w = 0.27
ax[0].bar(x - w, ana1, w, label="解析", color="tab:gray"); ax[0].bar(x, b1, w, label="B", color="tab:blue"); ax[0].bar(x + w, a1, w, label="A(r6)", color="tab:red")
ax[0].plot(np.arange(3) + w, a1_3, "k_", ms=14, mew=2, label="3 軸の A(r6)")
ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels([f"J{i + 1}" for i in x]); ax[0].set_ylabel("1 ステップ加速度 RMSE [rad/s²]"); ax[0].set_title("6 軸: 1 ステップ精度（test、自由）"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")
m6 = aw6["L200"]["models"]; wk = w6["L200/test/free"]["models"]; m3 = aw3["L200"]["models"]
ax[1].bar(x - w, m6["解析 サブステップ2"]["v_end_rmse"], w, label="解析", color="tab:gray"); ax[1].bar(x, wk["B_s0"]["v_end_rmse"], w, label="B", color="tab:blue"); ax[1].bar(x + w, m6["r6_w512d4_e60_s0"]["v_end_rmse"], w, label="A(r6)", color="tab:red")
ax[1].plot(np.arange(3) + w, m3["r6_w512d4_e60_s0"]["v_end_rmse"], "k_", ms=14, mew=2, label="3 軸の A(r6)")
ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels([f"J{i + 1}" for i in x]); ax[1].set_ylabel("終端速度の RMS 誤差 [rad/s]"); ax[1].set_title("6 軸: 自由窓 L=200 の終端速度誤差（test）"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(f"{out}/six_axis.png", dpi=130)
