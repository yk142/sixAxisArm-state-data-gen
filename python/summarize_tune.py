"""issue #57: 6 軸の A の改善（探索と 3 seed）の集計表と図。使い方: python python/summarize_tune.py [reports/issue-57]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
out = sys.argv[1] if len(sys.argv) > 1 else "reports/issue-57"
os.makedirs(out, exist_ok=True)
J = lambda p: json.load(open(p))
E = "data/models/a_exp6"
f = lambda x, n=3: " / ".join(f"{v:.{n}g}" for v in x)
TUNED = [f"r10_wd0.2_e100_s{s}" for s in range(3)]
ORIG = "r6_w512d4_e60_s0"

# 探索の一覧（seed 0、val 自由運動の加速度 RMSE の軸平均の最小値、test 自由運動の軸別）
explore = [("元: 幅 512・深さ 4・60 エポック", ORIG), ("深さ 6・40 エポック", "r7_w512d6_e40_s0"), ("100 エポック", "r7_w512d4_e100_s0"),
           ("ドロップアウト 0.1", "r8_drop01_s0"), ("重み減衰 0.05", "r8_wd_s0"), ("重み減衰 0.2", "r9_wd0.2_s0"), ("重み減衰 0.5", "r9_wd0.5_s0"),
           ("重み減衰 0.2・幅 1024", "r10_wd0.2_w1024_s0"), ("重み減衰 0.2・100 エポック", TUNED[0])]
lines = ["## 探索（seed 0。共通: 加速度表現・非接触のみ学習・lr 2e-3・バッチ 4096。val 平均は val 自由運動の加速度 RMSE の軸平均の最小値）", "", "| 設定 | val 平均 | test 自由 加速度 RMSE [rad/s²]（J1〜J6） |", "|---|---|---|"]
for lab, tag in explore:
    h = J(f"{E}/{tag}/history.json"); v = min(np.mean(e["val_free_acc_rmse"]) for e in h)
    lines.append(f"| {lab} | {v:.3f} | {f(J(f'{E}/{tag}/metrics.json')['test']['free']['acc_rmse'])} |")
lines.append("")

ana1 = np.array(J("data/eval/baseline6/baseline.json")["test"]["free"]["analytic"])
b1 = np.array(J("data/models/full6/B_s0/metrics.json")["test"]["free"]["acc_rmse"])
a1o = np.array(J(f"{E}/{ORIG}/metrics.json")["test"]["free"]["acc_rmse"])
a1t = np.array([J(f"{E}/{t}/metrics.json")["test"]["free"]["acc_rmse"] for t in TUNED])
lines += ["## 1 ステップ加速度 RMSE（test、自由運動、[rad/s²]、J1〜J6）", "", "| モデル | RMSE |", "|---|---|", f"| 解析（サブステップ 2） | {f(ana1)} |", f"| B（seed 0） | {f(b1)} |",
          f"| A 調整前（seed 0） | {f(a1o)} |"] + [f"| A 調整後 seed {s} | {f(a1t[s])} |" for s in range(3)] + [f"| A 調整後 3 seed の平均 | {f(a1t.mean(0))} |", f"| A 調整後 / 調整前（平均） | {f(a1t.mean(0) / a1o, 2)} |", ""]

aw0 = J("data/eval/a_window6/window_free.json"); awt = J("data/eval/a_window6_tuned/window_free.json")
w6 = J("data/eval/window6/window_metrics.json")
vend = {}
for Lw, lab in (("L50", "L=50"), ("L200", "L=200")):
    m = awt[Lw]["models"]; wk = w6[f"{Lw}/test/free"]["models"]
    lines += [f"## 自由窓の終端速度誤差（test、{lab}、窓 {awt[Lw]['n']} 本、開ループ、[rad/s]、J1〜J6）", "", "| モデル | 終端速度 |", "|---|---|",
              f"| 解析 サブステップ 2 | {f(m['解析 サブステップ2']['v_end_rmse'])} |", f"| B（seed 0） | {f(wk['B_s0']['v_end_rmse'])} |", f"| A 調整前 | {f(m[ORIG]['v_end_rmse'])} |"]
    for s, t in enumerate(TUNED):
        lines.append(f"| A 調整後 seed {s} | {f(m[t]['v_end_rmse'])} |")
    lines.append("")
    if Lw == "L200":
        vend = dict(ana=m["解析 サブステップ2"]["v_end_rmse"], b=wk["B_s0"]["v_end_rmse"], o=m[ORIG]["v_end_rmse"], t=[m[t]["v_end_rmse"] for t in TUNED])

# 閉ループ（軸別の位置 RMS の最大、[deg]）
base = J("data/eval/closed6/closed.json")
cl = {}
for g in ("ptp", "nearlimit"):
    mx = lambda d, m: np.array([d[g][s][m]["q_err_rms_deg"] for s in d[g]]).max(0)
    ta = [J(f"data/eval/closed6_a_s{s}/closed.json") for s in range(3)]
    cl[g] = dict(ana=mx(base, "解析"), b=mx(base, "B（4-2）"), o=mx(base, "A（4-3d の r6）"), t=[mx(d, "A") for d in ta])
    lines += [f"## 閉ループの位置誤差（{g}、{len(base[g])} 本、Simscape との差の RMS の最大、[deg]、J1〜J6）", "", "| モデル | 位置 RMS 最大 |", "|---|---|",
              f"| 解析 | {f(cl[g]['ana'])} |", f"| B（seed 0） | {f(cl[g]['b'])} |", f"| A 調整前 | {f(cl[g]['o'])} |"] + [f"| A 調整後 seed {s} | {f(cl[g]['t'][s])} |" for s in range(3)] + [""]
    lines.append(f"（発散: 調整後 A {sum(not d[g][s]['A']['ok'] for d in ta for s in d[g])} 件）\n")
open(f"{out}/summary.md", "w").write("\n".join(lines)); print("\n".join(lines))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6)); x = np.arange(6); w = 0.2
def panel(a, ana, b, o, t, title, yl):
    a.bar(x - 1.5 * w, ana, w, label="解析", color="tab:gray"); a.bar(x - .5 * w, b, w, label="B", color="tab:blue"); a.bar(x + .5 * w, o, w, label="A 調整前", color="tab:red")
    t = np.array(t); a.bar(x + 1.5 * w, t.mean(0), w, yerr=[t.mean(0) - t.min(0), t.max(0) - t.mean(0)], capsize=2, label="A 調整後（3 seed 平均、範囲）", color="tab:orange")
    a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels([f"J{i + 1}" for i in x]); a.set_ylabel(yl); a.set_title(title); a.grid(alpha=.3, axis="y")
panel(ax[0], ana1, b1, a1o, a1t, "1 ステップ精度（test、自由）", "加速度 RMSE [rad/s²]")
panel(ax[1], vend["ana"], vend["b"], vend["o"], vend["t"], "自由窓 L=200 の終端速度誤差（test）", "終端速度 RMS [rad/s]")
panel(ax[2], cl["ptp"]["ana"], cl["ptp"]["b"], cl["ptp"]["o"], cl["ptp"]["t"], "閉ループ PTP 12 本: 位置誤差", "Simscape との差 RMS の最大 [deg]")
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{out}/tune.png", dpi=130)
