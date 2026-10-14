"""切替方式（eval_switch.py）の集計: 表（Markdown）と図。全窓（自由＋接触）の合算 RMSE も出す。"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

IN = "data/eval/switch3/switch_metrics.json"
OUT = sys.argv[1] if len(sys.argv) > 1 else "reports/issue-37"
r = json.load(open(IN))
DS = [0.5, 1, 2, 5]
f = lambda a: " / ".join(f"{x:.3g}" for x in a)
lines, pooled = [], {}
for L in (50, 200):
    fr, ct = r[f"L{L}/test/free"], r[f"L{L}/test/contact"]
    nf, nc = fr["n"], ct["n"]
    def row(key, field):
        a = np.array([fr["models"][key][field], ct["models"][key][field]])
        return a, np.sqrt((nf * a[0] ** 2 + nc * a[1] ** 2) / (nf + nc))
    lines.append(f"\n### L={L}（test: 自由運動 {nf} 本 + 接触を含む {nc} 本）終端の速度誤差 RMSE [rad/s]、J1/J2/J3（A・B は 3 シード平均）\n")
    lines.append("| 方式 | 自由運動 | 接触を含む | 全窓の合算 |\n|---|---|---|---|")
    def rows(label, keys):
        out = []
        for fld in ["v_end_rmse"]:
            ms = [row(k, fld) for k in keys]
            out = [np.mean([m[0][0] for m in ms], 0), np.mean([m[0][1] for m in ms], 0), np.sqrt(np.mean([m[1] ** 2 for m in ms], 0))]
        lines.append(f"| {label} | {f(out[0])} | {f(out[1])} | {f(out[2])} |")
        return out
    base = {}
    for nm, lab in [("persist", "状態維持"), ("analytic", "解析"), ("hybrid", "hybrid")]:
        base[nm] = rows(lab, [nm])
    res = {"hybrid": base["hybrid"]}
    for k in "AB":
        rows(f"{k} 単独", [f"{k}_s{s}" for s in range(3)])
        for d in DS:
            res[f"{k}sw{d}"] = rows(f"{k} 切替 {d:g}°", [f"{k}sw{d:g}_s{s}" for s in range(3)])
    pooled[L] = res
    lines.append("\nL=%d 終端の位置誤差 RMSE [rad]（接触を含む窓、J1/J2/J3）: hybrid %s / B 切替 0.5° %s\n" % (
        L, f(ct["models"]["hybrid"]["q_end_rmse"]), f(np.mean([ct["models"][f"Bsw0.5_s{s}"]["q_end_rmse"] for s in range(3)], 0))))
    fm = r[f"L{L}/test/frac_model_steps"]
    lines.append(f"NN を使ったステップの割合（シード 0、test の全窓）: " + ", ".join(f"{k.split('_')[0]} {v:.1%}" for k, v in fm.items() if k.endswith("_s0")) + "\n")
open(f"{OUT}/switch_tables.md", "w").write("\n".join(lines) + "\n")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for j, L in enumerate((50, 200)):
    x = np.arange(3); w = 0.16
    items = [("hybrid", "hybrid", "tab:purple")] + [(f"Bsw{d}", f"B 切替 {d:g}°", plt.cm.Oranges(0.35 + 0.18 * i)) for i, d in enumerate(DS)]
    for i, (k, lab, c) in enumerate(items):
        ax[j].bar(x + (i - 2.5) * w, pooled[L][k][1], w, color=c, label=lab)      # 接触を含む窓
    ax[j].set_xticks(x); ax[j].set_xticklabels(["J1", "J2", "J3"]); ax[j].set_ylabel("終端の速度誤差 RMSE [rad/s]")
    ax[j].set_title(f"test 接触を含む窓 L={L}（B は 3 シード平均）"); ax[j].grid(alpha=.3, axis="y")
ax[0].legend(); fig.tight_layout(); fig.savefig(f"{OUT}/switch_contact.png", dpi=130)
