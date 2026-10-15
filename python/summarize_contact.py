"""issue #47: 接触を含む閉ループ評価の集計表と図。使い方: python python/summarize_contact.py [data/eval/closed_contact3] [reports/issue-47]"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
src = sys.argv[1] if len(sys.argv) > 1 else "data/eval/closed_contact3"
out = sys.argv[2] if len(sys.argv) > 2 else "reports/issue-47"
p = json.load(open("data/export/c8_params.json"))
d = json.load(open(f"{src}/closed.json"))["contact"]; z = np.load(f"{src}/traj.npz")
sc = list(d); models = [k for k in d[sc[0]] if k != "truth"]
cols = ["tab:gray", "tab:green", "tab:red", "tab:blue", "tab:orange"]
qmin, qmax = np.asarray(p["qMin"][:3]), np.asarray(p["qMax"][:3])
DEG = 180 / np.pi


def axis_side(name):                      # cp_nearlimit_j{a}{p|m}_{nnn}
    a = int(name.split("_j")[1][0]) - 1; return a, (1 if name.split("_j")[1][1] == "p" else -1)


def pen(q, a, side):                      # 目標軸のリミット超過量 [deg]（正 = リミットの外）
    return (side * (q[:, a] - (qmax[a] if side > 0 else qmin[a]))) * DEG


rows = []
for m in models + ["truth"]:
    ps, tgt, ctact = [], [], []
    for s in sc:
        a, side = axis_side(s); q = z[f"contact|{s}|{m}"]; t = z[f"contact|{s}|truth"]
        ps.append(pen(q, a, side).max()); 
        c = pen(t, a, side) > -0.1                                  # 真値が接触している区間
        tgt.append(np.sqrt(((np.degrees(q[:, a] - t[:, a]))[c] ** 2).mean()) if c.any() else np.nan)
    r = [d[s][m] for s in sc] if m != "truth" else None
    rows.append((m, np.median(ps), max(ps), np.median(tgt), np.nanmax(tgt),
                 None if r is None else max(max(x["q_err_rms_deg"]) for x in r),
                 None if r is None else max(max(x["q_err_max_deg"]) for x in r),
                 None if r is None else max(max(x["v_err_rms"]) for x in r),
                 None if r is None else max(max(x["q_err_end_deg"]) for x in r)))
f = lambda v, n=3: "-" if v is None else f"{v:.{n}f}"
lines = ["| モデル | リミット超過 中央値 [°] | 超過 最大 [°] | 接触中の目標軸誤差 RMS 中央値 [°] | 同 最大 [°] | 位置RMS 最大 [°] | 位置 最大誤差 [°] | 速度RMS 最大 [rad/s] | 終端誤差 最大 [°] |", "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r[0]} | {f(r[1], 2)} | {f(r[2], 2)} | {f(r[3], 2)} | {f(r[4], 2)} | {f(r[5], 2)} | {f(r[6], 2)} | {f(r[7])} | {f(r[8], 2)} |")
open(f"{out}/summary.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
x = np.arange(len(sc)); w = 0.16
for i, m in enumerate(models):
    ax[0].bar(x + (i - 2) * w, [max(max(d[s][m]["q_err_rms_deg"]), 1e-3) for s in sc], w, color=cols[i], label=m)
ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels([s.replace("cp_nearlimit_", "") for s in sc], rotation=60, fontsize=7)
ax[0].set_ylabel("位置誤差 RMS（軸最大）[deg]"); ax[0].set_title("Simscape 閉ループ（押し付け）との差"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3, axis="y")
for a_, s in zip(ax[1:], ("cp_nearlimit_j1p_002", "cp_nearlimit_j2m_001")):
    ai, side = axis_side(s); t = np.arange(len(z[f"contact|{s}|truth"])) * 1e-3
    a_.plot(t, pen(z[f"contact|{s}|ref"], ai, side), "k:", lw=.8, label="参照")
    a_.plot(t, pen(z[f"contact|{s}|truth"], ai, side), "k", lw=1.4, label="Simscape")
    for i, m in enumerate(models):
        a_.plot(t, pen(z[f"contact|{s}|{m}"], ai, side), color=cols[i], lw=.9, label=m)
    a_.axhline(0, color="m", lw=.6); a_.set_ylim(-8, 12); a_.set_xlabel("時刻 [s]"); a_.set_ylabel(f"J{ai+1} のリミット超過 [deg]（0 = リミット）"); a_.set_title(s); a_.grid(alpha=.3)
ax[1].legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"{out}/contact_closed_loop.png", dpi=130)
