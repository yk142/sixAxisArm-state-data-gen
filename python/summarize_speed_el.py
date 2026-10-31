"""Phase 5-2: 逐次の実時間比の図（issue #91）。reports/issue-91/speed_py.json と speed_simscape*.json から。"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
d = "reports/issue-91"
r = {"Simscape 弾性（真値）": json.load(open(f"{d}/speed_simscape.json"))["ms_per_step"],
     "Simscape 剛体": json.load(open(f"{d}/speed_simscape_rigid.json"))["ms_per_step"]}
r.update(json.load(open(f"{d}/speed_py.json"))["ms_per_step"])
G = {"2慣性解析 torch sub8": "○", "2慣性解析 numba sub8": "○", "2慣性解析 numba sub8（M 凍結）": "×（Δ整定 1 件）", "2慣性解析 numba sub4": "×", "2慣性解析 numba sub4（M 凍結）": "×"}
names = list(r); v = [r[k] for k in names]
fig, ax = plt.subplots(figsize=(10, 5.5))
cols = ["tab:gray" if "Simscape" in k else "tab:blue" if "解析" in k else "tab:orange" for k in names]
ax.barh(range(len(v)), v, color=cols); ax.set_xscale("log"); ax.axvline(1.0, color="r", ls="--", label="実時間（1 ms/step）")
ax.set_yticks(range(len(v))); ax.set_yticklabels([k + (f"  [水準G {G[k]}]" if k in G else "") for k in names], fontsize=8); ax.invert_yaxis()
for i, x in enumerate(v): ax.text(x * 1.1, i, f"{x:.3g}", va="center", fontsize=8)
ax.set_xlabel("1 ms 分の計算時間 [ms]（逐次 B=1、1 スレッド）＝ 実時間比"); ax.legend(); ax.grid(alpha=.3, axis="x", which="both")
ax.set_title("逐次の計算時間: Simscape・2 慣性解析・NN"); fig.tight_layout(); fig.savefig(f"{d}/speed.png", dpi=110)
