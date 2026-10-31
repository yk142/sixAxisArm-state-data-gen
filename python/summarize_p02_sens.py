"""Phase 5-4: 位置決め誤差（P02）のパラメータ感度の集計（issue #95）。data/eval/ptp95/*/ptp_pos.json を束ねる。"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
P02 = 0.02
PARAMS = [("fric", "摩擦（Fc・Bv）"), ("mass", "質量・慣性"), ("k", "ねじり剛性 k"), ("c", "減衰 c"), ("Jm", "モータ慣性 Jm")]
SCALES = [0.9, 0.97, 0.99, 1.01, 1.03, 1.1]


def main(out="reports/issue-95"):
    os.makedirs(out, exist_ok=True)
    E = {}
    for f in sorted(glob.glob("data/eval/ptp95/*/ptp_pos.json")):
        R = json.load(open(f))
        for k in R[0]:
            if isinstance(R[0][k], dict):
                E[k.replace("2慣性 ", "")] = np.array([r[k]["end_mm"] for r in R])
    base = E["k=1"]
    L = [f"位置決め 48 回（P-bench v1）の停留の終わりの先端誤差の **最大** [mm]（括弧内は中央値）。モデルは numba 2 慣性解析（サブステップ 4・M 凍結）。基準（倍率 1）: 最大 {base.max():.4f}、中央値 {np.median(base):.4f}。水準 P02 は ≤ {P02} mm。\n",
         "| パラメータ | " + " | ".join(f"×{s:g}" for s in SCALES) + " | 許容誤差の目安 |", "|---|" + "---|" * (len(SCALES) + 1)]
    tol = {}
    for p, nm in PARAMS:
        cells, xs, ys = [], [], []
        for s in SCALES:
            e = E.get(f"{p}={s:g}")
            if e is None: cells.append("-"); continue
            cells.append(f"{e.max():.3f}（{np.median(e):.3f}）" + (" ×" if e.max() > P02 else ""))
            xs.append(abs(s - 1)); ys.append(e.max())
        # 誤差 ≈ a·|δ|（小さい δ で比例）として、0.02 mm になる |δ| を ±1% の点から外挿（両側の大きい方）
        a1 = max(E[f"{p}=0.99"].max(), E[f"{p}=1.01"].max()) / 0.01
        tol[p] = P02 / a1
        L.append(f"| {nm} | " + " | ".join(cells) + f" | ±{tol[p] * 100:.2g}% |")
    L.append("\n許容誤差の目安: ±1% のときの最大誤差から、誤差がずれに比例するとして 0.02 mm になるずれを外挿した値（各パラメータ単独）。")
    open(f"{out}/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
    fig, ax = plt.subplots(figsize=(8, 5))
    for p, nm in PARAMS:
        xs = sorted([s for s in SCALES if f"{p}={s:g}" in E] + [1.0])
        ys = [base.max() if s == 1.0 else E[f"{p}={s:g}"].max() for s in xs]       # ずれ 0 は基準
        ax.plot([(s - 1) * 100 for s in xs], ys, "o-", label=nm)
    ax.axhline(P02, color="r", ls="--", label="水準 P02（0.02 mm）"); ax.axhline(base.max(), color="gray", ls=":", label="基準（ずれ 0）")
    ax.set_yscale("log"); ax.set_xlabel("パラメータのずれ [%]"); ax.set_ylabel("位置決め後の先端誤差 最大 [mm]（48 回）")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8, loc="lower left"); ax.set_title("位置決め誤差のパラメータ感度（2 慣性解析、P-bench v1）")
    fig.tight_layout(); fig.savefig(f"{out}/p02_sens.png", dpi=110)


main()
