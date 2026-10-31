"""Phase 5-7: ① 公称 / ② 同定 / ④ ハイブリッド の P-bench 比較とデータ量の軸（issue #101）。"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
P02 = 0.02


def get(path, prefix):
    R = json.load(open(path)); k = [x for x in R[0] if isinstance(R[0][x], dict) and x.startswith(prefix)][0]
    return np.array([r[k]["end_mm"] for r in R])


def main(out="reports/issue-101"):
    os.makedirs(out, exist_ok=True)
    rows = [("① 公称", "全", get("data/eval/ptp99/nom4_1/ptp_pos.json", "2慣性"), 0.128)]
    for f, nm in ((0.1, "10%"), (0.3, "30%"), (1.0, "100%")):
        p = f"data/eval/ptp101/f{f}/ptp_pos.json"
        rows.append(("② 同定（Bv≥0）", nm, get(p, "2慣性 同定"), 0.141))
        rows.append(("④ ハイブリッド", nm, get(p, "④"), 0.301))
    rows.append(("④ ハイブリッド（② は Bv 制約なし）", "100%", get("data/eval/ptp101/hy_full/ptp_pos.json", "④"), 0.301))
    L = ["位置決め 48 回（P-bench v1、モデル外の効果入り真値）の停留の終わりの先端誤差。データ量 = 同定・残差学習に使うシナリオの割合（PTP 8 本＋保持 19 本の中から無作為）。\n",
         "| モデル | データ量 | 中央値[mm] | 最大[mm] | ≤ 0.02 mm の割合 | 1 ms あたり計算時間[ms] | 水準 P02 |", "|---|---|---|---|---|---|---|"]
    for nm, d, e, sp in rows:
        L.append(f"| {nm} | {d} | {np.median(e):.4f} | {e.max():.4f} | {np.mean(e <= P02):.0%} | {sp:.3f} | {'○' if e.max() <= P02 else '×'} |")
    L.append("\n計算時間は numba（サブステップ 4・M 凍結）、逐次 B=1、1 スレッド。④ は残差 NN（軸ごとの 2×32 tanh）を各段で評価する分が増える。")
    open(f"{out}/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    lab = [f"{nm}\n{d}" for nm, d, e, sp in rows]
    ax[0].boxplot([np.maximum(e, 1e-6) for *_, e, sp in rows], vert=False, labels=lab, whis=(0, 100))
    ax[0].axvline(P02, color="r", ls="--", label="水準 P02"); ax[0].set_xscale("log"); ax[0].set_xlabel("停留の終わりの先端誤差 [mm]（48 回）"); ax[0].legend(); ax[0].grid(alpha=.3, which="both", axis="x")
    ax[0].tick_params(axis="y", labelsize=8); ax[0].set_title("① / ② / ④ の位置決め誤差")
    for nm, mk in (("② 同定（Bv≥0）", "s-"), ("④ ハイブリッド", "o-")):
        xs = [0.1, 0.3, 1.0]; r = [x for x in rows if x[0] == nm]
        ax[1].plot(xs, [np.median(x[2]) for x in r], mk, label=f"{nm} 中央値"); ax[1].plot(xs, [x[2].max() for x in r], mk.replace("-", "--"), label=f"{nm} 最大")
    ax[1].axhline(P02, color="r", ls="--"); ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("データ量（シナリオの割合）"); ax[1].set_ylabel("先端誤差 [mm]")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both"); ax[1].set_title("データ量の軸")
    fig.tight_layout(); fig.savefig(f"{out}/hybrid.png", dpi=110)


main()
