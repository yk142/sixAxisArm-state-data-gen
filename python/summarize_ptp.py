"""P-bench（PTP の位置決め後の先端誤差）の集計（issue #93）。data/eval/ptp93/*/ptp_pos.json を束ねる。"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
P02 = 0.02
SPEED = json.load(open("reports/issue-91/speed_py.json"))["ms_per_step"]
SPEED_KEY = {"2慣性 numba sub8": "2慣性解析 numba sub8", "2慣性 numba sub8（M 凍結）": "2慣性解析 numba sub8（M 凍結）",
             "2慣性 numba sub4": "2慣性解析 numba sub4", "2慣性 numba sub4（M 凍結）": "2慣性解析 numba sub4（M 凍結）"}


def main(src="data/eval/ptp93", out="reports/issue-93", title="PTP の位置決め後の先端誤差（Simscape 弾性真値との差）"):
    os.makedirs(out, exist_ok=True)
    E = {}
    for f in sorted(glob.glob(f"{src}/*/ptp_pos.json")):
        R = json.load(open(f))
        for k in R[0]:
            if isinstance(R[0][k], dict):
                nm = "真値どうし（許容誤差 1e-6/1e-8 対 1e-9/1e-11）" if k == "真値（--ref）" else k
                E[nm] = (np.array([r[k]["end_mm"] for r in R]), np.array([r[k]["last50_max_mm"] for r in R]))
    resid = np.array([r["truth_resid_mm"] for r in R])
    L = [f"位置決め {len(resid)} 回（PTP 12 本 × 4 到達点、各 0.2 s 停留の終わり）。先端 = フランジから 0.1 m 先。水準 P02: 真値との差 ≤ {P02} mm。\n",
         "| モデル | 停留の終わり 中央値[mm] | 停留の終わり 最大[mm] | 停留の最後 50 ms の最大[mm] | ≤ 0.02 mm の割合 | 1 ms あたり計算時間[ms]（#91） | 水準 P02 |",
         "|---|---|---|---|---|---|---|"]
    for k, (e, l) in E.items():
        sp = SPEED.get(SPEED_KEY.get(k, ""), None)
        L.append(f"| {k} | {np.median(e):.2e} | {e.max():.2e} | {l.max():.2e} | {np.mean(e <= P02):.0%} | {'-' if sp is None else f'{sp:.3f}'} | {'○' if e.max() <= P02 else '×'} |")
    L.append(f"\n参考: 真値の、参照（目標角）に対する先端の残差（停留の終わり）は中央値 {np.median(resid):.2f} mm、最大 {resid.max():.2f} mm（PD の定常偏差。モデルとの差とは別）。")
    open(f"{out}/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = list(E)
    ax.boxplot([np.maximum(E[k][0], 1e-9) for k in names], vert=False, labels=names, whis=(0, 100))
    ax.axvline(P02, color="r", ls="--", label="水準 P02（0.02 mm）"); ax.set_xscale("log"); ax.set_xlabel("停留の終わりの先端誤差 [mm]（48 回）")
    ax.grid(alpha=.3, which="both", axis="x"); ax.legend(loc="lower right"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(f"{out}/ptp_pos.png", dpi=110)



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/eval/ptp93"); ap.add_argument("--out", default="reports/issue-93")
    ap.add_argument("--title", default="PTP の位置決め後の先端誤差（Simscape 弾性真値との差）")
    a = ap.parse_args(); main(a.src, a.out, a.title)
