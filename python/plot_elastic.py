"""Phase 4-12: 弾性真値と剛体真値・剛体モデルのステップ応答の比較図（issue #71）。"""
import argparse, os
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

DEG = 180 / np.pi


def q_step(path, g, ax):
    with h5py.File(path, "r") as f:
        q = f[g]["truth_q"][:]
        if q.shape[0] < q.shape[1]: q = q.T
        at = f[g].attrs; sc = lambda k: float(np.ravel(at[k])[0])
        i0 = int(round((sc("tStep") - 0.01) * 8000)); t = np.arange(len(q)) / 8000 - sc("tStep")
        return t, (q[:, ax - 1] - q[i0, ax - 1]) / sc("step")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rigid", default="data/export/payload_eval6.h5")
    ap.add_argument("--elastic", default="data/export/payload_eval6_el20.h5")
    ap.add_argument("--traj", nargs="+", default=[f"data/eval/el20_s{i}/traj.npz" for i in range(4)])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    cases = [(0, 100, 100, 1), (8, 100, 100, 1), (8, 100, 100, 2), (0, 60, 100, 2), (8, 60, 100, 5), (8, 150, 50, 3)]
    for a_, (m, wn, z, ax) in zip(axs.ravel(), cases):
        g = f"/pl{m}_wn{wn}_z{z}_j{ax}"
        t, yr = q_step(a.rigid, g, ax); _, ye = q_step(a.elastic, g, ax)
        a_.plot(t, yr, label="剛体真値"); a_.plot(t, ye, label="弾性真値（リンク側）")
        a_.axhline(1, color="gray", lw=0.5); a_.set_xlim(-0.05, 0.6); a_.set_title(f"{m} kg, wn {wn}, ζ {z / 100:g}, 軸 {ax}"); a_.grid(alpha=0.3)
        a_.set_xlabel("ステップからの時間 [s]"); a_.set_ylabel("q / 目標")
    axs[0, 0].legend()
    fig.tight_layout(); fig.savefig(f"{a.out}/step_rigid_vs_elastic.png", dpi=110)


if __name__ == "__main__":
    main()
