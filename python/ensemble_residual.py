"""残差 NN（fit_residual_el.py の residual.npz）を複数つないで平均する（issue #105）。使い方:
  ~/.venvs/sa-train/bin/python python/ensemble_residual.py --in data/models/res105/eps data/models/res105/eps_s1 --out data/models/res105/ens_eps
軸ごとに、隠れ層を並べ（第 1 層は縦に積む、第 2 層はブロック対角）、出力層を 1/K 倍して足す。1 つの MLP（幅 K × width）になるので twomass_nb がそのまま読める（blk = width で第 2 層のブロックの外を飛ばす）。
入力・出力の尺度（xs, ys）は 1 つ目に揃え、他は重みに畳み込む。"""
import argparse, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Ds = [np.load(f"{p}/residual.npz") for p in a.inp]; K = len(Ds)
    blk = [int(d["blk"]) if "blk" in d else int(d["width"]) for d in Ds]           # 第 2 層のブロック幅（--split の NN は width/2）
    out = {"width": sum(int(d["width"]) for d in Ds), "blk": blk[0]}
    assert len(set(blk)) == 1 and all(int(d["width"]) % blk[0] == 0 for d in Ds), "ブロック幅を揃える"
    for i in range(6):
        xs, ys = Ds[0][f"xs_{i}"], Ds[0][f"ys_{i}"]
        W0 = np.concatenate([d[f"W0_{i}"] * (xs / d[f"xs_{i}"])[None, :] for d in Ds], 0)
        b0 = np.concatenate([d[f"b0_{i}"] for d in Ds])
        H = [d[f"W1_{i}"].shape[0] for d in Ds]; W1 = np.zeros((sum(H), sum(H))); o = 0
        for d, h in zip(Ds, H):
            W1[o:o + h, o:o + h] = d[f"W1_{i}"]; o += h
        b1 = np.concatenate([d[f"b1_{i}"] for d in Ds])
        W2 = np.concatenate([d[f"W2_{i}"] * (d[f"ys_{i}"] / ys)[:, None] / K for d in Ds], 1)
        b2 = sum(d[f"b2_{i}"] * d[f"ys_{i}"] / ys for d in Ds) / K
        out.update({f"W0_{i}": W0, f"b0_{i}": b0, f"W1_{i}": W1, f"b1_{i}": b1, f"W2_{i}": W2, f"b2_{i}": b2,
                    f"xs_{i}": xs, f"ys_{i}": ys, f"eps_{i}": Ds[0][f"eps_{i}"]})
    os.makedirs(a.out, exist_ok=True)
    np.savez(f"{a.out}/residual.npz", **out)
    print("完了", a.out, "幅", out["width"])


if __name__ == "__main__":
    main()
