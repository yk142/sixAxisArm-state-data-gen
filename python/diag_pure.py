"""③ 純 NN の静的なつり合いの精度（issue #111）。使い方:
  ~/.venvs/sa-train/bin/python python/diag_pure.py --data data/export/x_{speed,outer,payload}.h5 --pure data/models/pure111/w128/pure.npz --hybrid IDENT,NPZ --out data/eval/x111/static.json
X-bench の停留の終わり（P-bench は truth_tau が無い）（真値はほぼ静止、加速度 ≈ 0）の真値の状態・トルクで、モデルのリンク側加速度 ddq を出す。
モータ側を真値に固定したとき、ddq = 0 になる θ のずれ δθ = −(∂ddq/∂θ)⁻¹·ddq（線形化）と、それによる先端位置の差（q = qm − θ）を出す。
閉ループの位置決め誤差のうち「静止時のつり合いの誤り」で説明できる分の目安。"""
import argparse, json, os, sys
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.kinematics import Kinematics
from eval_ptp_pos import dwells


def model_f(kind, path):
    if kind == "pure":
        from sa_nsm.pure_nb import _load, _f
        P, _ = _load(path)
        return lambda y, tau, pl: _f(y, tau, pl, P)
    from sa_nsm.twomass_nb import TwoMassNB, _f, _mass_bias, _lu
    sj, npz = path.split(",")
    sc = {k: np.asarray(v, float) for k, v in json.load(open(sj))["scale"].items()}
    m = TwoMassNB("data/export/c8_params.json", "data/export/elastic20.json", substeps=2, frozen=1, scale=sc, residual=npz)
    def f(y, tau, pl):
        m.payload = pl; cs = m.consts()
        M, gb = _mass_bias(y[:6], y[6:12], *cs[:6]); LU, piv = _lu(M)
        return _f(y, tau, LU, piv, gb, *cs[6:], m.R, m._U())
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True); ap.add_argument("--pure", nargs="*", default=[]); ap.add_argument("--hybrid", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kin = Kinematics("data/export/c8_params.json")
    models = {f"③ {os.path.basename(os.path.dirname(p))}": model_f("pure", p) for p in a.pure} | {f"④ {os.path.basename(os.path.dirname(h.split(',')[1]))}": model_f("hyb", h) for h in a.hybrid}
    res = {k: {"ddq": [], "dth": [], "tip_mm": []} for k in models}
    for pth in a.data:
        with h5py.File(pth, "r", locking=False) as f:
            for k in f:
                if k == "controller": continue
                g = f[k]; pl = float(np.ravel(g.attrs.get("payload", 0.0))[0])
                ref = g["ref"][:].T; dqr = ref[::8, 7:13]
                S = {n: g[f"truth_{n}"][:].T for n in ("q", "dq", "qm", "dqm", "tau")}
                for (_, e) in dwells(dqr):
                    i8 = 8 * e
                    y = np.concatenate([S["q"][i8], S["dq"][i8], S["qm"][i8], S["dqm"][i8]]); tau = S["tau"][i8]
                    for nm, fm in models.items():
                        a0 = fm(y, tau, pl)[6:12]
                        A = np.empty((6, 6)); h = 1e-7
                        for j in range(6):
                            d = np.zeros(24); d[12 + j] = h
                            A[:, j] = (fm(y + d, tau, pl)[6:12] - fm(y - d, tau, pl)[6:12]) / (2 * h)
                        dth = -np.linalg.solve(A, a0)
                        tip = np.linalg.norm(kin.tip_pos(S["q"][i8] - dth) - kin.tip_pos(S["q"][i8])) * 1e3
                        res[nm]["ddq"].append(a0.tolist()); res[nm]["dth"].append(dth.tolist()); res[nm]["tip_mm"].append(float(tip))
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), ensure_ascii=False)
    for nm, r in res.items():
        t = np.array(r["tip_mm"]); d = np.abs(np.array(r["dth"])) * 1e3; aa = np.abs(np.array(r["ddq"]))
        print(f"{nm}: 停留 {len(t)} 回、静的なつり合いの先端誤差 中央値 {np.median(t):.4f} mm 最大 {t.max():.4f} mm、"
              f"|δθ| 中央値 [mrad] {np.round(np.median(d, 0), 4)}、|ddq| 中央値 {np.round(np.median(aa, 0), 2)}")


if __name__ == "__main__":
    main()
