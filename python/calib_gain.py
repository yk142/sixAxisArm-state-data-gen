"""トルクのゲイン誤差の校正（issue #113）。使い方:
  ~/.venvs/sa-train/bin/python python/calib_gain.py --data data/export/id_elum_{h1s,h2s,p1,p2,pl4a,pl4b}.h5 --meas rate=1000,...,tgain=0.02 --out data/eval/id113/cal_g20.json
測定トルクが τ_meas = (1 + g_i)·τ のとき、同定値はすべて (1 + g_i) 倍になり、モデルは止まる位置を誤る（P02 に効く）。
質量が正確に分かっているペイロード（重りを付けた運転）があれば、その分のトルクの係数から g_i を推定できる:
  τs_i = τ_i − Jm_i·ddqm_i ≈ a_i·rig0_i + b_i·rigP_i + Bv_i·dq_i + Fc_i·tanh(dq_i/ε_i)
  rig0 = ペイロードなしの M·ddq + g + b、rigP = ペイロードの分（rig(ペイロード込み) − rig0）。b_i = 1 + g_i（ペイロードの質量は既知）。
Jm はモータ側の最小二乗（τ = Jm·ddqm + k·θ + c·dθ、全データ）で求める（ゲイン誤差の分だけずれるが、τs への影響は小さい）。
出力 {gain: [6]}（推定した 1 + g_i）。identify_el.py / fit_residual_el.py の --tcal に渡すと τ をこれで割る。"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identify_el import load, rigid, parse_meas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--meas", default=""); ap.add_argument("--cubic", action="store_true", help="モータ側の式に θ³ を足して Jm を求める"); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--fric-basis", type=lambda s: [float(x) for x in s.split(",")], default=None, help="摩擦の基底 exp(−(dq/(mε))²)·tanh(dq/ε) の m の並び（例 1,3,6）。真値の Stribeck は m = 2")
    a = ap.parse_args()
    D = load(a.data, a.stride, meas=parse_meas(a.meas)); N = len(D["q"])
    eps = np.ravel(json.load(open("data/export/c8_params.json"))["frictionEps"])[:6]
    th, dth = D["qm"] - D["q"], D["dqm"] - D["dq"]
    Jm = np.array([np.linalg.lstsq(np.stack([D["ddqm"][:, i], th[:, i], dth[:, i]] + ([th[:, i] ** 3] if a.cubic else []), 1), D["tau"][:, i], rcond=None)[0][0] for i in range(6)])
    ts = D["tau"] - Jm * D["ddqm"]
    pl = D["pl"].copy()
    rig = rigid(D)                                  # ペイロード込み
    D0 = dict(D); D0["pl"] = np.zeros_like(pl); rig0 = rigid(D0)
    rigP = rig - rig0
    gain, rep = np.zeros(6), {}
    for i in range(6):
        v = D["dq"][:, i]; th_ = np.tanh(v / eps[i])
        cols = [rig0[:, i], rigP[:, i], v, th_] + ([np.exp(-(v / (m_ * eps[i])) ** 2) * th_ for m_ in a.fric_basis] if a.fric_basis else [])
        A = np.stack(cols, 1)
        x, *_ = np.linalg.lstsq(A, ts[:, i], rcond=None)
        gain[i] = x[1]; rep[f"axis{i+1}"] = {"a": float(x[0]), "b": float(x[1]), "rigP_rms": float(np.sqrt((rigP[:, i] ** 2).mean()))}
    print("推定したゲイン 1 + g:", np.round(gain, 5)); print("ペイロード分のトルク RMS [N·m]:", np.round([rep[k]["rigP_rms"] for k in rep], 2))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"gain": gain.tolist(), "detail": rep, "rows": N, "meas": a.meas, "data": a.data}, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
