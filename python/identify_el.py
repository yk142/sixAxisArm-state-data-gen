"""② 2 慣性解析のパラメータ同定（最小二乗、issue #99）。使い方:
  ~/.venvs/sa-train/bin/python python/identify_el.py --data data/export/id_elum_*.h5 --out data/eval/id99/ident.json
モデルの式（同定する側）:
  モータ側（軸ごと）: τ = Jm·ddqm + k·θ + c·dθ                     θ = qm − q
  リンク側        : τs = s·(M(q)·ddq + g(q) + b(q,dq)) + Bv·dq + Fc·tanh(dq/ε)   τs = τ − Jm·ddqm（モータ側の同定値）、s は質量倍率（全軸共通）
測定: q, dq, qm, dqm, τ（実印加トルク）を 8 kHz で得られる条件（最良の場合）。加速度は角速度の中心差分。
出力: 公称値に対する倍率（軸ごと）{k, c, Jm, Fc, Bv, mass}。TwoMassNB(scale=…) にそのまま渡せる。"""
import argparse, glob, json, os, sys
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.physics import Arm

DT8 = 1.0 / 8000


def load(paths, stride, frac=1.0, seed=0):
    """frac < 1: シナリオ単位で無作為に frac の割合だけ使う（データ量の軸、issue #101）。"""
    X = {k: [] for k in ("q", "dq", "qm", "dqm", "tau", "ddq", "ddqm")}
    keys = []
    for p in paths:
        with h5py.File(p, "r", locking=False) as f:
            keys += [(p, k) for k in f if k != "controller" and "truth_tau" in f[k]]
    if frac < 1.0:
        rs = np.random.RandomState(seed); keep = set(rs.permutation(len(keys))[:max(1, int(round(frac * len(keys))))])
        keys = [kk for i, kk in enumerate(keys) if i in keep]
    for p in paths:
        with h5py.File(p, "r", locking=False) as f:
            for k in f:
                if (p, k) not in keys: continue
                g = f[k]; a = lambda n: g[n][:].T
                q, dq, qm, dqm, tau = a("truth_q"), a("truth_dq"), a("truth_qm"), a("truth_dqm"), a("truth_tau")
                ddq, ddqm = np.gradient(dq, DT8, axis=0), np.gradient(dqm, DT8, axis=0)
                sl = slice(8, len(q) - 8, stride)                      # 端の差分を除く
                for n, v in zip(X, (q, dq, qm, dqm, tau, ddq, ddqm)): X[n].append(v[sl])
    return {k: np.concatenate(v) for k, v in X.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=4, help="8 kHz から間引く間隔")
    ap.add_argument("--frac", type=float, default=1.0, help="使うシナリオの割合（データ量の軸、issue #101）"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bv-nonneg", action="store_true", help="粘性摩擦 Bv を 0 以上に制約（負になった軸を 0 に固定して解き直す）")
    a = ap.parse_args()
    D = load(a.data, a.stride, frac=a.frac, seed=a.seed); N = len(D["q"]); print(f"行数 {N}")
    el = json.load(open("data/export/elastic20.json")); P = json.load(open("data/export/c8_params.json"))
    k0, c0, J0 = (np.ravel(el[x])[:6] for x in ("k", "c", "Jm"))
    Fc0, Bv0, eps = (np.ravel(P[x])[:6] for x in ("frictionFc", "frictionBv", "frictionEps"))
    th, dth = D["qm"] - D["q"], D["dqm"] - D["dq"]
    res = {"k": [], "c": [], "Jm": []}; fit = {}
    Jm = np.zeros(6)
    for i in range(6):                                                    # モータ側
        A = np.stack([D["ddqm"][:, i], th[:, i], dth[:, i]], 1)
        x, *_ = np.linalg.lstsq(A, D["tau"][:, i], rcond=None)
        Jm[i] = x[0]; res["Jm"].append(x[0] / J0[i]); res["k"].append(x[1] / k0[i]); res["c"].append(x[2] / c0[i])
        r = D["tau"][:, i] - A @ x; fit[f"motor_{i+1}_rms_Nm"] = float(np.sqrt((r ** 2).mean()))
    ts = D["tau"] - Jm * D["ddqm"]                                         # ばね＋ダンパのトルク（リンク側に伝わる）
    arm = Arm.from_json("data/export/c8_params.json", 6, 0.0); arm.armature = torch.zeros_like(arm.armature)
    rig = np.zeros_like(ts)
    for s in range(0, N, 50000):
        q, dq, ddq = (torch.from_numpy(D[n][s:s + 50000]) for n in ("q", "dq", "ddq"))
        with torch.no_grad():
            rig[s:s + 50000] = arm.rnea(q, dq, ddq).numpy()               # M·ddq + g + b（armature なし）
    # リンク側: 全軸まとめて [s（共通）, Bv_i, Fc_i]
    cols = 1 + 12; A = np.zeros((N * 6, cols)); y = ts.reshape(-1, order="F")
    for i in range(6):
        sl = slice(i * N, (i + 1) * N)
        A[sl, 0] = rig[:, i]; A[sl, 1 + i] = D["dq"][:, i]; A[sl, 7 + i] = np.tanh(D["dq"][:, i] / eps[i])
    free = np.ones(cols, bool)
    for _ in range(7):
        x = np.zeros(cols); x[free], *_ = np.linalg.lstsq(A[:, free], y, rcond=None)
        neg = [1 + i for i in range(6) if free[1 + i] and x[1 + i] < 0]
        if not a.bv_nonneg or not neg: break
        free[neg] = False
    res["mass"] = float(x[0]); res["Bv"] = (x[1:7] / Bv0).tolist(); res["Fc"] = (x[7:13] / Fc0).tolist()
    r = (y - A @ x).reshape(N, 6, order="F"); fit.update({f"link_{i+1}_rms_Nm": float(np.sqrt((r[:, i] ** 2).mean())) for i in range(6)})
    res = {k: (np.asarray(v).tolist() if not isinstance(v, float) else v) for k, v in res.items()}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"scale": res, "fit": fit, "rows": N, "data": a.data}, open(a.out, "w"), indent=1, ensure_ascii=False)
    for k, v in res.items(): print(k, np.round(v, 4))
    print("残差 RMS", {k: round(v, 3) for k, v in fit.items()})


if __name__ == "__main__":
    main()
