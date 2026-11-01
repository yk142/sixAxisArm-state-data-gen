"""② 2 慣性解析のパラメータ同定（最小二乗、issue #99）。使い方:
  ~/.venvs/sa-train/bin/python python/identify_el.py --data data/export/id_elum_*.h5 --out data/eval/id99/ident.json
モデルの式（同定する側）:
  モータ側（軸ごと）: τ = Jm·ddqm + k·θ + c·dθ                     θ = qm − q
  リンク側        : τs = s·(M(q)·ddq + g(q) + b(q,dq)) + Bv·dq + Fc·tanh(dq/ε)   τs = τ − Jm·ddqm（モータ側の同定値）、s は質量倍率（全軸共通）
測定: q, dq, qm, dqm, τ（実印加トルク）を 8 kHz で得られる条件（最良の場合）。加速度は角速度の中心差分。--meas で測定の模擬（1 kHz・角度だけ・量子化・トルクのノイズとゲイン誤差、issue #113）。
出力: 公称値に対する倍率（軸ごと）{k, c, Jm, Fc, Bv, mass}。TwoMassNB(scale=…) にそのまま渡せる。"""
import argparse, glob, json, os, sys
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.physics import Arm

DT8 = 1.0 / 8000


MEAS_DEFAULT = {"rate": 1000.0, "quant": 0.0, "tnoise": 0.0, "tgain": 0.0, "tgainr": 0.0, "smooth": 0.0, "seed": 0.0}


def parse_meas(s):
    """'rate=1000,quant=4.8e-7,tnoise=0.005,tgain=0.01,smooth=2' → dict（issue #113）。空文字・None は理想の測定（8 kHz、全状態）。"""
    if not s:
        return None
    m = dict(MEAS_DEFAULT)
    for kv in s.split(","):
        k, v = kv.split("="); assert k in m, k; m[k] = float(v)
    return m


def _gsmooth(x, sig):
    """零位相のガウス平滑（列ごと、端は端の値で延長）。sig: 標準偏差 [サンプル]。0 なら何もしない。"""
    if sig <= 0:
        return x
    r = int(np.ceil(4 * sig)); t = np.arange(-r, r + 1); w = np.exp(-0.5 * (t / sig) ** 2); w /= w.sum()
    xp = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    return np.stack([np.convolve(xp[:, j], w, mode="valid") for j in range(x.shape[1])], 1)


def _gain_dev(m):
    """軸ごとのゲイン誤差の向きと大きさ（−1〜1 の一様乱数、seed で固定。tgainr を掛ける）。"""
    return np.random.RandomState(1000 + int(m["seed"])).uniform(-1, 1, 6)


def measure(q, qm, tau, m, rated, rng):
    """測定の模擬（issue #113）。8 kHz の真値 → 1 kHz（rate）の q, qm, τ の測定値と、角度の差分から作る速度・加速度。
    量子化 quant [rad]、トルクのノイズ tnoise（定格トルクに対する比の標準偏差）、ゲイン誤差 tgain（全軸に 1 + tgain を掛ける）・tgainr（軸ごとに ±tgainr の一様乱数）。
    速度・加速度は角度に零位相のガウス平滑（smooth サンプル）をかけてから中心差分。τ にも同じ平滑をかける（式の両辺の帯域をそろえる）。
    返り値 (q, dq, qm, dqm, tau, ddq, ddqm, 端を除く添字の slice)。"""
    r = int(round(8000 / m["rate"])); dt = r / 8000.0
    q, qm, tau = q[::r].copy(), qm[::r].copy(), tau[::r].copy()
    if m["quant"] > 0:
        q = np.round(q / m["quant"]) * m["quant"]; qm = np.round(qm / m["quant"]) * m["quant"]
    tau = tau * (1.0 + m["tgain"] + m["tgainr"] * _gain_dev(m)) + (rng.standard_normal(tau.shape) * m["tnoise"] * rated if m["tnoise"] > 0 else 0.0)
    s = m["smooth"]
    qs, qms, tau = _gsmooth(q, s), _gsmooth(qm, s), _gsmooth(tau, s)
    dq, dqm = np.gradient(qs, dt, axis=0), np.gradient(qms, dt, axis=0)
    ddq, ddqm = np.gradient(dq, dt, axis=0), np.gradient(dqm, dt, axis=0)
    e = int(np.ceil(4 * s)) + 2
    return qs, dq, qms, dqm, tau, ddq, ddqm, slice(e, len(q) - e)


def tcal(D, path):
    """トルクのゲイン校正（issue #113、calib_gain.py の出力）: τ を推定した 1 + g で割る。校正できない軸（ペイロードのトルクが 0、gain ≤ 0）は 1。"""
    if path:
        g = np.asarray(json.load(open(path))["gain"], float); g = np.where(g > 0, g, 1.0)
        D["tau"] = D["tau"] / g
    return D


def load(paths, stride, frac=1.0, seed=0, meas=None):
    """frac < 1: シナリオ単位で無作為に frac の割合だけ使う（データ量の軸、issue #101）。
    meas: parse_meas の dict。与えると測定の模擬（issue #113）を通す（stride は無視し、測定の全サンプルを使う）。"""
    rated = np.ravel(json.load(open("data/export/c8_params.json"))["tauRated"])[:6]
    rng = np.random.RandomState(int(meas["seed"]) if meas else 0)
    X = {k: [] for k in ("q", "dq", "qm", "dqm", "tau", "ddq", "ddqm", "pl")}
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
                if meas:
                    q, dq, qm, dqm, tau, ddq, ddqm, sl = measure(a("truth_q"), a("truth_qm"), a("truth_tau"), meas, rated, rng)
                else:
                    q, dq, qm, dqm, tau = a("truth_q"), a("truth_dq"), a("truth_qm"), a("truth_dqm"), a("truth_tau")
                    ddq, ddqm = np.gradient(dq, DT8, axis=0), np.gradient(dqm, DT8, axis=0)
                    sl = slice(8, len(q) - 8, stride)                      # 端の差分を除く
                pl = float(np.ravel(g.attrs.get("payload", 0.0))[0])        # 手先ペイロード [kg]（issue #109）
                for n, v in zip(X, (q, dq, qm, dqm, tau, ddq, ddqm, np.full((len(q), 1), pl))): X[n].append(v[sl])
    D = {k: np.concatenate(v) for k, v in X.items()}
    D["pl"] = D["pl"][:, 0]
    return D


def rigid(D, mass=1.0):
    """M(q)·ddq + g + b（armature なし）。行ごとのペイロード D["pl"] に合わせる（issue #109）。
    mass: 質量・慣性の倍率（ペイロード込みの全リンクに掛ける。同定の s·rig、TwoMassNB の scale.mass と同じ扱い）。"""
    rig = np.zeros_like(D["q"])
    for m in np.unique(D["pl"]):
        arm = Arm.from_json("data/export/c8_params.json", 6, float(m)); arm.armature = torch.zeros_like(arm.armature)
        if mass != 1.0:
            arm.mass, arm.Ic = arm.mass * mass, arm.Ic * mass
        idx = np.where(D["pl"] == m)[0]
        for s in range(0, len(idx), 50000):
            j = idx[s:s + 50000]
            q, dq, ddq = (torch.from_numpy(D[n][j]) for n in ("q", "dq", "ddq"))
            with torch.no_grad():
                rig[j] = arm.rnea(q, dq, ddq).numpy()
    return rig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=4, help="8 kHz から間引く間隔")
    ap.add_argument("--frac", type=float, default=1.0, help="使うシナリオの割合（データ量の軸、issue #101）"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tcal", default="", help="トルクのゲイン校正 JSON（calib_gain.py、issue #113）")
    ap.add_argument("--meas", default="", help="測定の模擬（issue #113）。例 rate=1000,quant=4.8e-7,tnoise=0.005,tgain=0.01,smooth=2")
    ap.add_argument("--bv-nonneg", action="store_true", help="粘性摩擦 Bv を 0 以上に制約（負になった軸を 0 に固定して解き直す）")
    ap.add_argument("--cubic", action="store_true", help="モータ側の式に θ³ を足す（硬化ばね、issue #109）。出力 scale.k3 [N·m/rad³]（絶対値）")
    a = ap.parse_args()
    D = tcal(load(a.data, a.stride, frac=a.frac, seed=a.seed, meas=parse_meas(a.meas)), a.tcal); N = len(D["q"]); print(f"行数 {N}")
    el = json.load(open("data/export/elastic20.json")); P = json.load(open("data/export/c8_params.json"))
    k0, c0, J0 = (np.ravel(el[x])[:6] for x in ("k", "c", "Jm"))
    Fc0, Bv0, eps = (np.ravel(P[x])[:6] for x in ("frictionFc", "frictionBv", "frictionEps"))
    th, dth = D["qm"] - D["q"], D["dqm"] - D["dq"]
    res = {"k": [], "c": [], "Jm": []} | ({"k3": []} if a.cubic else {}); fit = {}
    Jm = np.zeros(6)
    for i in range(6):                                                    # モータ側
        cols = [D["ddqm"][:, i], th[:, i], dth[:, i]] + ([th[:, i] ** 3] if a.cubic else [])
        A = np.stack(cols, 1)
        x, *_ = np.linalg.lstsq(A, D["tau"][:, i], rcond=None)
        Jm[i] = x[0]; res["Jm"].append(x[0] / J0[i]); res["k"].append(x[1] / k0[i]); res["c"].append(x[2] / c0[i])
        if a.cubic: res["k3"].append(x[3])
        r = D["tau"][:, i] - A @ x; fit[f"motor_{i+1}_rms_Nm"] = float(np.sqrt((r ** 2).mean()))
    ts = D["tau"] - Jm * D["ddqm"]                                         # ばね＋ダンパのトルク（リンク側に伝わる）
    rig = rigid(D)                                                         # M·ddq + g + b（armature なし、ペイロード込み）
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
