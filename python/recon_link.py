"""リンク側の角度なしでの同定とリンク側の状態の再構成（issue #115）。使い方:
  ~/.venvs/sa-train/bin/python python/recon_link.py check --data data/export/id_elum_h1s.h5 --meas rate=1000,quant=4.8e-7,tnoise=0.005
  ~/.venvs/sa-train/bin/python python/recon_link.py fit --data data/export/id_elum_{h1s,h2s,p1,p2}.h5 --meas ... --out data/eval/id115/oe.json
  ~/.venvs/sa-train/bin/python python/recon_link.py recon --data data/export/id_elum_*.h5 --meas ... --params data/eval/id115/oe.json --out data/eval/id115/qhat_oe.npz
測定は qm と τ だけ（identify_el.measure、1 kHz）。q は使わない（check と診断の比較だけに使う）。
出力誤差法: 測った qm を入力にリンク側を模擬して θ̂ = qm − q̂ を作り（sa_nsm.linkobs_nb）、モータ側の式で予測したトルク
  τ̂ = Jm·ddqm + k·θ̂ + k3·θ̂³ + c·dθ̂
と測ったトルクの差（定格トルクで割る）を最小にする。Jm は線形なので毎回最小二乗で消す。残りのパラメータ（k, k3, c, 質量倍率 s, Bv, Fc,
摩擦の基底の係数）は Levenberg–Marquardt（差分のヤコビアン、プロセス 4 つで並列）。止まっているだけでは k が決まらない（θ = g/k で辻褄が合う）ので、
k は動的な応答（ばねの共振）から決まる。"""
import argparse, json, os, sys, time
import h5py
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identify_el import measure, parse_meas
from sa_nsm.linkobs_nb import simulate
from sa_nsm.physics import Arm

P = json.load(open("data/export/c8_params.json")); EL = json.load(open("data/export/elastic20.json"))
f6 = lambda x: np.asarray(x, float).reshape(-1)[:6].copy()
K0, C0, J0 = f6(EL["k"]), f6(EL["c"]), f6(EL["Jm"])
FC0, BV0, EPS, RATED = f6(P["frictionFc"]), f6(P["frictionBv"]), f6(P["frictionEps"]), f6(P["tauRated"])
TH_R = 0.005                                 # k3 の尺度（θr、真値の硬化ばねの基準と同じ桁。k3 = k0·x/θr²）
SKIP = 0.5                                   # 各シナリオの始めの 0.5 s は損失に入れない（真値の初期状態 θ = 0 からの過渡は実機には無い）


def load_scen(paths, meas):
    rng = np.random.RandomState(int(meas["seed"]))
    S = []
    for p in paths:
        with h5py.File(p, "r", locking=False) as f:
            for key in f:
                if key == "controller" or "truth_tau" not in f[key]: continue
                g = f[key]; a = lambda nm: g[nm][:].T
                qt = a("truth_q")[::int(round(8000 / meas["rate"]))]          # 比較・診断だけに使う（同定には使わない）
                _, _, qm, dqm, tau, _, ddqm, sl = measure(a("truth_q"), a("truth_qm"), a("truth_tau"), meas, RATED, rng)
                pl = float(np.ravel(g.attrs.get("payload", 0.0))[0])
                w = np.zeros(len(qm), bool); w[sl] = True; w[:int(SKIP * meas["rate"])] = False
                S.append(dict(name=f"{os.path.basename(p)}:{key}", qm=qm, dqm=dqm, ddqm=ddqm, tau=tau, q_true=qt, pl=pl, w=w, dt=1.0 / meas["rate"]))
    return S


_ARM = {}


def arm_consts(pl, s):
    if pl not in _ARM:
        a = Arm.from_json("data/export/c8_params.json", 6, pl); n = lambda t: t.detach().numpy().astype(float).copy()
        _ARM[pl] = (n(a.R0), n(a.pj), n(a.mass), n(a.com), n(a.Ic), n(a.r_base.T @ (-a.gravity)))
    R0, pj, m, com, Ic, g0 = _ARM[pl]
    return R0, pj, m * s, com, Ic * s, g0


class Model:
    """パラメータの並び x ↔ 物理量。x = [log k/k0 (6), k3·θr²/k0 (6), log c/c0 (6), log s (1), Bv/Bv0 (6), Fc/Fc0 (6), 基底 a/Fc0 (6·nb)]。"""
    def __init__(self, mb):
        self.mb = np.asarray(mb, float); self.nb = len(self.mb); self.n = 31 + 6 * self.nb

    def unpack(self, x):
        k = K0 * np.exp(x[0:6]); k3 = K0 * x[6:12] / TH_R ** 2; c = C0 * np.exp(x[12:18]); s = float(np.exp(x[18]))
        Bv = BV0 * x[19:25]; Fc = FC0 * x[25:31]; A = (FC0[:, None] * x[31:].reshape(6, self.nb)) if self.nb else np.zeros((6, 0))
        return k, k3, c, s, Bv, Fc, A

    def x0(self, k=1.0, c=1.0, s=1.0, fric=1.0):
        x = np.zeros(self.n); x[0:6] = np.log(k); x[12:18] = np.log(c); x[18] = np.log(s); x[19:25] = fric; x[25:31] = fric
        return x

    def to_json(self, x):
        k, k3, c, s, Bv, Fc, A = self.unpack(x)
        return {"k": (k / K0).tolist(), "k3": k3.tolist(), "c": (c / C0).tolist(), "mass": s, "Bv": (Bv / BV0).tolist(), "Fc": (Fc / FC0).tolist(),
                "fric_basis_m": self.mb.tolist(), "fric_basis_a": A.tolist()}


def sim_scen(sc, model, x, nsub=2):
    k, k3, c, s, Bv, Fc, A = model.unpack(x)
    q0 = sc["qm"][0].copy(); dq0 = np.zeros(6)                         # 真値の初期状態の決め方（θ = 0、静止）。実機なら静止時のつり合いを解く
    Q, DQ = simulate(sc["qm"], sc["dqm"], q0, dq0, nsub, *arm_consts(sc["pl"], s), k, k3, c, Bv, Fc, EPS, A, model.mb, sc["dt"])
    return Q, DQ


def spring(sc, model, x, Q, DQ):
    k, k3, c, *_ = model.unpack(x)
    th = sc["qm"] - Q; return k * th + k3 * th ** 3 + c * (sc["dqm"] - DQ)


# ---- 並列評価（プロセスごとにデータを持つ） ----
_G = {}


def _init(S, mb, nsub):
    _G["S"], _G["model"], _G["nsub"] = S, Model(mb), nsub


def _springs(x):
    """全シナリオの [損失に入れる行のばねのトルク]（連結）。"""
    out = []
    for sc in _G["S"]:
        Q, DQ = sim_scen(sc, _G["model"], x, _G["nsub"])
        out.append(spring(sc, _G["model"], x, Q, DQ)[sc["w"]])
    return np.concatenate(out)


def residual(TS, tau, ddqm):
    """Jm を軸ごとの最小二乗で消したあとの残差（定格で割る）と Jm。"""
    r = tau - TS; Jm = (ddqm * r).sum(0) / (ddqm ** 2).sum(0)
    return (r - Jm * ddqm) / RATED, Jm


def fit(a):
    meas = parse_meas(a.meas); S = load_scen(a.data, meas); mb = [float(v) for v in a.fric_basis.split(",")] if a.fric_basis else []
    model = Model(mb)
    tau = np.concatenate([sc["tau"][sc["w"]] for sc in S]); ddqm = np.concatenate([sc["ddqm"][sc["w"]] for sc in S])
    print(f"シナリオ {len(S)}、行数 {len(tau)}、パラメータ {model.n}", flush=True)
    x = model.x0(k=a.init_k, c=a.init_c, s=a.init_s, fric=a.init_fric)
    free = np.ones(model.n, bool)
    if not a.cubic: free[6:12] = False
    log = []
    with Pool(a.procs, initializer=_init, initargs=(S, mb, a.nsub)) as pool:
        r, Jm = residual(pool.map(_springs, [x])[0], tau, ddqm); cost = float((r ** 2).sum()); lam = 1e-2
        print(f"初期 損失 {cost:.6g}", flush=True)
        for it in range(a.iters):
            t0 = time.time(); idx = np.where(free)[0]; hs = np.full(model.n, a.fd)
            xs = [x + np.eye(model.n)[i] * hs[i] for i in idx]
            TSs = pool.map(_springs, xs)
            Jc = np.stack([(residual(T, tau, ddqm)[0] - r).ravel() / hs[i] for T, i in zip(TSs, idx)], 1)
            H = Jc.T @ Jc; g = Jc.T @ r.ravel(); D = np.diag(np.diag(H))
            acc = False
            for _ in range(8):
                dx = np.zeros(model.n); dx[idx] = -np.linalg.solve(H + lam * D, g)
                xn = x + dx; rn, Jmn = residual(pool.map(_springs, [xn])[0], tau, ddqm); cn = float((rn ** 2).sum())
                if cn < cost:
                    x, r, Jm, lam, acc = xn, rn, Jmn, max(lam / 3, 1e-7), True; break
                lam *= 4
            rel = (log[-1]["cost"] - cn) / log[-1]["cost"] if log and acc else None
            log.append({"it": it, "cost": cost if not acc else cn, "lam": lam, "sec": time.time() - t0})
            cost = log[-1]["cost"]; js = model.to_json(x)
            print(f"反復 {it}: 損失 {cost:.6g}、λ {lam:.1e}、{time.time() - t0:.0f} s | k {np.round(js['k'], 4)} s {js['mass']:.4f} c {np.round(js['c'], 3)}", flush=True)
            if not acc or (rel is not None and rel < a.tol): break
    out = {"scale": model.to_json(x) | {"Jm": (Jm / J0).tolist()}, "x": x.tolist(), "cost": cost, "log": log, "args": vars(a),
           "rms_Nm": np.sqrt(((r * RATED) ** 2).mean(0)).tolist()}
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("完了", a.out)


def check(a):
    """真値の式（oracle と同じ: k3 = k·0.3/θr²、Stribeck の基底 m = 2・係数 0.3·Fc）でリンク側を模擬し、q̂ と真の q を比べる。"""
    meas = parse_meas(a.meas); S = load_scen(a.data, meas); model = Model([2.0])
    x = model.x0(); x[6:12] = 0.3; x[31:] = 0.3
    for sc in S[:a.n]:
        t0 = time.time(); Q, DQ = sim_scen(sc, model, x, a.nsub); dt = time.time() - t0
        e = Q - sc["q_true"]; th = sc["qm"] - sc["q_true"]
        print(f"{sc['name']}: {len(Q)} ms を {dt:.2f} s、θ の誤差 RMS {np.sqrt((e[sc['w']] ** 2).mean(0))}（θ の RMS {np.sqrt((th[sc['w']] ** 2).mean(0))}）", flush=True)


def recon(a):
    """同定したパラメータ（fit の出力。--params truth なら真値の式）でリンク側を模擬し、全シナリオの q̂, dq̂ を npz に保存する（identify_el.load の qhat）。"""
    meas = parse_meas(a.meas); S = load_scen(a.data, meas)
    if a.params == "truth":
        model = Model([2.0]); x = model.x0(); x[6:12] = 0.3; x[31:] = 0.3
    else:
        js = json.load(open(a.params)); model = Model(js["scale"]["fric_basis_m"]); x = np.asarray(js["x"])
    out, rep = {}, {}
    for sc in S:
        Q, DQ = sim_scen(sc, model, x, a.nsub)
        out[sc["name"] + ":q"], out[sc["name"] + ":dq"] = Q, DQ
        e = (Q - sc["q_true"])[sc["w"]]; th = (sc["qm"] - sc["q_true"])[sc["w"]]
        rep[sc["name"]] = {"err_rms": np.sqrt((e ** 2).mean(0)).tolist(), "th_rms": np.sqrt((th ** 2).mean(0)).tolist()}
    os.makedirs(os.path.dirname(a.out), exist_ok=True); np.savez(a.out, **out)
    E = np.array([v["err_rms"] for v in rep.values()]); Tt = np.array([v["th_rms"] for v in rep.values()])
    print("θ̂ の誤差 RMS（シナリオの RMS の平均） [rad]:", np.round(np.sqrt((E ** 2).mean(0)), 7), "θ の RMS:", np.round(np.sqrt((Tt ** 2).mean(0)), 5))
    json.dump(rep, open(a.out.replace(".npz", ".json"), "w"), indent=1)


def refric(a):
    """摩擦だけを式の誤差で推定し直す（issue #115）。ばねのトルク τs = τ − Jm·ddqm は測定から決まり（q̂ によらない）、リンク側の式
    τs = s·(M(q̂)·ddq̂ + g + b) + 摩擦(dq̂) は q̂ の小さな誤差にほとんど影響されない。出力誤差法の摩擦の基底では形が足りず（Fc の約 2% のうねり）、
    それが動いている間の q̂ の誤差（軸 1 で 2.5e-5 rad）になっていた。基底を細かくして（--fric-basis）最小二乗で解き、q̂ を作り直す（recon）ことを繰り返す。
    ばね・c・質量・Jm は --params のまま。"""
    from identify_el import rigid
    meas = parse_meas(a.meas); S = load_scen(a.data, meas)
    js = json.load(open(a.params)); m0 = Model(js["scale"]["fric_basis_m"]); x0 = np.asarray(js["x"])
    k, k3, c, s, *_ = m0.unpack(x0); Jm = J0 * np.asarray(js["scale"]["Jm"])
    mb = [float(v) for v in a.fric_basis.split(",")]; m1 = Model(mb)
    x = m1.x0(); x[:19] = x0[:19]; x[19:31] = x0[19:31]
    if list(m1.mb) == list(m0.mb): x = x0.copy()                          # 同じ基底なら前回の続きから
    hist = []
    for it in range(a.iters):
        Qs = [sim_scen(sc, m1, x, a.nsub) for sc in S]
        q = np.concatenate([Q[sc["w"]] for (Q, _), sc in zip(Qs, S)]); dq = np.concatenate([DQ[sc["w"]] for (_, DQ), sc in zip(Qs, S)])
        ddq = np.concatenate([np.gradient(DQ, sc["dt"], axis=0)[sc["w"]] for (_, DQ), sc in zip(Qs, S)])
        pl = np.concatenate([np.full(sc["w"].sum(), sc["pl"]) for sc in S])
        ts = np.concatenate([(sc["tau"] - Jm * sc["ddqm"])[sc["w"]] for sc in S])
        r = ts - rigid({"q": q, "dq": dq, "ddq": ddq, "pl": pl}, s)
        err = np.sqrt(np.mean([((Q - sc["q_true"])[sc["w"]] ** 2).mean(0) for (Q, _), sc in zip(Qs, S)], 0))
        for i in range(6):
            v = dq[:, i]; t = np.tanh(v / EPS[i])
            X = np.stack([v, t] + [np.exp(-(v / (mm * EPS[i])) ** 2) * t for mm in mb], 1)
            b, *_ = np.linalg.lstsq(X, r[:, i], rcond=None)
            x[19 + i] = b[0] / BV0[i]; x[25 + i] = b[1] / FC0[i]; x[31 + i * len(mb): 31 + (i + 1) * len(mb)] = b[2:] / FC0[i]
        hist.append({"it": it, "qhat_err_rms": err.tolist()})
        print(f"反復 {it}: この摩擦の前の θ̂ の誤差 RMS {np.round(err, 7)}、Bv {np.round(x[19:25], 3)}、Fc {np.round(x[25:31], 3)}", flush=True)
    out = dict(js); out["x"] = x.tolist(); out["scale"] = m1.to_json(x) | {"Jm": js["scale"]["Jm"]}; out["refric"] = {"args": vars(a), "hist": hist}
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False); print("完了", a.out)


def scan(a):
    """見込みの確認: 真値の式（check と同じ）から、軸 i の k（と、それに合わせて k3）だけを倍率 f で変え、軸ごとの損失（Jm を消した残差の二乗和）を見る。"""
    meas = parse_meas(a.meas); S = load_scen(a.data, meas); mb = [2.0]
    x = Model(mb).x0(); x[6:12] = 0.3; x[31:] = 0.3
    tau = np.concatenate([sc["tau"][sc["w"]] for sc in S]); ddqm = np.concatenate([sc["ddqm"][sc["w"]] for sc in S])
    fs = [float(v) for v in a.factors.split(",")]; xs = []
    for i in a.axes:
        for f_ in fs:
            xx = x.copy(); xx[i - 1] += np.log(f_); xx[6 + i - 1] *= f_; xs.append(xx)
    with Pool(a.procs, initializer=_init, initargs=(S, mb, a.nsub)) as pool:
        TS = pool.map(_springs, xs)
    j = 0; out = {}
    for i in a.axes:
        row = []
        for f_ in fs:
            r, Jm = residual(TS[j], tau, ddqm); j += 1; row.append(float((r[:, i - 1] ** 2).sum()))
        out[f"axis{i}"] = row
        print(f"軸 {i}: " + "  ".join(f"{f_:.3f}: {v:.5g}" for f_, v in zip(fs, row)), flush=True)
    if a.out: json.dump({"factors": fs, "cost": out}, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan"); s.add_argument("--data", nargs="+", required=True); s.add_argument("--meas", default="rate=1000"); s.add_argument("--axes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    s.add_argument("--factors", default="0.97,0.98,0.99,0.995,1,1.005,1.01,1.02,1.03"); s.add_argument("--procs", type=int, default=4); s.add_argument("--out", default=""); s.add_argument("--nsub", type=int, default=4)
    c = sp.add_parser("check"); c.add_argument("--data", nargs="+", required=True); c.add_argument("--meas", default="rate=1000"); c.add_argument("--n", type=int, default=3); c.add_argument("--nsub", type=int, default=2)
    f = sp.add_parser("fit"); f.add_argument("--data", nargs="+", required=True); f.add_argument("--meas", default="rate=1000"); f.add_argument("--out", required=True)
    f.add_argument("--cubic", action="store_true"); f.add_argument("--fric-basis", default=""); f.add_argument("--iters", type=int, default=30); f.add_argument("--tol", type=float, default=1e-6)
    f.add_argument("--procs", type=int, default=4); f.add_argument("--nsub", type=int, default=16); f.add_argument("--fd", type=float, default=2e-3, help="差分のヤコビアンの刻み（軸 5 の損失の細かい凹凸を拾わない大きさ）")
    for nm in ("k", "c", "s", "fric"): f.add_argument(f"--init-{nm}", type=float, default=1.0, help="初期値（公称に対する倍率）")
    q = sp.add_parser("refric"); q.add_argument("--data", nargs="+", required=True); q.add_argument("--meas", default="rate=1000"); q.add_argument("--params", required=True)
    q.add_argument("--out", required=True); q.add_argument("--nsub", type=int, default=16); q.add_argument("--iters", type=int, default=3); q.add_argument("--fric-basis", default="0.5,1,1.5,2.5,3,4,6,10")
    r = sp.add_parser("recon"); r.add_argument("--data", nargs="+", required=True); r.add_argument("--meas", default="rate=1000"); r.add_argument("--params", required=True, help="fit の出力 JSON、または truth")
    r.add_argument("--out", required=True); r.add_argument("--nsub", type=int, default=4)
    a = ap.parse_args()
    {"check": check, "fit": fit, "scan": scan, "recon": recon, "refric": refric}[a.cmd](a)


if __name__ == "__main__":
    main()
