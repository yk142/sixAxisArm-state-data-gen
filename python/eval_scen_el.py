"""2 慣性解析モデル（と A_el）の、PTP・保持（閉ループ）と励振（開ループ再生）での評価（issue #79）。使い方:
  ~/.venvs/sa-train/bin/python python/eval_scen_el.py --data data/export/eval6_el_scen.h5 --out reports/issue-79 [--ael data/models/a_el/el_e30_s0]
閉ループ（type が ptp / hold）: closed_loop_el（iters 反復で区間の平均トルクを整合）で全長を進め、真値とのリンク角 q・ねじれ角 θ の誤差を軸別に。
励振（開ループ）: 真値の印加トルクを 1 ms の区間平均で再生し、真値の状態から K ステップ進めた誤差（複数の開始点）。"""
import argparse, json, os, sys
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.closedloop import Scenario, closed_loop_el, load_controller, R, DT
from sa_nsm.twomass import TwoMass
from eval_payload_el import ElPlant

DEG = 180 / np.pi


def closed(f, names, plants, iters):
    ctl = load_controller(f)
    out = {}
    for pn, pf in plants.items():
        sc = [Scenario(f[n], ctl) for n in names]
        # ペイロードはシナリオごとに異なりうるので、ペイロード別にまとめて回す
        pls = [int(round(float(f[n].attrs["payload"]))) for n in names]
        for m in sorted(set(pls)):
            ix = [i for i, p in enumerate(pls) if p == m]
            pf.payload = float(m)
            q, dq, qm, dqm = closed_loop_el(pf, [sc[i] for i in ix], ctl, iters=iters if hasattr(pf, "Jm") else 2)
            for j, i in enumerate(ix):
                s = sc[i]; K = s.K
                eq = (q[j][:K + 1] - s.q[:K + 1]) * DEG
                eth = ((qm[j] - q[j])[:K + 1] - (s.qm - s.q)[:K + 1]) * 1e3
                th_true = np.abs((s.qm - s.q)[:K + 1]) * 1e3
                t_idx = {f"{t}s": min(int(t * 1000), K) for t in (1, 2, 5, 10)}
                out.setdefault(names[i], {})[pn] = {
                    "q_err_max_deg": np.abs(eq).max(0).tolist(), "q_err_rms_deg": np.sqrt((eq ** 2).mean(0)).tolist(),
                    "theta_err_max_mrad": np.abs(eth).max(0).tolist(), "theta_true_max_mrad": th_true.max(0).tolist(),
                    "q_err_at_deg": {k: np.abs(eq[v]).max().item() for k, v in t_idx.items()}, "finite": bool(np.isfinite(q[j]).all())}
            print(pn, "payload", m, "done", flush=True)
    return out


def excite(f, names, plants, Ks, nstart):
    out = {}
    for pn, pf in plants.items():
        for n in names:
            g = f[n]; m = int(round(float(g.attrs["payload"]))); pf.payload = float(m)
            q8, dq8, qm8, dqm8, tau8 = (g[k][:].T for k in ("truth_q", "truth_dq", "truth_qm", "truth_dqm", "truth_tau"))
            K = (len(q8) - 1) // R; idx = R * np.arange(K + 1)
            pair = 0.5 * (tau8[:-1] + tau8[1:]); tau1 = pair[:R * K].reshape(K, R, -1).mean(1)
            q, dq, qm, dqm = q8[idx], dq8[idx], qm8[idx], dqm8[idx]
            Kmax = max(Ks)
            starts = [int(x) for x in np.linspace(0, max(0, K - Kmax), nstart)]
            if K < Kmax: Kmax = K
            T = lambda a: torch.from_numpy(np.stack([a[s] for s in starts]))
            s = (T(q), T(dq), T(qm), T(dqm))
            qs = [s[0].numpy()]; ths = [(s[2] - s[0]).numpy()]
            for k in range(Kmax):
                tau = torch.from_numpy(np.stack([tau1[st + k] for st in starts]))
                with torch.no_grad(): s = pf.step(*s, tau)
                qs.append(s[0].numpy()); ths.append((s[2] - s[0]).numpy())
            qs, ths = np.stack(qs, 1), np.stack(ths, 1)                                    # [W, Kmax+1, n]
            rec = {}
            for Kw in Ks:
                kw = min(Kw, Kmax)
                tq = np.stack([q[st:st + kw + 1] for st in starts]); tth = np.stack([(qm - q)[st:st + kw + 1] for st in starts])
                eq = np.abs(qs[:, :kw + 1] - tq) * DEG; eth = np.abs(ths[:, :kw + 1] - tth) * 1e3
                rec[str(Kw)] = {"q_err_max_deg": eq.max((0, 1)).tolist(), "theta_err_max_mrad": eth.max((0, 1)).tolist(), "theta_true_max_mrad": (np.abs(tth) * 1e3).max((0, 1)).tolist(), "finite": bool(np.isfinite(qs).all())}
            out.setdefault(n, {})[pn] = rec
            print(pn, n, "done", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/export/eval6_el_scen.h5")
    ap.add_argument("--out", default="reports/issue-79")
    ap.add_argument("--ael", default=None)
    ap.add_argument("--h5", default="data/export/sa_flat_fullplpu6el.h5")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", default="both", choices=["both", "closed", "excite"])
    ap.add_argument("--shard", default="0/1", help="シナリオの分割 i/k（閉ループ・励振とも）")
    ap.add_argument("--Ks", nargs="+", type=int, default=[100, 1000])
    ap.add_argument("--nstart", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    si, sk = (int(x) for x in a.shard.split("/"))
    plants = {"2慣性解析": TwoMass("data/export/c8_params.json", "data/export/elastic20.json", substeps=a.substeps)}
    if a.ael:
        with h5py.File(a.h5, "r") as f:
            dq_max, tau_max = f["scale/DQ_MAX"][:].ravel(), f["scale/TAU_MAX"][:].ravel()
        plants["A_el"] = ElPlant(a.ael, dq_max, tau_max)
    res = {}
    with h5py.File(a.data, "r") as f:
        names = [k for k in f if k != "controller"]
        typ = {n: f[n].attrs["type"] for n in names}
        typ = {n: (t.decode() if isinstance(t, bytes) else str(t)) for n, t in typ.items()}
        cn = [n for n in names if typ[n] in ("ptp", "hold")][si::sk]
        en = [n for n in names if typ[n] == "excitation"][si::sk]
        meta = {n: {"type": typ[n], "pattern": str(f[n].attrs["pattern"]), "contact": float(f[n].attrs["contact"])} for n in names}
        if a.only in ("both", "closed") and cn: res["closed"] = closed(f, cn, plants, a.iters)
        if a.only in ("both", "excite") and en: res["excite"] = excite(f, en, plants, a.Ks, a.nstart)
    res["meta"] = meta
    json.dump(res, open(f"{a.out}/scen_{si}of{sk}_{a.only}.json", "w"), indent=1, ensure_ascii=False)
    print("完了")


if __name__ == "__main__":
    main()
