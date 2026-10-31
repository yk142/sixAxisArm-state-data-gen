"""PTP の位置決め後の先端誤差（P-bench、issue #93）。使い方:
  ~/.venvs/sa-train/bin/python python/eval_ptp_pos.py --data data/export/ptp_el.h5 --twomass-nb 8:0 8:1 --out data/eval/ptp93/nb
  ~/.venvs/sa-train/bin/python python/eval_ptp_pos.py --data data/export/ptp_el.h5 --ref data/export/ptp_el_tight.h5 --out data/eval/ptp93/tol   （真値どうしの比較）
位置決め = 参照の速度が全軸 0 の停留区間（各到達点で 0.2 s）。指標は停留の終わり（次の移動の直前）の先端位置（フランジから 0.1 m 先）の差 [mm]。
停留の最後 50 ms の差の最大も出す。モデルは真値の初期状態から閉ループ（PD はモータ側、iters=1）で全長を回す。"""
import argparse, json, os, sys, time
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.closedloop import DT, Scenario, closed_loop_el, load_controller
from sa_nsm.kinematics import Kinematics


def dwells(dq_ref, min_len=100):
    """参照速度が全軸 0 の区間 [(start, end)]（1 kHz の添字、end は区間の最後）。先頭（初期姿勢）の区間は除く。"""
    z = np.all(np.abs(dq_ref) < 1e-12, 1)
    out, i, K = [], 0, len(z)
    while i < K:
        if z[i]:
            j = i
            while j + 1 < K and z[j + 1]: j += 1
            if i > 0 and j - i + 1 >= min_len: out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="真値（比較の基準）。複数ファイル可")
    ap.add_argument("--ref", nargs="*", default=None, help="もう一つの真値（許容誤差違い）。複数ファイル可。同名のシナリオを比べる")
    ap.add_argument("--twomass-nb", nargs="*", default=[], help="'サブステップ:凍結' の組")
    ap.add_argument("--perturb", nargs="*", default=[], help="パラメータ倍率の組（例 fric=1.01 mass=0.97,k=1.1）。numba sub4・M 凍結で評価（issue #95）")
    ap.add_argument("--scale-json", nargs="*", default=[], help="identify_el.py の出力（倍率、軸ごと）。numba 2 慣性解析に与えて評価（issue #99）")
    ap.add_argument("--pt-spec", default="4:1", help="--perturb のサブステップ:凍結")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    kin = Kinematics("data/export/c8_params.json")
    plants = {}
    for spec in a.twomass_nb:
        from sa_nsm.twomass_nb import TwoMassNB
        sub, fz = (int(x) for x in spec.split(":"))
        plants[f"2慣性 numba sub{sub}" + ("（M 凍結）" if fz else "")] = TwoMassNB("data/export/c8_params.json", "data/export/elastic20.json", substeps=sub, frozen=fz)
    for pt in a.perturb:
        from sa_nsm.twomass_nb import TwoMassNB
        sub, fz = (int(x) for x in a.pt_spec.split(":"))
        sc_ = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in pt.split(",")}
        plants[f"2慣性 {pt}"] = TwoMassNB("data/export/c8_params.json", "data/export/elastic20.json", substeps=sub, frozen=fz, scale=sc_)
    for sj in a.scale_json:
        from sa_nsm.twomass_nb import TwoMassNB
        sub, fz = (int(x) for x in a.pt_spec.split(":"))
        sc_ = {k: np.asarray(v, float) for k, v in json.load(open(sj))["scale"].items()}
        plants[f"2慣性 同定（{os.path.basename(os.path.dirname(sj))}/{os.path.basename(sj)}）"] = TwoMassNB("data/export/c8_params.json", "data/export/elastic20.json", substeps=sub, frozen=fz, scale=sc_)
    res, tim = [], []
    def load(paths):
        out = {}
        for pth in paths:
            with h5py.File(pth, "r") as f:
                c = load_controller(f)
                out.update({k: Scenario(f[k], c) for k in f if k != "controller"})
        return out, c
    D, ctl = load(a.data)
    names = sorted(D); sc = [D[k] for k in names]
    q_by = {}
    if a.ref:
        Rf, _ = load(a.ref)
        q_by["真値（--ref）"] = [Rf[k].q for k in names]
    for pn, pl in plants.items():
        pl.payload = 0.0; t0 = time.time()
        q = closed_loop_el(pl, sc, ctl, iters=a.iters)[0]
        q_by[pn] = [q[i][:s.K + 1] for i, s in enumerate(sc)]
        tim.append({"plant": pn, "sec": time.time() - t0, "sim_s": float(sum(s.K for s in sc) * DT)})
        print(f"{pn}: {time.time() - t0:.0f} s", flush=True)
    for i, (nm, s) in enumerate(zip(names, sc)):
        pt = kin.tip_pos(s.q)
        for (b, e) in dwells(s.dq_ref):
            row = {"scenario": nm, "start": b, "end": e, "move_deg": float(np.abs(s.q_ref[b] - s.q_ref[max(0, b - 1)]).max())}
            for pn, qs in q_by.items():
                pm = kin.tip_pos(qs[i])
                d = np.linalg.norm(pm - pt, axis=1) * 1e3                      # mm
                row[pn] = {"end_mm": float(d[e]), "last50_max_mm": float(d[max(b, e - 50):e + 1].max()),
                           "joint_end_deg": np.rad2deg(np.abs(qs[i][e] - s.q[e])).tolist()}
            row["truth_resid_mm"] = float(np.linalg.norm(pt[e] - kin.tip_pos(s.q_ref[e])) * 1e3)    # 真値の、参照に対する残差（参考）
            res.append(row)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/ptp_pos.json", "w"), indent=1, ensure_ascii=False)
    json.dump(tim, open(f"{a.out}/timing.json", "w"), indent=1, ensure_ascii=False)
    for pn in q_by:
        e = np.array([r[pn]["end_mm"] for r in res]); l = np.array([r[pn]["last50_max_mm"] for r in res])
        print(f"{pn}: 位置決め {len(e)} 回、停留の終わりの先端誤差 中央値 {np.median(e):.4f} mm 最大 {e.max():.4f} mm、最後 50 ms の最大 {l.max():.4f} mm、0.02 mm 以下 {np.mean(e <= 0.02):.0%}")
    print("完了", a.out)


if __name__ == "__main__":
    main()
