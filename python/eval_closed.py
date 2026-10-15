"""閉ループ評価（issue #45）。使い方:
  ~/.venvs/sa-train/bin/python python/eval_closed.py --out data/eval/closed3
PD + RNEA FF の制御器（1 kHz 移植）で、各モデルをプラントにして閉ループを回し、Simscape の閉ループ（真値）と比べる。
対象: ベンチマーク PTP 12 本、リミット近傍の保持（学習に使っていない: 新規 12 本 ＋ test の既存 2 本）。"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_window
from eval_a_window import AccelNet
from sa_nsm.closedloop import closed_loop, load_scenarios
from sa_nsm.data import Dataset
from sa_nsm.physics_fast import FastArm
from sa_nsm.rollout import Predictor
from sa_nsm.train_lib import prepare

DEG = 180 / np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/export/sa_flat_full3.h5")
    ap.add_argument("--cl", default="data/export/closed_eval3.h5")
    ap.add_argument("--params", default="data/export/c8_params.json")
    ap.add_argument("--models", default="data/models/full3")
    ap.add_argument("--a-model", default="data/models/a_exp/r6_w512d4_e60_s0")
    ap.add_argument("--hold-name", default="nearlimit", help="保持シナリオ群の名前（接触評価では contact）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--d-switch", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ds = Dataset(a.h5, splits=("train",)); n = ds.n_ax
    arm = FastArm.from_json(a.params, n)
    if a.compile:
        comp = torch.compile(lambda q, v, t, dt, sub: FastArm.step(arm, q, v, t, dt, sub), dynamic=True)
        arm.step = comp
    data = prepare(ds, arm, a.h5.replace(".h5", "_analytic.npz"), splits=("train",))
    scen, ctl = load_scenarios(a.cl)
    groups = {"ptp": [s for s in scen if s.kind == "ptp"], a.hold_name: [s for s in scen if s.kind == "hold"]}
    groups = {k: v for k, v in groups.items() if v}
    models = {
        "解析": Predictor(arm, ds, "analytic"),
        "hybrid（リミットで状態維持）": Predictor(arm, ds, "hybrid"),
        "A（4-3d の r6）": AccelNet(ds, a.a_model),
        "B（4-2）": Predictor(arm, ds, "model", eval_window.load_model(a.models, "B_s0", data["train"])),
        f"B 切替（d={a.d_switch}°）": Predictor(arm, ds, "switch", eval_window.load_model(a.models, "B_s0", data["train"]), d_switch_deg=a.d_switch),
    }
    os.makedirs(a.out, exist_ok=True)
    res, traj = {}, {}
    for gname, sc in groups.items():
        print(f"== {gname}: {len(sc)} 本、最長 {max(s.K for s in sc)} ステップ", flush=True)
        for mname, pred in models.items():
            t0 = time.time()
            q, dq = closed_loop(pred, sc, ctl)
            for i, s in enumerate(sc):
                K = s.K
                eq, ev = q[i, :K + 1] - s.q, dq[i, :K + 1] - s.dq
                ok = bool(np.isfinite(q[i, :K + 1]).all() and np.abs(q[i, :K + 1]).max() < 10)
                qmin, qmax = ds_limits = (np.asarray(arm.q_min), np.asarray(arm.q_max))
                lim = lambda x: ((x <= qmin + np.radians(0.1)) | (x >= qmax - np.radians(0.1))).any(1).mean()
                res.setdefault(gname, {}).setdefault(s.name, {})[mname] = {
                    "ok": ok, "q_err_rms_deg": (np.sqrt((eq ** 2).mean(0)) * DEG).tolist(), "q_err_max_deg": (np.abs(eq).max(0) * DEG).tolist(),
                    "v_err_rms": np.sqrt((ev ** 2).mean(0)).tolist(), "q_err_end_deg": (np.abs(eq[-1]) * DEG).tolist(),
                    "track_rms_deg": (np.sqrt(((q[i, :K + 1] - s.q_ref) ** 2).mean(0)) * DEG).tolist(),
                    "limit_frac": float(lim(q[i, :K + 1])), "K": K}
                if i == 0 or gname != "ptp":
                    traj[f"{gname}/{s.name}/{mname}"] = q[i, :K + 1].astype(np.float32)
            print(f"  {mname}: {time.time() - t0:.0f} s", flush=True)
        for s in sc:
            res[gname][s.name]["truth"] = {"track_rms_deg": (np.sqrt(((s.q - s.q_ref) ** 2).mean(0)) * DEG).tolist(),
                                           "limit_frac": float((((s.q <= np.asarray(arm.q_min) + np.radians(0.1)) | (s.q >= np.asarray(arm.q_max) - np.radians(0.1))).any(1)).mean()), "K": s.K, "split": s.split}
            traj[f"{gname}/{s.name}/truth"] = s.q.astype(np.float32); traj[f"{gname}/{s.name}/ref"] = s.q_ref.astype(np.float32)
    json.dump(res, open(f"{a.out}/closed.json", "w"), indent=1)
    np.savez_compressed(f"{a.out}/traj.npz", **{k.replace("/", "|"): v for k, v in traj.items()})
    print("完了", a.out)


if __name__ == "__main__":
    main()
