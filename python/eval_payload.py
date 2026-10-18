"""手先ペイロード × 制御ゲインの閉ループステップ応答評価（issue #61）。使い方:
  ~/.venvs/sa-train/bin/python python/eval_payload.py --out data/eval/payload6
データ: tools/sa_export_payload_eval.m の HDF5。各グループ = 1 軸の 0.5° ステップ（t = 0.6 s）。
プラント（Simscape の真値と比べる）:
  解析（ペイロード反映） : c8_params_pl<kg>.json の解析 RK4
  解析（公称）           : ペイロードなしの c8_params_pl0.json（ペイロードを知らない解析）
  A・B                   : ペイロードなしで学習した 6 軸モデル（重りは入力にない）
指標（ステップ応答）: オーバーシュート [%]、整定時間（2% 帯）[s]、定常偏差（t = 1.8 s）[°]、軌道の真値との差（RMS、最大）[°]。"""
import argparse, json, os, sys, time
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_window
from eval_a_window import AccelNet
from sa_nsm.closedloop import R, DT, Scenario, closed_loop
from sa_nsm.data import Dataset
from sa_nsm.physics_fast import FastArm
from sa_nsm.rollout import Predictor
from sa_nsm.train_lib import prepare

DEG = 180 / np.pi


def step_metrics(t, q, q_pre, step, t_step, band=0.02):
    """q: [K+1]（1 kHz）。ステップ前の値 q_pre からのステップ応答の指標。
    ペイロードの重力（フィードフォワードは公称）で定常偏差が出るので、オーバーシュートと整定は最終値（t = 末尾）を基準にする。
    final_gain = 最終値 / 目標（1 なら偏差なし）。"""
    y = (q - q_pre) / step
    m = t >= t_step
    ts, ys = t[m] - t_step, y[m]
    yf = ys[-1]
    over = max(0.0, (ys.max() - yf) / abs(yf)) * 100
    out = np.where(np.abs(ys - yf) > band * abs(yf))[0]
    settle = 0.0 if len(out) == 0 else (float(ts[out[-1] + 1]) if out[-1] + 1 < len(ts) else float("nan"))
    return {"overshoot_pct": float(over), "settle_s": settle, "final_gain": float(yf), "final_err_deg": float((yf - 1.0) * step * DEG)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/export/sa_flat_full6.h5")
    ap.add_argument("--data", default="data/export/payload_eval6.h5")
    ap.add_argument("--params-fmt", default="data/export/c8_params_pl{m}.json")
    ap.add_argument("--b-model", default="data/models/full6")
    ap.add_argument("--a-model", default="data/models/a_exp6/r10_wd0.2_e100_s0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-nn", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ds = Dataset(a.h5, splits=("train",)); n = ds.n_ax
    nominal = FastArm.from_json(a.params_fmt.format(m=0), n)
    data = prepare(ds, nominal, a.h5.replace(".h5", "_analytic.npz"), splits=("train",))
    nn = {} if a.no_nn else {"A（調整後）": AccelNet(ds, a.a_model),
                             "B": Predictor(nominal, ds, "model", eval_window.load_model(a.b_model, "B_s0", data["train"]))}
    os.makedirs(a.out, exist_ok=True)
    res, traj = [], {}
    with h5py.File(a.data, "r") as f:
        peak = f["controller/tauPeak"][:].ravel()
        groups = sorted({k.rsplit("_j", 1)[0] for k in f if k != "controller"})
        arms = {}
        for gk in groups:
            m = int(f[gk + "_j1"].attrs["payload"])
            arms.setdefault(m, FastArm.from_json(a.params_fmt.format(m=m), n))
            ctl_g = {"Kp": f[gk + "_j1/Kp"][:].ravel(), "Kd": f[gk + "_j1/Kd"][:].ravel(), "peak": peak}
            sc = [Scenario(f[f"{gk}_j{ax}"], ctl_g) for ax in range(1, n + 1)]
            at = f[gk + "_j1"].attrs
            wn, zeta, step, t_step = float(at["wn"]), float(at["zeta"]), float(at["step"]), float(at["tStep"])
            plants = {"解析（ペイロード反映）": Predictor(arms[m], ds, "analytic"), "解析（公称）": Predictor(nominal, ds, "analytic")}
            plants.update(nn)
            t = np.arange(sc[0].K + 1) * DT
            q_by = {"Simscape": np.stack([s.q for s in sc])}
            for pn, pred in plants.items():
                q_by[pn], _ = closed_loop(pred, sc, ctl_g)
            i_pre = int(round((t_step - 0.01) / DT))
            for ax in range(n):
                row = {"payload": m, "wn": wn, "zeta": zeta, "axis": ax + 1}
                for pn, q in q_by.items():
                    qa = q[ax][:, ax]
                    row[pn] = step_metrics(t, qa, qa[i_pre], step, t_step)
                    if pn != "Simscape":
                        e = (q[ax] - q_by["Simscape"][ax]) * DEG
                        dq_ = (q[ax][:, ax] - q[ax][i_pre, ax]) - (q_by["Simscape"][ax][:, ax] - q_by["Simscape"][ax][i_pre, ax])        # ステップ分（ステップ前の値を引く）の差
                        row[pn].update({"step_err_max_deg": float(np.abs(dq_[i_pre:]).max() * DEG), "err_rms_deg": np.sqrt((e ** 2).mean(0)).tolist(), "err_max_deg": np.abs(e).max(0).tolist(), "err_axis_max_deg": float(np.abs(e[:, ax]).max()),
                                        "ok": bool(np.isfinite(q[ax]).all())})
                res.append(row)
                if m in (0, 8) and wn == 60 and zeta == 1.0 and ax + 1 in (2, 5):
                    for pn, q in q_by.items(): traj[f"{gk}|j{ax + 1}|{pn}"] = q[ax].astype(np.float32)
            print(f"{gk}: done", flush=True)
    json.dump(res, open(f"{a.out}/payload.json", "w"), indent=1)
    np.savez_compressed(f"{a.out}/traj.npz", **traj)
    print("完了", a.out)


if __name__ == "__main__":
    main()
