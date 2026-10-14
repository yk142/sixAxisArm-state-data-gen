"""NN A の改善版（train_a.py）の自由窓評価（issue #42）。使い方:
  ~/.venvs/sa-train/bin/python python/eval_a_window.py --out data/eval/a_window --models r2_acc_free_s0 r3_acc_free_w512d4_s0 r6_w512d4_e60_s0
窓: test の自由窓（窓内に at_limit なし）、L=50/200、開ループ。基準線: persist / 解析（サブステップ 1・2）/ 4-2 の A。"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.data import Dataset
from sa_nsm.physics_fast import FastArm
from sa_nsm.rollout import Predictor, make_windows, rollout, true_states
from sa_nsm.train_lib import make_stats, prepare
from train_a import Net, DT


class Analytic:
    def __init__(self, arm, sub): self.arm, self.sub = arm, sub
    def step(self, q, dq, tau):
        a, b = self.arm.step(q, dq, tau, DT, self.sub)
        return q + a, dq + b


class AccelNet:
    """train_a.py の accel 表現のモデル: Δdq = tv·dt、Δq = dq·dt + tq·dt²。"""
    def __init__(self, ds, path):
        cfg = json.load(open(f"{path}/config.json")); n = cfg["n_ax"]
        d = ds.part["train"]; tr = np.where(~ds.at_limit("train"))[0]
        z = {k: np.zeros(1) for k in ("t_mean", "t_std")}
        act = {"silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh}[cfg["act"]]
        self.m = Net(n, d["X_spec"][tr].mean(0), d["X_spec"][tr].std(0), np.zeros(2 * n), np.ones(2 * n), cfg["width"], cfg["depth"], act, cfg["drop"])
        self.m.load_state_dict(torch.load(f"{path}/model.pt")); self.m.eval(); self.n, self.sc = n, ds.scale
    @torch.no_grad()
    def step(self, q, dq, tau):
        xs = torch.cat([torch.sin(q), torch.cos(q), dq / torch.as_tensor(self.sc["DQ_MAX"]), tau / torch.as_tensor(self.sc["TAU_MAX"])], -1).float()
        p = self.m.raw(xs).double(); n = self.n
        return q + dq * DT + p[:, :n] * DT ** 2, dq + p[:, n:] * DT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/export/sa_flat_full3.h5")
    ap.add_argument("--params", default="data/export/c8_params.json")
    ap.add_argument("--exp", default="data/models/a_exp")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, nargs="+", default=[50, 200])
    a = ap.parse_args()
    torch.set_num_threads(4)
    ds = Dataset(a.h5); n = ds.n_ax
    arm = FastArm.from_json(a.params, n)
    preds = {"persist": Predictor(arm, ds, "persist"), "解析 サブステップ1": Analytic(arm, 1), "解析 サブステップ2": Analytic(arm, 2)}
    # 4-2 の A（標準化 Δ、全データ学習）
    import eval_window
    data = prepare(ds, arm, a.h5.replace(".h5", "_analytic.npz"), splits=("train",))
    preds["A 4-2"] = Predictor(arm, ds, "model", eval_window.load_model("data/models/full3", "A_s0", data["train"]))
    for m in a.models:
        preds[m] = AccelNet(ds, f"{a.exp}/{m}")
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for L in a.L:
        sp = "test"
        starts = make_windows(ds, sp, L, stride=L // 2)
        qt, vt, tau = true_states(ds, sp, starts, L)
        at = ds.at_limit(sp)
        free = ~np.array([at[s:s + L].any() for s in starts])
        qt, vt, tau = qt[free], vt[free], tau[free]
        print(f"L={L}: 自由窓 {free.sum()}", flush=True)
        err = {k: {"q": [], "v": []} for k in preds}
        for b in range(0, len(qt), 500):
            sl = slice(b, b + 500)
            q0, v0, T = torch.from_numpy(qt[sl, 0]), torch.from_numpy(vt[sl, 0]), torch.from_numpy(tau[sl])
            for k, p in preds.items():
                qs, vs = rollout(p, q0, v0, T)
                err[k]["q"].append(qs.numpy() - qt[sl]); err[k]["v"].append(vs.numpy() - vt[sl])
        res[f"L{L}"] = {"n": int(free.sum()), "models": {}}
        for k in preds:
            eq, ev = np.concatenate(err[k]["q"]), np.concatenate(err[k]["v"])
            res[f"L{L}"]["models"][k] = {"q_end_rmse": np.sqrt((eq[:, -1] ** 2).mean(0)).tolist(), "v_end_rmse": np.sqrt((ev[:, -1] ** 2).mean(0)).tolist(),
                                         "v_curve": np.sqrt((ev ** 2).mean((0, 2))).tolist()}
            print(f"  {k:28s} 終端 速度 {np.round(res[f'L{L}']['models'][k]['v_end_rmse'], 5)} 位置 {np.round(res[f'L{L}']['models'][k]['q_end_rmse'], 6)}", flush=True)
    json.dump(res, open(f"{a.out}/window_free.json", "w"), indent=1)


if __name__ == "__main__":
    main()
