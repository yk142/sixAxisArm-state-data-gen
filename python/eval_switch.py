"""切替方式の窓評価（issue #37）: リミットから d_switch 以内なら NN（A または B）、他は解析モデル。
  ~/.venvs/sa-train/bin/python python/eval_switch.py --out data/eval/switch3 [--L 50 200] [--d 0.5 1 2 5]
窓の作り方・区分・指標は eval_window.py と同じ。基準線 hybrid・解析・状態維持と、切替なしの B も同じ窓で再計算して並べる。"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_window import load_model
from sa_nsm.data import Dataset
from sa_nsm.physics import Arm
from sa_nsm.rollout import Predictor, make_windows, rollout, true_states
from sa_nsm.train_lib import prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/export/sa_flat_full3.h5")
    ap.add_argument("--params", default="data/export/c8_params.json")
    ap.add_argument("--models", default="data/models/full3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, nargs="+", default=[50, 200])
    ap.add_argument("--d", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ds = Dataset(a.h5); n = ds.n_ax
    arm = Arm.from_json(a.params, n)
    data = prepare(ds, arm, a.h5.replace(".h5", "_analytic.npz"))
    preds = {"persist": Predictor(arm, ds, "persist"), "analytic": Predictor(arm, ds, "analytic"), "hybrid": Predictor(arm, ds, "hybrid")}
    for k in "AB":
        for s in range(3):
            m = load_model(a.models, f"{k}_s{s}", data["train"])
            preds[f"{k}_s{s}"] = Predictor(arm, ds, "model", m)
            for d in a.d:
                preds[f"{k}sw{d:g}_s{s}"] = Predictor(arm, ds, "switch", m, d)
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for L in a.L:
        for sp in ("test", "benchmark"):
            starts = make_windows(ds, sp, L, stride=L // 2)
            qt, vt, tau = true_states(ds, sp, starts, L)
            at = ds.at_limit(sp)
            contact = np.array([at[s:s + L].any() for s in starts])
            print(f"L={L} {sp}: 窓 {len(starts)}（接触を含む {contact.sum()}）", flush=True)
            err = {k: {"q": [], "v": []} for k in preds}
            for p in preds.values():
                p.n_model = p.n_all = 0
            for b in range(0, len(starts), 1000):
                sl = slice(b, b + 1000)
                q0, v0 = torch.from_numpy(qt[sl, 0]), torch.from_numpy(vt[sl, 0])
                T = torch.from_numpy(tau[sl])
                for k, p in preds.items():
                    qs, vs = rollout(p, q0, v0, T)
                    err[k]["q"].append(qs.numpy() - qt[sl]); err[k]["v"].append(vs.numpy() - vt[sl])
            for k in err:
                for c in ("q", "v"):
                    err[k][c] = np.concatenate(err[k][c])
            groups = {"free": ~contact, "contact": contact} if sp == "test" else {"ptp": np.ones(len(starts), bool)}
            for gname, m in groups.items():
                if not m.any():
                    continue
                r = {}
                for k in preds:
                    eq, ev = err[k]["q"][m], err[k]["v"][m]
                    r[k] = {"q_end_rmse": np.sqrt((eq[:, -1] ** 2).mean(0)).tolist(), "v_end_rmse": np.sqrt((ev[:, -1] ** 2).mean(0)).tolist(),
                            "q_path_rmse": np.sqrt((eq ** 2).mean((0, 1))).tolist(), "v_path_rmse": np.sqrt((ev ** 2).mean((0, 1))).tolist()}
                res[f"L{L}/{sp}/{gname}"] = {"n": int(m.sum()), "models": r}
            res[f"L{L}/{sp}/frac_model_steps"] = {k: p.n_model / max(1, p.n_all) for k, p in preds.items() if p.kind == "switch"}
    json.dump(res, open(f"{a.out}/switch_metrics.json", "w"), indent=1)
    print("完了")


if __name__ == "__main__":
    main()
