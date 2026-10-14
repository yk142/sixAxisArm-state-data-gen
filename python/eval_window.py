"""短い窓の開ループ評価（issue #37）。使い方:
  ~/.venvs/sa-train/bin/python python/eval_window.py --out data/eval/window3 [--L 50 200]
窓: test と benchmark（PTP）。区分: 窓内に接触（at_limit）が無ければ free、有れば contact。"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.data import Dataset
from sa_nsm.model import MODELS
from sa_nsm.physics import Arm
from sa_nsm.rollout import Predictor, make_windows, rollout, true_states
from sa_nsm.train_lib import make_stats, prepare


def load_model(root, name, train):
    cfg = json.load(open(f"{root}/{name}/config.json"))
    kw = {"width_c": cfg["width_c"]} if "width_c" in cfg else {}
    m = MODELS[cfg["kind"]](cfg["n_ax"], make_stats(cfg["kind"], train), cfg["width"], cfg["depth"], **kw)
    m.load_state_dict(torch.load(f"{root}/{name}/model.pt")); m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/export/sa_flat_full3.h5")
    ap.add_argument("--params", default="data/export/c8_params.json")
    ap.add_argument("--models", default="data/models/full3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, nargs="+", default=[50, 200])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ds = Dataset(a.h5); n = ds.n_ax
    arm = Arm.from_json(a.params, n)
    data = prepare(ds, arm, a.h5.replace(".h5", "_analytic.npz"))
    preds = {"persist": Predictor(arm, ds, "persist"), "analytic": Predictor(arm, ds, "analytic"), "hybrid": Predictor(arm, ds, "hybrid")}
    for k in "ABC":
        for s in range(3):
            preds[f"{k}_s{s}"] = Predictor(arm, ds, "model", load_model(a.models, f"{k}_s{s}", data["train"]))
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for L in a.L:
        for sp in ("test", "benchmark"):
            starts = make_windows(ds, sp, L, stride=L // 2)
            qt, vt, tau = true_states(ds, sp, starts, L)
            at = ds.at_limit(sp)
            contact = np.array([at[s:s + L].any() for s in starts])
            pat = ds.pattern(sp)[starts]
            print(f"L={L} {sp}: 窓 {len(starts)}（接触を含む {contact.sum()}）", flush=True)
            err = {k: {"q": [], "v": []} for k in preds}
            for b in range(0, len(starts), 1000):
                sl = slice(b, b + 1000)
                q0, v0 = torch.from_numpy(qt[sl, 0]), torch.from_numpy(vt[sl, 0])
                T = torch.from_numpy(tau[sl])
                for k, p in preds.items():
                    qs, vs = rollout(p, q0, v0, T)
                    err[k]["q"].append((qs.numpy() - qt[sl]))
                    err[k]["v"].append((vs.numpy() - vt[sl]))
            for k in err:
                for c in ("q", "v"):
                    err[k][c] = np.concatenate(err[k][c])           # [B, L+1, n]
            groups = {"all": np.ones(len(starts), bool), "free": ~contact, "contact": contact}
            if sp == "benchmark":
                groups = {"ptp": np.ones(len(starts), bool)}
            for gname, m in groups.items():
                if not m.any():
                    continue
                r = {}
                for k in preds:
                    eq, ev = err[k]["q"][m], err[k]["v"][m]
                    r[k] = {"q_end_rmse": np.sqrt((eq[:, -1] ** 2).mean(0)).tolist(),
                            "v_end_rmse": np.sqrt((ev[:, -1] ** 2).mean(0)).tolist(),
                            "q_path_rmse": np.sqrt((eq ** 2).mean((0, 1))).tolist(),
                            "v_path_rmse": np.sqrt((ev ** 2).mean((0, 1))).tolist()}
                res[f"L{L}/{sp}/{gname}"] = {"n": int(m.sum()), "models": r}
            if sp == "test":        # 軸別・horizon 別の誤差曲線（図用）
                for k in ("persist", "analytic", "hybrid", "A_s0", "B_s0", "C_s0"):
                    for gname in ("free", "contact"):
                        m = groups[gname]
                        if m.any():
                            np.save(f"{a.out}/curve_L{L}_{gname}_{k}_v.npy", np.sqrt((err[k]["v"][m] ** 2).mean(0)))
                            np.save(f"{a.out}/curve_L{L}_{gname}_{k}_q.npy", np.sqrt((err[k]["q"][m] ** 2).mean(0)))
    json.dump(res, open(f"{a.out}/window_metrics.json", "w"), indent=1)
    print("完了")


if __name__ == "__main__":
    main()
