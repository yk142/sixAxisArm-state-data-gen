"""1 ステップ（1 ms）の計算時間: 解析モデル（RK4、サブステップ数違い）と NN（A/B/C）。issue #39。
  ~/.venvs/sa-train/bin/python python/bench_speed.py --out data/eval/speed3 [--threads 4]
全方式を同じ入力 (q, dq, tau) → 次状態 の関数にして、バッチサイズ別のスループットとレイテンシを測る。"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.data import Dataset
from sa_nsm.model import MODELS, limit_features
from sa_nsm.physics import Arm
from sa_nsm.train_lib import make_stats, prepare


def load_model(root, name, train):
    cfg = json.load(open(f"{root}/{name}/config.json"))
    kw = {"width_c": cfg["width_c"]} if "width_c" in cfg else {}
    m = MODELS[cfg["kind"]](cfg["n_ax"], make_stats(cfg["kind"], train), cfg["width"], cfg["depth"], **kw)
    m.load_state_dict(torch.load(f"{root}/{name}/model.pt")); m.eval()
    return m


def make_fns(arm, sc, models):
    DQ, TAU, D_Q, D_DQ = (torch.as_tensor(sc[k]) for k in ("DQ_MAX", "TAU_MAX", "D_Q_MAX", "D_DQ_MAX"))
    fns = {}
    for sub in (1, 2, 4):
        fns[f"解析 RK4 サブステップ{sub}"] = (lambda q, v, t, sub=sub: arm.step(q, v, t, 1e-3, sub))
    def nn(kind, m):
        def f(q, v, t):
            xs = torch.cat([torch.sin(q), torch.cos(q), v / DQ, t / TAU], -1).float()
            if kind == "A":
                ana = lim = None
            else:
                dq_, dv_ = arm.step(q, v, t, 1e-3, 2)
                ana = torch.cat([dq_ / D_Q, dv_ / D_DQ], -1).float(); lim = limit_features(q, arm.q_min, arm.q_max).float()
            return m(xs, ana, lim)
        return f
    for k, m in models.items():
        fns[f"NN {k}"] = nn(k, m)
    return fns


def timeit(fn, args, min_time=0.6, warm=3):
    for _ in range(warm):
        fn(*args)
    n, t0 = 0, time.perf_counter()
    while True:
        fn(*args); n += 1
        dt = time.perf_counter() - t0
        if dt >= min_time and n >= 5:
            return dt / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/export/sa_flat_full3.h5")
    ap.add_argument("--params", default="data/export/c8_params.json")
    ap.add_argument("--models", default="data/models/full3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 256, 4096, 65536])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ds = Dataset(a.h5, splits=("train", "test")); n = ds.n_ax
    arm = Arm.from_json(a.params, n)
    data = prepare(ds, arm, a.h5.replace(".h5", "_analytic.npz"), splits=("train", "test"))
    models = {k: load_model(a.models, f"{k}_s0", data["train"]) for k in "ABC"}
    fns = make_fns(arm, ds.scale, models)
    q, dq, tau = (torch.from_numpy(x) for x in ds.q_dq_tau("test"))
    out = {"threads": a.threads, "torch": torch.__version__, "cpu": os.cpu_count(), "batches": a.batches, "time_per_call_s": {}}
    with torch.no_grad():
        for name, fn in fns.items():
            r = []
            for B in a.batches:
                args = (q[:B].contiguous(), dq[:B].contiguous(), tau[:B].contiguous())
                r.append(timeit(fn, args))
                print(f"{name:20s} B={B:6d}: {r[-1] * 1e3:9.3f} ms/呼び出し  {B / r[-1]:12.0f} step/s", flush=True)
            out["time_per_call_s"][name] = r
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/bench_t{a.threads}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
