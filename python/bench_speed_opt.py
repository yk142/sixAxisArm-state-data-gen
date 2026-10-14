"""最適化した解析モデルと NN の 1 ステップ（1 ms）の計算時間。issue #41（4-3b の bench_speed.py の続き）。
  ~/.venvs/sa-train/bin/python python/bench_speed_opt.py --out data/eval/speed3_opt [--threads 4]
解析: 旧（physics.Arm）/ 高速（FastArm、eager）/ 高速 + torch.compile。B・C は解析部に各実装を使った版を測る。"""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_speed import load_model, timeit as _timeit
from sa_nsm.data import Dataset
from sa_nsm.model import limit_features
from sa_nsm.physics import Arm
from sa_nsm.physics_fast import FastArm
from sa_nsm.train_lib import prepare


def timeit(fn, args, reps=5):
    """雑音対策: 短い測定を reps 回行い最小値（最良値）を取る。"""
    return min(_timeit(fn, args, min_time=0.3) for _ in range(reps))


def make_fns(arm_old, arm_fast, sc, models):
    DQ, TAU, D_Q, D_DQ = (torch.as_tensor(sc[k]) for k in ("DQ_MAX", "TAU_MAX", "D_Q_MAX", "D_DQ_MAX"))
    step = {}                                                    # 名前 → 解析の step 関数(q, v, t, sub)
    step["旧"] = lambda q, v, t, sub: arm_old.step(q, v, t, 1e-3, sub)
    step["高速"] = lambda q, v, t, sub: arm_fast.step(q, v, t, 1e-3, sub)
    comp = {sub: torch.compile(lambda q, v, t, sub=sub: arm_fast.step(q, v, t, 1e-3, sub), dynamic=True) for sub in (1, 2)}
    step["高速+compile"] = lambda q, v, t, sub: comp[sub](q, v, t)
    fns = {}
    for tag, st in step.items():
        for sub in (1, 2):
            fns[f"解析 {tag} サブステップ{sub}"] = (lambda q, v, t, st=st, sub=sub: st(q, v, t, sub))

    def nn(kind, m, st):
        def f(q, v, t):
            xs = torch.cat([torch.sin(q), torch.cos(q), v / DQ, t / TAU], -1).float()
            if kind == "A":
                ana = lim = None
            else:
                dq_, dv_ = st(q, v, t, 2)
                ana = torch.cat([dq_ / D_Q, dv_ / D_DQ], -1).float(); lim = limit_features(q, arm_old.q_min, arm_old.q_max).float()
            return m(xs, ana, lim)
        return f
    fns["NN A"] = nn("A", models["A"], None)
    for k in "BC":
        for tag in ("旧", "高速+compile"):
            fns[f"NN {k} (解析 {tag})"] = nn(k, models[k], step[tag])
    return fns


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
    arm_old, arm_fast = Arm.from_json(a.params, n), FastArm.from_json(a.params, n)
    data = prepare(ds, arm_old, a.h5.replace(".h5", "_analytic.npz"), splits=("train", "test"))
    models = {k: load_model(a.models, f"{k}_s0", data["train"]) for k in "ABC"}
    fns = make_fns(arm_old, arm_fast, ds.scale, models)
    q, dq, tau = (torch.from_numpy(x) for x in ds.q_dq_tau("test"))
    out = {"threads": a.threads, "torch": torch.__version__, "cpu": os.cpu_count(), "batches": a.batches, "time_per_call_s": {}}
    with torch.no_grad():
        for name, fn in fns.items():
            r = []
            for B in a.batches:
                args = (q[:B].contiguous(), dq[:B].contiguous(), tau[:B].contiguous())
                r.append(timeit(fn, args))
                print(f"{name:28s} B={B:6d}: {r[-1] * 1e3:9.3f} ms/呼び出し  {B / r[-1]:12.0f} step/s", flush=True)
            out["time_per_call_s"][name] = r
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/bench_t{a.threads}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
