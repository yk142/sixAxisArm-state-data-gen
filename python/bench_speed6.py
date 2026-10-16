"""6 軸と 3 軸の 1 ステップ（1 ms）計算時間の比較（issue #53）。解析（FastArm + torch.compile、サブステップ 1・2）と NN A（4-3d の r6）。
  ~/.venvs/sa-train/bin/python python/bench_speed6.py --out data/eval/speed6 [--threads 4]
バッチ 1（レイテンシ）と 4096（スループット）、best-of-5（bench_speed_opt.py と同じ測り方）。"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_speed import timeit as _timeit
from eval_a_window import AccelNet
from sa_nsm.data import Dataset
from sa_nsm.physics_fast import FastArm


def timeit(fn, args, reps=5):
    return min(_timeit(fn, args, min_time=0.3) for _ in range(reps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4096])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cases = {3: ("data/export/sa_flat_full3.h5", "data/models/a_exp/r6_w512d4_e60_s0"), 6: ("data/export/sa_flat_full6.h5", "data/models/a_exp6/r6_w512d4_e60_s0")}
    res = {}
    for n, (h5, apath) in cases.items():
        ds = Dataset(h5, splits=("train", "test"))
        arm = FastArm.from_json("data/export/c8_params.json", n)
        q, dq, tau = (torch.from_numpy(x) for x in ds.q_dq_tau("test"))
        fns = {f"解析 compile サブステップ{sub}": (lambda f: (lambda q, v, t: f(q, v, t)))(torch.compile(lambda q, v, t, sub=sub: arm.step(q, v, t, 1e-3, sub), dynamic=True)) for sub in (1, 2)}
        net = AccelNet(ds, apath)
        fns["NN A（r6）"] = lambda q, v, t: net.step(q, v, t)
        with torch.no_grad():
            for name, fn in fns.items():
                r = []
                for B in a.batches:
                    r.append(timeit(fn, (q[:B].contiguous(), dq[:B].contiguous(), tau[:B].contiguous())))
                    print(f"{n} 軸 {name:24s} B={B:5d}: {r[-1] * 1e3:8.3f} ms/呼び出し（{r[-1] * 1e3 / B:.5f} ms/ステップ）", flush=True)
                res[f"{n}軸 {name}"] = r
    os.makedirs(a.out, exist_ok=True)
    json.dump({"batches": a.batches, "time_per_call_s": res, "threads": a.threads}, open(f"{a.out}/bench6_t{a.threads}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
