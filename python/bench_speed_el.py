"""逐次（B=1、1 スレッド）の 1 ms あたり計算時間と実時間比（issue #91）。
  ~/.venvs/sa-train/bin/python python/bench_speed_el.py --out reports/issue-91/speed_py.json
対象: 2 慣性解析（torch 版・numba 版、サブステップ 4/8）、NN（A_el と同じ入出力の MLP、幅×深さ別）。
各対象は「プラントの 1 ステップ（1 ms 分）」の時間。閉ループで PD 反復（iters=1）を使うと 2 回呼ぶので 2 倍になる。
測定: 連続 N ステップを状態を進めながら実行し、1 ステップの平均。雑音対策に reps 回の最小値を取る。"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.twomass import TwoMass
from sa_nsm.twomass_nb import TwoMassNB
from train_a_el import ElNet

P, E = "data/export/c8_params.json", "data/export/elastic20.json"
Q0 = np.array([0.0, -0.5, 0.5, 0.4, 0.8, -0.4])


def bench(fn, n, reps):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter(); fn(n); best = min(best, (time.perf_counter() - t0) / n)
    return best * 1e3          # ms / step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    torch.set_num_threads(1)
    tau = np.array([0.0, 30.0, 10.0, 0.5, 0.5, 0.1]); res = {}
    # 2 慣性解析 torch
    for sub in (4, 8):
        tm = TwoMass(P, E, substeps=sub)
        st = [torch.tensor(Q0)[None], torch.zeros(1, 6, dtype=torch.float64), torch.tensor(Q0)[None], torch.zeros(1, 6, dtype=torch.float64)]
        tt = torch.tensor(tau)[None]
        def run(n, tm=tm, st=st, tt=tt):
            s = st
            for _ in range(n): s = list(tm.step(*s, tt))
        run(3); res[f"2慣性解析 torch sub{sub}"] = bench(run, 30, a.reps)
    # 2 慣性解析 numba（numpy 1 行、変換なし）
    for sub, fz in ((4, 0), (8, 0), (8, 1), (4, 1)):
        nb = TwoMassNB(P, E, substeps=sub, frozen=fz)
        y0 = np.concatenate([Q0, np.zeros(6), Q0, np.zeros(6)])
        def run(n, nb=nb, y0=y0):
            y = y0
            for _ in range(n): y = nb.step_np(y, tau)
        run(3); res[f"2慣性解析 numba sub{sub}" + ("（M 凍結）" if fz else "")] = bench(run, 2000, a.reps)
    # NN（A_el と同じ特徴量 37 次元 → 24 出力、float32、torch eager、no_grad）
    for w, d in ((256, 3), (512, 4)):
        m = ElNet(6, 37, np.zeros(37), np.ones(37), np.zeros(24), np.ones(24), w, d).eval()
        x = torch.zeros(1, 37)
        def run(n, m=m, x=x):
            with torch.no_grad():
                for _ in range(n):
                    p = m.raw(x); x = x + 0 * p[:, :1]           # 前の出力に依存させて逐次にする
        run(10); res[f"NN MLP {w}×{d}（eager）"] = bench(run, 2000, a.reps)
        # numpy での行列積（torch の呼び出しオーバーヘッドを除いた参考値）
        Ws = [(l.weight.detach().numpy().astype(np.float32), l.bias.detach().numpy().astype(np.float32)) for l in m.net if isinstance(l, torch.nn.Linear)]
        xn = np.zeros(37, np.float32)
        def run_np(n, Ws=Ws, xn=xn):
            v = xn
            for _ in range(n):
                h = v
                for i, (W, b) in enumerate(Ws):
                    h = W @ h + b
                    if i < len(Ws) - 1: h = h / (1 + np.exp(-h))
                v = xn + 0 * h[0]
        run_np(10); res[f"NN MLP {w}×{d}（numpy）"] = bench(run_np, 2000, a.reps)
    for k, v in res.items():
        print(f"{k}: {v:.4f} ms/step（実時間比 {v:.3f}）")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"ms_per_step": res, "note": "B=1、1 スレッド、1 ms 分のプラント 1 ステップ。実時間比 = ms/step ÷ 1 ms"}, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
