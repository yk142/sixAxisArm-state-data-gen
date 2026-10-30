"""閉ループ評価器のバッチ化の効果（issue #89）: 2 慣性解析（サブステップ 8、iters=1）の 1 ステップあたりの時間を、行数 B=6（1 グループ）と B=72（12 グループ）で測る。"""
import sys, os, time, json
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.twomass import TwoMass

torch.set_num_threads(1)
tm = TwoMass("data/export/c8_params.json", "data/export/elastic20.json", substeps=8)
q0 = torch.tensor([0.0, -0.5, 0.5, 0.4, 0.8, -0.4], dtype=torch.float64)
res = {}
for B in (1, 6, 72):
    q = q0.expand(B, 6).clone(); z = torch.zeros(B, 6, dtype=torch.float64); tau = torch.zeros(B, 6, dtype=torch.float64)
    for _ in range(5): tm.step(q, z, q, z, tau)
    n = 100; t0 = time.perf_counter()
    for _ in range(n): tm.step(q, z, q, z, tau)
    dt = (time.perf_counter() - t0) / n
    res[B] = {"ms_per_step": dt * 1e3, "ms_per_row_step": dt * 1e3 / B}
    print(f"B={B}: {dt*1e3:.2f} ms/step、1 行あたり {dt*1e3/B:.3f} ms", flush=True)
print(f"B=72 は B=6 の 12 回分に対して {12 * res[6]['ms_per_step'] / res[72]['ms_per_step']:.1f} 倍速い")
json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), indent=1)
