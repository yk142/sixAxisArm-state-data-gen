"""③ 純 NN（連続時間、neural ODE）の学習（issue #111）。使い方:
  ~/.venvs/sa-train/bin/python python/fit_pure_el.py --data data/export/id_elum_{h1s,h2s,p1,p2,pl4a,pl4b}.h5 --out data/models/pure111/w128
物理の式（M・g・k・Jm・摩擦）を使わず、状態・トルク・ペイロードから加速度を出す 6 軸まとめての MLP:
  入力 [sin q, cos q, dq, θ = qm − q, dθ, τ, payload]（37 列、平均・標準偏差で標準化） → 出力 [ddq, ddqm]（12 列、標準化）
教師の加速度は 8 kHz の真値の角速度の中心差分（identify_el.load と同じ）。データは ④（#109）と同じ。
積分は numba 版（sa_nsm/pure_nb.py）が ROS2 で行い、ヤコビアンは NN の微分から作る。
閉ループ評価器が使うモータ慣性は、NN の ∂ddqm_i/∂τ_i の平均の逆数（学習した値。公称の Jm は使わない）。
--wlow: 低速の行に重み 1 + wlow·mean_i exp(−(dq_i/(4ε_i))²)（④ の --wlow と同じ考え）。--seed を変えて複数学習し、--members でまとめる（アンサンブル）。"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identify_el import load


def features(D):
    th, dth = D["qm"] - D["q"], D["dqm"] - D["dq"]
    return np.concatenate([np.sin(D["q"]), np.cos(D["q"]), D["dq"], th, dth, D["tau"], D["pl"][:, None]], 1)


class Net(nn.Module):
    def __init__(self, nin, width, nout=12):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nin, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, nout))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=4); ap.add_argument("--frac", type=float, default=1.0, help="使うシナリオの割合（データ量の軸）"); ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--iters", type=int, default=20000); ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--wlow", type=float, default=0.0, help="低速の重み")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    D = load(a.data, a.stride, frac=a.frac, seed=a.seed)
    X = features(D); Y = np.concatenate([D["ddq"], D["ddqm"]], 1); N = len(X)
    print(f"行数 {N}", flush=True)
    idx = rng.permutation(N); nv = N // 10; vi, ti = idx[:nv], idx[nv:]
    xm, xs = X[ti].mean(0), X[ti].std(0) + 1e-12; ym, ys = Y[ti].mean(0), Y[ti].std(0) + 1e-12
    eps = np.ravel(json.load(open("data/export/c8_params.json"))["frictionEps"])[:6]
    w = 1.0 + a.wlow * np.exp(-(D["dq"] / (4 * eps)) ** 2).mean(1)
    Xt, Yt, Wt = (torch.tensor(v, dtype=torch.float32) for v in ((X - xm) / xs, (Y - ym) / ys, w))
    m = Net(X.shape[1], a.width)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    for it in range(a.iters):
        j = torch.from_numpy(rng.choice(ti, a.batch))
        opt.zero_grad(); l = (Wt[j, None] * (m(Xt[j]) - Yt[j]) ** 2).mean(); l.backward(); opt.step(); sch.step()
        if it % 2000 == 0: print(f"{it} 損失 {l.item():.3e}", flush=True)
    with torch.no_grad():
        pv = m(Xt[vi]).numpy() * ys + ym
    e = Y[vi] - pv; low = np.abs(D["dq"][vi]).max(1) < 4 * eps.max()
    rep = {"rel_rms": (np.sqrt((e ** 2).mean(0)) / ys).tolist(), "rms": np.sqrt((e ** 2).mean(0)).tolist(),
           "low_rms": np.sqrt((e[low] ** 2).mean(0)).tolist() if low.any() else None}
    # 学習したモータ慣性: Jm_i = 1 / mean ∂ddqm_i/∂τ_i（閉ループ評価器のトルク反復の緩和に使う）
    xv = Xt[vi[:20000]].clone().requires_grad_(True)
    g = np.zeros(6)
    for i in range(6):
        (gr,) = torch.autograd.grad(m(xv)[:, 6 + i].sum(), xv)
        g[i] = (gr[:, 30 + i].mean().item()) * ys[6 + i] / xs[30 + i]
    jm = 1.0 / g
    print("相対 RMS（ddq, ddqm）", np.round(rep["rel_rms"], 4)); print("学習したモータ慣性", np.round(jm, 4))
    L = [l_ for l_ in m.net if isinstance(l_, nn.Linear)]
    out = {f"W{i}": l_.weight.detach().numpy().astype(float)[None] for i, l_ in enumerate(L)} | {f"b{i}": l_.bias.detach().numpy().astype(float)[None] for i, l_ in enumerate(L)}
    out |= {"xm": xm, "xs": xs, "ym": ym, "ys": ys, "jm": jm}
    os.makedirs(a.out, exist_ok=True)
    np.savez(f"{a.out}/pure.npz", **out)
    json.dump({"args": vars(a), "rows": N, "val": rep, "jm": jm.tolist()}, open(f"{a.out}/fit.json", "w"), indent=1, ensure_ascii=False)
    print("完了", a.out)


def ensemble(ins, out):
    """複数の種の pure.npz を 1 つに（第 0 軸に積む。標準化は種 0 のもの: 同じデータ・同じ分割なので種ごとの差は乱数の分割だけ）。"""
    Z = [np.load(f"{p}/pure.npz") for p in ins]
    o = {k: np.concatenate([z[k] for z in Z]) for k in ("W0", "b0", "W1", "b1", "W2", "b2")}
    # 標準化は種ごとに違いうるので、各メンバーの xm/xs/ym/ys も積む
    o |= {k: np.stack([z[k] for z in Z]) for k in ("xm", "xs", "ym", "ys")}
    o["jm"] = np.mean([z["jm"] for z in Z], 0)
    os.makedirs(out, exist_ok=True); np.savez(f"{out}/pure.npz", **o); print("完了", out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ens":
        ensemble(sys.argv[3:], sys.argv[2])
    else:
        main()
