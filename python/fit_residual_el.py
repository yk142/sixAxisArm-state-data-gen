"""④ ハイブリッドの残差トルク NN の学習（issue #101）。使い方:
  ~/.venvs/sa-train/bin/python python/fit_residual_el.py --data data/export/id_elum_*.h5 --ident data/eval/id99/ident_free.json --out data/models/res101/full
同定（identify_el.py）と同じ測定から、同定済みの式で説明できないトルクを軸ごとに計算して回帰する（#85 の 1 ステップ状態損失ではなく、トルクの式の残差を直接）:
  モータ側 r_m = τ − (Jm·ddqm + k·θ + c·dθ)                          → ばねの非線形分 Δs（モデルでは τs = k·θ + c·dθ + Δs）
  リンク側 r_l = τs − (s·(M·ddq + g + b) + Bv·dq + Fc·tanh(dq/ε))    → 摩擦の非線形分 Δf（モデルでは摩擦に足す）
  τs = τ − Jm·ddqm。パラメータは同定値。
モデル: 軸ごとの MLP（入力 [θ_i/θs, dθ_i/ws, dq_i/vs]、隠れ層 2 × width、tanh、出力 [Δs_i, Δf_i]/τsc）。出力は numba 版（twomass_nb）が読む npz。
issue #105: --dq-eps で dq を ε で割る（Stribeck の幅が NN から見えるように）、--tanh-feat で 4 番目の入力 tanh(dq/ε) を足す、
--wlow で低速の行に重み 1 + wlow·exp(−(dq/(4ε))²) を付ける。
issue #109: --split でばねと摩擦を別の NN にする（ばね Δs ← [θ, dθ]、摩擦 Δf ← [dq, tanh(dq/ε)]）。1 つの NN だと摩擦が θ（姿勢の負荷）を
手がかりにし、ペイロードで θ が学習範囲を超えると止まる位置がずれた。npz は 1 つの MLP（幅 2·width、第 1 層・出力層をマスク、第 2 層はブロック対角）にまとめる。"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identify_el import load, rigid, parse_meas, tcal


def residuals(D, scale):
    el = json.load(open("data/export/elastic20.json")); P = json.load(open("data/export/c8_params.json"))
    sc = {k: np.asarray(v, float) for k, v in scale.items()}
    k = np.ravel(el["k"])[:6] * sc["k"]; c = np.ravel(el["c"])[:6] * sc["c"]; Jm = np.ravel(el["Jm"])[:6] * sc["Jm"]
    Fc = np.ravel(P["frictionFc"])[:6] * sc["Fc"]; Bv = np.ravel(P["frictionBv"])[:6] * sc["Bv"]; eps = np.ravel(P["frictionEps"])[:6]
    th, dth = D["qm"] - D["q"], D["dqm"] - D["dq"]
    k3 = sc.get("k3", np.zeros(6))                                          # 同定した θ³ の係数（issue #109、無ければ 0）
    rm = D["tau"] - (Jm * D["ddqm"] + k * th + k3 * th ** 3 + c * dth)
    ts = D["tau"] - Jm * D["ddqm"]
    rig = rigid(D, float(sc["mass"]))
    rl = ts - rig - Bv * D["dq"] - Fc * np.tanh(D["dq"] / eps)
    return th, dth, rm, rl


class AxisNet(nn.Module):
    def __init__(self, width, nin=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nin, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 2))

    def forward(self, x):
        return self.net(x)

    def pack(self):
        L = [l for l in self.net if isinstance(l, nn.Linear)]
        return {k: v for li, l in enumerate(L) for k, v in ((f"W{li}", l.weight.detach().numpy().astype(float)), (f"b{li}", l.bias.detach().numpy().astype(float)))}


class SubNet(nn.Module):
    """入力の一部の列 cols だけを見て、出力 out（0: Δs、1: Δf）だけを出す MLP（issue #109）。"""
    def __init__(self, width, cols, nin, out):
        super().__init__()
        self.cols, self.nin, self.out = cols, nin, out
        self.net = nn.Sequential(nn.Linear(len(cols), width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x[:, self.cols])


class SplitNet(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = a, b

    def forward(self, x):
        return torch.cat([self.a(x), self.b(x)], 1)

    def pack(self):
        """1 つの MLP（幅 2·width）にまとめる。第 1 層は見ない列を 0、第 2 層はブロック対角、出力層は担当の出力だけ。"""
        g = lambda t: t.detach().numpy().astype(float)
        H = self.a.net[0].out_features; nin = self.a.nin
        W0 = np.zeros((2 * H, nin)); b0 = np.zeros(2 * H); W1 = np.zeros((2 * H, 2 * H)); b1 = np.zeros(2 * H); W2 = np.zeros((2, 2 * H)); b2 = np.zeros(2)
        for k, sn in enumerate((self.a, self.b)):
            sl = slice(k * H, (k + 1) * H); l0, l1, l2 = sn.net[0], sn.net[2], sn.net[4]
            W0[sl, sn.cols] = g(l0.weight); b0[sl] = g(l0.bias); W1[sl, sl] = g(l1.weight); b1[sl] = g(l1.bias)
            W2[sn.out, sl] = g(l2.weight)[0]; b2[sn.out] = g(l2.bias)[0]
        return {"W0": W0, "b0": b0, "W1": W1, "b1": b1, "W2": W2, "b2": b2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True); ap.add_argument("--ident", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=4); ap.add_argument("--frac", type=float, default=1.0, help="使うシナリオの割合（データ量の軸）")
    ap.add_argument("--width", type=int, default=32); ap.add_argument("--iters", type=int, default=4000); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--meas", default="", help="測定の模擬（issue #113、identify_el.parse_meas）")
    ap.add_argument("--tcal", default="", help="トルクのゲイン校正 JSON（calib_gain.py）")
    ap.add_argument("--dq-eps", action="store_true", help="dq の尺度を ε にする（既定は標準偏差）")
    ap.add_argument("--tanh-feat", action="store_true", help="入力に tanh(dq/ε) を足す")
    ap.add_argument("--wlow", type=float, default=0.0, help="低速の重み")
    ap.add_argument("--split", action="store_true", help="ばね（θ, dθ）と摩擦（dq, tanh）を別の NN にする（issue #109）")
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    D = tcal(load(a.data, a.stride, frac=a.frac, seed=a.seed, meas=parse_meas(a.meas)), a.tcal)
    th, dth, rm, rl = residuals(D, json.load(open(a.ident))["scale"])
    N = len(th); print(f"行数 {N}（frac {a.frac}）", flush=True)
    idx = rng.permutation(N); nv = N // 10; vi, ti = idx[:nv], idx[nv:]
    out = {"width": 2 * a.width if a.split else a.width, "blk": a.width}; rep = {}
    eps = np.ravel(json.load(open("data/export/c8_params.json"))["frictionEps"])[:6]
    for i in range(6):
        cols = [th[:, i], dth[:, i], D["dq"][:, i]] + ([np.tanh(D["dq"][:, i] / eps[i])] if a.tanh_feat else [])
        X = np.stack(cols, 1); Y = np.stack([rm[:, i], rl[:, i]], 1)
        xs = X[ti].std(0) + 1e-12; ys = Y[ti].std(0) + 1e-12
        if a.dq_eps: xs[2] = eps[i]
        if a.tanh_feat: xs[3] = 1.0
        Xt, Yt = torch.tensor(X / xs, dtype=torch.float32), torch.tensor(Y / ys, dtype=torch.float32)
        Wt = torch.tensor(1.0 + a.wlow * np.exp(-(D["dq"][:, i] / (4 * eps[i])) ** 2), dtype=torch.float32)
        if a.split:
            nin = X.shape[1]; ms_, mf_ = SubNet(a.width, [0, 1], nin, 0), SubNet(a.width, [2] + ([3] if a.tanh_feat else []), nin, 1)
            m = SplitNet(ms_, mf_)
        else:
            m = AxisNet(a.width, X.shape[1])
        opt = torch.optim.Adam(m.parameters(), lr=a.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
        for it in range(a.iters):
            j = torch.from_numpy(rng.choice(ti, 4096))
            opt.zero_grad(); l = (Wt[j, None] * (m(Xt[j]) - Yt[j]) ** 2).mean(); l.backward(); opt.step(); sch.step()
        with torch.no_grad():
            pv = m(Xt[vi]).numpy() * ys
        e = Y[vi] - pv; lo = np.abs(D["dq"][vi, i]) < 4 * eps[i]
        rep[f"axis{i+1}"] = {"r_l_low_rms": float(np.sqrt((Y[vi, 1][lo] ** 2).mean())), "r_l_low_after": float(np.sqrt((e[lo, 1] ** 2).mean())),"r_m_rms": float(np.sqrt((Y[vi, 0] ** 2).mean())), "r_m_after": float(np.sqrt((e[:, 0] ** 2).mean())),
                             "r_l_rms": float(np.sqrt((Y[vi, 1] ** 2).mean())), "r_l_after": float(np.sqrt((e[:, 1] ** 2).mean()))}
        print(f"軸 {i+1}: モータ側残差 RMS {rep[f'axis{i+1}']['r_m_rms']:.3f} → {rep[f'axis{i+1}']['r_m_after']:.3f} N·m、リンク側 {rep[f'axis{i+1}']['r_l_rms']:.3f} → {rep[f'axis{i+1}']['r_l_after']:.3f} N·m（低速 {rep[f'axis{i+1}']['r_l_low_rms']:.3f} → {rep[f'axis{i+1}']['r_l_low_after']:.4f}）", flush=True)
        for key, v in m.pack().items():
            out[f"{key}_{i}"] = v
        out[f"xs_{i}"] = xs; out[f"ys_{i}"] = ys; out[f"eps_{i}"] = np.array([eps[i]])
    os.makedirs(a.out, exist_ok=True)
    np.savez(f"{a.out}/residual.npz", **out)
    json.dump({"args": vars(a), "rows": N, "val": rep}, open(f"{a.out}/fit.json", "w"), indent=1, ensure_ascii=False)
    print("完了", a.out)


if __name__ == "__main__":
    main()
