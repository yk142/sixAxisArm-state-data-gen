"""2 慣性解析 + 残差 NN（B 型）の学習（issue #85）。使い方:
  ~/.venvs/sa-train/bin/python python/train_b_el.py --tag b --perturb k=1.2,c=2,fric=0.7,mass=1.1 --h5 data/export/sa_flat_fullplpu6el.h5 --out data/models/b_el
基準モデルは `TwoMass`（パラメータ誤差つき）。NN は状態・トルク・ペイロードからリンク側 rl とモータ側 rm の残差トルク [N·m]（1 ステップ一定）を出し、
ROS2 の中でダイナミクスに足す。損失は 1 ステップ後の 4 状態の真値との差（成分ごとに「基準モデルの 1 ステップ誤差の標準偏差」で割る）。
g, b, M と W は勾配に通さない（凍結係数）。入力は train_a_el.py と同じ特徴量 + ペイロード。出力層は 0 初期化なので、学習前は基準モデルそのもの。"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.data import Dataset
from sa_nsm.twomass import TwoMass
from train_a_el import features, PAYLOAD_SCALE, THETA_SCALE


class ResNet(nn.Module):
    def __init__(self, n_ax, d_in, x_mean, x_std, tau_scale, width, depth):
        super().__init__()
        self.n_ax = n_ax
        for k, v in dict(x_mean=x_mean, x_std=x_std, tau_scale=tau_scale).items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        out = nn.Linear(d, 2 * n_ax)
        nn.init.zeros_(out.weight); nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, xs):
        o = self.net((xs - self.x_mean) / self.x_std).double()
        s = self.tau_scale.double()
        return o[:, :self.n_ax] * s, o[:, self.n_ax:] * s       # (rl, rm) [N·m]


def parse_scale(s):
    return {kv.split("=")[0]: float(kv.split("=")[1]) for kv in s.split(",")} if s else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--perturb", default="", help="基準モデルのパラメータ倍率（例 k=1.2,c=2,fric=0.7,mass=1.1）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=256); ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--iters", type=int, default=3000); ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--theta-max", type=float, default=0.06)
    ap.add_argument("--val-n", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    ds = Dataset(a.h5, splits=("train", "val")); n = ds.n_ax
    tm = TwoMass("data/export/c8_params.json", "data/export/elastic20.json", n_ax=n, substeps=a.substeps, scale=parse_scale(a.perturb))
    tau_scale = np.asarray(ds.scale["TAU_MAX"]).ravel()[:n]

    def prep(sp):
        q, dq, tau = ds.q_dq_tau(sp); qm, dqm = ds.motor(sp)
        d_q, d_dq = ds.delta(sp); d_qm, d_dqm = ds.motor_delta(sp)
        th = np.abs(qm - q).max(1)
        ok = ~ds.at_limit(sp) & (th < a.theta_max)
        pl = ds.payload(sp)
        x = np.concatenate([features(ds, sp), pl[:, None] / PAYLOAD_SCALE], 1).astype(np.float32)
        st = np.concatenate([q, dq, qm, dqm], 1); nx = st + np.concatenate([d_q, d_dq, d_qm, d_dqm], 1)
        return dict(x=x, st=st, nx=nx, tau=tau, pl=pl, ok=ok)
    D = {sp: prep(sp) for sp in ("train", "val")}
    pls = sorted(set(D["train"]["pl"][D["train"]["ok"]].tolist()))
    idx = {p: np.where(D["train"]["ok"] & (D["train"]["pl"] == p))[0] for p in pls}
    print(f"[{a.tag}] ペイロード別の行数 " + ", ".join(f"{p:g}kg:{len(i)}" for p, i in idx.items()), flush=True)

    def base_pred(d, j, res=None, diff=False):
        tm.payload = float(d["pl"][j[0]])
        st = torch.from_numpy(d["st"][j]); tau = torch.from_numpy(d["tau"][j])
        o = tm.step(st[:, :n], st[:, n:2 * n], st[:, 2 * n:3 * n], st[:, 3 * n:], tau, res, diff)
        return torch.cat(o, -1)

    # 基準モデルの 1 ステップ誤差の標準偏差（成分ごと）でスケール
    errs = []
    for p in pls:
        j = rng.choice(idx[p], min(len(idx[p]), 2048), replace=False)
        errs.append((base_pred(D["train"], j) - torch.from_numpy(D["train"]["nx"][j])).numpy())
    scale = np.concatenate(errs).std(0) + 1e-12
    print(f"[{a.tag}] 基準モデルの 1 ステップ誤差 std q {scale[:n].mean():.3g} dq {scale[n:2*n].mean():.3g} qm {scale[2*n:3*n].mean():.3g} dqm {scale[3*n:].mean():.3g}", flush=True)
    sc_t = torch.from_numpy(scale)
    rows_all = np.concatenate(list(idx.values())); xm = D["train"]["x"][rows_all].mean(0); xs_ = D["train"]["x"][rows_all].std(0) + 1e-8
    model = ResNet(n, D["train"]["x"].shape[1], xm, xs_, tau_scale, a.width, a.depth)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * (1e-2 + 0.5 * (1 - 1e-2) * (1 + np.cos(np.pi * s / a.iters))))
    w = np.array([len(idx[p]) for p in pls], float); w /= w.sum()

    def loss_on(d, j, diff):
        res = model(torch.from_numpy(d["x"][j])) if diff or True else None
        pred = base_pred(d, j, res, diff=diff)
        return (((pred - torch.from_numpy(d["nx"][j])) / sc_t) ** 2).mean(0)        # [4n]

    vj = {p: np.where(D["val"]["ok"] & (D["val"]["pl"] == p))[0] for p in pls}
    vj = {p: rng.choice(v, min(len(v), a.val_n), replace=False) for p, v in vj.items() if len(v)}

    def val():
        model.eval(); out = []
        with torch.no_grad():
            for p, j in vj.items():
                out.append(loss_on(D["val"], j, False).numpy())
        model.train(); return np.mean(out, 0)

    v0 = np.mean([(((base_pred(D["val"], j) - torch.from_numpy(D["val"]["nx"][j])) / sc_t) ** 2).mean(0).numpy() for j in vj.values()], 0)
    print(f"[{a.tag}] val 基準（NN なし）標準化 MSE {v0.mean():.4g}", flush=True)
    best, best_state, hist, t0 = np.inf, None, [], time.time()
    for it in range(1, a.iters + 1):
        p = pls[rng.choice(len(pls), p=w)]
        j = rng.choice(idx[p], a.bs, replace=False)
        opt.zero_grad(); l = loss_on(D["train"], j, True); l.mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        if it % 100 == 0 or it == a.iters:
            v = val(); hist.append({"it": it, "train": float(l.mean()), "val": float(v.mean()), "val_comp": v.tolist()})
            if v.mean() < best: best, best_state = v.mean(), {k: x.clone() for k, x in model.state_dict().items()}
            print(f"[{a.tag}] it {it}/{a.iters} train {float(l.mean()):.4g} val {v.mean():.4g} best {best:.4g} (q {v[:n].mean():.3g} dq {v[n:2*n].mean():.3g} qm {v[2*n:3*n].mean():.3g} dqm {v[3*n:].mean():.3g}) {time.time() - t0:.0f}s", flush=True)
    model.load_state_dict(best_state)
    out = os.path.join(a.out, f"{a.tag}_s{a.seed}"); os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), f"{out}/model.pt")
    cfg = vars(a).copy(); cfg.update(n_ax=n, d_in=int(D["train"]["x"].shape[1]), err_std=scale.tolist(), val_base=v0.tolist())
    json.dump(cfg, open(f"{out}/config.json", "w")); json.dump(hist, open(f"{out}/history.json", "w"))
    print("完了", out, "val", best, "基準", v0.mean())


if __name__ == "__main__":
    main()
