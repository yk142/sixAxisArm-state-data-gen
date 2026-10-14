"""NN A（純 MLP、解析モデルなし）の自由運動精度の改善実験。issue #42。使い方:
  ~/.venvs/sa-train/bin/python python/train_a.py --tag free_acc --target accel --subset free --h5 data/export/sa_flat_full3.h5 --out data/models/a_exp
入力は 4-2 の A と同じ X_spec。変えるのは (1) 出力の表現（--target std: 4-2 と同じ標準化 Δ、accel: Δdq/dt と (Δq − dq·dt)/dt² を自由運動の σ で標準化）、
(2) 学習行（--subset all / free）、(3) 幅・深さ・正則化・エポック。モデルの選択は val の自由運動（非接触）の加速度誤差で行う。"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.baseline import accel_rms, groups
from sa_nsm.data import Dataset
from sa_nsm.model import mlp

DT = 1e-3


def targets(ds, sp, kind):
    """学習の目標（float64 で作る）。accel: [(Δq − dq·dt)/dt², Δdq/dt]、std: Y_spec（4-2 と同じ）。"""
    n = ds.n_ax
    d = ds.part[sp]
    if kind == "std":
        return d["Y_spec"].astype(np.float64)
    q, dq, _ = ds.q_dq_tau(sp)
    d_q, d_dq = ds.delta(sp)
    return np.concatenate([(d_q - dq * DT) / DT ** 2, d_dq / DT], 1)


class Net(nn.Module):
    def __init__(self, n_ax, x_mean, x_std, t_mean, t_std, width, depth, act, drop):
        super().__init__()
        self.n_ax = n_ax
        for k, v in dict(x_mean=x_mean, x_std=x_std, t_mean=t_mean, t_std=t_std).items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
        layers, d = [], 4 * n_ax
        for _ in range(depth):
            layers += [nn.Linear(d, width), act()] + ([nn.Dropout(drop)] if drop > 0 else [])
            d = width
        layers.append(nn.Linear(d, 2 * n_ax))
        self.net = nn.Sequential(*layers)

    def forward(self, xs):          # 標準化された出力 z
        return self.net((xs - self.x_mean) / self.x_std)

    def raw(self, xs):              # 目標の単位（accel: rad/s² 相当）
        return self.forward(xs) * self.t_std + self.t_mean


def val_free_rmse(model, xs, t, free):
    """val の自由運動の加速度誤差 RMSE（Δdq/dt の軸別。accel 表現のとき）。"""
    model.eval()
    with torch.no_grad():
        p = model.raw(torch.from_numpy(xs[free])).numpy().astype(np.float64)
    return np.sqrt(np.mean((p - t[free]) ** 2, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/export/sa_flat_full3.h5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", choices=["std", "accel"], default="accel")
    ap.add_argument("--subset", choices=["all", "free"], default="free")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--act", choices=["silu", "gelu", "tanh"], default="silu")
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    ds = Dataset(a.h5, splits=("train", "val", "test"))
    n = ds.n_ax
    T = {sp: targets(ds, sp, a.target) for sp in ("train", "val", "test")}
    X = {sp: ds.part[sp]["X_spec"].astype(np.float32) for sp in T}
    AT = {sp: ds.at_limit(sp) for sp in T}
    rows = np.where(~AT["train"])[0] if a.subset == "free" else np.arange(len(X["train"]))
    ref = ~AT["train"]                                   # 標準化は自由運動の統計（accel）／全行（std：4-2 と同じ）
    sel = ref if a.target == "accel" else np.ones(len(ref), bool)
    t_mean, t_std = T["train"][sel].mean(0), T["train"][sel].std(0)
    xm, xs_ = X["train"][rows].mean(0), X["train"][rows].std(0) + 1e-8
    act = {"silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh}[a.act]
    model = Net(n, xm, xs_, t_mean, t_std, a.width, a.depth, act, a.drop)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd) if a.wd > 0 else torch.optim.Adam(model.parameters(), lr=a.lr)
    N = len(rows); spe = (N + a.bs - 1) // a.bs; steps = a.epochs * spe; warm, lr_end = 200, 1e-5
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else lr_end / a.lr + 0.5 * (1 - lr_end / a.lr) * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    Xt = torch.from_numpy(X["train"]); Zt = torch.from_numpy(((T["train"] - t_mean) / t_std).astype(np.float32))
    vfree = ~AT["val"]
    # 選択用の指標: val の自由運動の標準化損失（std 表現でも比較できるよう、加速度換算の RMSE を軸平均で使う）
    def val_metric():
        model.eval()
        with torch.no_grad():
            p = model.raw(torch.from_numpy(X["val"][vfree])).numpy().astype(np.float64)
        if a.target == "std":
            d_dq = (p[:, n:] - T["val"][vfree][:, n:]) * ds.scale["D_DQ_MAX"] / DT
        else:
            d_dq = p[:, n:] - T["val"][vfree][:, n:]
        return np.sqrt(np.mean(d_dq ** 2, 0))
    best, best_state, hist = np.inf, None, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        perm = rows[rng.permutation(N)]
        for s in range(0, N, a.bs):
            j = perm[s:s + a.bs]
            opt.zero_grad()
            loss = ((model(Xt[j]) - Zt[j]) ** 2).mean()
            loss.backward(); opt.step(); sch.step()
            tot += loss.item() * len(j)
        v = val_metric(); vm = float(v.mean())
        hist.append({"epoch": ep + 1, "train": tot / N, "val_free_acc_rmse": v.tolist(), "sec": time.time() - t0})
        if vm < best:
            best, best_state = vm, {k: x.clone() for k, x in model.state_dict().items()}
        print(f"[{a.tag}] epoch {ep + 1}/{a.epochs} train {tot / N:.4g} val 自由 加速度RMSE {np.round(v, 4)} best {best:.4f} ({time.time() - t0:.0f} s)", flush=True)
    model.load_state_dict(best_state)
    out = os.path.join(a.out, f"{a.tag}_s{a.seed}"); os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), f"{out}/model.pt")
    cfg = vars(a).copy(); cfg["n_ax"] = n
    json.dump(cfg, open(f"{out}/config.json", "w")); json.dump(hist, open(f"{out}/history.json", "w"))
    res = {}
    for sp in ("val", "test"):
        model.eval()
        with torch.no_grad():
            p = model.raw(torch.from_numpy(X[sp])).numpy().astype(np.float64)
        if a.target == "std":
            e_v = (p[:, n:] - T[sp][:, n:]) * ds.scale["D_DQ_MAX"] / DT
            e_q = (p[:, :n] - T[sp][:, :n]) * ds.scale["D_Q_MAX"]
        else:
            e_v = p[:, n:] - T[sp][:, n:]
            e_q = (p[:, :n] - T[sp][:, :n]) * DT ** 2
        d_q, d_dq = ds.delta(sp)
        res[sp] = {}
        for k, m in groups(ds, sp, DT).items():
            res[sp][k] = {"n": int(m.sum()), "acc_rmse": np.sqrt(np.mean(e_v[m] ** 2, 0)).tolist(),
                          "q_rmse": np.sqrt(np.mean(e_q[m] ** 2, 0)).tolist(), "target": accel_rms(d_dq, DT, m).tolist()}
    json.dump(res, open(f"{out}/metrics.json", "w"), indent=1)
    print("完了", out, "val 自由 平均", best, "test 自由", np.round(res["test"]["free"]["acc_rmse"], 4))


if __name__ == "__main__":
    main()
