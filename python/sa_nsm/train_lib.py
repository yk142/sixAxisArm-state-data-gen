"""学習データの準備（解析モデルの事前計算・特徴量・標準化統計）、学習ループ、評価。"""
import json
import os
import time

import numpy as np
import torch

from .baseline import accel_rmse, accel_rms, analytic_delta, groups
from .model import MODELS, limit_features
from .physics import Arm


def prepare(ds, arm, cache_path=None, splits=("train", "val", "test", "benchmark"), dt=1e-3):
    """split ごとのテンソルを返す。解析モデルの予測は倍精度で事前計算し、cache_path（npz）に保存・再利用する。"""
    n = ds.n_ax
    sc = ds.scale
    cache = dict(np.load(cache_path)) if cache_path and os.path.exists(cache_path) else {}
    out = {}
    for sp in splits:
        if sp not in ds.part:
            continue
        d = ds.part[sp]
        q, dq, tau = ds.q_dq_tau(sp)
        if f"{sp}_dq" not in cache:
            t = time.time()
            a, b = analytic_delta(arm, q, dq, tau, dt)
            cache[f"{sp}_dq"], cache[f"{sp}_ddq"] = a, b
            print(f"解析モデルの事前計算 {sp}: {len(q):,} 遷移 {time.time() - t:.0f} s", flush=True)
            if cache_path:
                np.savez(cache_path, **cache)
        ana = np.concatenate([cache[f"{sp}_dq"] / sc["D_Q_MAX"], cache[f"{sp}_ddq"] / sc["D_DQ_MAX"]], 1)
        lim = limit_features(torch.from_numpy(q), arm.q_min, arm.q_max).float().numpy()
        out[sp] = {"xs": d["X_spec"].astype(np.float32), "y": d["Y_spec"].astype(np.float32),
                   "ana": ana.astype(np.float32), "lim": lim, "at": d["at_limit"].astype(bool)}
    return out


def make_stats(kind, tr):
    """train から標準化の統計を作る。"""
    f = lambda a: torch.as_tensor(a, dtype=torch.float32)
    xs, y, ana, lim, at = (tr[k] for k in ("xs", "y", "ana", "lim", "at"))
    if kind == "A":
        return {"x_mean": xs.mean(0), "x_std": xs.std(0) + 1e-8, "y_mean": y.mean(0), "y_std": y.std(0)}
    z = np.concatenate([xs, ana, lim], 1)
    st = {"x_mean": z.mean(0), "x_std": z.std(0) + 1e-8}
    res = y - ana
    if kind == "B":
        st["sig_all"] = res.std(0)
    else:
        st["sig_free"], st["sig_contact"] = res[~at].std(0), res[at].std(0)
    return {k: f(v) for k, v in st.items()}


def _batches(data, idx, bs):
    for s in range(0, len(idx), bs):
        j = idx[s:s + bs]
        yield tuple(torch.from_numpy(data[k][j]) for k in ("xs", "ana", "lim", "y", "at"))


@torch.no_grad()
def predict(model, data, bs=65536):
    model.eval()
    return torch.cat([model(xs, ana, lim) for xs, ana, lim, _, _ in _batches(data, np.arange(len(data["y"])), bs)]).numpy()


@torch.no_grad()
def mean_loss(model, data, bs=65536):
    model.eval()
    tot, n = 0.0, 0
    for xs, ana, lim, y, at in _batches(data, np.arange(len(data["y"])), bs):
        tot += model.loss(model(xs, ana, lim), y, at).item() * len(y)
        n += len(y)
    return tot / n


def train(kind, data, n_ax, seed, epochs=40, bs=4096, lr=2e-3, lr_end=1e-5, warmup=200, width=256, depth=3, log=print):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = MODELS[kind](n_ax, make_stats(kind, data["train"]), width, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = len(data["train"]["y"])
    steps = epochs * ((N + bs - 1) // bs)
    sched = lambda s: (s + 1) / warmup if s < warmup else lr_end / lr + 0.5 * (1 - lr_end / lr) * (1 + np.cos(np.pi * (s - warmup) / max(1, steps - warmup)))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, sched)
    best, best_state, hist = np.inf, None, []
    for ep in range(epochs):
        model.train()
        t = time.time()
        tot = 0.0
        for xs, ana, lim, y, at in _batches(data["train"], rng.permutation(N), bs):
            opt.zero_grad()
            loss = model.loss(model(xs, ana, lim), y, at)
            loss.backward()
            opt.step(); sch.step()
            tot += loss.item() * len(y)
        vl = mean_loss(model, data["val"])
        hist.append({"epoch": ep + 1, "train": tot / N, "val": vl, "sec": time.time() - t})
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        log(f"[{kind} s{seed}] epoch {ep + 1}/{epochs} train {tot / N:.4g} val {vl:.4g} best {best:.4g} ({time.time() - t:.0f} s)")
    model.load_state_dict(best_state)
    return model, hist


def evaluate_model(model, ds, data, dt=1e-3, splits=("val", "test", "benchmark")):
    """区分別の 1 ステップ誤差（Δdq の加速度換算 RMSE と Δq の RMSE [rad]、軸別）。"""
    sc = ds.scale
    res = {}
    for sp in splits:
        if sp not in data:
            continue
        p = predict(model, data[sp])
        y = data[sp]["y"]
        n = ds.n_ax
        e_v = (p[:, n:] - y[:, n:]) * sc["D_DQ_MAX"]
        e_q = (p[:, :n] - y[:, :n]) * sc["D_Q_MAX"]
        d_dq = y[:, n:] * sc["D_DQ_MAX"]
        r = {}
        for k, m in groups(ds, sp, dt).items():
            r[k] = {"n": int(m.sum()), "acc_rmse": (np.sqrt(np.mean(e_v[m] ** 2, 0)) / dt).tolist(),
                    "q_rmse": np.sqrt(np.mean(e_q[m] ** 2, 0)).tolist(), "target": accel_rms(d_dq, dt, m).tolist()}
        res[sp] = r
    return res
