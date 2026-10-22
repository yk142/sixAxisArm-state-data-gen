"""モータ側の状態を含む NN A_el の学習（issue #73）。使い方:
  ~/.venvs/sa-train/bin/python python/train_a_el.py --tag el --h5 data/export/sa_flat_fullplpu6el.h5 --out data/models/a_el --payload
状態 (q, dq, qm, dqm)（qm, dqm = モータ側をリンク側に換算した角度・角速度。実機ではエンコーダで測れる）と tau、payload から、
1 ms 後の 4 状態の変化を出す。出力は train_a.py の accel 表現を 4 状態に拡張したもの:
  [(Δq − dq·dt)/dt², Δdq/dt, (Δqm − dqm·dt)/dt², Δdqm/dt]（各 n 列）を、自由運動（リミット拘束でない行）の平均・標準偏差で標準化。
入力: [sin q, cos q, dq/DQ_MAX, (qm − q)/THETA_SCALE, dqm/DQ_MAX, tau/TAU_MAX, (payload/8)]。モデルの選択は val の標準化 MSE（4 状態の平均）。"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.data import Dataset

DT = 1e-3
PAYLOAD_SCALE = 8.0
THETA_SCALE = 0.02          # ねじれ角 θ = qm − q の目安 [rad]（saElasticParams の thetaMax）


def features(ds, sp):
    """入力行列 [N, 6n(+1)]（payload は後で足す）。"""
    n = ds.n_ax
    q, dq, tau = ds.q_dq_tau(sp)
    qm, dqm = ds.motor(sp)
    sc = ds.scale
    return np.concatenate([np.sin(q), np.cos(q), dq / sc["DQ_MAX"], (qm - q) / THETA_SCALE, dqm / sc["DQ_MAX"], tau / sc["TAU_MAX"]], 1)


def targets(ds, sp):
    q, dq, _ = ds.q_dq_tau(sp)
    qm, dqm = ds.motor(sp)
    d_q, d_dq = ds.delta(sp)
    d_qm, d_dqm = ds.motor_delta(sp)
    return np.concatenate([(d_q - dq * DT) / DT ** 2, d_dq / DT, (d_qm - dqm * DT) / DT ** 2, d_dqm / DT], 1)


class ElNet(nn.Module):
    def __init__(self, n_ax, d_in, x_mean, x_std, t_mean, t_std, width, depth):
        super().__init__()
        self.n_ax = n_ax
        for k, v in dict(x_mean=x_mean, x_std=x_std, t_mean=t_mean, t_std=t_std).items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(d, 4 * n_ax))
        self.net = nn.Sequential(*layers)

    def forward(self, xs):
        return self.net((xs - self.x_mean) / self.x_std)

    def raw(self, xs):
        return self.forward(xs) * self.t_std + self.t_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.2)
    ap.add_argument("--payload", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--theta-max", type=float, default=0.06, help="学習・選択に使う行の上限: 全軸で |qm − q| < この値 [rad]。励振（チリープ等）が共振を叩くと θ が 0.7 rad に達するが、閉ループ（保持・ステップ）の p99 は 0.005 rad 程度で実運転の範囲外。0 で無効")
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    ds = Dataset(a.h5, splits=("train", "val", "test"))
    assert ds.has_motor("train"), "Xm がありません（弾性つきのデータではない）"
    n = ds.n_ax
    X, T, AT = {}, {}, {}
    for sp in ("train", "val", "test"):
        x = features(ds, sp)
        if a.payload: x = np.concatenate([x, ds.payload(sp)[:, None] / PAYLOAD_SCALE], 1)
        X[sp], T[sp], AT[sp] = x.astype(np.float32), targets(ds, sp), ds.at_limit(sp)
    TH = {sp: np.abs(ds.motor(sp)[0] - ds.q_dq_tau(sp)[0]).max(1) for sp in ("train", "val", "test")}
    inr = {sp: (TH[sp] < a.theta_max) if a.theta_max > 0 else np.ones(len(TH[sp]), bool) for sp in TH}
    print(f"[{a.tag}] θ<{a.theta_max} の行: train {inr['train'].mean() * 100:.1f}% val {inr['val'].mean() * 100:.1f}%", flush=True)
    rows = np.where(~AT["train"] & inr["train"])[0]
    t_mean, t_std = T["train"][rows].mean(0), T["train"][rows].std(0)
    xm, xs_ = X["train"][rows].mean(0), X["train"][rows].std(0) + 1e-8
    model = ElNet(n, X["train"].shape[1], xm, xs_, t_mean, t_std, a.width, a.depth)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    N = len(rows); spe = (N + a.bs - 1) // a.bs; steps = a.epochs * spe; warm, lr_end = 200, 1e-5
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else lr_end / a.lr + 0.5 * (1 - lr_end / a.lr) * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    Xt = torch.from_numpy(X["train"]); Zt = torch.from_numpy(((T["train"] - t_mean) / t_std).astype(np.float32))
    vfree = ~AT["val"] & inr["val"]

    def val_metric():
        model.eval()
        with torch.no_grad():
            p = model.raw(torch.from_numpy(X["val"][vfree])).numpy().astype(np.float64)
        e = p - T["val"][vfree]
        return np.sqrt((e ** 2).mean(0)), float((((e) / t_std) ** 2).mean())

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
        rm, vm = val_metric()
        hist.append({"epoch": ep + 1, "train": tot / N, "val_norm_mse": vm, "val_rmse": rm.tolist(), "sec": time.time() - t0})
        if vm < best:
            best, best_state = vm, {k: x.clone() for k, x in model.state_dict().items()}
        print(f"[{a.tag}] epoch {ep + 1}/{a.epochs} train {tot / N:.4g} val 標準化MSE {vm:.4g} best {best:.4g} 加速度RMSE(リンク dq) {np.round(rm[n:2 * n], 3)} ({time.time() - t0:.0f} s)", flush=True)
    model.load_state_dict(best_state)
    out = os.path.join(a.out, f"{a.tag}_s{a.seed}"); os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), f"{out}/model.pt")
    cfg = vars(a).copy(); cfg.update(n_ax=n, d_in=int(X["train"].shape[1]))
    json.dump(cfg, open(f"{out}/config.json", "w")); json.dump(hist, open(f"{out}/history.json", "w"))
    res = {}
    for sp in ("val", "test"):
        model.eval()
        with torch.no_grad():
            p = model.raw(torch.from_numpy(X[sp])).numpy().astype(np.float64)
        e = p - T[sp]; fr = ~AT[sp] & inr[sp]
        res[sp] = {"free_rmse": np.sqrt((e[fr] ** 2).mean(0)).tolist(), "target_rms": np.sqrt((T[sp][fr] ** 2).mean(0)).tolist()}
    json.dump(res, open(f"{out}/metrics.json", "w"), indent=1)
    print("完了", out, "val", best)


if __name__ == "__main__":
    main()
