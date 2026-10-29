"""A_el（モータ側の状態を含む NN）の弾性真値での閉ループステップ評価（issue #73）。使い方:
  ~/.venvs/sa-train/bin/python python/eval_payload_el.py --models data/models/a_el/el_s0 --data data/export/payload_eval6_el20.h5 --out data/eval/el20_ael --shard 0/4
eval_payload.py と同じ指標（ステップ応答のリンク側角度 q）。データは #71 の弾性真値（truth_qm 付き）。PD は実機と同じくモータ側でフィードバック。"""
import argparse, json, os, sys
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_payload import step_metrics, DEG
from sa_nsm.closedloop import DT, Scenario, closed_loop_el
from train_a_el import ElNet, DT as DT_, THETA_SCALE, PAYLOAD_SCALE


class ElPlant:
    """train_a_el.py のモデル。payload [kg] を属性で与える。"""
    def __init__(self, path, dq_max, tau_max):
        cfg = json.load(open(f"{path}/config.json")); n = cfg["n_ax"]; d = cfg["d_in"]
        z = np.zeros(d)
        self.m = ElNet(n, d, z, np.ones(d), np.zeros(4 * n), np.ones(4 * n), cfg["width"], cfg["depth"])
        self.m.load_state_dict(torch.load(f"{path}/model.pt")); self.m.eval()
        self.n, self.use_payload, self.payload = n, bool(cfg["payload"]), 0.0
        self.dq_max, self.tau_max = torch.as_tensor(dq_max), torch.as_tensor(tau_max)

    @torch.no_grad()
    def step(self, q, dq, qm, dqm, tau):
        xs = torch.cat([torch.sin(q), torch.cos(q), dq / self.dq_max, (qm - q) / THETA_SCALE, dqm / self.dq_max, tau / self.tau_max], -1).float()
        if self.use_payload:
            xs = torch.cat([xs, torch.full((len(xs), 1), self.payload / PAYLOAD_SCALE)], -1)
        p = self.m.raw(xs).double(); n = self.n; h = DT
        return (q + dq * h + p[:, :n] * h ** 2, dq + p[:, n:2 * n] * h,
                qm + dqm * h + p[:, 2 * n:3 * n] * h ** 2, dqm + p[:, 3 * n:] * h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--twomass", type=int, default=0, help="2 慣性の解析モデル（issue #77）を追加する。値はサブステップ数（0 で無効、推奨 8）")
    ap.add_argument("--data", default="data/export/payload_eval6_el20.h5")
    ap.add_argument("--h5", default="data/export/sa_flat_fullplpu6el.h5", help="スケール（DQ_MAX, TAU_MAX）を読む学習データ")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--iters", type=int, default=0, help="closed_loop_el の平均トルク整合の反復回数（issue #77）")
    ap.add_argument("--wn-min", type=float, default=0, help="この wn 以上のグループだけ評価する")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    si, sk = (int(x) for x in a.shard.split("/"))
    torch.set_num_threads(a.threads)
    with h5py.File(a.h5, "r") as f:
        dq_max, tau_max = f["scale/DQ_MAX"][:].ravel(), f["scale/TAU_MAX"][:].ravel()
    plants = {f"A_el（{os.path.basename(p)}）": ElPlant(p, dq_max, tau_max) for p in a.models}
    if a.twomass:
        from sa_nsm.twomass import TwoMass
        plants[f"2慣性解析（sub{a.twomass}）"] = TwoMass("data/export/c8_params.json", "data/export/elastic20.json", substeps=a.twomass)
    os.makedirs(a.out, exist_ok=True)
    res = []
    with h5py.File(a.data, "r") as f:
        peak = f["controller/tauPeak"][:].ravel()
        groups = sorted({k.rsplit("_j", 1)[0] for k in f if k != "controller"})
        groups = [g for g in groups if float(np.ravel(f[g + "_j1"].attrs["wn"])[0]) >= a.wn_min]
        for gi, gk in enumerate(groups):
            if gi % sk != si: continue
            m = int(f[gk + "_j1"].attrs["payload"])
            ctl = {"Kp": f[gk + "_j1/Kp"][:].ravel(), "Kd": f[gk + "_j1/Kd"][:].ravel(), "peak": peak}
            sc = [Scenario(f[f"{gk}_j{ax}"], ctl) for ax in range(1, len(peak) + 1)]
            at = f[gk + "_j1"].attrs
            wn, zeta, step, t_step = float(at["wn"]), float(at["zeta"]), float(at["step"]), float(at["tStep"])
            t = np.arange(sc[0].K + 1) * DT
            i_pre = int(round((t_step - 0.01) / DT))
            q_by = {"Simscape": np.stack([s.q for s in sc])}
            for pn, pl in plants.items():
                pl.payload = float(m)
                q_by[pn] = closed_loop_el(pl, sc, ctl, iters=a.iters)[0]
            for ax in range(len(peak)):
                row = {"payload": m, "wn": wn, "zeta": zeta, "axis": ax + 1}
                for pn, q in q_by.items():
                    qa = q[ax][:, ax]
                    row[pn] = step_metrics(t, qa, qa[i_pre], step, t_step)
                    if pn != "Simscape":
                        e = (q[ax] - q_by["Simscape"][ax]) * DEG
                        dq_ = (q[ax][:, ax] - q[ax][i_pre, ax]) - (q_by["Simscape"][ax][:, ax] - q_by["Simscape"][ax][i_pre, ax])
                        row[pn].update({"step_err_max_deg": float(np.abs(dq_[i_pre:]).max() * DEG), "err_rms_deg": np.sqrt((e ** 2).mean(0)).tolist(), "err_max_deg": np.abs(e).max(0).tolist(),
                                        "err_axis_max_deg": float(np.abs(e[:, ax]).max()), "ok": bool(np.isfinite(q[ax]).all())})
                res.append(row)
            print(f"{gk}: done", flush=True)
    json.dump(res, open(f"{a.out}/payload.json", "w"), indent=1)
    print("完了", a.out)


if __name__ == "__main__":
    main()
