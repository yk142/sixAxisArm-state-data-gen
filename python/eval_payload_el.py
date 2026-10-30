"""A_el（モータ側の状態を含む NN）の弾性真値での閉ループステップ評価（issue #73）。使い方:
  ~/.venvs/sa-train/bin/python python/eval_payload_el.py --models data/models/a_el/el_s0 --data data/export/payload_eval6_el20.h5 --out data/eval/el20_ael --shard 0/4
eval_payload.py と同じ指標（ステップ応答のリンク側角度 q）。データは #71 の弾性真値（truth_qm 付き）。PD は実機と同じくモータ側でフィードバック。"""
import argparse, json, os, sys, time
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


class BPlant:
    """2 慣性解析（パラメータ誤差つき）+ 残差 NN（train_b_el.py。issue #85）。payload [kg] を属性で与える。"""
    def __init__(self, path, substeps):
        from sa_nsm.twomass import TwoMass
        from train_b_el import ResNet, parse_scale
        cfg = json.load(open(f"{path}/config.json")); n = cfg["n_ax"]; d = cfg["d_in"]; z = np.zeros(d)
        self.tm = TwoMass("data/export/c8_params.json", "data/export/elastic20.json", n_ax=n, substeps=substeps or cfg["substeps"], scale=parse_scale(cfg["perturb"]))
        self.m = ResNet(n, d, z, np.ones(d), np.ones(n), cfg["width"], cfg["depth"])
        self.m.load_state_dict(torch.load(f"{path}/model.pt")); self.m.eval()
        self.n, self.payload, self.Jm = n, 0.0, self.tm.Jm

    @torch.no_grad()
    def step(self, q, dq, qm, dqm, tau):
        self.tm.payload = self.payload
        xs = torch.cat([torch.sin(q), torch.cos(q), dq / self.dq_max, (qm - q) / THETA_SCALE, dqm / self.dq_max, tau / self.tau_max,
                        torch.full((len(q), 1), self.payload / PAYLOAD_SCALE, dtype=q.dtype)], -1).float()
        return self.tm.step(q, dq, qm, dqm, tau, self.m(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--twomass", type=int, default=0, help="2 慣性の解析モデル（issue #77）を追加する。値はサブステップ数（0 で無効、推奨 8）")
    ap.add_argument("--perturb", nargs="*", default=[], help="2 慣性のパラメータ倍率の組（例 k=1.2 c=0.5,Jm=1.1）。issue #81")
    ap.add_argument("--bmodels", nargs="*", default=[], help="B 型（2 慣性 + 残差 NN）のモデルディレクトリ（issue #85）")
    ap.add_argument("--pt-sub", type=int, default=8, help="--perturb のサブステップ数")
    ap.add_argument("--groups", nargs="*", default=None, help="評価するグループ名の接頭辞/部分文字列（例 pl0_ pl8_）")
    ap.add_argument("--data", nargs="+", default=["data/export/payload_eval6_el20.h5"], help="真値の HDF5（複数可。グループ名が重なれば先のファイルを優先）")
    ap.add_argument("--zetas", nargs="*", type=float, default=None, help="評価する ζ（省略で全部）")
    ap.add_argument("--payloads", nargs="*", type=float, default=None, help="評価するペイロード [kg]（省略で全部）")
    ap.add_argument("--batch", action="store_true", help="同じペイロードの全グループ×6 軸を 1 回の閉ループで計算する（issue #89）。シャードはペイロード単位")
    ap.add_argument("--wn-max", type=float, default=1e9)
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
    for p in a.bmodels:
        pl = BPlant(p, a.pt_sub); pl.dq_max, pl.tau_max = torch.as_tensor(dq_max), torch.as_tensor(tau_max)
        plants[f"B型（{os.path.basename(p)}）"] = pl
    for pt in a.perturb:
        sc_ = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in pt.split(",")}
        from sa_nsm.twomass import TwoMass
        plants[f"2慣性 {pt}"] = TwoMass("data/export/c8_params.json", "data/export/elastic20.json", substeps=a.pt_sub, scale=sc_)
    os.makedirs(a.out, exist_ok=True)
    res, tim = [], []
    files = [h5py.File(p, "r") for p in a.data]
    peak = files[0]["controller/tauPeak"][:].ravel(); n = len(peak)
    G = {}                                                    # グループ名 → ファイル
    for f in files:
        for k in f:
            if k != "controller": G.setdefault(k.rsplit("_j", 1)[0], f)
    att = lambda gk, key: float(np.ravel(G[gk][gk + "_j1"].attrs[key])[0])
    groups = sorted(g for g in G if a.wn_min <= att(g, "wn") <= a.wn_max
                    and (a.zetas is None or any(abs(att(g, "zeta") - z) < 1e-9 for z in a.zetas))
                    and (a.payloads is None or att(g, "payload") in a.payloads))
    if a.groups: groups = [g for g in groups if any(x in g for x in a.groups)]
    if a.batch:
        pls = sorted({att(g, "payload") for g in groups})
        units = [[g for g in groups if att(g, "payload") == p] for p in pls]
    else:
        units = [[g] for g in groups]
    for ui, unit in enumerate(units):
        if ui % sk != si: continue
        m = att(unit[0], "payload")
        sc, Kp, Kd, meta = [], [], [], []
        for gk in unit:
            f = G[gk]
            ctl = {"Kp": f[gk + "_j1/Kp"][:].ravel(), "Kd": f[gk + "_j1/Kd"][:].ravel(), "peak": peak}
            for ax in range(1, n + 1):
                sc.append(Scenario(f[f"{gk}_j{ax}"], ctl)); Kp.append(ctl["Kp"]); Kd.append(ctl["Kd"]); meta.append((gk, ax))
        ctl_b = {"Kp": np.stack(Kp), "Kd": np.stack(Kd), "peak": peak}            # 行ごとのゲイン [B, n]
        q_by = {"Simscape": [s.q for s in sc]}
        for pn, pl in plants.items():
            pl.payload = float(m); t0 = time.time()
            q_by[pn] = list(closed_loop_el(pl, sc, ctl_b, iters=a.iters)[0])
            tim.append({"plant": pn, "payload": m, "groups": len(unit), "rows": len(sc), "sec": time.time() - t0})
        for i, (gk, ax) in enumerate(meta):
            at = G[gk][gk + "_j1"].attrs
            step, t_step = float(np.ravel(at["step"])[0]), float(np.ravel(at["tStep"])[0])
            K = sc[i].K; t = np.arange(K + 1) * DT; i_pre = int(round((t_step - 0.01) / DT))
            row = {"payload": int(m), "wn": att(gk, "wn"), "zeta": att(gk, "zeta"), "axis": ax}
            qt = q_by["Simscape"][i]
            for pn, qq in q_by.items():
                q = qq[i][:K + 1]; qa = q[:, ax - 1]
                row[pn] = step_metrics(t, qa, qa[i_pre], step, t_step)
                if pn != "Simscape":
                    e = (q - qt) * DEG
                    dq_ = (qa - qa[i_pre]) - (qt[:, ax - 1] - qt[i_pre, ax - 1])
                    row[pn].update({"step_err_max_deg": float(np.abs(dq_[i_pre:]).max() * DEG), "err_rms_deg": np.sqrt((e ** 2).mean(0)).tolist(), "err_max_deg": np.abs(e).max(0).tolist(),
                                    "err_axis_max_deg": float(np.abs(e[:, ax - 1]).max()), "ok": bool(np.isfinite(q).all())})
            res.append(row)
        print(f"{unit[0]}..（{len(unit)} グループ）: done", flush=True)
    json.dump(tim, open(f"{a.out}/timing.json", "w"), indent=1)
    json.dump(res, open(f"{a.out}/payload.json", "w"), indent=1)
    print("完了", a.out)


if __name__ == "__main__":
    main()
