"""Phase 4-17: 2 慣性解析モデルのパラメータ感度の集計（issue #81）。data/eval/sens81/<倍率>/payload.json を束ねて水準 G の各指標を出す。"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
OS_LIM, G_DOS, G_DST_REL, G_DST_ABS, G_POS = 5.0, 2.0, 0.10, 0.020, 0.1
ORDER = ["k=1", "k=0.8", "k=1.2", "c=0.5", "c=2", "Jm=0.85", "Jm=1.15", "fric=0.7", "fric=1.3", "mass=0.9", "mass=1.1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/eval/sens81"); ap.add_argument("--out", default="reports/issue-81")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    D = {}
    for v in ORDER:
        f = f"{a.dir}/{v}/payload.json"
        if os.path.isfile(f): D[v] = json.load(open(f))
    key = lambda r: (r["payload"], r["wn"], r["zeta"], r["axis"])
    tru = {key(r): r["Simscape"] for r in next(iter(D.values()))}
    gains = sorted({k[1:3] for k in tru}); pls = sorted({k[0] for k in tru}); axes = sorted({k[3] for k in tru})
    def score(get, p, w, z):
        rs = [get(p, w, z, ax) for ax in axes]
        return max(r["overshoot_pct"] for r in rs), float(np.nanmean([r["settle_s"] for r in rs]))
    def best(get, p):
        c = [(score(get, p, w, z)[1], w, z) for w, z in gains if score(get, p, w, z)[0] <= OS_LIM]
        return min(c)[1:]
    tget = lambda p, w, z, ax: tru[p, w, z, ax]
    tb = {p: best(tget, p) for p in pls}
    rows = []
    for v, R in D.items():
        mk = [k for k in R[0] if k.startswith("2慣性")][0]
        m = {key(r): r[mk] for r in R}
        dos = max(abs(m[k]["overshoot_pct"] - tru[k]["overshoot_pct"]) for k in tru)
        d = [(abs(m[k]["settle_s"] - tru[k]["settle_s"]), tru[k]["settle_s"]) for k in tru]
        d = [(x, t) for x, t in d if np.isfinite(x)]
        dabs, drel = max(x for x, t in d), max(x / t for x, t in d)
        se = max(m[k]["step_err_max_deg"] for k in tru); ae = max(m[k]["err_axis_max_deg"] for k in tru)
        mget = lambda p, w, z, ax: m[p, w, z, ax]
        mb = {p: best(mget, p) for p in pls}
        match = sum(mb[p] == tb[p] for p in pls)
        regret = max(score(tget, p, *mb[p])[1] / score(tget, p, *tb[p])[1] - 1 for p in pls)       # 選んだゲインの真値整定の悪化
        osv = max(score(tget, p, *mb[p])[0] for p in pls)                                         # 選んだゲインの真値 OS 最大
        rows.append((v, dos, dabs, drel, se, ae, match, regret, osv, mb))
    L = [f"真値の最適ゲイン(wn/ζ): " + "、".join(f"{p} kg → {tb[p][0]:g}/{tb[p][1]:g}" for p in pls),
         f"評価ゲイン: wn {sorted({g[0] for g in gains})}、ζ {sorted({g[1] for g in gains})}（{len(gains)} 点）\n",
         f"| 倍率 | ΔOS 最大[pt] | Δ整定 最大[ms] | Δ整定 相対最大 | ステップ位置誤差 最大[deg] | 全区間 q 誤差 最大[deg] | 最適ゲイン一致 | 選択ゲインの整定悪化 | 選択ゲインの真値OS最大[%] | 水準 G |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for v, dos, dabs, drel, se, ae, match, regret, osv, mb in rows:
        ok = dos <= G_DOS and (drel <= G_DST_REL or dabs <= G_DST_ABS) and se <= G_POS and match == len(pls)
        L.append(f"| {v} | {dos:.2f} | {dabs*1e3:.1f} | {drel:.1%} | {se:.3f} | {ae:.3f} | {match}/{len(pls)} | {regret:.1%} | {osv:.2f} | {'○' if ok else '×'} |")
    open(f"{a.out}/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
    names = [r[0] for r in rows]; x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a_, i, t, lim in ((ax[0], 1, "|ΔOS| 最大 [pt]", G_DOS), (ax[1], 3, "Δ整定 相対最大", G_DST_REL), (ax[2], 4, "ステップ位置誤差 最大 [deg]", G_POS)):
        a_.bar(x, [r[i] for r in rows], color="tab:blue"); a_.axhline(lim, color="r", ls="--", label="水準 G")
        a_.set_xticks(x); a_.set_xticklabels(names, rotation=60, fontsize=8); a_.set_title(t); a_.grid(alpha=.3, axis="y"); a_.set_yscale("log")
    ax[0].legend(); fig.suptitle("2 慣性解析のパラメータ感度（弾性真値、ペイロード 0/8 kg、wn 60–150）"); fig.tight_layout(); fig.savefig(f"{a.out}/sens.png", dpi=110)


main()
