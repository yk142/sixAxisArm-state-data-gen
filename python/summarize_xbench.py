"""Phase 5-10: 外挿の評価（X-bench、issue #107）。区分（速度・ペイロード・ゲイン・姿勢）× モデルの位置決め誤差と、ねじれ角の範囲（学習 vs X-bench）。"""
import json, os
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
P02 = 0.02
E = "data/eval/x107"
MODELS = [("① 公称", "a", "2慣性 numba"), ("② 同定", "a", "2慣性 同定"), ("④ #101", "a", "④"), ("④ #105（採用）", "b", "④"), ("オラクル（正解の式）", "b", "オラクル")]
CATS = [("速度 0.1", "w60", "xs_v10_"), ("速度 1.0", "w60", "xs_v100_"), ("ペイロード 3 kg", "w60", "xp_pl3"), ("ペイロード 6 kg", "w60", "xp_pl6"),
        ("ゲイン wn 40", "wn40", "xg"), ("ゲイン wn 90", "wn90", "xg"), ("姿勢（学習範囲の外）", "w60", "xo")]
TRAIN = ["data/export/id_elum_h1s.h5", "data/export/id_elum_h2s.h5", "data/export/id_elum_p1.h5", "data/export/id_elum_p2.h5"]


def errs(ds, part, key, prefix):
    R = json.load(open(f"{E}/{ds}_{part}/ptp_pos.json")); k = [x for x in R[0] if isinstance(R[0][x], dict) and x.startswith(key)][0]
    return np.array([r[k]["end_mm"] for r in R if r["scenario"].startswith(prefix)])


def twist_max(files, prefix=""):
    m = np.zeros(6)
    for f in files:
        with h5py.File(f, "r", locking=False) as h:
            for k in h:
                if k == "controller" or not k.startswith(prefix): continue
                m = np.maximum(m, np.abs(h[k]["truth_qm"][:] - h[k]["truth_q"][:]).max(1))
    return m * 1e3


def main(out="reports/issue-107"):
    os.makedirs(out, exist_ok=True)
    E_ = {(c, m): errs(ds, part, key, pf) for c, ds, pf in CATS for m, part, key in MODELS}
    L = ["X-bench（モデル外の効果入り真値、PTP・各到達点 0.2 s 停留）の停留の終わりの先端誤差 [mm]。各セルは 中央値 / 最大（≤ 0.02 mm の割合）。学習は速度 0.3〜0.9・ペイロード 0・wn 60・到達点は学習範囲の内側。\n",
         "| 区分 | 回数 | " + " | ".join(m for m, *_ in MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
    for c, *_ in CATS:
        n = len(E_[(c, MODELS[0][0])])
        L.append(f"| {c} | {n} | " + " | ".join(f"{np.median(E_[(c, m)]):.4f} / {E_[(c, m)].max():.4f}（{np.mean(E_[(c, m)] <= P02):.0%}）" for m, *_ in MODELS) + " |")
    ex = [c for c, *_ in CATS if not c.startswith("ペイロード")]
    L.append("| ペイロード以外の計 | " + str(sum(len(E_[(c, MODELS[0][0])]) for c in ex)) + " | " +
             " | ".join(f"{np.median(np.concatenate([E_[(c, m)] for c in ex])):.4f} / {np.concatenate([E_[(c, m)] for c in ex]).max():.4f}（{np.mean(np.concatenate([E_[(c, m)] for c in ex]) <= P02):.0%}）" for m, *_ in MODELS) + " |")
    files = {"w60": ["data/export/x_speed.h5", "data/export/x_payload.h5", "data/export/x_outer.h5"], "wn40": ["data/export/x_wn40.h5"], "wn90": ["data/export/x_wn90.h5"]}
    tw_tr = twist_max(TRAIN); tw = {c: twist_max(files[ds], pf) for c, ds, pf in CATS}
    L.append("\nねじれ角 θ = qm − q の最大 [mrad]（軸 1〜6）:\n")
    L.append("| データ | 軸 1 | 軸 2 | 軸 3 | 軸 4 | 軸 5 | 軸 6 |"); L.append("|---|---|---|---|---|---|---|")
    L.append("| 学習（同定・残差 NN） | " + " | ".join(f"{v:.2f}" for v in tw_tr) + " |")
    for c, *_ in CATS:
        L.append(f"| {c} | " + " | ".join(f"**{v:.2f}**" if v > tw_tr[i] * 1.05 else f"{v:.2f}" for i, v in enumerate(tw[c])) + " |")
    L.append("\n太字: 学習の最大を 5% 超える（ばねの外挿）。")
    open(f"{out}/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [2, 1]})
    w = 0.16; x = np.arange(len(CATS))
    for j, (m, *_ ) in enumerate(MODELS):
        for i, (c, *_ ) in enumerate(CATS):
            e = np.maximum(E_[(c, m)], 1e-5)
            ax[0].scatter(np.full(len(e), i + (j - 2) * w), e, s=14, color=f"C{j}", label=m if i == 0 else None)
    ax[0].axhline(P02, color="r", ls="--", label="水準 P02"); ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels([c for c, *_ in CATS], fontsize=9)
    ax[0].set_ylabel("停留の終わりの先端誤差 [mm]"); ax[0].set_title("X-bench（学習範囲の外）"); ax[0].legend(fontsize=8, ncol=3); ax[0].grid(alpha=.3, which="both", axis="y")
    for i, (c, *_ ) in enumerate([CATS[2], CATS[3], CATS[1]]):
        ax[1].plot(np.arange(1, 7), tw[c], "o-", label=c)
    ax[1].plot(np.arange(1, 7), tw_tr, "k-", lw=2.5, label="学習（最大）")
    ax[1].set_xlabel("軸"); ax[1].set_ylabel("ねじれ角の最大 [mrad]"); ax[1].set_title("ねじれ角の範囲"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}/xbench.png", dpi=110)


main()
