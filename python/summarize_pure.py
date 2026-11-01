"""Phase 5-12: ③ 純 NN との比較（issue #111）。X-bench（#107、39 回）と P-bench v1（48 回）の位置決め誤差、静的なつり合いの誤差、計算時間。"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.twomass_nb import TwoMassNB
from sa_nsm.pure_nb import PureNB

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
P02 = 0.02
ID0 = "data/eval/id101/ident_f1.0.json"
X7, X9, X11 = "data/eval/x107", "data/eval/x109", "data/eval/x111"
PARTS = ("w60", "wn40", "wn90")
# (名前, {部分: (結果ディレクトリ, 列名の頭)}, P-bench (ディレクトリ, 列名の頭), 静的 json, 計算時間の作り方)
x7 = lambda key: {p: (f"{X7}/{p}_a", key) for p in PARTS}
x11 = lambda n: {p: (f"{X11}/{n}_s2_{p}", "③") for p in PARTS}
MODELS = [
    ("① 公称（解析）", x7("2慣性 numba"), ("data/eval/ptp99/nom4_1", "2慣性"), None, ("tm", None, None, 4)),
    ("② 同定（解析）", x7("2慣性 同定"), ("data/eval/ptp101/f1.0", "2慣性 同定"), None, ("tm", ID0, None, 4)),
    ("③ 純 NN 幅 128", x11("w128"), (f"{X11}/w128_s2_pb", "③"), f"{X11}/static.json:③ w128", ("pure", "data/models/pure111/w128/pure.npz")),
    ("③ 純 NN 幅 128・低速の重み 4", x11("w128_w4"), (f"{X11}/w128_w4_s2_pb", "③"), f"{X11}/static.json:③ w128_w4", ("pure", "data/models/pure111/w128_w4/pure.npz")),
    ("③ 純 NN 幅 128・データ半分", None, (f"{X11}/w128_f05_s2_pb", "③"), f"{X11}/static_f05.json:③ w128_f05", ("pure", "data/models/pure111/w128_f05/pure.npz")),
    ("③ 純 NN 幅 256", x11("w256"), (f"{X11}/w256_s2_pb", "③"), f"{X11}/static_w256.json:③ w256", ("pure", "data/models/pure111/w256/pure.npz")),
    ("**④ ハイブリッド（#109 推奨）**", {p: (f"{X9}/split_pl4b_w16_{p}", "④") for p in PARTS}, (f"{X9}/split_pl4b_w16_pb", "④"),
     f"{X11}/static.json:④ ens_split_pl4b_w16", ("tm", ID0, "data/models/res109/ens_split_pl4b_w16/residual.npz", 2)),
]


def errs(d, key):
    R = json.load(open(f"{d}/ptp_pos.json")); k = [x for x in R[0] if isinstance(R[0][x], dict) and x.startswith(key)][0]
    return np.array([r[k]["end_mm"] for r in R])


def ms_per_step(spec):
    if spec[0] == "pure":
        m = PureNB(spec[1], substeps=2)
    else:
        _, idf, npz, sub = spec
        sc = {k: np.asarray(v, float) for k, v in json.load(open(idf))["scale"].items()} if idf else None
        m = TwoMassNB("data/export/c8_params.json", "data/export/elastic20.json", substeps=sub, frozen=1, scale=sc, residual=npz)
    y = np.zeros(24); y[:6] = 0.3; y[12:18] = 0.3; t = np.zeros(6)
    for _ in range(20): y = m.step_np(y, t)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(3000): y = m.step_np(y, t)
        best = min(best, (time.perf_counter() - t0) / 3000 * 1e3)
    return best


def main(out="reports/issue-111"):
    os.makedirs(out, exist_ok=True)
    L = ["停留の終わりの先端誤差 [mm]（中央値 / 最大）。X-bench（#107、39 回: 速度・ペイロード・ゲイン・姿勢）と P-bench v1（48 回）。"
         "静的なつり合い = X-bench の停留 27 回（ペイロード 0・3・6 kg）の真値の状態でモデルのリンク側加速度が 0 になる θ のずれによる先端誤差（`diag_pure.py`）。\n",
         "| モデル | X-bench 中央値 / 最大 | X-bench ≤ 0.02 mm | P-bench 中央値 / 最大 | P-bench ≤ 0.02 mm | 静的なつり合いの誤差 中央値 / 最大 | 1 ms あたり計算時間[ms] |",
         "|---|---|---|---|---|---|---|"]
    res = {}
    for nm, X, PB, st, sp in MODELS:
        x = np.concatenate([errs(*X[p]) for p in PARTS]) if X else None
        pb = errs(*PB)
        s = None
        if st:
            f, key = st.split(":"); s = np.array(json.load(open(f))[key]["tip_mm"])
        t = ms_per_step(sp); res[nm] = (x, pb, s, t)
        fm = lambda v: "—" if v is None else f"{np.median(v):.4g} / {v.max():.4g}"
        cnt = lambda v: "—" if v is None else f"{np.sum(v <= P02)}/{len(v)}"
        L.append(f"| {nm} | {fm(x)} | {cnt(x)} | {fm(pb)} | {cnt(pb)} | {fm(s)} | {t:.3f} |")
    L.append("\n③ 純 NN: 物理の式を使わず [sin q, cos q, dq, θ, dθ, τ, ペイロード] → [ddq, ddqm] の 6 軸 MLP（2 層 tanh）。積分は ④ と同じ ROS2 sub2、ヤコビアンは NN の微分（1 ms に 1 回）。"
             "学習データは ④ と同じ（ペイロード 0 kg と 4 kg、`fit_pure_el.py`）。① ② は sub4・M 凍結、④ は sub2・M 凍結。計算時間は numba、逐次 B=1、1 スレッド、3 回の最小。")
    open(f"{out}/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

    fig, ax = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.6, 1]})
    names = [m[0] for m in MODELS]
    for j, nm in enumerate(names):
        x, pb, s, _ = res[nm]
        for k, (v, mk, lab) in enumerate(((x, "o", "X-bench"), (pb, "s", "P-bench"))):
            if v is None: continue
            ax[0].scatter(np.full(len(v), j + (k - 0.5) * 0.25), np.maximum(v, 1e-4), s=10, marker=mk, color=f"C{k}", alpha=0.6, label=lab if j == 0 else None)
    ax[0].axhline(P02, color="r", ls="--", label="水準 P02"); ax[0].set_yscale("log"); ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels([n.strip("*").replace("（", "\n（").replace("・", "\n") for n in names], fontsize=8)
    ax[0].set_ylabel("停留の終わりの先端誤差 [mm]"); ax[0].set_title("位置決め誤差（閉ループ）"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which="both", axis="y")
    sv = [(nm, res[nm][2]) for nm in names if res[nm][2] is not None]
    ax[1].boxplot([np.maximum(v, 1e-4) for _, v in sv], vert=False, tick_labels=[n.strip("*") for n, _ in sv], whis=(0, 100))
    ax[1].axvline(P02, color="r", ls="--"); ax[1].set_xscale("log"); ax[1].tick_params(axis="y", labelsize=8)
    ax[1].set_xlabel("静的なつり合いの先端誤差 [mm]（X-bench の停留 27 回）"); ax[1].set_title("止まる位置の誤り（加速度 0 となる θ のずれ）"); ax[1].grid(alpha=.3, which="both", axis="x")
    fig.tight_layout(); fig.savefig(f"{out}/pure.png", dpi=110)


main()
