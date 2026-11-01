"""Phase 5-11: 残差モデルの外挿の改善（issue #109）。X-bench（#107）の区分ごとの最大と P-bench、計算時間、摩擦 NN の θ 依存。"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.twomass_nb import TwoMassNB, _resid

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
P02 = 0.02
ID0, IDC, IDP = "data/eval/id101/ident_f1.0.json", "data/eval/id109/ident_cubic.json", "data/eval/id109/ident_pl4.json"
X9 = "data/eval/x109"
# (名前, 同定, 残差 npz, {区分: 結果 json}, P-bench json)
def x9(n):
    return {"w60": f"{X9}/{n}_w60", "wn40": f"{X9}/{n}_wn40", "wn90": f"{X9}/{n}_wn90", "pb": f"{X9}/{n}_pb"}
VARS = [
    ("#105（1 つの NN）", ID0, "data/models/res105/ens_w4", {"w60": "data/eval/x107/w60_b", "wn40": "data/eval/x107/wn40_b", "wn90": "data/eval/x107/wn90_b", "pb": "data/eval/ptp105/ens_w4_sub2"}),
    ("θ³ を同定", IDC, "data/models/res109/ens_cubic", {"w60": f"{X9}/cubic_w60", "wn40": f"{X9}/cubic_wn40", "wn90": f"{X9}/cubic_wn90", "pb": f"{X9}/cubic_pb"}),
    ("NN を分ける", ID0, "data/models/res109/ens_split", x9("split")),
    ("NN を分ける＋θ³ を同定（楽観的）", IDC, "data/models/res109/ens_split_cubic", x9("split_cubic")),
    ("NN を分ける＋4 kg データ（同定もやり直す）", IDP, "data/models/res109/ens_split_pl4", x9("split_pl4")),
    ("NN を分ける＋4 kg データ（残差だけ）", ID0, "data/models/res109/ens_split_pl4b", x9("split_pl4b")),
    ("**NN を分ける＋4 kg データ（残差だけ）、幅 16**", ID0, "data/models/res109/ens_split_pl4b_w16", x9("split_pl4b_w16")),
]
CATS = [("速度 0.1", "w60", "xs_v10_"), ("速度 1.0", "w60", "xs_v100_"), ("ペイロード 3 kg", "w60", "xp_pl3"), ("ペイロード 6 kg", "w60", "xp_pl6"),
        ("wn 40", "wn40", "xg"), ("wn 90", "wn90", "xg"), ("姿勢（外）", "w60", "xo")]


def errs(d, prefix=""):
    R = json.load(open(f"{d}/ptp_pos.json")); k = [x for x in R[0] if isinstance(R[0][x], dict)][0]
    return np.array([r[k]["end_mm"] for r in R if r["scenario"].startswith(prefix)])


def ms_per_step(ident, npz):
    sc = {k: np.asarray(v, float) for k, v in json.load(open(ident))["scale"].items()}
    m = TwoMassNB("data/export/c8_params.json", "data/export/elastic20.json", substeps=2, frozen=1, scale=sc, residual=f"{npz}/residual.npz")
    y = np.zeros(24); y[:6] = 0.3; y[12:18] = 0.3; t = np.zeros(6)
    for _ in range(20): y = m.step_np(y, t)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(3000): y = m.step_np(y, t)
        best = min(best, (time.perf_counter() - t0) / 3000 * 1e3)
    return best, m


def main(out="reports/issue-109"):
    os.makedirs(out, exist_ok=True)
    L = ["④ ハイブリッド（3 つの種のアンサンブル、sub2・M 凍結）の停留の終わりの先端誤差の**最大** [mm]。X-bench（#107、39 回）の区分ごとと P-bench v1（48 回）。太字なし = ≤ 0.02 mm。\n",
         "| ④ の残差モデル | " + " | ".join(c for c, *_ in CATS) + " | X-bench ≤ 0.02 mm | P-bench 最大 | 1 ms あたり計算時間[ms] |", "|---|" + "---|" * (len(CATS) + 3)]
    res = {}
    for nm, idf, npz, D in VARS:
        e = {c: errs(D[ds], pf) for c, ds, pf in CATS}; pb = errs(D["pb"]); sp, m = ms_per_step(idf, npz)
        allx = np.concatenate(list(e.values())); res[nm] = (e, pb, sp, m)
        f = lambda v: f"**{v:.4f}**" if v > P02 else f"{v:.4f}"
        L.append(f"| {nm} | " + " | ".join(f(e[c].max()) for c, *_ in CATS) + f" | {np.sum(allx <= P02)}/{len(allx)} | {f(pb.max())} | {sp:.3f} |")
    L.append("\n4 kg データ = ペイロード 4 kg の同定用 PTP 4 本（速度 0.3〜0.9）を学習に足す。X-bench のペイロードは 3 kg（内挿）と 6 kg（外挿）。"
             "「同定もやり直す」は ② の同定にも 4 kg データを使う、「残差だけ」は ② はペイロードなしの同定のまま、残差 NN の学習にだけ使う。")
    L.append("θ³ を同定: モータ側の式に k3·θ³ を足す。真値のばねがちょうど 3 次式なので k3 は真値と一致し（比 0.999〜1.000）、楽観的。")
    open(f"{out}/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

    fig, ax = plt.subplots(1, 2, figsize=(17, 5.6), gridspec_kw={"width_ratios": [2.2, 1]})
    names = [v[0].strip("*") for v in VARS]; w = 0.8 / len(VARS); x = np.arange(len(CATS) + 1)
    for j, nm in enumerate([v[0] for v in VARS]):
        e, pb, *_ = res[nm]
        ax[0].bar(x + (j - len(VARS) / 2 + 0.5) * w, [e[c].max() for c, *_ in CATS] + [pb.max()], w, label=names[j], color=f"C{j}")
    ax[0].axhline(P02, color="r", ls="--", label="水準 P02"); ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels([c for c, *_ in CATS] + ["P-bench"], fontsize=9)
    ax[0].set_ylabel("先端誤差の最大 [mm]"); ax[0].set_title("区分ごとの最大"); ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=.3, which="both", axis="y")
    # 摩擦の補正 Δf の θ 依存（軸 5、dq = dθ = 0）。物理的には θ に依らないはず
    th = np.linspace(-9e-3, 9e-3, 181)
    for nm, st in ((VARS[0][0], "C0-"), (VARS[2][0], "C2-"), (VARS[6][0], "C6-")):
        m = res[nm][3]; y = []
        for t in th:
            v = np.zeros(6); v[4] = t; z = np.zeros(6)
            y.append(_resid(v, z, z, m.R)[1][4])
        ax[1].plot(th * 1e3, y, st, label=nm.strip("*"))
    ax[1].axvspan(-4.07, 4.07, color="0.9", label="学習の範囲（軸 5、ペイロード 0）")
    ax[1].set_xlabel("軸 5 のねじれ角 θ [mrad]"); ax[1].set_ylabel("摩擦の補正 Δf [N·m]（dq = 0）"); ax[1].set_title("摩擦 NN の θ 依存（本来は 0）"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}/spring.png", dpi=110)


main()
