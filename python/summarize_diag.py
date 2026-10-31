"""Phase 5-8: ④ の残る誤差の切り分け（issue #103）。正解の式（オラクル）と ④ を計算の細かさごとに比べ、④ が学んだ摩擦・ばねの形を正解と並べる。"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.twomass_nb import TwoMassNB, _resid

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
P02 = 0.02
E = "data/eval/ptp103"
IDENT, NPZ = "data/eval/id101/ident_f1.0.json", "data/models/res101/g1.0/residual.npz"


def get(path, key):
    R = json.load(open(path)); k = [x for x in R[0] if isinstance(R[0][x], dict) and x.startswith(key)][0]
    return np.array([r[k]["end_mm"] for r in R])


def ms_per_step(**kw):
    m = TwoMassNB("data/export/c8_params.json", "data/export/elastic20.json", **kw)
    y = np.zeros(24); y[:6] = 0.3; y[12:18] = 0.3; t = np.zeros(6)
    for _ in range(20): y = m.step_np(y, t)
    t0 = time.perf_counter()
    for _ in range(3000): y = m.step_np(y, t)
    return (time.perf_counter() - t0) / 3000 * 1e3


def main(out="reports/issue-103"):
    os.makedirs(out, exist_ok=True)
    sc = {k: np.asarray(v, float) for k, v in json.load(open(IDENT))["scale"].items()}
    O = {"stiffA": 0.3, "stribeck": 0.3}
    rows = [  # (モデル, 細かさ, J, 誤差, 計算時間)
        ("オラクル（正解の式）", "sub2・M 凍結", "傾き ≥ 0", get(f"{E}/or_fz_b/ptp_pos.json", "オラクル（all）sub2"), ms_per_step(substeps=2, frozen=1, oracle=O)),
        ("オラクル（正解の式）", "sub4・M 凍結", "傾き ≥ 0", get(f"{E}/or_fz_a/ptp_pos.json", "オラクル（all）sub4"), ms_per_step(substeps=4, frozen=1, oracle=O)),
        ("オラクル（正解の式）", "sub8・M 凍結", "傾き ≥ 0", get(f"{E}/or_fz_a/ptp_pos.json", "オラクル（all）sub8"), ms_per_step(substeps=8, frozen=1, oracle=O)),
        ("オラクル（正解の式）", "sub16・M 凍結", "傾き ≥ 0", get(f"{E}/or_fz_b/ptp_pos.json", "オラクル（all）sub16"), ms_per_step(substeps=16, frozen=1, oracle=O)),
        ("オラクル（正解の式）", "sub4・M 凍結", "なし（④ と同じ）", get(f"{E}/or_nojac/ptp_pos.json", "オラクル（all、J なし）sub4"), None),
        ("オラクル（正解の式）", "sub16・M 凍結", "なし（④ と同じ）", get(f"{E}/or_nojac/ptp_pos.json", "オラクル（all、J なし）sub16"), None),
        ("オラクル（ばねだけ）", "sub4・M 凍結", "あり", get(f"{E}/or_part/ptp_pos.json", "オラクル（spring）"), None),
        ("オラクル（Stribeck だけ）", "sub4・M 凍結", "あり", get(f"{E}/or_part/ptp_pos.json", "オラクル（stribeck）"), None),
        ("④ ハイブリッド（#101）", "sub4・M 凍結", "なし", get("data/eval/ptp101/f1.0/ptp_pos.json", "④"), ms_per_step(substeps=4, frozen=1, scale=sc, residual=NPZ)),
        ("④ ハイブリッド", "sub8", "なし", get(f"{E}/hy_sub/ptp_pos.json", "④ ハイブリッド（g1.0） sub8"), ms_per_step(substeps=8, frozen=0, scale=sc, residual=NPZ)),
        ("④ ハイブリッド", "sub16", "なし", get(f"{E}/hy_sub/ptp_pos.json", "④ ハイブリッド（g1.0） sub16"), None),
    ]
    neg = get(f"{E}/or_fz_Jneg/ptp_pos.json", "オラクル（all）sub8")
    L = ["位置決め 48 回（P-bench v1、モデル外の効果入り真値）の停留の終わりの先端誤差。オラクル = 公称パラメータ＋真値に入れた硬化ばね・Stribeck 摩擦の式そのもの。④ は #101 の Bv≥0・データ 100% のモデル。\n",
         "| モデル | 計算の細かさ | 効果の微分のヤコビアン | 中央値[mm] | 最大[mm] | ≤ 0.02 mm の割合 | 1 ms あたり計算時間[ms] | 水準 P02 |", "|---|---|---|---|---|---|---|---|"]
    for nm, d, j, e, sp in rows:
        L.append(f"| {nm} | {d} | {j} | {np.median(e):.4f} | {e.max():.4f} | {np.mean(e <= P02):.0%} | {'—' if sp is None else f'{sp:.3f}'} | {'○' if e.max() <= P02 else '×'} |")
    L.append(f"\n参考: Stribeck の負の傾きもヤコビアンに入れると sub8 で 48 回中 {int(np.isnan(neg).sum())} 回が発散（NaN）した（W = I − γhJ が特異に近づく）。以後は傾き ≥ 0 の部分だけを入れる。")
    L.append("計算時間は numba、逐次 B=1、1 スレッド、このマシンで測った値。")
    open(f"{out}/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

    # ④ が学んだ形と正解の比較（全体の摩擦・ばねトルク。θ・dθ・dq の他の入力は 0）
    P = json.load(open("data/export/c8_params.json")); el = json.load(open("data/export/elastic20.json"))
    f6 = lambda x: np.ravel(x)[:6].astype(float)
    Fc0, Bv0, ep = f6(P["frictionFc"]), f6(P["frictionBv"]), f6(P["frictionEps"]); k0 = f6(el["k"])
    m = TwoMassNB("data/export/c8_params.json", "data/export/elastic20.json", substeps=4, frozen=1, scale=sc, residual=NPZ)
    _, _, _, _, _, _, kI, _, _, BvI, FcI, _ = m.consts()
    axes = (3, 5)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    e_or = rows[1][3]; e_hy = rows[8][3]
    ax[0, 0].boxplot([np.maximum(r[3], 1e-6) for r in rows], vert=False, labels=[f"{r[0]}\n{r[1]}・J {r[2]}" for r in rows], whis=(0, 100))
    ax[0, 0].axvline(P02, color="r", ls="--", label="水準 P02"); ax[0, 0].set_xscale("log"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].tick_params(axis="y", labelsize=6); ax[0, 0].set_xlabel("先端誤差 [mm]（48 回）"); ax[0, 0].set_title("切り分け"); ax[0, 0].grid(alpha=.3, which="both", axis="x")
    ax[1, 0].scatter(e_or, e_hy, s=12); ax[1, 0].axhline(P02, color="r", ls="--"); ax[1, 0].axvline(P02, color="r", ls="--")
    ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log"); ax[1, 0].set_xlabel("オラクル sub4 [mm]"); ax[1, 0].set_ylabel("④ sub4 [mm]"); ax[1, 0].set_title("位置決めごと（同じ刻み）"); ax[1, 0].grid(alpha=.3, which="both")
    for c, i in enumerate(axes, start=1):
        v = np.linspace(-8 * ep[i], 8 * ep[i], 401)
        tru = Bv0[i] * v + Fc0[i] * np.tanh(v / ep[i]) + 0.3 * Fc0[i] * np.exp(-(v / (2 * ep[i])) ** 2) * np.tanh(v / ep[i])
        hyb = np.empty_like(v); idn = BvI[i] * v + FcI[i] * np.tanh(v / ep[i])
        for j, vv in enumerate(v):
            z = np.zeros(6); dq = z.copy(); dq[i] = vv
            hyb[j] = idn[j] + _resid(z, z, dq, m.R)[1][i]
        ax[0, c].plot(v / ep[i], tru, "k-", lw=2, label="正解（真値の式）"); ax[0, c].plot(v / ep[i], idn, "b--", label="② 同定の式"); ax[0, c].plot(v / ep[i], hyb, "r-", label="④ 同定＋NN")
        ax[0, c].set_xlabel("関節速度 / ε"); ax[0, c].set_ylabel("摩擦トルク [N·m]"); ax[0, c].set_title(f"軸 {i+1} の摩擦（θ = dθ = 0）"); ax[0, c].legend(fontsize=8); ax[0, c].grid(alpha=.3)
        th = np.linspace(-0.006, 0.006, 401)
        tru = k0[i] * th * (1 + 0.3 * th ** 2 / 0.005 ** 2); hyb = np.empty_like(th)
        for j, t in enumerate(th):
            z = np.zeros(6); tt = z.copy(); tt[i] = t
            hyb[j] = kI[i] * t + _resid(tt, z, z, m.R)[0][i]
        ax[1, c].plot(th * 1e3, tru - k0[i] * th, "k-", lw=2, label="正解 − 公称の線形分"); ax[1, c].plot(th * 1e3, kI[i] * th - k0[i] * th, "b--", label="② − 公称の線形分")
        ax[1, c].plot(th * 1e3, hyb - k0[i] * th, "r-", label="④ − 公称の線形分")
        ax[1, c].set_xlabel("ねじれ角 θ [mrad]"); ax[1, c].set_ylabel("ばねトルクの非線形分 [N·m]"); ax[1, c].set_title(f"軸 {i+1} のばね（dθ = dq = 0）"); ax[1, c].legend(fontsize=8); ax[1, c].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}/diag.png", dpi=110)


main()
