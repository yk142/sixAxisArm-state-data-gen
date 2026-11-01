"""Phase 5-9: 残差 NN の学習の改善（issue #105）。入力の尺度・低速の重み・種・アンサンブル・刻みの P-bench 比較と、学んだ摩擦の形。"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.twomass_nb import TwoMassNB, _resid

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
P02 = 0.02
E, M = "data/eval/ptp105", "data/models/res105"
IDENT = "data/eval/id101/ident_f1.0.json"


def get(path):
    R = json.load(open(path)); k = [x for x in R[0] if isinstance(R[0][x], dict)][0]
    return np.array([r[k]["end_mm"] for r in R])


def ms_per_step(sub, npz=None, oracle=None):
    sc = {k: np.asarray(v, float) for k, v in json.load(open(IDENT))["scale"].items()} if npz else None
    m = TwoMassNB("data/export/c8_params.json", "data/export/elastic20.json", substeps=sub, frozen=1, scale=sc, residual=npz, oracle=oracle)
    y = np.zeros(24); y[:6] = 0.3; y[12:18] = 0.3; t = np.zeros(6)
    for _ in range(20): y = m.step_np(y, t)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(3000): y = m.step_np(y, t)
        best = min(best, (time.perf_counter() - t0) / 3000 * 1e3)
    return best


def main(out="reports/issue-105"):
    os.makedirs(out, exist_ok=True)
    V = [("#101 と同じ（dq を標準偏差で割る）", "base", 4), ("dq を ε で割る", "eps", 4), ("＋ tanh(dq/ε) 入力", "eps_tanh", 4), ("＋ tanh(dq/ε) 入力 ＋ 低速の重み 4", "eps_tanh_w4", 4)]
    rows = []
    for nm, v, sub in V:
        rows.append((nm, "0", f"sub{sub}", get(f"{E}/{v}/ptp_pos.json"), ms_per_step(sub, f"{M}/{v}/residual.npz")))
    for nm, v in (("dq を ε で割る", "eps"), ("＋ tanh ＋ 低速の重み", "eps_tanh_w4")):
        for s in (1, 2):
            rows.append((nm, str(s), "sub4", get(f"{E}/{v}_s{s}/ptp_pos.json"), None))
    rows.append(("dq を ε で割る（アンサンブル）", "0,1,2 の平均", "sub4", get(f"{E}/ens_eps/ptp_pos.json"), ms_per_step(4, f"{M}/ens_eps/residual.npz")))
    rows.append(("＋ tanh ＋ 低速の重み（アンサンブル）", "0,1,2 の平均", "sub4", get(f"{E}/ens_w4/ptp_pos.json"), ms_per_step(4, f"{M}/ens_w4/residual.npz")))
    rows.append(("＋ tanh ＋ 低速の重み", "0", "sub2", get(f"{E}/eps_tanh_w4_sub2/ptp_pos.json"), ms_per_step(2, f"{M}/eps_tanh_w4/residual.npz")))
    rows.append(("**＋ tanh ＋ 低速の重み（アンサンブル）**", "0,1,2 の平均", "sub2", get(f"{E}/ens_w4_sub2/ptp_pos.json"), ms_per_step(2, f"{M}/ens_w4/residual.npz")))
    R = json.load(open("data/eval/ptp103/or_fz_b/ptp_pos.json")); k = [x for x in R[0] if isinstance(R[0][x], dict) and "sub2" in x][0]
    rows.append(("（参考）オラクル（正解の式、#103）", "—", "sub2", np.array([r[k]["end_mm"] for r in R]), ms_per_step(2, oracle={"stiffA": 0.3, "stribeck": 0.3})))
    L = ["位置決め 48 回（P-bench v1、モデル外の効果入り真値）の停留の終わりの先端誤差。④ = ② 同定（Bv≥0、データ 100%）＋ 残差トルク NN（軸ごと 2×32 tanh）。すべて M 凍結。\n",
         "| ④ の残差 NN | 種 | 計算の細かさ | 中央値[mm] | 最大[mm] | ≤ 0.02 mm の割合 | 1 ms あたり計算時間[ms] | 水準 P02 |", "|---|---|---|---|---|---|---|---|"]
    for nm, s, sub, e, sp in rows:
        L.append(f"| {nm} | {s} | {sub} | {np.median(e):.4f} | {e.max():.4f} | {np.mean(e <= P02):.0%} | {'—' if sp is None else f'{sp:.3f}'} | {'○' if e.max() <= P02 else '×'} |")
    L.append("\n計算時間は numba、逐次 B=1、1 スレッド、このマシンで 3 回測った最小。アンサンブルは 3 つの NN を幅 96 の 1 つの MLP にまとめ、第 2 層のブロックの外を飛ばす（`ensemble_residual.py`）。")
    open(f"{out}/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    ax[0].boxplot([np.maximum(r[3], 1e-6) for r in rows], vert=False, tick_labels=[f"{r[0].strip('*')}\n種 {r[1]}・{r[2]}" for r in rows], whis=(0, 100))
    ax[0].axvline(P02, color="r", ls="--", label="水準 P02"); ax[0].set_xscale("log"); ax[0].legend(fontsize=8); ax[0].tick_params(axis="y", labelsize=6)
    ax[0].set_xlabel("停留の終わりの先端誤差 [mm]（48 回）"); ax[0].set_title("残差 NN の学習の改善"); ax[0].grid(alpha=.3, which="both", axis="x")
    # 学んだ摩擦の形（軸 4・軸 6、θ = dθ = 0 の断面）
    P = json.load(open("data/export/c8_params.json")); f6 = lambda x: np.ravel(x)[:6].astype(float)
    Fc0, Bv0, ep = f6(P["frictionFc"]), f6(P["frictionBv"]), f6(P["frictionEps"])
    sc = {k: np.asarray(v, float) for k, v in json.load(open(IDENT))["scale"].items()}
    for c, i in ((1, 3), (2, 5)):
        v = np.linspace(-8 * ep[i], 8 * ep[i], 401)
        ax[c].plot(v / ep[i], Bv0[i] * v + Fc0[i] * np.tanh(v / ep[i]) + 0.3 * Fc0[i] * np.exp(-(v / (2 * ep[i])) ** 2) * np.tanh(v / ep[i]), "k-", lw=2.5, label="正解（真値の式）")
        for nm, v_, st in (("④ #101（標準偏差で割る）", "base", "b--"), ("④ ε で割る＋tanh＋重み（アンサンブル）", "ens_w4", "r-")):
            m = TwoMassNB("data/export/c8_params.json", "data/export/elastic20.json", substeps=2, frozen=1, scale=sc, residual=f"{M}/{v_}/residual.npz")
            BvI, FcI = m.consts()[9], m.consts()[10]
            y = np.empty_like(v)
            for j, vv in enumerate(v):
                z = np.zeros(6); dq = z.copy(); dq[i] = vv
                y[j] = BvI[i] * vv + FcI[i] * np.tanh(vv / ep[i]) + _resid(z, z, dq, m.R)[1][i]
            ax[c].plot(v / ep[i], y, st, label=nm)
        ax[c].set_xlabel("関節速度 / ε"); ax[c].set_ylabel("摩擦トルク [N·m]"); ax[c].set_title(f"軸 {i+1} の摩擦（θ = dθ = 0 の断面）"); ax[c].legend(fontsize=8); ax[c].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}/resfit.png", dpi=110)


main()
