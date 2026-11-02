"""Phase 5-14: リンク側の角度なし（モータ側だけ）での同定（issue #115）。X-bench（39 回）と P-bench v1（48 回）、q̂ の再構成誤差、静的なばねの誤差。"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
P02 = 0.02
CATS = [("速度 0.1", "w60", "xs_v10_"), ("速度 1.0", "w60", "xs_v100_"), ("ペイロード 3 kg", "w60", "xp_pl3"), ("ペイロード 6 kg", "w60", "xp_pl6"),
        ("wn 40", "wn40", "xg"), ("wn 90", "wn90", "xg"), ("姿勢（外）", "w60", "xo")]


def errs(d, key, prefix=""):
    R = json.load(open(f"{d}/ptp_pos.json")); k = [x for x in R[0] if isinstance(R[0][x], dict) and x.startswith(key)][0]
    return np.array([r[k]["end_mm"] for r in R if r["scenario"].startswith(prefix)])
X13, X15, I15 = "data/eval/x113", "data/eval/x115", "data/eval/id115"
# (名前, 結果の頭, q̂ の再構成の記録（None は q を測る）)
CONDS = [
    ("q を測る（#113 の基本）", f"{X13}/c3_", None),
    ("q̂: 真値の式で再構成（上限）", f"{X15}/q_truth_", "qhat_truth"),
    ("q̂: 出力誤差法", f"{X15}/q_oe_", "qhat_oe"),
    ("　切り分け A: 質量を真値に", f"{X15}/q_A_s1_", "qhat_A_s1"),
    ("　切り分け B: A ＋ 軸 4・5 のばねを真値に", f"{X15}/q_B_s1w_", "qhat_B_s1w"),
    ("　切り分け C: ばね・質量をすべて真値に", f"{X15}/q_C_", "qhat_C"),
    ("　切り分け D: 摩擦だけ真値に", f"{X15}/q_D_", "qhat_D"),
    ("**q̂: 出力誤差法 ＋ 摩擦の再推定**", f"{X15}/q_rf2_", "qhat_rf2"),
]
EL = json.load(open("data/export/elastic20.json")); K0 = np.ravel(EL["k"])[:6]


def theta_static(tau, k, k3):
    t = tau / k
    for _ in range(50): t = t - (k * t + k3 * t ** 3 - tau) / (k + 3 * k3 * t ** 2)
    return t


def main(out="reports/issue-115"):
    os.makedirs(out, exist_ok=True)
    f = lambda v: f"**{v:.4f}**" if v > P02 else f"{v:.4f}"
    L = ["④ ハイブリッド（#109 の推奨構成。測定は #113 の基本 = 1 kHz・量子化・トルクのノイズ 0.5%）の停留の終わりの先端誤差の**最大** [mm]。太字 = 0.02 mm 超。"
         "q̂ の行は、q を測らずに再構成した q̂ で ② を同定し、残差 NN を学習したもの。\n",
         "| 条件 | " + " | ".join(c for c, *_ in CATS) + " | X-bench ≤ 0.02 mm | P-bench 中央値 / 最大 | P-bench ≤ 0.02 mm |",
         "|---|" + "---|" * (len(CATS) + 3)]
    res = {}
    for nm, pf, qh in CONDS:
        e = {c: errs(pf + ds, "④", px) for c, ds, px in CATS}; pb = errs(pf + "pb", "④"); allx = np.concatenate(list(e.values()))
        res[nm] = pb
        L.append(f"| {nm} | " + " | ".join(f(e[c].max()) for c, *_ in CATS) + f" | {np.sum(allx <= P02)}/{len(allx)} | {np.median(pb):.4f} / {f(pb.max())} | {np.sum(pb <= P02)}/{len(pb)} |")
    L += ["\nq̂ の再構成誤差（θ̂ − θ の RMS、ペイロード 4 kg を含む全シナリオ、始めの 0.5 s を除く） [rad]:\n",
          "| 再構成 | 軸 1 | 軸 2 | 軸 3 | 軸 4 | 軸 5 | 軸 6 |", "|---|---|---|---|---|---|---|"]
    qe = {}
    for nm, _, qh in CONDS:
        if qh is None: continue
        R = json.load(open(f"{I15}/{qh}.json")); E = np.array([v["err_rms"] for v in R.values()]); qe[nm] = np.sqrt((E ** 2).mean(0))
        L.append(f"| {nm} | " + " | ".join(f"{v:.1e}" for v in qe[nm]) + " |")
    Th = np.array([v["th_rms"] for v in R.values()]); L.append("| （参考）θ の RMS | " + " | ".join(f"{v:.1e}" for v in np.sqrt((Th ** 2).mean(0))) + " |")
    # 出力誤差法のパラメータ
    js = json.load(open(f"{I15}/oe_c3.json")); sc = js["scale"]; k = K0 * np.array(sc["k"]); k3 = np.array(sc["k3"])
    L += ["\n出力誤差法の推定値（真値に対する比。初期値は k 0.9・c 1.5・質量 0.97・摩擦 0.8 倍）:\n",
          "| 量 | 軸 1 | 軸 2 | 軸 3 | 軸 4 | 軸 5 | 軸 6 |", "|---|---|---|---|---|---|---|"]
    L.append("| k | " + " | ".join(f"{v:.4f}" for v in sc["k"]) + " |")
    L.append("| k3（真値 k·0.3/θr² に対する比） | " + " | ".join(f"{v / (k0 * 0.3 / 0.005 ** 2):.3f}" for v, k0 in zip(k3, K0)) + " |")
    L.append("| c | " + " | ".join(f"{v:.3f}" for v in sc["c"]) + " |")
    L.append("| Jm | " + " | ".join(f"{v:.4f}" for v in sc["Jm"]) + " |")
    for th in (0.0015, 0.003):
        tau = K0 * th * (1 + 0.3 * (th / 0.005) ** 2); e = theta_static(tau, k, k3) - th
        L.append(f"| 静止時の θ の誤差（θ = {th} rad） | " + " | ".join(f"{100 * v / th:+.2f}%" for v in e) + " |")
    L.append(f"\n質量の倍率 {sc['mass']:.4f}（真値 1）。損失 {js['cost']:.2f}（真値のパラメータで約 33）。")
    open(f"{out}/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
    names = [c[0] for c in CONDS]
    ax[0].boxplot([np.maximum(res[n], 1e-4) for n in names][::-1], vert=False, tick_labels=[n.replace("*", "").strip() for n in names][::-1], whis=(0, 100))
    ax[0].axvline(P02, color="r", ls="--", label="水準 P02"); ax[0].set_xscale("log"); ax[0].tick_params(axis="y", labelsize=8); ax[0].legend(fontsize=8)
    ax[0].set_xlabel("④ の P-bench の先端誤差 [mm]（48 回）"); ax[0].set_title("q を測らない同定（P-bench）"); ax[0].grid(alpha=.3, which="both", axis="x")
    for j, nm in enumerate(["q̂: 真値の式で再構成（上限）", "q̂: 出力誤差法", "　切り分け D: 摩擦だけ真値に", "**q̂: 出力誤差法 ＋ 摩擦の再推定**"]):
        ax[1].semilogy(np.arange(1, 7), qe[nm], "o-", label=nm.replace("*", "").strip())
    ax[1].axhline(1.2e-5, color="r", ls="--", label="P02 の目安（静止時、1 軸）"); ax[1].set_xlabel("軸"); ax[1].set_ylabel("θ̂ の誤差 RMS [rad]（動いている間を含む）")
    ax[1].set_title("q̂ の再構成誤差"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(f"{out}/motoronly.png", dpi=110)


main()
