"""Phase 5-13: 測定条件を実機に近づけたときの ② / ④ の劣化（issue #113）。X-bench（39 回）と P-bench v1（48 回）。"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
P02 = 0.02
X9, X13 = "data/eval/x109", "data/eval/x113"
B = "1 kHz・角度だけ・量子化・トルクのノイズ 0.5%"
# (名前, 結果の頭, ② の結果があるか, ゲイン誤差 [%]（図の横軸。None は図に出さない）, 校正)
CONDS = [
    ("理想（8 kHz・全状態・ノイズなし、#109）", f"{X9}/split_pl4b_w16_", False, None, False),
    ("1 kHz・角度だけ", f"{X13}/c1_", True, None, False),
    (B, f"{X13}/c3_", True, 0.0, False),
    (f"{B}＋ゲイン誤差 0.5%", f"{X13}/g05_", True, 0.5, False),
    (f"{B}＋ゲイン誤差 1%", f"{X13}/g10_", True, 1.0, False),
    (f"{B}＋ゲイン誤差 2%", f"{X13}/g20_", True, 2.0, False),
    (f"{B}＋軸ごとのゲイン誤差 ±2%", f"{X13}/r20_", True, None, False),
    (f"{B}＋ゲイン誤差 2%、校正あり", f"{X13}/g20cal_", True, 2.0, True),
    (f"{B}＋軸ごとのゲイン誤差 ±2%、校正あり", f"{X13}/r20cal_", True, None, True),
]
CATS = [("速度 0.1", "w60", "xs_v10_"), ("速度 1.0", "w60", "xs_v100_"), ("ペイロード 3 kg", "w60", "xp_pl3"), ("ペイロード 6 kg", "w60", "xp_pl6"),
        ("wn 40", "wn40", "xg"), ("wn 90", "wn90", "xg"), ("姿勢（外）", "w60", "xo")]


def errs(d, key, prefix=""):
    R = json.load(open(f"{d}/ptp_pos.json")); k = [x for x in R[0] if isinstance(R[0][x], dict) and x.startswith(key)][0]
    return np.array([r[k]["end_mm"] for r in R if r["scenario"].startswith(prefix)])


def main(out="reports/issue-113"):
    os.makedirs(out, exist_ok=True)
    f = lambda v: f"**{v:.4f}**" if v > P02 else f"{v:.4f}"
    L = ["④ ハイブリッド（#109 の推奨構成を各測定条件で同定・学習し直す。sub2・M 凍結）の停留の終わりの先端誤差の**最大** [mm]。太字 = 0.02 mm 超。② は同じ測定での同定（sub4・M 凍結）。\n",
         "| 測定条件 | " + " | ".join(c for c, *_ in CATS) + " | ④ X-bench ≤ 0.02 mm | ④ P-bench 中央値 / 最大 | ④ P-bench ≤ 0.02 mm | ② P-bench 中央値 / 最大 |",
         "|---|" + "---|" * (len(CATS) + 4)]
    res = {}
    for nm, pf, has2, g, cal in CONDS:
        e = {c: errs(pf + ds, "④", px) for c, ds, px in CATS}; pb = errs(pf + "pb", "④"); allx = np.concatenate(list(e.values()))
        p2 = errs(pf + "pb", "2慣性 同定") if has2 else None; res[nm] = (pb, allx, p2)
        L.append(f"| {nm} | " + " | ".join(f(e[c].max()) for c, *_ in CATS) + f" | {np.sum(allx <= P02)}/{len(allx)} | {np.median(pb):.4f} / {f(pb.max())} | {np.sum(pb <= P02)}/{len(pb)} | "
                 + ("—" if p2 is None else f"{np.median(p2):.3f} / {p2.max():.3f}") + " |")
    L.append("\n測定の模擬（`identify_el.py --meas`）: 8 kHz の真値を 1 kHz に間引き、角度（q, qm）と τ だけを測る。速度・加速度は角度の中心差分（平滑化なし）。"
             "量子化 4.8e-7 rad、トルクのノイズは定格の 0.5%（標準偏差）。ゲイン誤差は測定トルクに 1 + g を掛ける（全軸同じ、または軸ごとに ±2% の一様乱数: "
             "+0.61, −1.54, +1.80, −0.07, +1.49, −1.15%）。校正 = 質量既知のペイロード 4 kg の運転から軸ごとのゲインを推定して τ を割る（`calib_gain.py`、軸 6 はペイロードのトルクが 0 で校正できず補正なし）。")
    open(f"{out}/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
    for cal, st, lab in ((False, "o-", "校正なし"), (True, "s", "校正あり")):
        pts = [(g, res[nm][0]) for nm, _, _, g, c in CONDS if g is not None and c == cal]
        if cal: pts = [(0.0, res[CONDS[2][0]][0])] + pts
        gs = [p[0] for p in pts]
        ax[0].plot(gs, [np.median(p[1]) for p in pts], st, color="C0" if not cal else "C2", label=f"④ 中央値（{lab}）")
        ax[0].plot(gs, [p[1].max() for p in pts], st, color="C1" if not cal else "C3", mfc="none", label=f"④ 最大（{lab}）")
    ax[0].axhline(P02, color="r", ls="--", label="水準 P02"); ax[0].set_xlabel("トルクのゲイン誤差（全軸同じ） [%]"); ax[0].set_ylabel("P-bench の先端誤差 [mm]")
    ax[0].set_title("トルクのゲイン誤差と位置決め誤差（P-bench 48 回）"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    names = [c[0] for c in CONDS]
    ax[1].boxplot([np.maximum(res[n][0], 1e-4) for n in names], vert=False, tick_labels=[n.replace(B, "基本").replace("（8 kHz・全状態・ノイズなし、#109）", "") for n in names], whis=(0, 100))
    ax[1].axvline(P02, color="r", ls="--"); ax[1].set_xscale("log"); ax[1].tick_params(axis="y", labelsize=8)
    ax[1].set_xlabel("④ の P-bench の先端誤差 [mm]（48 回）"); ax[1].set_title("測定条件ごと（基本 = 1 kHz・角度だけ・量子化・ノイズ 0.5%）"); ax[1].grid(alpha=.3, which="both", axis="x")
    fig.tight_layout(); fig.savefig(f"{out}/meas.png", dpi=110)


main()
