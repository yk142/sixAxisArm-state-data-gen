"""Phase 4-7: ペイロード×ゲインのステップ応答の集計（Simscape 真値と各モデルの比較）。"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
PL = ["Simscape", "解析（ペイロード反映）", "解析（公称）", "A（調整後）", "B"]
OS_LIM = 5.0   # 許容オーバーシュート [%]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval/payload6/payload.json")
    ap.add_argument("--out", default="reports/issue-61")
    a = ap.parse_args()
    R = json.load(open(a.data)); os.makedirs(a.out, exist_ok=True)
    pls = sorted({r["payload"] for r in R}); wns = sorted({r["wn"] for r in R}); zs = sorted({r["zeta"] for r in R})
    axes = sorted({r["axis"] for r in R})
    G = lambda p, w, z, ax: next(r for r in R if r["payload"] == p and r["wn"] == w and r["zeta"] == z and r["axis"] == ax)
    L = []
    # 1. 真値のメトリクス（軸平均）: ペイロード×ゲイン
    L.append("## 1. Simscape 真値: ステップ応答（2°、軸 1–6 の平均 / 最大）\n")
    L.append("| ペイロード[kg] | wn | ζ | OS 平均[%] | OS 最大[%] | 整定 平均[s] | 整定 最大[s] | 最終ゲイン 最小 |")
    L.append("|---|---|---|---|---|---|---|---|")
    M = {}
    for p in pls:
        for w in wns:
            for z in zs:
                rs = [G(p, w, z, ax)["Simscape"] for ax in axes]
                os_ = np.array([r["overshoot_pct"] for r in rs]); st = np.array([r["settle_s"] for r in rs]); fg = np.array([r["final_gain"] for r in rs])
                M[p, w, z] = (os_.mean(), os_.max(), np.nanmean(st), np.nanmax(st), fg.min())
                L.append(f"| {p} | {w:g} | {z:g} | {os_.mean():.2f} | {os_.max():.2f} | {np.nanmean(st):.3f} | {np.nanmax(st):.3f} | {fg.min():.4f} |")
    # 2. 最適ゲイン（OS 最大 ≤ OS_LIM のうち整定平均が最小）
    L.append(f"\n## 2. 最適ゲイン（OS 最大 ≤ {OS_LIM:g}% の中で整定時間の軸平均が最小）\n")
    L.append("| ペイロード[kg] | 最適 wn | 最適 ζ | 整定 平均[s] | 公称ゲイン(60,1) の整定[s] |")
    L.append("|---|---|---|---|---|")
    best = {}
    for p in pls:
        c = [(M[p, w, z][2], w, z) for w in wns for z in zs if M[p, w, z][1] <= OS_LIM]
        s, w, z = min(c) if c else (np.nan, None, None)
        best[p] = (w, z)
        L.append(f"| {p} | {w} | {z} | {s:.3f} | {M[p, 60, 1.0][2]:.3f} |")
    # 3. 各モデルの真値との差
    L.append("\n## 3. 真値との差（全 72 ケース×6 軸）\n")
    L.append("| モデル | ステップ区間 位置誤差 最大[deg] | 全区間 位置誤差 最大[deg] | |ΔOS| 最大[%] | |Δ整定| 最大[s] |")
    L.append("|---|---|---|---|---|")
    for k in PL[1:]:
        se = max(r[k]["step_err_max_deg"] for r in R); ae = max(r[k]["err_axis_max_deg"] for r in R)
        dos = max(abs(r[k]["overshoot_pct"] - r["Simscape"]["overshoot_pct"]) for r in R)
        dst = np.nanmax([abs(r[k]["settle_s"] - r["Simscape"]["settle_s"]) for r in R])
        L.append(f"| {k} | {se:.2e} | {ae:.2e} | {dos:.2f} | {dst:.3f} |")
    L.append("\n### ペイロード別（全ゲイン・全軸の最大。位置誤差[deg]）\n")
    L.append("| モデル | " + " | ".join(f"{p} kg" for p in pls) + " |"); L.append("|---|" + "---|" * len(pls))
    for k in PL[1:]:
        L.append(f"| {k} | " + " | ".join(f"{max(r[k]['err_axis_max_deg'] for r in R if r['payload'] == p):.2e}" for p in pls) + " |")
    # 4. ブラインド（公称）モデルが予測する最適ゲインとの一致
    L.append("\n## 4. 最適ゲインの一致（ペイロード別）\n")
    L.append("| ペイロード[kg] | 真値 | 解析（反映） | 解析（公称） | A | B |"); L.append("|---|---|---|---|---|---|")
    agree = {k: 0 for k in PL[1:]}
    for p in pls:
        row = [f"{best[p][0]}/{best[p][1]}"]
        for k in PL[1:]:
            sc = {}
            for w in wns:
                for z in zs:
                    rs = [G(p, w, z, ax)[k] for ax in axes]
                    sc[w, z] = (max(r["overshoot_pct"] for r in rs), np.nanmean([r["settle_s"] for r in rs]))
            c = [(v[1], w, z) for (w, z), v in sc.items() if v[0] <= OS_LIM]
            b = min(c)[1:] if c else (None, None)
            agree[k] += (b == best[p]); row.append(f"{b[0]}/{b[1]}")
        L.append(f"| {p} | " + " | ".join(row) + " |")
    L.append("\n一致数（/%d）: " % len(pls) + "、".join(f"{k} {v}" for k, v in agree.items()))
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    # 図
    fig, ax_ = plt.subplots(2, len(wns), figsize=(4.5 * len(wns), 8))
    for j, w in enumerate(wns):
        for i, (nm, idx) in enumerate([("オーバーシュート平均 [%]", 0), ("整定時間 平均 [s]", 2)]):
            for z in zs:
                ax_[i, j].plot(pls, [M[p, w, z][idx] for p in pls], "o-", label=f"ζ={z:g}")
            ax_[i, j].set_title(f"wn={w:g}: {nm}"); ax_[i, j].set_xlabel("ペイロード [kg]"); ax_[i, j].grid(alpha=.3)
    ax_[0, 0].legend(); fig.suptitle("Simscape 真値: ペイロードとゲインによるステップ応答の変化（FF はペイロード込み）")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "payload_gain.png"), dpi=110); plt.close(fig)
    # モデル誤差 vs ペイロード
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in PL[1:]:
        ax.semilogy(pls, [max(r[k]["err_axis_max_deg"] for r in R if r["payload"] == p) for p in pls], "o-", label=k)
    ax.set_xlabel("ペイロード [kg]"); ax.set_ylabel("真値との位置誤差 最大 [deg]"); ax.grid(alpha=.3, which="both"); ax.legend()
    ax.set_title("モデル別: 真値との差 vs ペイロード"); fig.tight_layout(); fig.savefig(os.path.join(a.out, "payload_model_err.png"), dpi=110)


main()
