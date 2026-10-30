"""eval_scen_el.py の結果の集計（issue #79）。使い方:
  ~/.venvs/sa-train/bin/python python/summarize_scen.py --data data/eval/scen79 --out reports/issue-79"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval/scen79")
    ap.add_argument("--out", default="reports/issue-79")
    a = ap.parse_args()
    R, meta = {"closed": {}, "excite": {}}, {}
    for f in sorted(glob.glob(f"{a.data}/scen_*of*_both.json")):
        d = json.load(open(f)); meta.update(d["meta"])
        for k in R: R[k].update(d.get(k, {}))
    os.makedirs(a.out, exist_ok=True)
    models = ["2慣性解析", "A_el"]
    L = ["## 1. 閉ループ（PTP・保持、全長）: 軸最大の誤差\n", "q 誤差 = リンク角の真値との差の最大 [°]（RMS の軸最大も併記）。θ 誤差 = ねじれ角 θ=qm−q の差の最大 [mrad]。θ 真値最大は参考。\n",
         "| シナリオ | 種別 | 継続[s] | " + " | ".join(f"{m} q最大[°] | {m} q RMS[°] | {m} θ誤差[mrad]" for m in models) + " | θ真値最大[mrad] |", "|---|---|---|" + "---|---|---|" * len(models) + "---|"]
    cn = sorted(R["closed"], key=lambda n: (meta[n]["type"], n))
    for n in cn:
        v = R["closed"][n]
        dur = None
        row = f"| {n} | {meta[n]['type']} | - | "
        for m in models:
            r = v.get(m)
            row += (f"{max(r['q_err_max_deg']):.3g} | {max(r['q_err_rms_deg']):.3g} | {max(r['theta_err_max_mrad']):.3g} | " if r else "- | - | - | ")
        ref = next(iter(v.values()))
        L.append(row + f"{max(ref['theta_true_max_mrad']):.1f} |")
    L += ["\n## 2. 励振（開ループ再生）: 真値状態から K ステップ進めた誤差（4 つの開始点、軸最大）\n",
          "| シナリオ | θ真値最大[mrad] | " + " | ".join(f"{m} K=100 q[°] | {m} K=100 θ[mrad] | {m} K=1000 q[°] | {m} K=1000 θ[mrad]" for m in models) + " |", "|---|---|" + "---|---|---|---|" * len(models)]
    en = sorted(R["excite"])
    for n in en:
        v = R["excite"][n]; ref = next(iter(v.values()))
        row = f"| {n} | {max(ref['1000']['theta_true_max_mrad']):.0f} | "
        for m in models:
            r = v.get(m)
            row += (" | ".join(f"{max(r[K][key]):.3g}" for K in ("100", "1000") for key in ("q_err_max_deg", "theta_err_max_mrad")) + " | ") if r else "- | - | - | - | "
        L.append(row)
    open(f"{a.out}/summary.md", "w").write("\n".join(L) + "\n")
    # 図: 励振 K=1000 と 閉ループの q 誤差（対数）
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for j, (title, names, getter) in enumerate((("閉ループ（全長）の q 誤差 最大[°]", cn, lambda n, m: max(R["closed"][n][m]["q_err_max_deg"])),
                                                 ("励振の開ループ再生（K=1000）の q 誤差 最大[°]", en, lambda n, m: max(R["excite"][n][m]["1000"]["q_err_max_deg"])))):
        x = np.arange(len(names)); w = 0.38
        for i, m in enumerate(models):
            vals = [getter(n, m) if m in (R["closed"] if j == 0 else R["excite"]).get(n, {}) else np.nan for n in names]
            ax[j].bar(x + (i - 0.5) * w, np.maximum(vals, 1e-5), w, label=m)
        ax[j].set_yscale("log"); ax[j].set_xticks(x); ax[j].set_xticklabels(names, rotation=70, ha="right", fontsize=8); ax[j].set_title(title); ax[j].set_ylim(top=ax[j].get_ylim()[1] * 6); ax[j].legend(ncol=2, loc="upper left"); ax[j].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{a.out}/scen_err.png", dpi=130)
    print("\n".join(L))


if __name__ == "__main__":
    main()
