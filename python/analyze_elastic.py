"""Phase 4-12: 関節弾性つき真値と剛体真値の閉ループステップ応答の差（issue #71）。使い方:
  ~/.venvs/sa-train/bin/python python/analyze_elastic.py --elastic data/export/payload_eval6_el20.h5 --out reports/issue-71
同じ 360 ケース（ペイロード × wn × ζ × 軸）で、リンク側角度 q の OS・整定・残留振動・振動周波数を弾性／剛体の真値で比べる。
モデル（解析・A_pl）が弾性真値からどれだけ外れるかは eval_payload.py の出力（--model-json）で別に集計する。"""
import argparse, json, os
import h5py
import numpy as np

DEG = 180 / np.pi
DT = 1e-3


def metrics(q, qpre, step, t_step, band=0.02, fs=8000):
    t = np.arange(len(q)) / fs
    y = (q - qpre) / step
    m = t >= t_step
    ts, ys = t[m] - t_step, y[m]
    yf = ys[-1]
    over = max(0.0, (ys.max() - yf) / abs(yf)) * 100
    out = np.where(np.abs(ys - yf) > band * abs(yf))[0]
    settle = 0.0 if len(out) == 0 else (float(ts[out[-1] + 1]) if out[-1] + 1 < len(ts) else float("nan"))
    # 振動成分: ステップ後 0.05 s 以降の応答から、50 ms の移動平均（ゆっくりした整定）を引いた高域成分の最大振幅（ステップ量に対する %）と主周波数 [Hz]
    r = ys[ts >= 0.05] - yf
    w = int(0.05 * fs); ker = np.ones(w) / w
    hp = (r - np.convolve(np.pad(r, (w // 2, w - w // 2 - 1), mode="edge"), ker, mode="valid"))[w:-w]
    res = float(np.abs(hp).max() * 100)
    f = 0.0
    if res > 0.05:
        sp = np.abs(np.fft.rfft(hp * np.hanning(len(hp)))); fr = np.fft.rfftfreq(len(hp), 1 / fs)
        f = float(fr[1:][sp[1:].argmax()])
    return {"overshoot_pct": float(over), "settle_s": settle, "resid_pct": res, "freq_hz": f, "final_gain": float(yf)}


def load(path):
    rows = {}
    with h5py.File(path, "r") as f:
        for g in f:
            if g == "controller": continue
            at = f[g].attrs
            q = f[g]["truth_q"][:]
            if q.shape[0] < q.shape[1]: q = q.T
            sc = lambda k: float(np.ravel(at[k])[0])
            ax = int(sc("axis"))
            rows[(int(sc("payload")), sc("wn"), sc("zeta"), ax)] = metrics(q[:, ax - 1], q[int(round((sc("tStep") - 0.01) * 8000)), ax - 1], sc("step"), sc("tStep"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rigid", default="data/export/payload_eval6.h5")
    ap.add_argument("--elastic", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R, E = load(a.rigid), load(a.elastic)
    keys = sorted(set(R) & set(E))
    os.makedirs(a.out, exist_ok=True)
    res = [{"payload": k[0], "wn": k[1], "zeta": k[2], "axis": k[3], "rigid": R[k], "elastic": E[k]} for k in keys]
    json.dump(res, open(f"{a.out}/elastic_vs_rigid.json", "w"), indent=1)
    lines = [f"ケース数 {len(keys)}（弾性: {os.path.basename(a.elastic)}）", "",
             "| 軸 | ΔOS 最大 [pt] | Δ整定 最大 [ms] | 弾性の振動成分 最大 [%] | 剛体の振動成分 最大 [%] | 弾性の主周波数 [Hz]（振動>1% の中央値） |", "|---|---|---|---|---|---|"]
    for ax in range(1, 7):
        s = [r for r in res if r["axis"] == ax]
        dos = max(abs(r["elastic"]["overshoot_pct"] - r["rigid"]["overshoot_pct"]) for r in s)
        dst = max(abs(r["elastic"]["settle_s"] - r["rigid"]["settle_s"]) for r in s) * 1e3
        fr = [r["elastic"]["freq_hz"] for r in s if r["elastic"]["resid_pct"] > 1]
        lines.append(f"| {ax} | {dos:.2f} | {dst:.0f} | {max(r['elastic']['resid_pct'] for r in s):.2f} | {max(r['rigid']['resid_pct'] for r in s):.2f} | {np.median(fr) if fr else float('nan'):.1f}（{len(fr)} ケース） |")
    lines += ["", "ペイロード別（全軸・全ゲインの最大）:", "", "| ペイロード [kg] | ΔOS [pt] | Δ整定 [ms] | 弾性の振動成分 [%] |", "|---|---|---|---|"]
    for m in sorted({r["payload"] for r in res}):
        s = [r for r in res if r["payload"] == m]
        lines.append(f"| {m} | {max(abs(r['elastic']['overshoot_pct'] - r['rigid']['overshoot_pct']) for r in s):.2f} | {max(abs(r['elastic']['settle_s'] - r['rigid']['settle_s']) for r in s) * 1e3:.0f} | {max(r['elastic']['resid_pct'] for r in s):.2f} |")
    open(f"{a.out}/elastic_vs_rigid.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
