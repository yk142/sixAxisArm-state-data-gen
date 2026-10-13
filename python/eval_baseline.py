"""学習前の基準線の 1 ステップ誤差（加速度換算 RMSE [rad/s²]）を出す。

使い方: ~/.venvs/sa-train/bin/python python/eval_baseline.py data/export/sa_flat_full3.h5 data/export/c8_params.json OUT_DIR [--limit N]
出力: OUT_DIR/baseline.json、baseline.md（表）。解析モデル＝RNEA＋摩擦、トルクは区間一定の RK4（1 ms を 2 分割）。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.baseline import evaluate_baselines  # noqa: E402
from sa_nsm.data import Dataset  # noqa: E402
from sa_nsm.physics import Arm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5"); ap.add_argument("params"); ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=None, help="split ごとの最大遷移数（既定は全件）")
    ap.add_argument("--splits", default="val,test,benchmark")
    a = ap.parse_args()
    torch.set_num_threads(4)
    ds = Dataset(a.h5)
    arm = Arm.from_json(a.params, ds.n_ax)
    t = time.time()
    res = evaluate_baselines(ds, arm, tuple(a.splits.split(",")), limit=a.limit)
    os.makedirs(a.out, exist_ok=True)
    ser = {sp: {k: {kk: (v.tolist() if isinstance(v, np.ndarray) else v) for kk, v in d.items()} for k, d in r.items()}
           for sp, r in res.items()}
    with open(os.path.join(a.out, "baseline.json"), "w") as f:
        json.dump(ser, f, indent=1)
    lines = [f"# 基準線の 1 ステップ誤差（加速度換算 RMSE [rad/s²]、軸別。{os.path.basename(a.h5)}、{time.time() - t:.0f} s）", ""]
    for sp, r in res.items():
        lines += [f"## {sp}", "", "| 区分 | 遷移数 | 目標の RMS | 解析モデル | 解析＋非弾性ストッパ | 状態を変えない |", "|---|---|---|---|---|---|"]
        for k, d in r.items():
            f = lambda v: " / ".join(f"{x:.3g}" for x in v)
            lines.append(f"| {k} | {d['n']:,} | {f(d['target'])} | {f(d['analytic'])} | {f(d['analytic_stop'])} | {f(d['persist'])} |")
        lines.append("")
    open(os.path.join(a.out, "baseline.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
