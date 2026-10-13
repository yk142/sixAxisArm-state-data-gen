"""学習 1 回（モデル A/B/C、シード指定）。使い方:
  ~/.venvs/sa-train/bin/python python/train.py --model B --seed 0 --h5 data/export/sa_flat_full3.h5 --params data/export/c8_params.json --out data/models/full3
出力: OUT/<model>_s<seed>/{model.pt, config.json, history.json, metrics.json}
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_nsm.data import Dataset  # noqa: E402
from sa_nsm.physics import Arm  # noqa: E402
from sa_nsm.train_lib import evaluate_model, prepare, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["A", "B", "C"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="train の遷移数を減らす（動作確認用）")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ds = Dataset(a.h5)
    arm = Arm.from_json(a.params, ds.n_ax)
    cache = os.path.join(os.path.dirname(a.h5), os.path.basename(a.h5).replace(".h5", "_analytic.npz"))
    data = prepare(ds, arm, cache)
    if a.limit:
        import numpy as np
        sel = np.sort(np.random.RandomState(0).permutation(len(data["train"]["y"]))[:a.limit])
        data["train"] = {k: v[sel] for k, v in data["train"].items()}
    out = os.path.join(a.out, f"{a.model}_s{a.seed}")
    os.makedirs(out, exist_ok=True)
    model, hist = train(a.model, data, ds.n_ax, a.seed, epochs=a.epochs)
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    json.dump(model.config(), open(os.path.join(out, "config.json"), "w"))
    json.dump(hist, open(os.path.join(out, "history.json"), "w"))
    json.dump(evaluate_model(model, ds, data), open(os.path.join(out, "metrics.json"), "w"), indent=1)
    print("完了", out)


if __name__ == "__main__":
    main()
