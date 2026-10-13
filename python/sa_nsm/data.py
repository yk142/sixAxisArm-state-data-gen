"""HDF5（`saExportFlat` の出力、3 軸／6 軸共通）の読み込み。行は遷移、(N,d)、ID は 0 始まり。"""
import h5py
import numpy as np

SPLITS = ("train", "val", "test", "benchmark")


def _txt(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def _attr(f, k):
    v = f.attrs[k]
    v = np.ravel(v)[0] if np.ndim(v) else v
    return v.decode() if isinstance(v, bytes) else v


class Dataset:
    """遅延ではなく全 split をメモリに載せる（3 軸 full で約 150 MB、6 軸で約 280 MB）。"""

    def __init__(self, path, splits=SPLITS):
        with h5py.File(path, "r") as f:
            self.n_ax = int(_attr(f, "nAx"))
            self.gen_clean = bool(_attr(f, "gen_clean"))
            self.scenarios = {k: _txt(f["scenarios"][k][:]) if f["scenarios"][k].dtype == object else f["scenarios"][k][:]
                              for k in f["scenarios"]}
            self.scale = {k: f["scale"][k][:] for k in f["scale"]}
            self.stats = {k: f["stats"][k][:] for k in f["stats"]}
            self.part = {}
            for sp in splits:
                if sp not in f:
                    continue
                g = f[sp]
                self.part[sp] = {k: g[k][:] for k in g}

    def q_dq_tau(self, split):
        d, n = self.part[split], self.n_ax
        X = d["X"].astype(np.float64)
        return X[:, :n], X[:, n:2 * n], X[:, 2 * n:]

    def delta(self, split):
        """真の遷移の (Δq, Δdq)（物理量。float32 の q の丸めを避けるため Y を使う）。"""
        Y, n = self.part[split]["Y"].astype(np.float64), self.n_ax
        return Y[:, :n], Y[:, n:]

    def pattern(self, split):
        """遷移ごとのパターン名（`scenario_id` から引く。ptp_v20 などは ptp にまとめる）。"""
        pat = self.scenarios["pattern"][self.part[split]["scenario_id"]]
        return np.array([("ptp" if p.startswith("ptp") else "hold" if p.startswith("hold") else p) for p in pat])

    def at_limit(self, split):
        return self.part[split]["at_limit"].astype(bool)
