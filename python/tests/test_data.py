"""HDF5 読み込みと基準線の評価区分。3 軸 full の HDF5 があるときだけ実行する。"""
import os

import numpy as np
import pytest

from sa_nsm.baseline import groups
from sa_nsm.data import Dataset

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H5 = os.path.join(REPO, "data", "export", "sa_flat_full3.h5")
pytestmark = pytest.mark.skipif(not os.path.exists(H5), reason="data/export/sa_flat_full3.h5 が無い")


@pytest.fixture(scope="module")
def ds():
    return Dataset(H5, splits=("val", "benchmark"))


def test_shapes_and_consistency(ds):
    n = ds.n_ax
    assert n == 3
    for sp in ("val", "benchmark"):
        q, dq, tau = ds.q_dq_tau(sp)
        d_q, d_dq = ds.delta(sp)
        assert q.shape == dq.shape == tau.shape == d_q.shape == d_dq.shape
        assert np.isfinite(q).all() and np.isfinite(d_dq).all()
        assert len(ds.pattern(sp)) == len(q) == len(ds.at_limit(sp))


def test_benchmark_is_ptp_only_and_has_no_contact(ds):
    assert set(ds.pattern("benchmark")) == {"ptp"}
    assert not ds.at_limit("benchmark").any()


def test_groups_partition(ds):
    g = groups(ds, "val")
    assert g["free"].sum() + g["contact"].sum() == g["all"].sum()
    assert g["dynamic"].sum() + g["static"].sum() == g["contact"].sum()
    assert not (g["free"] & g["contact"]).any()
