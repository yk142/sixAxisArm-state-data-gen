#!/usr/bin/env python3
"""saExportFlat で書き出した HDF5 を h5py で読み、整合を検証する（学習は含まない）。

使い方: ~/.venvs/sa-h5/bin/python tools/verify_h5.py data/export/sa_flat_full3.h5
        （<名前>_check.mat があれば、MATLAB 側の倍精度の値との一致も見る。scipy が必要）
検証: 形状・型・有限性 / spec 形式の再計算 / 正規化統計と train の一致 / シナリオ境界（start, n_transition の連続性、
      split の所属、step）/ シナリオ内で X の q[k+1] − q[k] が Y の Δq と一致 / MATLAB の抜き取り行との一致。
終了コード: 全項目 OK なら 0。
"""
import os
import sys
import numpy as np
import h5py


def check(cond, msg, log):
    log.append(bool(cond))
    print(('  [OK] ' if cond else '  [NG] ') + msg)


def attr(f, k):
    v = f.attrs[k]
    v = np.ravel(v)[0] if np.ndim(v) else v
    return v.decode() if isinstance(v, bytes) else v


def txt(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def main(path):
    log = []
    with h5py.File(path, 'r') as f:
        nax = int(attr(f, 'nAx'))
        print(f'ファイル: {path}（nAx={nax}、dtype={attr(f, "dtype")}、preset={attr(f, "preset")}、gen_clean={attr(f, "gen_clean")}）')
        name, sc_split = txt(f['scenarios/name'][:]), txt(f['scenarios/split'][:])
        ntr, start = f['scenarios/n_transition'][:].astype(np.int64), f['scenarios/start'][:].astype(np.int64)
        scale = {k: f['scale'][k][:] for k in f['scale']}
        splits = [s for s in ('train', 'val', 'test', 'benchmark') if s in f]
        tol = 1.5e-7 if attr(f, 'dtype') == 'single' else 1e-14
        total = 0
        for sp in splits:
            g = f[sp]
            X, Y, Xs, Ys, qr = g['X'][:], g['Y'][:], g['X_spec'][:], g['Y_spec'][:], g['qref'][:]
            sid, stp, at = g['scenario_id'][:], g['step'][:], g['at_limit'][:]
            N = X.shape[0]
            total += N
            print(f'[{sp}] N = {N:,}')
            check(X.shape == (N, 3 * nax) and Y.shape == (N, 2 * nax) and Xs.shape == (N, 4 * nax) and Ys.shape == (N, 2 * nax)
                  and qr.shape == (N, nax), f'形状 X{X.shape} Y{Y.shape} X_spec{Xs.shape} Y_spec{Ys.shape} qref{qr.shape}', log)
            check(all(np.isfinite(a).all() for a in (X, Y, Xs, Ys)), 'X, Y, X_spec, Y_spec は有限', log)
            q, dq, tau = X[:, :nax], X[:, nax:2 * nax], X[:, 2 * nax:]
            check(np.allclose(Xs[:, :nax] ** 2 + Xs[:, nax:2 * nax] ** 2, 1.0, atol=1e-5), 'sin²+cos² = 1', log)
            check(np.allclose(Xs[:, 2 * nax:3 * nax], dq / scale['DQ_MAX'], rtol=1e-5, atol=1e-6), 'X_spec の dq/DQ_MAX', log)
            check(np.allclose(Xs[:, 3 * nax:], tau / scale['TAU_MAX'], rtol=1e-5, atol=1e-6), 'X_spec の tau/TAU_MAX', log)
            check(np.allclose(Ys[:, :nax], Y[:, :nax] / scale['D_Q_MAX'], rtol=1e-5, atol=1e-6), 'Y_spec の Δq/D_Q_MAX', log)
            check(np.allclose(Ys[:, nax:], Y[:, nax:] / scale['D_DQ_MAX'], rtol=1e-5, atol=1e-6), 'Y_spec の Δdq/D_DQ_MAX', log)
            # シナリオ境界
            ids = np.unique(sid)
            check(ids.min() >= 0 and ids.max() < len(name) and np.all(sc_split[ids] == sp), f"全行が split == '{sp}' のシナリオに属する", log)
            check(np.array_equal(ids, np.where(sc_split == sp)[0]), 'split のシナリオがすべて現れる', log)
            ok_blocks = True
            for s in ids:
                a, n = start[s], ntr[s]
                ok_blocks &= bool(np.all(sid[a:a + n] == s) and np.array_equal(stp[a:a + n], np.arange(n)))
                ok_blocks &= bool(a + n <= N and (a + n == N or sid[a + n] != s))
            check(ok_blocks, 'シナリオは [start, start+n_transition) の連続区間で、step が 0..n-1', log)
            check(sum(ntr[sc_split == sp]) == N, 'n_transition の合計 = 行数', log)
            # シナリオ内の連続性: q[k+1]-q[k] = Δq[k]
            same = (sid[1:] == sid[:-1])
            dqq = q[1:] - q[:-1]
            err = np.abs(dqq[same] - Y[:-1, :nax][same])
            u = 6.0e-8 if tol > 1e-10 else 1e-16              # single の丸め単位（半 ulp / 値）。q の 2 点の丸め ＋ Δq の丸め
            bound = u * (np.abs(q[1:][same]) + np.abs(q[:-1][same]) + np.abs(Y[:-1, :nax][same])) * 1.01 + 1e-12
            check(bool(np.all(err <= bound)), f'シナリオ内で q[k+1]−q[k] = Δq（丸めの範囲内、最大差 {err.max():.2e}）', log)
            check(set(np.unique(at)) <= {0, 1}, 'at_limit は 0/1', log)
            print(f'     at_limit {at.mean() * 100:.2f}%、qref あり {np.isfinite(qr[:, 0]).mean() * 100:.1f}%、|dq|/qdMax 最大 {np.abs(dq / f["physics/qdMax"][:]).max():.3f}、|tau|/tauPeak 最大 {np.abs(tau / f["physics/tauPeak"][:]).max():.3f}')
        check(total == ntr.sum(), f'全 split の行数 {total:,} = n_transition の合計', log)
        # 正規化統計（train の全遷移）
        Xt, Yt = f['train/X'][:].astype(np.float64), f['train/Y'][:].astype(np.float64)
        for nm, A in (('x', Xt), ('y', Yt)):
            check(np.allclose(f[f'stats/{nm}_mean'][:], A.mean(0), rtol=1e-4, atol=1e-6 * np.abs(A).max()), f'stats/{nm}_mean = train の平均', log)
            check(np.allclose(f[f'stats/{nm}_std'][:], A.std(0, ddof=1), rtol=1e-4), f'stats/{nm}_std = train の標準偏差', log)
        # MATLAB の倍精度の値との一致
        cp = os.path.splitext(path)[0] + '_check.mat'
        if os.path.isfile(cp):
            from scipy.io import loadmat
            m = loadmat(cp)
            worst = 0.0
            for sp in splits:
                ii = m[f'{sp}_idx'].ravel().astype(np.int64)
                for k in ('X', 'Y'):
                    a, b = f[f'{sp}/{k}'][:][ii].astype(np.float64), m[f'{sp}_{k}']
                    rel = np.abs(a - b) / (np.abs(b) * tol + 1e-30) if tol > 1e-10 else np.abs(a - b)
                    worst = max(worst, rel.max())
                check(np.array_equal(f[f'{sp}/scenario_id'][:][ii], m[f'{sp}_scenario_id'].ravel().astype(np.int32)), f'[{sp}] 抜き取り行の scenario_id が MATLAB と一致', log)
            check(worst <= 1.0, f'MATLAB の倍精度との差が丸め（相対 {tol:g}）以内（最大 {worst:.2f} 倍）', log)
        else:
            print('  （_check.mat なし: MATLAB との照合は省略）')
    ok = all(log)
    print(f'\n結果: {sum(log)}/{len(log)} 項目 OK' + ('' if ok else '（NG あり）'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
