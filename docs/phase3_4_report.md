# Phase 3-4 レポート — HDF5 出力と h5py 読み戻し検証（3 軸）（issue #25）

実施日: 2026-10-13 / MATLAB R2025a / Python 3 + h5py 3.16.0 + numpy 2.5.4 + scipy（`~/.venvs/sa-h5`、リポジトリ外）

## 成果物
| ファイル | 内容 |
|---|---|
| `src/data/saExportFlat.m` | データセット（`sa_dataset_full3.mat`）を平坦な HDF5（single、gzip 4）に書き出す。検証用に各 split から等間隔に 2,000 行を倍精度で `*_check.mat` にも保存 |
| `tools/verify_h5.py` | h5py で読み戻して検証（学習は含まない）。全項目 OK で終了コード 0 |
| `test/test_sa_export.m` | 3 件（配置、Python 検証が通る、壊したファイルを Python が検出する）合格。合成データ・シミュレーションなし |
| `data/export/sa_flat_full3.h5`（git 管理外） | 3 軸 full の出力。129 MB（参考: `sa_dataset_full3.mat` は 108 MB。X_spec・Y_spec・qref を余分に持つぶん大きい、と見ているが未検証）、書き出し 16 s |

## HDF5 の構成（h5py から見た形、インデックスは 0 始まり、d = 軸数 = 3）
| 項目 | 形 | 内容 |
|---|---|---|
| `/<split>/X` | (N, 3d) | [q(d), dq(d), tau(d)]（SI、τ は区間の台形平均） |
| `/<split>/Y` | (N, 2d) | [Δq(d), Δdq(d)]（1 ステップ = 1 ms。倍精度で差を取ってから single） |
| `/<split>/X_spec`, `Y_spec` | (N, 4d), (N, 2d) | [sin q, cos q, dq/DQ_MAX, tau/TAU_MAX]、[Δq/D_Q_MAX, Δdq/D_DQ_MAX] |
| `/<split>/scenario_id`, `step` | (N,) int32 | `/scenarios/*` の添字、シナリオ内の遷移番号 |
| `/<split>/at_limit`, `qref` | (N,) uint8, (N, d) | リミット拘束フラグ、開始時刻の参照角（励振は NaN） |
| `/scenarios/{name,pattern,type,split,seed,n_transition,start}` | (シナリオ数,) | **start = そのシナリオの先頭行（所属 split の配列内）**。行は split 内でシナリオ順・step 順なので、シナリオは [start, start+n_transition) の連続区間 |
| `/stats/*`, `/scale/*`, `/physics/*` | | train のみの正規化統計、スケール、可動範囲・トルク・摩擦の係数 |
| ルート属性 | | nAx, preset, scenario_hash, gen_clean, gen_repo, parent/repo commit, matlab, layout など |

split は train / val / test / benchmark の 4 つ（行数 691,000 / 99,000 / 140,000 / 107,294）。

## 検証結果（`tools/verify_h5.py data/export/sa_flat_full3.h5`、3 秒）
**62 / 62 項目 OK**。内容:
- 形状 (N, 9) (N, 6) (N, 12) (N, 6) (N, 3)、型 float32、X・Y・spec が有限
- spec 形式を X・Y・scale から再計算して一致、sin²+cos² = 1
- シナリオ境界: 全行が自分の split のシナリオに属する、各 split のシナリオが全部現れる、各シナリオが [start, start+n_transition) の連続区間で step が 0..n−1、行数の合計が n_transition の合計（1,037,294）に一致
- シナリオ内で X の q[k+1] − q[k] が Y の Δq と、single の丸めの範囲で一致（q の 2 点と Δq の丸めから計算した上限以内。J1 の |q| が 4 rad を超えるため最大差は 4.7e-7）
- `/stats` が train の X・Y の平均・標準偏差（不偏）と一致
- MATLAB 側の倍精度の値（各 split 2,000 行）との差が single の丸め（相対 1.5e-7）以内（最大でも丸め幅の 0.40 倍）、scenario_id も一致

## 判断・発見
1. シナリオ境界を `start`・`n_transition` で持たせた（ユーザー了承済み）。時間窓の学習で窓がシナリオをまたがないように、`[start, start+n−K+1)` の範囲から窓を取れる
2. J1 は可動範囲が ±240°（|q| > 4 rad）で、single の q の精度は約 5e-7 rad。1 ステップの Δq は数 e-4 rad のオーダーなので、**q の差分から Δq を再計算する用途には単精度は不十分**（Y に Δq を別に持たせているのはこのため）。ネットワークの入力・目標の精度としては影響が小さいと見ているが、未評価（Phase 4）
3. 壊したファイル（scenario_id をずらす）を Python 側の検証が検出することをテストで確認した

## 限界
- 検証は MATLAB からの出力の整合（形・値・境界）。学習用の Python 前処理（DataLoader、正規化）は含まない（Phase 4）
- `at_limit` は 3 軸 full では全て 0（接触なし）。接触シナリオは 3-5
- Python 環境（h5py など）はリポジトリ外 `~/.venvs/sa-h5`。再現にはインストールが必要（`pip install h5py numpy scipy`）
