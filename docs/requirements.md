# 要件定義（叩き台）— 6 軸 Neural State Model 学習データ生成

状態: **一部決定**。決定済み: ①真値は Simscape/URDF、RNEA は残差型の解析モデル、②段階は腕 3 軸（J1〜J3）→ 6 軸（2026-10-11）。[handover.md](handover.md) §5 の 3〜7 は未決（Phase 0 の実測後に決める）。

## 目的
EPSON C8-A901S の 6 軸について、Neural State Model（次時刻状態予測）の学習データを生成し、学習・評価する。

## 機能要件（案）
| ID | 要件 |
|---|---|
| FR-1 | 真値プラント: C8-A901S の 6 軸（またはその部分集合）。8 kHz 固定ステップ。物理検証（重力トルク、自由落下、リミット到達、RNEA との一致）に合格 |
| FR-2 | シナリオ: 励振（BLN / chirp / step / micro）、姿勢保持（代表姿勢）、PTP ベンチマーク、リミット接触。シード固定で再現 |
| FR-3 | 1 kHz 変換: 状態は瞬時値の間引き、トルクは区間平均。遷移 `(x_k, τ_k) → x_{k+1}` |
| FR-4 | split: シナリオ単位の train / val / test / benchmark。正規化統計は train のみ |
| FR-5 | 検証関数: 有限性、トルク・速度・角度の範囲、解析モデルとの 1 ステップ整合、split リーク |
| FR-6 | カバレッジ評価: 時系列、位相平面（軸別）、周波数、PTP の包含率。6 軸向けに指標を再設計する |
| FR-7 | 他言語出力: HDF5（h5py から (N,d)、0 始まり ID） |
| FR-8 | 学習・評価（Python / PyTorch）: 1 ステップ誤差、短窓の開ループ、PTP 閉ループ。解析モデル＋残差 NN を基準に |

## 非機能要件（案）
- 再現性: キャッシュなし再生成でビット一致を確認
- 作業サイクル: handover.md §3 の 1
- 使用禁止 Toolbox: Deep Learning / System Identification / Statistics & ML。Parallel Computing は未許諾

## 確認事項
handover.md §5 の 3〜7（モデルの形、J2 版の未解決、実用性の確認、データ規模、親資産への依存）。
