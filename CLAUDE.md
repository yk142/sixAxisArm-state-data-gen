# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 言語
ユーザーとのやり取りは日本語で行うこと。

## 参照先
- `docs/handover.md`（J2 単軸版・sixAxisArmSurrogate からの引継ぎ。**最初に読む**）、`docs/requirements.md`、`docs/plan.md`（真値と段階は決定済み。他は叩き台）
- J2 版: `~/matlab-projects/j2-state-data-gen`（参照のみ。変更しない）

## 作業サイクル（必須）
1. issue 作成 → ブランチ作成 → 作業
2. 作業終了後、**ユーザーに報告**（ここで止まる）
3. **ユーザー指示があって初めて**コミットし、結果を issue にコメント投稿
4. **ユーザー指示があって初めて** PR を作成（**base は main**）→ ユーザーが承認

- **main への直接 push は禁止**。ユーザーの明示指示なしにコミット・push・PR 作成・issue 投稿をしない。
- 要件定義 → 計画 → 実装の順。いきなり実装しない。
- 評価結果は PNG を `reports/issue-N/` にコミットし、SHA 固定の raw URL で issue コメントに埋め込む。

## 方針（J2 版からの教訓。詳細は handover.md §3）
- 全長の開ループ再生で評価しない。閉ループと短い窓で評価する
- 弱い基準線との倍率を主張しない
- Deep Learning / System Identification / Statistics & ML Toolbox は使用禁止。Parallel Computing は未許諾
- 親資産 `~/matlab-projects/robotArmSurrogate-matlab` は path 参照のみ（コピー・変更しない）

## 現状
- Phase 0〜4 完了（#1〜#85）。真値は弾性つき Simscape（#71）、比較基準は 2 慣性解析（`python/sa_nsm/twomass.py`）。
- 次は Phase 5（docs/plan.md）: 逐次で実時間より速く、水準 G を満たす NN。NN の役割は計算の高速化（#87 で決定）。
- MATLAB: `src/`・`config/`・`tools/`・`test/`。Python: `python/`（venv `~/.venvs/sa-train`）。データは `data/`（git 管理外）。
