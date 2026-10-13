#!/bin/bash
# 3 モデル × シード（既定 0 1 2）の学習を 2 本並行（各 2 スレッド）で回す。使い方: tools/sa_train_run.sh [シード...]
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-$HOME/.venvs/sa-train/bin/python}
SEEDS=${@:-0 1 2}
OUT=data/models/full3; mkdir -p "$OUT" data/logs
RUNS=(); for s in $SEEDS; do for m in A B C; do RUNS+=("$m:$s"); done; done
lane() { for r in "$@"; do m=${r%%:*}; s=${r##*:}
  [ -f "$OUT/${m}_s${s}/metrics.json" ] && continue                 # 完了済みは飛ばす（再開可）
  $PY -u python/train.py --model "$m" --seed "$s" --h5 data/export/sa_flat_full3.h5 --params data/export/c8_params.json --out "$OUT" --threads 2 > "data/logs/train_${m}_s${s}.log" 2>&1
done; }
L1=(); L2=(); i=0; for r in "${RUNS[@]}"; do if (( i % 2 == 0 )); then L1+=("$r"); else L2+=("$r"); fi; i=$((i+1)); done
echo "[$(date +%T)] 学習開始: ${RUNS[*]}"
lane "${L1[@]}" & lane "${L2[@]}" &
wait
echo "[$(date +%T)] 学習すべて終了"
