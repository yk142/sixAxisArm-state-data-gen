#!/bin/bash
# 再現性確認: 別プロセスでキャッシュなしに再生成する（data/repro）。使い方: tools/sa_repro_run.sh <nAx> <preset> <sample|all> <shard 数>
cd "$(dirname "$0")/.." || exit 1
M=${MATLAB:-/usr/local/MATLAB/R2025a/bin/matlab}
NAX=${1:-3}; PRESET=${2:-full}; MODE=${3:-sample}; N=${4:-2}
mkdir -p data/logs
echo "[$(date +%T)] 再生成開始 nAx=$NAX preset=$PRESET mode=$MODE shard=$N"
for i in $(seq 1 "$N"); do
  $M -batch "addpath('tools'); sa_repro_worker($NAX,'$PRESET','$MODE',$i,$N)" > "data/logs/repro_${PRESET}${NAX}_${MODE}_$i.log" 2>&1 &
done
wait
echo "[$(date +%T)] 再生成終了"
