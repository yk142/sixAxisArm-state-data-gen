#!/bin/bash
# shard 数ぶんの作業プロセスで未生成のシナリオを生成し、続けて組み立て・検証を行う。
# 使い方: tools/sa_run_all.sh <nAx> <preset> <shard 数> [ペイロード複製の組数（既定 0）] [パルス 軸1 本数（既定 0）] [パルス 他軸 本数（既定 0）]
cd "$(dirname "$0")/.." || exit 1
M=${MATLAB:-/usr/local/MATLAB/R2025a/bin/matlab}
NAX=${1:-3}; PRESET=${2:-full}; N=${3:-2}; REP=${4:-0}; PU1=${5:-0}; PUO=${6:-0}
TAG=${PRESET}${NAX}; [ "$REP" -gt 0 ] && TAG=${PRESET}pl${NAX}
if [ "$PU1" -gt 0 ]; then TAG=${PRESET}$([ "$REP" -gt 0 ] && echo pl)pu${NAX}; fi
mkdir -p data/logs
echo "[$(date +%T)] 生成開始 nAx=$NAX preset=$PRESET shard=$N"
for i in $(seq 1 "$N"); do
  $M -batch "addpath('tools'); sa_gen_worker($NAX,'$PRESET',$i,$N,$REP,$PU1,$PUO)" > "data/logs/gen_${TAG}_$i.log" 2>&1 &
done
wait
echo "[$(date +%T)] 生成終了。組み立て・検証"
$M -batch "addpath('tools'); sa_finish($NAX,'$PRESET',$REP,$PU1,$PUO)" > "data/logs/finish_${TAG}.log" 2>&1
echo "[$(date +%T)] すべて終了（ログ: data/logs）"
