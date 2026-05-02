#!/bin/bash
# Pull experiment outputs from CARC scratch back to local for analysis.
# Run from repo root locally.
set -e

REMOTE="endeavour:/scratch1/zizhaoh/context-is-the-new-weight"
LOCAL_OUT="outputs/01_synth_distill_kvdw"
LOCAL_DATA="data/synth"
LOCAL_LOG="logs"

mkdir -p "$LOCAL_OUT" "$LOCAL_DATA" "$LOCAL_LOG"

echo "[pull] outputs/"
scp -q -r "$REMOTE/outputs/01_synth_distill_kvdw/." "$LOCAL_OUT/" 2>&1 | tail -5

echo "[pull] data/synth/"
scp -q -r "$REMOTE/data/synth/." "$LOCAL_DATA/" 2>&1 | tail -5

echo "[pull] logs/ (tails only)"
ssh endeavour 'cd /project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs && for f in cinw_*.log; do tail -200 "$f" > "/tmp/${f}.tail"; done && ls /tmp/cinw_*.log.tail'
scp -q "endeavour:/tmp/cinw_*.log.tail" "$LOCAL_LOG/" 2>&1

echo "[pull] done. local outputs at $LOCAL_OUT/"
ls -la "$LOCAL_OUT/"
