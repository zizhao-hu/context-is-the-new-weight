#!/bin/bash
# Re-invoke when a specific SLURM job leaves the queue (finished/failed).
JID=${1:?jobid}; MAXPOLL=${2:-25}
for i in $(seq 1 "$MAXPOLL"); do
  q=$(ssh -o ConnectTimeout=20 endeavour "squeue -u zizhaoh -h -o '%i' 2>/dev/null | grep -c '^$JID'" 2>/dev/null)
  echo "[wait poll $i] job $JID in_queue=${q:-?}"
  [ "${q:-1}" -eq 0 ] && { echo "[wait] job $JID finished"; exit 0; }
  sleep 90
done
echo "[wait] maxpoll reached"
