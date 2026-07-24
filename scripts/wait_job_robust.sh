#!/bin/bash
# Re-invoke only when a job is CONFIRMED finished: absent from squeue AND sacct shows a terminal
# state (COMPLETED/TIMEOUT/FAILED/CANCELLED). Avoids transient-squeue false fires.
JID=${1:?jobid}; NAME=${2:?jobname}; MAXPOLL=${3:-80}
for i in $(seq 1 "$MAXPOLL"); do
  read inq state <<<"$(ssh -o ConnectTimeout=20 endeavour "
    q=\$(squeue -u zizhaoh -h -o '%i' 2>/dev/null | grep -c '^$JID')
    s=\$(sacct -n --name=$NAME --format=State -X 2>/dev/null | tail -1 | tr -d ' ')
    echo \${q:-x} \${s:-x}" 2>/dev/null)"
  echo "[wait poll $i] in_queue=$inq sacct=$state"
  case "$inq:$state" in
    0:COMPLETED|0:TIMEOUT|0:FAILED|0:CANCELLED|0:CANCELLED+*) echo "[wait] $NAME finished ($state)"; exit 0;;
  esac
  sleep 300
done
echo "[wait] maxpoll reached"
