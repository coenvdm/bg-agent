#!/usr/bin/env bash
# Pull a running vast.ai training job's artifacts down to data/ on a loop, so
# the local live graph (tools/live_graph.py --serve) always shows fresh numbers.
#
#   tools/sync_from_vast.sh <ssh_host> <ssh_port> [remote_dir] [interval_s]
#
# Example:
#   tools/sync_from_vast.sh ssh4.vast.ai 12345 /workspace/bg-agent 60
#
# What it pulls and why the cadences differ:
#   - fresh_training_history.json  every tick   (small-ish, drives every chart)
#   - training_progress.png        every tick   (trainer-side matplotlib view)
#   - training.log tail            every tick   (so a crash is visible locally)
#   - bg_agent_ppo*.pt             every CKPT_EVERY ticks (~42 MB each)
#
# The checkpoint is synced at a slower cadence AND to checkpoint_backups/ under
# a distinct name, because losing the checkpoint is how a previous run was lost.
# rsync is retried on failure rather than aborting the loop: a mid-`torch.save`
# read or a dropped SSH connection is expected and must not kill the sync.
set -uo pipefail

HOST="${1:?usage: sync_from_vast.sh <ssh_host> <ssh_port> [remote_dir] [interval_s]}"
PORT="${2:?missing ssh port}"
RDIR="${3:-/workspace/bg-agent}"
INTERVAL="${4:-60}"
CKPT_EVERY="${CKPT_EVERY:-10}"     # checkpoint every N ticks

cd "$(dirname "$0")/.." || exit 1
mkdir -p data checkpoint_backups

SSH_OPTS="-p ${PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=20"
REMOTE="root@${HOST}"

pull() {  # pull <remote_relpath> <local_path>
  rsync -az --partial --timeout=90 -e "ssh ${SSH_OPTS}" \
        "${REMOTE}:${RDIR}/$1" "$2" 2>/dev/null
}

tick=0
echo "[sync] ${REMOTE}:${RDIR} -> $(pwd)/data every ${INTERVAL}s (checkpoint every ${CKPT_EVERY} ticks)"
while true; do
  tick=$((tick + 1))
  ok=""

  pull "data/fresh_training_history.json" "data/fresh_training_history.json" && ok="${ok}history "
  pull "data/training_progress.png"       "data/training_progress.png"       && ok="${ok}chart "

  # Log tail only -- the full log grows without bound and is not worth the bytes.
  ssh ${SSH_OPTS} "${REMOTE}" "tail -n 400 ${RDIR}/training.log 2>/dev/null" \
      > data/training_remote.log.tmp 2>/dev/null \
      && mv data/training_remote.log.tmp data/training_remote.log && ok="${ok}log "

  if [ $((tick % CKPT_EVERY)) -eq 0 ]; then
    for f in bg_agent_ppo.pt bg_agent_ppo_best.pt; do
      pull "$f" "checkpoint_backups/live_${f}" && ok="${ok}${f} "
    done
  fi

  printf '[sync %s] tick %d: %s\n' "$(date +%H:%M:%S)" "$tick" "${ok:-NOTHING (instance down?)}"
  sleep "$INTERVAL"
done
