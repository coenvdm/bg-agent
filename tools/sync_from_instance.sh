#!/usr/bin/env bash
# Pull training artefacts from the vast.ai box to local disk.
#
# Syncs the HISTORY + CHART every pass and the CHECKPOINT every 6th (~2 min):
# the chart/history tell you what happened, but only the .pt lets you resume,
# and an instance can vanish at any time. A chart-only loop silently turns
# "resume" into "restart from scratch" -- that has cost a run before.
#
# The checkpoint lands on a live_-prefixed name so a pull that catches a
# mid-torch.save() write can never clobber a known-good archived checkpoint.
set -u
HOST=vast-bg-agent
REMOTE=/workspace/bg-agent
i=0
while true; do
  rsync -az -e ssh "$HOST:$REMOTE/data/fresh_training_history.json" ./data/ 2>/dev/null \
    || echo "$(date +%T) history sync failed"
  rsync -az -e ssh "$HOST:$REMOTE/data/training_progress.png" ./data/ 2>/dev/null || true
  rsync -az -e ssh "$HOST:$REMOTE/data/training_remote.log" ./data/ 2>/dev/null || true
  if [ $((i % 6)) -eq 0 ]; then
    # A "read errors mapping ... No data available" failure here is the
    # mid-torch.save() race, not a real error -- the next pass picks it up.
    rsync -az -e ssh "$HOST:$REMOTE/bg_agent_ppo.pt" ./checkpoint_backups/live_bg_agent_ppo.pt 2>/dev/null \
      || echo "$(date +%T) checkpoint sync failed (likely mid-save; will retry)"
  fi
  if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$HOST" "tmux has-session -t train" 2>/dev/null; then
    echo "$(date +%T) TRAINING SESSION GONE"
  fi
  i=$((i+1))
  sleep 20
done
