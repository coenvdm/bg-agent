---
name: vast-ai-training
description: Rent, configure, and manage vast.ai GPU instances for running this project's training jobs (or similar CPU+GPU self-play/RL workloads). Covers finding an instance that's actually a good deal (not just cheap-looking), verifying its real CPU/GPU capacity before trusting it, deploying code safely, and avoiding the specific failure modes hit in practice: stuck image pulls, fractional/shared hosts that throttle CPU far below their advertised vCPU count, CUDA+multiprocessing fork crashes, and spawn re-executing a launch script's setup code in every worker. Use whenever the task is renting a vast.ai instance, speeding up training, diagnosing why a rented instance is slow, or migrating a running job to a different instance.
---

# vast.ai training instances

Hard-won operational knowledge from actually running this project's training on vast.ai.
Follow this end-to-end rather than improvising — most of it exists because the naive approach
failed in a specific, documented way.

## 0. Prerequisites (once per machine)

```bash
pip install --quiet vastai
vastai set api-key <API_KEY>          # ask the user for this if not already set
vastai show user                       # confirms it worked, shows credit balance
```

## 1. Finding a good offer

**Do not just sort by price.** The nominal `cpu_cores` a listing advertises is frequently a lie
for CPU-bound workloads (self-play game simulation, symbolic computation, anything that isn't
pure GPU matmuls) — see §3. Search like this:

```bash
vastai search offers 'num_gpus=1 reliability>0.99 dph<<PRICE_CEILING> disk_space>=30' -o 'dph' --raw \
  | python3 -c "
import json,sys
data = json.load(sys.stdin)
rows = []
for o in data:
    cc, cce = o.get('cpu_cores') or 0, o.get('cpu_cores_effective') or 0
    if cc <= 0: continue
    rows.append((cce/cc, o))
rows.sort(key=lambda r: (-r[0], -(r[1].get('total_flops') or 0)))
for ratio, o in rows[:20]:
    print(o['id'], o['gpu_name'], 'cores=',o['cpu_cores'],'effective=',round(o.get('cpu_cores_effective',0),1),
          'ratio=',round(ratio,2),'dph=',round(o['dph_total'],4),'reliability=',round(o['reliability2'],3),
          'tflops=',round(o.get('total_flops',0),1))
"
```

Picking rules, in priority order:
1. **`cpu_cores_effective / cpu_cores` ratio close to 1.0** — this is the single strongest signal
   that a host is dedicated rather than fractionally shared. A listing showing `cpu_cores=80` with
   `cpu_cores_effective=20` (ratio 0.25) means you're renting a quarter of a shared box, and the
   *real* enforced quota (checked after boot, see §3) can be even lower than the effective number.
2. **`reliability2 > 0.99`** — below ~0.97 the host is meaningfully more likely to be flaky.
3. Among the dedicated (ratio≈1.0) candidates, pick the best `total_flops` (or the GPU model you
   actually want) within budget. For CPU-bound workloads, also weigh raw `cpu_cores` — a dedicated
   32-core host is more valuable than a dedicated 12-core host even with a weaker GPU, if the
   self-play/simulation phase dominates wall-clock time (check GPU util with `nvidia-smi` on the
   *current* instance before assuming you need a faster GPU — it may already be well-utilized, and
   the real bottleneck may be CPU).
4. Only after that, minimize `dph_total`.

## 2. Renting and connecting

```bash
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime \
  --disk 40 --ssh --direct --label <descriptive-label>
# -> returns {"new_contract": <ID>}

# Poll for boot (background wait, single notification, don't poll manually):
until [ "$(vastai show instance <ID> --raw | python3 -c "import json,sys; print(json.load(sys.stdin).get('actual_status'))")" = "running" ]; do sleep 10; done
```

**If it stays `loading` for 15-20+ minutes with the status message stuck on the same Docker layer**,
it's likely genuinely stalled (bad host network path to the registry) rather than just slow — destroy
it (`vastai destroy instance <ID> -y`) and try a different offer rather than waiting indefinitely.
This happened twice in practice; the fix was re-rolling, not patience. But don't panic-destroy after
5 minutes either — large CUDA base images can legitimately take 10+ minutes on a slow-but-fine host.

Once running, attach your key and connect:

```bash
vastai attach ssh <ID> "$(cat ~/.ssh/id_ed25519.pub)"
# Get ssh_host/ssh_port:
vastai show instance <ID> --raw | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['ssh_host'], d['ssh_port'])"
```

Add/update a **stable alias** in `~/.ssh/config` (reuse the same `Host` name across migrations —
every script, monitor loop, and rsync command below then keeps working unchanged when you swap
instances, you only edit Hostname/Port):

```
Host vast-bg-agent
Hostname <ssh_host>
Port <ssh_port>
User root
ServerAliveInterval 10
StrictHostKeyChecking accept-new
```

Retry SSH a few times after attach — key propagation takes a few seconds:
```bash
for i in $(seq 1 15); do
  timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=10 vast-bg-agent "echo ok" && break
  sleep 10
done
```

## 3. Verify real capacity BEFORE deploying anything (critical)

**Immediately after SSH is up, before installing anything:**

```bash
ssh vast-bg-agent "nproc; cat /sys/fs/cgroup/cpu.max; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
```

`cpu.max` is `<quota_us> <period_us>`; real usable cores = quota/period. This is the ONLY number
to trust — not the listing's `cpu_cores`, not even `cpu_cores_effective` from the search API (in
practice these overstated the real quota: a listing showing `cpu_cores_effective=20` turned out to
have a hard cgroup quota of 8.64 cores once actually rented). If this number is much lower than
expected, no amount of raising `N_WORKERS`/parallelism will recover throughput — the constraint is
structural. Destroy and pick a different (genuinely dedicated) offer instead of fighting it.

Size your worker/parallelism count to this real number, not the advertised one. Rule of thumb from
this project: `N_WORKERS ≈ real_cores / 3-4`, leaving headroom for the main process, tmux, and OS.

## 4. Deploying code

```bash
rsync -az \
  --exclude='.git/' --exclude='__pycache__/' --exclude='**/__pycache__/' --exclude='*.pyc' \
  --exclude='*.pt' --exclude='*.pth' --exclude='data/' --exclude='checkpoint_backups/' \
  --exclude='*.ipynb' --exclude='*.ipynb_checkpoints/' --exclude='.env' \
  -e ssh /path/to/repo/ vast-bg-agent:/workspace/bg-agent/

ssh vast-bg-agent "cd /workspace/bg-agent && pip install --no-cache-dir -r requirements.txt"
ssh vast-bg-agent "which tmux >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)"
```

If resuming a training run, separately rsync the checkpoint/history files (they're excluded above
on purpose — don't want them clobbered by the code-only sync, and don't want to accidentally sync
them on every code push). **If a checkpoint pull fails with `read errors mapping ... No data
available`**, the training process was mid-`torch.save()` — just retry a few seconds later, don't
treat it as a real failure.

Launch in tmux with a generous file-descriptor limit (multiprocessing + CUDA workers can exhaust
the default 1024 surprisingly fast, especially if anything triggers repeated worker-pool rebuilds):

```bash
ssh vast-bg-agent "cd /workspace/bg-agent && tmux new-session -d -s train 'ulimit -n 65536; python3 -u train_script.py 2>&1 | tee -a training.log'"
```

## 5. Two specific bugs to check for if you touch multiprocessing + CUDA

These are not vast.ai-specific but *will* bite you the first time a training script that worked
fine on CPU gets pointed at a GPU with a multi-worker self-play/rollout loop:

1. **`ProcessPoolExecutor` defaults to `fork` on Linux.** Forking a process that has already
   touched CUDA (or that will touch CUDA in the child) crashes with
   `RuntimeError: Cannot re-initialize CUDA in forked subprocess`. Fix: build a spawn context and
   pass it explicitly — `mp_context=multiprocessing.get_context("spawn")` — everywhere a
   `ProcessPoolExecutor`/`Pool` is constructed (including in any "rebuild the pool after an error"
   retry path).
2. **Spawn re-executes the launching script in every worker.** Unlike fork, spawn does not inherit
   parent memory — it re-imports your entry-point script as `__mp_main__`. Any module-level code
   *outside* an `if __name__ == "__main__":` guard runs again in every single worker: rebuilding
   models on CUDA, reloading checkpoints, re-reading data files. This is silently wasteful at best
   and causes `OSError: [Errno 24] Too many open files` at worst (each worker opening its own CUDA
   context + checkpoint file handles, compounded by any pool-rebuild-on-error retry loop). Fix:
   wrap the entire body of the script (everything after imports) in
   `if __name__ == "__main__":` — an `if` block doesn't introduce a new scope, so this is a pure
   indentation change, not a refactor; module-level function *defs* can safely live inside the
   guard too if nothing outside the guard calls them.

If a training script's history log/checkpoint may have been written by an older version of the
code that tracked fewer fields (e.g. resuming into a run where a new metric's array is empty while
an older, related array already has many entries), guard any code that zips/pairs same-index
values across two arrays (e.g. `matplotlib`'s `fill_between`) against length mismatches — don't
assume "resumed run" means "all history arrays are the same length."

## 6. Monitoring

Don't poll manually. Use a background quiet loop that only speaks up on state changes:

```bash
# One-shot "wait until X" -> use Bash run_in_background with an until-loop, single notification.
# Recurring "sync chart back to local disk" -> Monitor with persistent:true, silent unless it fails:
while true; do
  rsync -az -e ssh vast-bg-agent:/workspace/bg-agent/data/chart.png ./data/ || echo "sync failed"
  ssh -o ConnectTimeout=10 vast-bg-agent "tmux has-session -t train" 2>/dev/null || { echo "training session gone"; break; }
  sleep 20
done
```

When tailing a live training log for a go/no-go check after (re)launching, grep for both progress
*and* every failure signature in one alternation (`^update |Traceback|Error|Too many open files|
Killed`) — a filter that only matches the happy path stays silent through a crash.

Don't trust the first batch/iteration's timing — cold-start (CUDA context init, first connections)
inflates it. Wait for 2-3 steady-state samples before concluding an instance is fast or slow.

## 7. Cleanup

```bash
vastai show instances --raw   # list what's currently running/billing
vastai destroy instance <ID> -y   # -y skips the interactive confirmation prompt
```

**When migrating to a new instance, stop the old training job (`tmux kill-session -t train` over
SSH) as soon as you've pulled its checkpoint** — it's easy to forget and end up paying for two
instances training in parallel while you're busy setting up the new one. Destroy the old instance
once the new one is confirmed stable (a few clean update cycles), not before — keep it as a
fallback until then.

## Summary checklist for "speed up training" / "set up a vast.ai instance" requests

1. Check current GPU/CPU utilization first (`nvidia-smi`, `uptime`/`top`) — don't assume a bigger
   GPU is the answer without evidence the current one is actually the bottleneck.
2. Search offers, filter by `cpu_cores_effective/cpu_cores` ratio ≈ 1.0, sort by flops/reliability
   within budget.
3. Rent, wait for `running` (destroy and retry if stuck loading >15-20 min).
4. **Before installing anything**: check `/sys/fs/cgroup/cpu.max` for the real core count.
5. Size worker/parallelism count to that real number, not the advertised one.
6. Deploy code (exclude weights/data/git), install deps, migrate checkpoint if resuming.
7. Launch in tmux with a high `ulimit -n`.
8. Confirm 2-3 steady-state batches/updates before trusting the setup.
9. Stop old instance's job, verify new one, then destroy old instance.
10. Set up a quiet background sync/monitor loop; don't poll manually.
