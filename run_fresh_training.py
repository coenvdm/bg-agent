"""Standalone fresh-start training run, mirroring explore.ipynb cells 46-49
exactly (same hyperparameters, same checkpoint paths), minus the IPython
dependency -- writes PNG charts to disk instead of clear_output/show, and can
resume itself (re-running this script picks up bg_agent_ppo.pt if present).

Not part of the permanent project structure -- a one-off verification script,
deleted after use.
"""
import json
import logging
import random
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent.policy import BGPolicyNetwork
from agent.ppo import PPOConfig, PPOTrainer
from train import _train_parallel, N_PLAYERS, evaluate_policy

for _ln in ('env.game_loop', 'env.tavern_pool', 'agent.ppo',
            'symbolic.firestone_client', 'symbolic.effect_handler'):
    logging.getLogger(_ln).setLevel(logging.WARNING)

if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training device: {DEVICE}', flush=True)

    PPO_PATH         = Path('bg_agent_ppo.pt')
    PPO_BACKUP_PATH  = Path('bg_agent_ppo_backup.pt')
    HISTORY_PATH     = Path('data/fresh_training_history.json')
    CHART_PATH       = Path('data/training_progress.png')

    with open('bg_card_definitions.json', encoding='utf-8') as _f:
        _raw = json.load(_f)
    if isinstance(_raw, dict) and 'cards' in _raw:
        _raw = _raw['cards']
    if isinstance(_raw, list):
        _raw = {d.get('card_id', str(i)): d for i, d in enumerate(_raw)}
    card_defs_train = _raw
    print(f'Loaded {len(card_defs_train)} card defs', flush=True)

    # ---- Run sizing / LR + entropy anneal schedule -------------------------
    # N_GAMES raised 5000->40000 (2026-08-31): the prior 5000-game/312-update
    # run produced ZERO placement improvement -- part of that is the starved
    # optimiser fixed in _make_ppo_trainer below, but even a fixed optimiser
    # needs enough games to actually learn something. 40000 games gives it
    # room to do that.
    # Raised again 40000->150000 after benchmarking self-play at 8.64 games/sec
    # on this host (was 2.79 with the old 8-CUDA-worker config): 40000 games is
    # only ~1.3h of wall-clock here, far too short to judge whether the fixes
    # worked. 150000 games ~= 4.8h of self-play plus ~12% eval overhead
    # (~5.4h total, ~$0.64 at $0.118/hr).
    N_GAMES = 150000

    # anneal_steps sizing: the prior run averaged ~2,345 transitions per PPO
    # update at UPDATE_INTERVAL=16 games/update, i.e.
    #     2345 transitions/update / 16 games/update ~= 146.6 ~= ~150 steps/game
    # Expected total steps for this run, at that rate:
    #     N_GAMES * ~150 steps/game = 40000 * 150 = 6,000,000 steps
    # anneal_steps is set to that estimate so lr (2.5e-4 -> 5e-5) and
    # entropy_coef (0.015 -> 0.004) anneal gradually across the ENTIRE run,
    # reaching their floor right around the last games -- instead of
    # collapsing to the floor early (starving exploration for most of
    # training) or barely annealing at all (never settling down for a
    # cleaner late-training policy).
    # Resized for N_GAMES=150000. The prior run's actual rate was
    # 731,858 transitions / 4,128 games ~= 178 transitions/game (2 training
    # seats per game), so:
    #     150000 games * ~178 steps/game ~= 26,700,000 steps
    ANNEAL_STEPS = 26_000_000


    def _make_fresh_policy():
        return BGPolicyNetwork(
            card_dim=44, d_model=256, nhead=8, num_layers=4, scalar_dim=100, dropout=0.1
        ).to(DEVICE)


    def _make_ppo_trainer(policy):
        # 2026-08-31: the previous config here (lr=3e-5, clip_eps=0.10,
        # entropy_coef=0.05, batch_size=1024) starved the optimiser. With
        # ~2,345 transitions/update and batch_size=1024 that's only 3
        # minibatches * 4 epochs = 12 gradient steps/update (~3,700 gradient
        # steps across the entire 312-update run). Measured effect: max
        # importance weight stayed at 1.005 (policy essentially frozen) and
        # entropy ROSE from 1.44->1.70 over the run (entropy_coef=0.05
        # dominating a policy gradient that was barely moving), and average
        # placement never left 4.58 (worse than the scripted heuristic/greedy
        # baselines). Relying on PPOConfig's new defaults instead (lr
        # 2.5e-4->5e-5 anneal, clip_eps=0.2, entropy_coef 0.015->0.004 anneal,
        # batch_size=256) fixes all three at once: batch_size alone takes
        # minibatches/epoch from 3 to ~9 (36 gradient steps/update), and the
        # much higher initial lr/entropy_coef give the policy room to actually
        # move early, annealing down later for a cleaner late-run policy. See
        # CONTEXT.md 2026-08-31 for the full diagnosis.
        cfg = PPOConfig(device=DEVICE, anneal_steps=ANNEAL_STEPS)
        return PPOTrainer(policy, cfg)


    policy_train = _make_fresh_policy()
    ppo_trainer  = _make_ppo_trainer(policy_train)

    # Resume if a checkpoint from THIS run already exists (e.g. this script was
    # stopped and re-launched). This is the only checkpoint this run will ever see,
    # since the pre-fix checkpoints were moved aside to checkpoint_backups/ before
    # this run's very first launch -- so "resume" here never means "resume the old
    # pre-fix training," only "continue this fresh run."
    # Per-game series
    game_rewards         = []
    game_lengths          = []
    action_counts          = {}
    board_sizes            = []   # avg board size at END_TURN
    sell_counts            = []   # SELL actions this game
    place_counts           = []   # PLACE actions this game
    level_counts           = []   # LEVEL_UP actions this game
    total_action_counts    = []   # all actions this game (for rate denominators)
    sell_place_ratios      = []   # SELL/PLACE this game (nan if no PLACE actions)
    level_rates            = []   # LEVEL_UP / total actions this game
    action_type_game_rates = {k: [] for k in range(9)}  # per-game % of actions of each type
    train_placements       = []   # avg placement (1=best..8=worst) of train_current seats
    heuristic_placements   = []   # avg placement of heuristic seats
    greedy_placements      = []   # avg placement of greedy seats
    # Per-update series
    ppo_losses         = []
    ppo_values         = []
    entropies          = []
    max_ws             = []
    skip_flags         = []       # 1 if update had 0 minibatches (fully skipped), else 0
    # New PPO diagnostics (agent/ppo.py PPOTrainer.update() return keys, added
    # 2026-08-31 alongside the optimiser-starvation fix -- these are what
    # confirm the fix actually took effect: clip_frac near 0 means the policy
    # still isn't moving; explained_var is the value function's health check.
    approx_kls         = []
    clip_fracs         = []
    explained_vars     = []
    n_minibatches_hist = []
    # Fixed-opponent eval series (train.py's evaluate_policy()) -- the only
    # honest, un-gameable progress signal: one point every EVAL_EVERY updates,
    # so these are indexed by their own eval_updates list, not by update index.
    eval_updates          = []
    eval_mean_placement   = []
    eval_top1_rate        = []
    eval_top4_rate        = []
    # Per-update mean/std pairs (mean+std over exactly the games that fed each
    # update, so charts show one aggregated point per update instead of one raw
    # point per game -- game-to-game noise was dominating the picture).
    update_reward_avg,     update_reward_std     = [], []
    update_length_avg,     update_length_std     = [], []
    update_board_avg,      update_board_std      = [], []
    update_sellplace_avg,  update_sellplace_std  = [], []
    update_levelrate_avg,  update_levelrate_std  = [], []
    update_action_rate_avg = {k: [] for k in range(9)}   # per-update, % of actions of each type
    update_train_plc_avg,      update_train_plc_std      = [], []
    update_heuristic_plc_avg,  update_heuristic_plc_std  = [], []
    update_greedy_plc_avg,     update_greedy_plc_std     = [], []
    best_avg10    = float('-inf')

    if PPO_PATH.exists():
        loaded = ppo_trainer.load_checkpoint(str(PPO_PATH))
        if loaded and HISTORY_PATH.exists():
            hist = json.loads(HISTORY_PATH.read_text())
            game_rewards         = hist.get('game_rewards', [])
            game_lengths         = hist.get('game_lengths', [])
            action_counts        = {int(k): v for k, v in hist.get('action_counts', {}).items()}
            board_sizes           = hist.get('board_sizes', [])
            sell_counts            = hist.get('sell_counts', [])
            place_counts           = hist.get('place_counts', [])
            level_counts           = hist.get('level_counts', [])
            total_action_counts    = hist.get('total_action_counts', [])
            sell_place_ratios       = hist.get('sell_place_ratios', [])
            level_rates              = hist.get('level_rates', [])
            _loaded_atgr = hist.get('action_type_game_rates', {})
            action_type_game_rates = {k: _loaded_atgr.get(str(k), []) for k in range(9)}
            train_placements       = hist.get('train_placements', [])
            heuristic_placements   = hist.get('heuristic_placements', [])
            greedy_placements       = hist.get('greedy_placements', [])
            ppo_losses               = hist.get('ppo_losses', [])
            ppo_values               = hist.get('ppo_values', [])
            entropies               = hist.get('entropies', [])
            max_ws                  = hist.get('max_ws', [])
            skip_flags               = hist.get('skip_flags', [])
            # New keys (2026-08-31): .get(..., []) so an older history file
            # written before this session simply resumes with empty series
            # here instead of crashing.
            approx_kls              = hist.get('approx_kls', [])
            clip_fracs              = hist.get('clip_fracs', [])
            explained_vars          = hist.get('explained_vars', [])
            n_minibatches_hist      = hist.get('n_minibatches_hist', [])
            eval_updates            = hist.get('eval_updates', [])
            eval_mean_placement     = hist.get('eval_mean_placement', [])
            eval_top1_rate          = hist.get('eval_top1_rate', [])
            eval_top4_rate          = hist.get('eval_top4_rate', [])
            update_reward_avg       = hist.get('update_reward_avg', [])
            update_reward_std       = hist.get('update_reward_std', [])
            update_length_avg       = hist.get('update_length_avg', [])
            update_length_std       = hist.get('update_length_std', [])
            update_board_avg        = hist.get('update_board_avg', [])
            update_board_std        = hist.get('update_board_std', [])
            update_sellplace_avg    = hist.get('update_sellplace_avg', [])
            update_sellplace_std    = hist.get('update_sellplace_std', [])
            update_levelrate_avg    = hist.get('update_levelrate_avg', [])
            update_levelrate_std    = hist.get('update_levelrate_std', [])
            _loaded_uara = hist.get('update_action_rate_avg', {})
            update_action_rate_avg = {k: _loaded_uara.get(str(k), []) for k in range(9)}
            update_train_plc_avg      = hist.get('update_train_plc_avg', [])
            update_train_plc_std      = hist.get('update_train_plc_std', [])
            update_heuristic_plc_avg  = hist.get('update_heuristic_plc_avg', [])
            update_heuristic_plc_std  = hist.get('update_heuristic_plc_std', [])
            update_greedy_plc_avg     = hist.get('update_greedy_plc_avg', [])
            update_greedy_plc_std     = hist.get('update_greedy_plc_std', [])
            best_avg10    = hist.get('best_avg10', float('-inf'))
            print(f'Resumed this run: steps={ppo_trainer.total_steps}, '
                  f'games={len(game_rewards)}, updates={ppo_trainer.update_count}', flush=True)
        else:
            print('WARNING: checkpoint present but incompatible or history missing -- '
                  'starting fresh anyway', flush=True)
    else:
        max_w = max(p.abs().max().item() for p in policy_train.parameters())
        print(f'Fresh start: step 0, max_w={max_w:.4f}', flush=True)

    ACTION_NAMES = ['BUY', 'SELL', 'PLACE', 'REROLL', 'FREEZE', 'LEVEL', 'HERO_PWR', 'END_TURN', 'ACTIVATE']

    # N_GAMES defined earlier alongside ANNEAL_STEPS (both needed before
    # _make_ppo_trainer() is called above).
    # N_WORKERS / WORKER_DEVICE retuned 2026-08-31 by direct benchmark on the
    # CURRENT host (RTX 4060 Ti, contract 49396123). The previous value of 8
    # was inherited from the fractional RTX 3090 host that was cgroup-capped at
    # 8.64 real cores; this box has ~30.7 real cores (/sys/fs/cgroup/cpu.max =
    # "3071999 100000"), so 8 workers left ~22 cores idle.
    #
    # Measured (32 timed games after an 8-game warmup, ulimit -n 65536):
    #     8 workers  on cuda -> 2.79 games/sec   <- the old config
    #    12 workers  on cpu  -> 4.67 games/sec
    #    20 workers  on cpu  -> 6.80 games/sec
    #    26 workers  on cpu  -> 8.64 games/sec   <- 3.1x the old config
    #
    # Workers run the policy on CPU while the PPO trainer stays on DEVICE
    # (cuda): _train_parallel's `device` argument is passed ONLY to
    # _worker_init, so it sets the WORKERS' device and nothing else (the
    # trainer's device comes from PPOConfig). Each CUDA worker otherwise holds
    # its own ~0.3-0.6GB CUDA context, which caps worker count on an 8GB card
    # long before the CPU cores run out. Self-play is single-sample inference
    # on a small net (d_model=256, 4 layers) plus CPU-heavy game logic, so it
    # is latency-bound rather than throughput-bound and does not need the GPU.
    #
    # NOTE: >12 workers requires a raised file-descriptor limit. At the default
    # ulimit -n 1024 this dies with "OSError: [Errno 24] Too many open files"
    # during pool construction -- the launch command must set `ulimit -n 65536`
    # (see the vast-ai-training skill).
    N_WORKERS       = 26
    WORKER_DEVICE   = 'cpu'
    UPDATE_INTERVAL = 26     # == N_WORKERS: games are dispatched in batches of
                              # N_WORKERS, so this fires exactly one PPO update
                              # per dispatched batch -- predictable, and at
                              # ~178 transitions/game gives ~4.6k transitions
                              # per update (~18 minibatches x 4 epochs ~= 72
                              # optimizer steps/update, vs the 12 that starved
                              # the previous run).
    BACKUP_EVERY    = 5
    # EVAL_EVERY / EVAL_N_GAMES / EVAL_WORKERS -- retuned 2026-08-31 after
    # measuring the single-process evaluate_policy() at 3.23s/game: at the old
    # EVAL_EVERY=25 / EVAL_N_GAMES=32 that was ~103s of eval per ~400 training
    # games, AND n=32 gives a placement standard error of ~0.35 (placement std
    # ~2.0) against an effect size we care about of only ~0.3-0.5 placement --
    # noise on the same order as the signal (a freshly-initialised random
    # network scored 3.875 and 4.125 on two different 8-game evals, spanning
    # the whole effect size on pure noise). n=128 gives a placement standard
    # error of ~0.18 (vs 0.35 at n=32) -- small enough to actually resolve the
    # 0.3-0.5 placement improvement we need to detect.
    # evaluate_policy() now parallelises across a CPU ProcessPoolExecutor (see
    # train.py's _evaluate_policy_parallel / _worker_run_eval_game): measured
    # locally (8-core box) at n_games=32, per-game cost drops from 2.69s
    # (n_workers=1) to ~1.37s (n_workers=8), a ~2x speedup, and the SAME
    # policy/seed gives bit-identical mean_placement at n_workers in
    # {1,4,8,12}. IMPORTANT — this does NOT make one eval call itself cheaper
    # in absolute wall-clock: n_games=128 at n_workers=8-12 measured
    # ~150-159s locally (~180-190s scaled to the 3.23s/game target box),
    # versus ~86s locally (~103s scaled) for the old n_games=32 sequential
    # call -- quadrupling n_games outweighs the ~2x per-game parallel
    # speedup, so a single eval call now costs MORE wall-clock, not less.
    # What actually pays for the 4x larger n_games is EVAL_EVERY=50 (vs 25):
    # doubling the interval between eval calls keeps the eval-overhead
    # FRACTION of total wall-clock roughly flat versus the old cadence at any
    # training throughput in the 4-12 games/sec range (verified by
    # measurement -- see the session notes), rather than doubling it as
    # keeping EVAL_EVERY=25 would have. Net effect: ~4x better eval
    # resolution (SE 0.35 -> 0.18) for roughly the same overhead fraction as
    # before, not "free" or "cheaper" eval.
    # Retuned 2026-08-31 from a benchmark on THIS host rather than the 8-core
    # dev box the previous values came from. Measured, 64 eval games:
    #     sequential      -> 87.4s
    #     12 eval workers -> 22.3s
    #     20 eval workers -> 18.1s   <- knee of the curve
    #     28 eval workers -> 18.2s   (no further gain)
    # mean_placement was byte-identical across all worker counts, confirming
    # the seed-by-game-index aggregation really is order-independent.
    #
    # So 128 eval games now costs ~36s. At 8.64 training games/sec and
    # UPDATE_INTERVAL=26, EVAL_EVERY=100 means 2600 training games (~301s)
    # between evals -> ~12% wall-clock overhead, for a placement standard
    # error of ~0.18 (n=128). EVAL_EVERY=50 would have cost ~24%, which is not
    # worth it: the high-frequency progress signal comes for free from the
    # fixed GreedyPlayAgent/HeuristicAgent seats present in EVERY training game
    # (train_plc - greedy_plc), averaged over thousands of games. This eval is
    # the low-frequency ABSOLUTE anchor that no shaping term can inflate.
    EVAL_EVERY      = 100
    EVAL_N_GAMES    = 128
    EVAL_WORKERS    = 20     # CPU workers for evaluate_policy's pool -- eval
                              # workers always run the policy on CPU regardless
                              # of DEVICE (see train.py), so this is independent
                              # of N_WORKERS/training GPU memory. Measured
                              # locally, n_workers=8 was ~6% faster than 12 at
                              # n_games=128 (an 8-core dev box oversubscribes
                              # at 12) -- the target box's ~8.64 real cores
                              # (see N_WORKERS comment above) suggest 8-10
                              # may edge out 12 there too; kept at 12 as
                              # specified, worth an A/B on the target host.

    _seed_offset = ppo_trainer.total_steps
    random.seed(_seed_offset); np.random.seed(_seed_offset); torch.manual_seed(_seed_offset)

    # Windowed (last-200-game) action mix, since the cumulative mix converges too
    # slowly to show whether the fix is changing recent behaviour.
    recent_actions = deque(maxlen=200 * 100)  # ~100 actions/game * 200 games headroom

    # scalar_context[19] = board_size/7.0 -- own-board scalar block, index confirmed
    # against symbolic/board_computer.py's to_scalar_vector() (own_scalar is the
    # first 24 dims of the 100-dim scalar_context, per game_loop.py's concatenate
    # order) -- and directly verified against ground-truth len(ps.board) this session.
    BOARD_SIZE_SCALAR_IDX = 19

    # rolling-window sizes for the ratio/rate panels (games for per-game series,
    # updates for per-update series) -- wide enough to smooth game-to-game noise,
    # narrow enough to stay responsive to real trend shifts.
    UPDATE_WINDOW      = 10
    UPDATE_TREND_WINDOW = 10   # rolling-trend window (in updates) for the reward/length panels


    def _save_history():
        HISTORY_PATH.write_text(json.dumps({
            'game_rewards': game_rewards,
            'game_lengths': game_lengths,
            'action_counts': action_counts,
            'best_avg10': best_avg10,
            'board_sizes': board_sizes,
            'sell_counts': sell_counts,
            'place_counts': place_counts,
            'level_counts': level_counts,
            'total_action_counts': total_action_counts,
            'sell_place_ratios': sell_place_ratios,
            'level_rates': level_rates,
            'action_type_game_rates': action_type_game_rates,
            'train_placements': train_placements,
            'heuristic_placements': heuristic_placements,
            'greedy_placements': greedy_placements,
            'ppo_losses': ppo_losses,
            'ppo_values': ppo_values,
            'entropies': entropies,
            'max_ws': max_ws,
            'skip_flags': skip_flags,
            'approx_kls': approx_kls,
            'clip_fracs': clip_fracs,
            'explained_vars': explained_vars,
            'n_minibatches_hist': n_minibatches_hist,
            'eval_updates': eval_updates,
            'eval_mean_placement': eval_mean_placement,
            'eval_top1_rate': eval_top1_rate,
            'eval_top4_rate': eval_top4_rate,
            'update_reward_avg': update_reward_avg,
            'update_reward_std': update_reward_std,
            'update_length_avg': update_length_avg,
            'update_length_std': update_length_std,
            'update_board_avg': update_board_avg,
            'update_board_std': update_board_std,
            'update_sellplace_avg': update_sellplace_avg,
            'update_sellplace_std': update_sellplace_std,
            'update_levelrate_avg': update_levelrate_avg,
            'update_levelrate_std': update_levelrate_std,
            'update_action_rate_avg': update_action_rate_avg,
            'update_train_plc_avg': update_train_plc_avg,
            'update_train_plc_std': update_train_plc_std,
            'update_heuristic_plc_avg': update_heuristic_plc_avg,
            'update_heuristic_plc_std': update_heuristic_plc_std,
            'update_greedy_plc_avg': update_greedy_plc_avg,
            'update_greedy_plc_std': update_greedy_plc_std,
        }))


    def _append_update_stat(series, avg_list, std_list, window=UPDATE_INTERVAL):
        """Append mean/std of the last *window* entries of a per-game series.

        NaNs (e.g. a game with 0 PLACE actions, undefined sell:place ratio) are
        dropped before averaging. No-op if the window has nothing valid to average.
        """
        recent = [v for v in series[-window:] if v is not None and not np.isnan(v)]
        if recent:
            avg_list.append(float(np.mean(recent)))
            std_list.append(float(np.std(recent)))


    def _append_update_avg(series, avg_list, window=UPDATE_INTERVAL):
        """Same as _append_update_stat but skips std -- used for the 9-way action-mix
        breakdown, where std bands on every series would just be visual noise.
        """
        recent = [v for v in series[-window:] if v is not None and not np.isnan(v)]
        if recent:
            avg_list.append(float(np.mean(recent)))


    def _rolling_mean_series(values, window):
        """(x_indices, means) for a simple rolling mean, window in list-index units."""
        n = len(values)
        if n < window:
            return [], []
        sm = np.convolve(values, np.ones(window) / window, mode='valid')
        return list(range(window - 1, n)), sm.tolist()


    def _plot_avg_trend_std(ax, avg_series, std_series, color, trend_window=UPDATE_TREND_WINDOW, label=None):
        """One point per PPO update: faint mean line, ±1 std shaded band, bold
        rolling trend line. Replaces plotting individual per-game data points,
        which is too noisy to read once there are more than a few hundred games.
        """
        if not avg_series:
            return
        xs_all = list(range(len(avg_series)))
        if std_series:
            # std_series can be shorter than avg_series when resuming a run
            # whose saved history predates std-tracking -- shade only the
            # tail where both exist instead of crashing on the length mismatch.
            n = min(len(std_series), len(avg_series))
            xs_std = xs_all[-n:]
            lo = [a - s for a, s in zip(avg_series[-n:], std_series[-n:])]
            hi = [a + s for a, s in zip(avg_series[-n:], std_series[-n:])]
            ax.fill_between(xs_std, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(xs_all, avg_series, alpha=0.35, color=color)
        xs, ys = _rolling_mean_series(avg_series, trend_window)
        if xs:
            ax.plot(xs, ys, color=color, lw=2, label=label or f'{trend_window}-update trend')
            ax.legend()


    def _save_chart():
        fig, axes = plt.subplots(2, 6, figsize=(36, 8))

        # ---- Row 1: training internals / health ----
        # Reward: one point per PPO update (mean over exactly the games whose
        # transitions fed that update), not one point per game -- per-game noise
        # was dominating the picture. Trend line is a rolling mean over the
        # latest UPDATE_TREND_WINDOW updates on top of that already-averaged series.
        ax = axes[0][0]
        if update_reward_avg:
            _plot_avg_trend_std(ax, update_reward_avg, update_reward_std, 'steelblue')
            ax.axhline(0, color='gray', lw=0.8, ls='--')
            ax.set_xlabel('PPO update'); ax.set_ylabel(f'Mean reward ({UPDATE_INTERVAL} games/update)')
            ax.set_title(f'Training Reward (steps={ppo_trainer.total_steps:,}, games={len(game_rewards)})')

        ax = axes[0][1]
        if ppo_losses:
            ax.plot(ppo_losses, color='crimson', label='total loss')
            ax.plot(ppo_values, color='orange', label='value loss')
            ax.set_xlabel('PPO update'); ax.set_ylabel('Loss')
            ax.set_title(f'PPO Losses ({ppo_trainer.update_count} updates)')
            ax.legend()

        ax = axes[0][2]
        if entropies:
            ax.plot(entropies, color='mediumpurple')
            ax.set_xlabel('PPO update'); ax.set_ylabel('Entropy')
            ax.set_title(f'Policy Entropy (last={entropies[-1]:.3f})')

        ax = axes[0][3]
        if max_ws:
            ax.plot(max_ws, color='firebrick')
            ax.axhline(2.0, color='gray', lw=0.8, ls='--')  # notebook's divergence warning threshold
            ax.set_xlabel('PPO update'); ax.set_ylabel('Max |weight|')
            ax.set_title(f'Max Weight Magnitude (last={max_ws[-1]:.3f})')

        ax = axes[0][4]
        xs, ys = _rolling_mean_series(skip_flags, UPDATE_WINDOW)
        if xs:
            ys_pct = [100 * y for y in ys]
            ax.plot(xs, ys_pct, color='slategray')
            ax.set_ylim(-2, 102)
            ax.set_xlabel('PPO update'); ax.set_ylabel('% updates skipped')
            ax.set_title(f'Skipped-Update Rate ({UPDATE_WINDOW}-update window)')

        ax = axes[0][5]
        # PPO health diagnostics added 2026-08-31 alongside the optimiser-
        # starvation fix -- this is what actually confirms the fix took:
        # clip_frac near 0 means the policy still isn't moving (the exact
        # failure being fixed), a healthy run shows ~0.05-0.30. explained_var
        # is the value function's health check (1.0 = perfect predictions,
        # <=0 = value net is not tracking returns at all).
        if clip_fracs or explained_vars:
            if clip_fracs:
                ax.plot(clip_fracs, color='crimson', label='clip_frac')
            if explained_vars:
                ax.plot(explained_vars, color='seagreen', label='explained_var')
            ax.axhline(0.0, color='gray', lw=0.8, ls='--')
            ax.set_xlabel('PPO update'); ax.set_ylabel('Fraction / R²')
            ax.set_title('PPO Health (clip_frac, explained_var)')
            ax.legend(fontsize=7)

        # ---- Row 2: game shape / strategy ----
        ax = axes[1][0]
        if update_length_avg:
            _plot_avg_trend_std(ax, update_length_avg, update_length_std, 'teal')
            ax.set_xlabel('PPO update'); ax.set_ylabel(f'Mean rounds ({UPDATE_INTERVAL} games/update)')
            ax.set_title(f'Game Length (avg last 50 updates={np.mean(update_length_avg[-50:]):.1f})')

        ax = axes[1][1]
        if update_board_avg:
            _plot_avg_trend_std(ax, update_board_avg, update_board_std, 'darkorange')
            ax.axhline(7, color='gray', lw=0.8, ls='--')  # max possible board size
            ax.set_xlabel('PPO update'); ax.set_ylabel(f'Minions ({UPDATE_INTERVAL} games/update)')
            ax.set_title(f'Avg Board Size @ END_TURN (last 50 updates={np.mean(update_board_avg[-50:]):.2f})')

        ax = axes[1][2]
        if update_sellplace_avg:
            _plot_avg_trend_std(ax, update_sellplace_avg, update_sellplace_std, 'crimson')
            ax.axhline(1.0, color='gray', lw=0.8, ls='--')  # sell == place
            ax.set_xlabel('PPO update'); ax.set_ylabel(f'SELL / PLACE ({UPDATE_INTERVAL} games/update)')
            ax.set_title(f'Sell:Place Ratio (last={update_sellplace_avg[-1]:.2f})')

        ax = axes[1][3]
        # Stacked share of each action type per update -- replaces the old
        # LEVEL_UP-only panel with the full BUY/SELL/PLACE/.../ACTIVATE breakdown
        # in one view, since it's a composition (sums to ~100%) not independent rates.
        _action_colors = ['steelblue', 'crimson', 'seagreen', 'darkorange', 'mediumpurple',
                           'goldenrod', 'teal', 'slategray', 'deeppink']
        _lens = [len(update_action_rate_avg[k]) for k in range(9)]
        if all(_lens):
            n = min(_lens)
            xs = list(range(n))
            series = [[100 * v for v in update_action_rate_avg[k][:n]] for k in range(9)]
            ax.stackplot(xs, *series, labels=ACTION_NAMES, colors=_action_colors, alpha=0.85)
            ax.set_ylim(0, 100)
            ax.set_xlabel('PPO update'); ax.set_ylabel(f'% of actions ({UPDATE_INTERVAL} games/update)')
            ax.set_title('Action Mix')
            ax.legend(loc='upper left', fontsize=6, ncol=3, framealpha=0.7)

        ax = axes[1][4]
        # Per-update mean + trend line, same style as the other row-2 panels --
        # no std shading here since 3 overlapping bands on one axes reads as
        # clutter rather than signal.
        for avg_series, label, color in [
            (update_train_plc_avg, 'train_current', 'steelblue'),
            (update_heuristic_plc_avg, 'heuristic', 'darkorange'),
            (update_greedy_plc_avg, 'greedy', 'seagreen'),
        ]:
            _plot_avg_trend_std(ax, avg_series, None, color, label=f'{label} trend')
        if update_train_plc_avg or update_heuristic_plc_avg or update_greedy_plc_avg:
            ax.axhline(4.5, color='gray', lw=0.8, ls='--')  # mid-table
            ax.invert_yaxis()  # lower placement = better = "up" on the chart
            ax.set_xlabel('PPO update'); ax.set_ylabel(f'Avg placement ({UPDATE_INTERVAL} games/update, inverted, up=better)')
            ax.set_title('Placement vs Baselines')

        ax = axes[1][5]
        # The honest metric (train.py's evaluate_policy): one deterministic
        # policy seat vs. 7 FIXED greedy-scripted seats, evaluated every
        # EVAL_EVERY updates. Unlike the "Placement vs Baselines" panel to its
        # left -- which is measured against a SnapshotPool that co-evolves
        # with the training policy, so it can plateau at ~4.5 even while the
        # whole pool keeps improving together -- this opponent never changes,
        # so a drop below 4.5 here can only mean the policy itself got better.
        # Same inverted-y convention as the placement panel: up = better.
        if eval_mean_placement:
            ax.plot(eval_updates, eval_mean_placement, color='darkorchid', marker='o', ms=3)
            ax.axhline(4.5, color='gray', lw=0.8, ls='--')  # random-play expectation, 8 players
            ax.invert_yaxis()
            ax.set_xlabel('PPO update'); ax.set_ylabel(f'Mean placement ({EVAL_N_GAMES} games, inverted, up=better)')
            ax.set_title(f'Eval vs Fixed Greedy (last={eval_mean_placement[-1]:.2f})')

        plt.tight_layout()
        fig.savefig(CHART_PATH, dpi=100)
        plt.close(fig)


    def _on_batch(game_idx, summaries, transitions, elapsed):
        for summary, game_trans in zip(summaries, transitions):
            # final_rewards has one entry per seat (all 8 players: training agents,
            # HeuristicAgent, GreedyPlayAgent, frozen snapshots) -- averaging all of
            # them mixes in scripted/frozen opponents whose reward never changes,
            # diluting the actual training-policy signal. Filter to train_current
            # seats using agent_labels (falls back to the old all-seats average if
            # labels are ever missing, so this never crashes on unexpected input).
            labels = summary.get('agent_labels', {})
            placements = summary.get('placements', {})
            train_pids = [pid for pid, lbl in labels.items() if lbl == 'train_current']
            train_rewards = [summary['final_rewards'][pid] for pid in train_pids
                              if pid in summary['final_rewards']]
            game_rewards.append(float(np.mean(train_rewards)) if train_rewards
                                 else float(np.mean(list(summary['final_rewards'].values()))))
            game_lengths.append(summary['n_rounds'])

            # Avg board size at END_TURN (type_action==7) across this game's training
            # seats -- the same "board size going into combat" diagnostic used
            # manually all session, now tracked live instead of needing a separate
            # instrumented script each time.
            endturn_sizes = [t.scalar_context[BOARD_SIZE_SCALAR_IDX] * 7.0
                              for t in game_trans if t.type_action == 7]
            if endturn_sizes:
                board_sizes.append(float(np.mean(endturn_sizes)))

            # Per-game action tallies for the sell:place, level-up, and action-mix panels.
            type_counts_this_game = [0] * 9
            for t in game_trans:
                type_counts_this_game[t.type_action] += 1
            n_sell, n_place, n_level = type_counts_this_game[1], type_counts_this_game[2], type_counts_this_game[5]
            sell_counts.append(n_sell)
            place_counts.append(n_place)
            level_counts.append(n_level)
            total_action_counts.append(len(game_trans))
            sell_place_ratios.append(n_sell / n_place if n_place > 0 else float('nan'))
            level_rates.append(n_level / len(game_trans) if game_trans else float('nan'))
            for k in range(9):
                action_type_game_rates[k].append(
                    type_counts_this_game[k] / len(game_trans) if game_trans else float('nan')
                )

            # Avg placement (1=best..8=worst) for train_current vs. the fixed-quality
            # scripted baselines, per seat-type present this game -- the most direct
            # "is it actually getting good" signal, more interpretable than raw reward.
            for pids, sink in [
                (train_pids, train_placements),
                ([pid for pid, lbl in labels.items() if lbl == 'heuristic'], heuristic_placements),
                ([pid for pid, lbl in labels.items() if lbl == 'greedy'], greedy_placements),
            ]:
                vals = [placements[pid] for pid in pids if pid in placements]
                if vals:
                    sink.append(float(np.mean(vals)))

            for t in game_trans:
                action_counts[t.type_action] = action_counts.get(t.type_action, 0) + 1
                recent_actions.append(t.type_action)

        _ac_total = sum(action_counts.values()) or 1
        _ac_str = ' '.join(f'{ACTION_NAMES[k]}={100*action_counts.get(k,0)/_ac_total:.0f}%'
                            for k in range(len(ACTION_NAMES)))
        _rc_total = len(recent_actions) or 1
        _rc_counts = {}
        for a in recent_actions:
            _rc_counts[a] = _rc_counts.get(a, 0) + 1
        _rc_str = ' '.join(f'{ACTION_NAMES[k]}={100*_rc_counts.get(k,0)/_rc_total:.0f}%'
                            for k in range(len(ACTION_NAMES)))
        print(f'Games {game_idx-len(summaries)+1:4d}-{game_idx:4d}/{N_GAMES}  '
              f'batch={elapsed:.1f}s ({len(summaries)}w)  '
              f'reward={game_rewards[-1]:+.3f}  avg10={np.mean(game_rewards[-10:]):+.3f}  '
              f'steps={ppo_trainer.total_steps:,}', flush=True)
        print(f'  cumulative: {_ac_str}', flush=True)
        print(f'  recent~200: {_rc_str}', flush=True)


    def _on_update(metrics, update_count):
        global best_avg10
        ppo_losses.append(metrics.get('total_loss', 0.0))
        ppo_values.append(metrics.get('value_loss', 0.0))
        entropies.append(metrics.get('entropy', 0.0))
        skip_flags.append(1 if metrics.get('n_minibatches', 1) == 0 else 0)
        max_w = max(p.abs().max().item() for p in policy_train.parameters())
        max_ws.append(max_w)

        # New PPO diagnostics (see agent/ppo.py PPOTrainer.update()) -- use
        # .get(key, nan) throughout so this can never crash if a key is
        # missing (e.g. against an older/newer PPOTrainer build).
        approx_kls.append(metrics.get('approx_kl', float('nan')))
        clip_fracs.append(metrics.get('clip_frac', float('nan')))
        explained_vars.append(metrics.get('explained_var', float('nan')))
        n_minibatches_hist.append(metrics.get('n_minibatches', float('nan')))

        # Mean/std for exactly the games that fed this update's gradient step
        # (the last UPDATE_INTERVAL games recorded) -- one aggregated point per
        # update instead of one raw point per game for every game-shape panel.
        _append_update_stat(game_rewards, update_reward_avg, update_reward_std)
        _append_update_stat(game_lengths, update_length_avg, update_length_std)
        _append_update_stat(board_sizes, update_board_avg, update_board_std)
        _append_update_stat(sell_place_ratios, update_sellplace_avg, update_sellplace_std)
        _append_update_stat(level_rates, update_levelrate_avg, update_levelrate_std)
        for k in range(9):
            _append_update_avg(action_type_game_rates[k], update_action_rate_avg[k])
        _append_update_stat(train_placements, update_train_plc_avg, update_train_plc_std)
        _append_update_stat(heuristic_placements, update_heuristic_plc_avg, update_heuristic_plc_std)
        _append_update_stat(greedy_placements, update_greedy_plc_avg, update_greedy_plc_std)

        ppo_trainer.save_checkpoint(str(PPO_PATH), extra={'game': len(game_rewards)})
        if update_count % BACKUP_EVERY == 0:
            ppo_trainer.save_checkpoint(str(PPO_BACKUP_PATH), extra={'game': len(game_rewards)})
        _cur_avg10 = float(np.mean(game_rewards[-10:])) if len(game_rewards) >= 10 else float('-inf')
        if _cur_avg10 > best_avg10:
            best_avg10 = _cur_avg10
            ppo_trainer.save_checkpoint('bg_agent_ppo_best.pt',
                                         extra={'game': len(game_rewards), 'best_avg10': best_avg10})
        # Fixed-opponent eval (train.py's evaluate_policy) -- the only progress
        # signal here no shaping term can inflate. Collects no PPO
        # transitions and never touches ppo_trainer, so it's safe to run
        # mid-training. Uses ppo_trainer.total_steps as the seed so each
        # eval call sees a fresh, still-reproducible set of games.
        if update_count % EVAL_EVERY == 0:
            eval_result = evaluate_policy(
                policy_train, card_defs_train,
                n_games=EVAL_N_GAMES, opponent='greedy',
                device=DEVICE, seed=ppo_trainer.total_steps,
                n_workers=EVAL_WORKERS,
            )
            eval_updates.append(update_count)
            eval_mean_placement.append(eval_result['mean_placement'])
            eval_top1_rate.append(eval_result['top1_rate'])
            eval_top4_rate.append(eval_result['top4_rate'])
            print(f'  EVAL @ update {update_count}: '
                  f'mean_placement={eval_result["mean_placement"]:.2f} '
                  f'top1={eval_result["top1_rate"]:.2f} top4={eval_result["top4_rate"]:.2f} '
                  f'(vs greedy, {EVAL_N_GAMES} games)', flush=True)

        _save_history()
        _save_chart()
        _w_warn = '  WARNING: high weights' if max_w > 2.0 else ''
        print(f'update {update_count}: total_loss={metrics.get("total_loss"):.4f} '
              f'value_loss={metrics.get("value_loss"):.4f} entropy={metrics.get("entropy", 0.0):.3f} '
              f'max_w={max_w:.4f}{_w_warn} best_avg10={best_avg10:.3f}', flush=True)
        print(f'  diagnostics: approx_kl={metrics.get("approx_kl", float("nan")):.4f} '
              f'clip_frac={metrics.get("clip_frac", float("nan")):.3f} '
              f'explained_var={metrics.get("explained_var", float("nan")):.3f} '
              f'n_minibatches={metrics.get("n_minibatches", float("nan"))} '
              f'lr={metrics.get("lr", float("nan")):.2e} '
              f'entropy_coef={metrics.get("entropy_coef", float("nan")):.4f} '
              f'ret_std={metrics.get("ret_std", float("nan")):.3f}', flush=True)


    _train_parallel(
        N_GAMES, policy_train, ppo_trainer, card_defs_train,
        n_workers=N_WORKERS,
        update_interval=UPDATE_INTERVAL,
        seed=_seed_offset,
        # Workers only. _train_parallel passes this straight to _worker_init,
        # so it sets the worker processes' device; the PPO trainer keeps using
        # DEVICE (cuda) for updates. See the N_WORKERS comment for the
        # benchmark that motivated CPU workers.
        device=WORKER_DEVICE,
        on_batch=_on_batch,
        on_update=_on_update,
    )
    print(f'\nDone -- steps={ppo_trainer.total_steps:,}, updates={ppo_trainer.update_count}', flush=True)
