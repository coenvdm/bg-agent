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

from agent.policy import (make_policy, BGPolicyNetwork, N_ACTION_TYPES,
                          ACTION_TYPE_NAMES, PTR_SHOP_OFF)
from agent.ppo import PPOConfig, PPOTrainer
from train import (
    _train_parallel, N_PLAYERS, evaluate_policy,
    N_HEURISTIC_SLOTS, N_GREEDY_SLOTS,
    REDUCED_HEURISTIC_SLOTS, REDUCED_GREEDY_SLOTS,
)

for _ln in ('env.game_loop', 'env.tavern_pool', 'agent.ppo',
            'symbolic.firestone_client', 'symbolic.effect_handler'):
    logging.getLogger(_ln).setLevel(logging.WARNING)


class AdaptiveOpponentMix:
    """Hysteresis state machine switching between the full and reduced
    scripted-opponent mixes, off the honest fixed-opponent eval only.

    Background: every training game seats N_TRAIN_PLAYERS current-policy
    agents against 6 opponent slots, historically fixed at
    N_HEURISTIC_SLOTS HeuristicAgent + N_GREEDY_SLOTS GreedyPlayAgent + the
    rest sampled SnapshotPool policies. Once the policy has clearly beaten
    both scripted baselines, those 4 fixed slots are wasted compute -- see
    train.py's REDUCED_HEURISTIC_SLOTS / REDUCED_GREEDY_SLOTS for why the
    reduced mix keeps exactly one GreedyPlayAgent rather than dropping every
    scripted seat.

    This class owns ONLY the decision of which mix is active; it never
    touches SnapshotPool, _train_parallel, or any training state, which is
    what makes it possible to unit test with a synthetic sequence of eval
    values instead of running real games.

    Trigger source: the caller MUST feed this eval_mean_placement from
    train.py's evaluate_policy(..., opponent='greedy') -- the only progress
    signal that a co-evolving SnapshotPool or a shaping term cannot inflate.
    In-game placement vs the pool is confounded (the pool co-evolves with
    the policy, so "everyone improved together" looks identical to "nobody
    improved") and must never drive this trigger.

    Hysteresis: two DISTINCT thresholds (`low` < `high`) are required, not
    one -- a single threshold sitting right where the eval is expected to
    hover would flip the mix back and forth on ordinary eval noise. `low`
    and `high` bound a dead zone the eval must clear on `streak` CONSECUTIVE
    points (default 2) in the same direction before the mix actually
    switches, so one noisy point past a threshold is never enough on its
    own to flip it, and a run sitting between the thresholds never
    oscillates.
    """

    def __init__(self, full_mix, reduced_mix, low=2.0, high=3.0, streak=2,
                 reduced=False, log=None):
        self.full_mix    = tuple(full_mix)
        self.reduced_mix = tuple(reduced_mix)
        self.low    = low
        self.high   = high
        self.streak_required = streak
        self.reduced = reduced   # which mix is currently active
        self.low_streak  = 0
        self.high_streak = 0
        self.switch_count = 0    # tests / callers can assert on this directly
        # Injectable so tests can capture switch events without touching
        # stdout; production default matches the task's logging requirement
        # (print(..., flush=True)) exactly.
        self._log = log if log is not None else (lambda msg: print(msg, flush=True))

    @property
    def mix(self):
        """The currently active (n_heuristic, n_greedy) mix."""
        return self.reduced_mix if self.reduced else self.full_mix

    def update(self, update_count, eval_value):
        """Feed ONE fixed-opponent eval point; return the active mix after
        considering it (whether or not this point caused a switch).

        A None/NaN eval_value (evaluate_policy's failure path) never moves
        the hysteresis state -- it is treated as "no data for this point",
        not as a data point that happens to fail both thresholds, so a
        transient eval failure can never itself trigger or suppress a
        switch.
        """
        if eval_value is None or (isinstance(eval_value, float) and eval_value != eval_value):
            return self.mix

        if not self.reduced:
            self.low_streak = self.low_streak + 1 if eval_value < self.low else 0
            if self.low_streak >= self.streak_required:
                old, new = self.full_mix, self.reduced_mix
                self.reduced = True
                self.low_streak = self.high_streak = 0
                self.switch_count += 1
                self._log(
                    f'OPPONENT MIX SWITCH @ update {update_count}: '
                    f'eval_greedy={eval_value:.2f} < {self.low} for '
                    f'{self.streak_required} consecutive eval points -- '
                    f'reducing scripted seats (n_heuristic, n_greedy) {old} -> {new}'
                )
        else:
            self.high_streak = self.high_streak + 1 if eval_value > self.high else 0
            if self.high_streak >= self.streak_required:
                old, new = self.reduced_mix, self.full_mix
                self.reduced = False
                self.low_streak = self.high_streak = 0
                self.switch_count += 1
                self._log(
                    f'OPPONENT MIX SWITCH @ update {update_count}: '
                    f'eval_greedy={eval_value:.2f} > {self.high} for '
                    f'{self.streak_required} consecutive eval points -- '
                    f'restoring full scripted seats (n_heuristic, n_greedy) {old} -> {new}'
                )
        return self.mix


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
    # worked. [HISTORICAL, stale -- that host is gone and this predates the
    # engine fixes: 150000 games ~= 4.8h of self-play plus ~12% eval overhead,
    # ~5.4h total, ~$0.64 at $0.118/hr.]
    # Current projection (2026-09-01, contract 49546957, Xeon Gold 5418Y /
    # RTX 5000 Ada, 48 physical cores): measured 5.03 games/sec at
    # N_WORKERS=90. 150000 games / 5.03 games/sec ~= 8.3h of self-play, plus
    # ~2% eval overhead at the current EVAL_EVERY=100 / UPDATE_INTERVAL=90
    # cadence ~= 8.5h total, ~$3.40 at $0.401/hr.
    # Resized 2026-09-02 for the real-combat + REORDER engine. Measured on
    # THIS host (i9-14900KF, 24 physical cores, contract 49669406) with PPO
    # updates ON: 3.79 games/s at 16 workers, 5.77 at 24. Real combat makes
    # games END sooner (15.8 rounds vs 20.5 under the mock), which more than
    # pays for the simulator's cost -- throughput went UP, not down.
    # 200,000 games at ~4.5 games/s ~= 12.3h of self-play plus eval overhead,
    # inside the ~28h of runway $3.02 buys at $0.1081/hr.
    N_GAMES = 200000

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
    # Resized for the new engine. steps/game MEASURED across six
    # configurations on this host: 172.6, 156, 115, 113, 86.4, 63.8 -- a wide
    # spread, because transitions/game depends on how long the training seats
    # SURVIVE, which is high-variance (it is not drift: transitions/game was
    # flat within each run, e.g. 148 -> 146 -> 187 -> 144 by quarter).
    # Taking ~110 steps/game as a central estimate: 200,000 x 110 = 22M.
    # Deliberately set slightly BELOW that: if the true total overshoots,
    # lr/entropy simply sit at their floors for the tail (fine), whereas
    # setting it too HIGH means the anneal never completes and the run ends
    # with lr still elevated (not fine).
    ANNEAL_STEPS = 20_000_000


    def _make_fresh_policy():
        return make_policy().to(DEVICE)


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

    # Round-number buckets for the "action mix by game round" breakdown below.
    # Width-4 buckets with a catch-all top bucket, not raw round numbers --
    # median game length is ~15-21 rounds (see CLAUDE.md's Reward Shaping
    # section), so a per-exact-round breakdown would get sparse and noisy
    # past round ~20 while most of the interesting strategy shift (forced
    # early buys -> mid-game economy management -> late-game board-locking)
    # is already visible at this resolution. Defined here, before the series
    # declarations and resume-load block right below, since both need it.
    ROUND_BUCKET_LABELS = ['R1-4', 'R5-8', 'R9-12', 'R13-16', 'R17-20', 'R21+']
    N_ROUND_BUCKETS = len(ROUND_BUCKET_LABELS)

    def _round_bucket(round_num):
        """Map a 1-indexed round number to a bucket index, or None if unknown.

        round_num is None for any transition recorded before this field
        existed (old resumed history) or for a caller that never had a
        player_state to read it from -- always drop those rather than
        guessing, so the per-bucket rates stay exact instead of silently
        diluted.
        """
        if round_num is None:
            return None
        return min((int(round_num) - 1) // 4, N_ROUND_BUCKETS - 1)

    # Per-game series
    game_rewards         = []
    game_lengths          = []
    action_counts          = {}
    board_sizes            = []   # avg board size at END_TURN
    endturn_golds          = []   # avg UNSPENT gold at END_TURN (see GOLD_SCALAR_IDX)
    endturn_tiers          = []   # avg tavern tier at END_TURN
    combat_winrates        = []   # per-game combat win rate, training seats
    combat_damage          = []   # per-game mean damage taken per combat
    choice_events          = []   # per-game count of real choice actions (types 10/11)
    choice_branch0_rates   = []   # share of CHOOSE_OPTION picks that took branch 0
    sell_counts            = []   # SELL actions this game
    place_counts           = []   # PLACE actions this game
    level_counts           = []   # LEVEL_UP actions this game
    total_action_counts    = []   # all actions this game (for rate denominators)
    sell_place_ratios      = []   # SELL/PLACE this game (nan if no PLACE actions)
    level_rates            = []   # LEVEL_UP / total actions this game
    action_type_game_rates = {k: [] for k in range(N_ACTION_TYPES)}  # per-game % of actions of each type
    # Per-game action-mix rate, broken down by round bucket too -- same shape as
    # action_type_game_rates, indexed [bucket][action_type] -> list of per-game
    # rates (nan for a game with 0 actions in that bucket, e.g. a game that never
    # reached R21+). Needs round_num on each Transition (added 2026-09-03).
    round_bucket_action_game_rates = {b: {k: [] for k in range(N_ACTION_TYPES)}
                                       for b in range(N_ROUND_BUCKETS)}
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
    eval_mean_placement   = []      # vs greedy   -- KEEP: the historical series
    eval_top1_rate        = []
    eval_top4_rate        = []
    eval_heur_mean_placement = []   # vs heuristic
    eval_ref_mean_placement  = []   # vs 7 frozen copies of an early self
    eval_gauntlet_placement  = []   # placement vs 7 DIFFERENT past selves
    eval_gauntlet_elo        = []   # Bradley-Terry rating vs that same set
    # Adaptive opponent-mix series: one entry per EVAL point (same x-axis as
    # eval_updates), recording which opponent mix was ACTIVE for games
    # dispatched right after that eval -- so this can be overlaid directly on
    # the eval-vs-fixed-opponents curve to check that a metric change tracks
    # a mix switch rather than coincidence. See AdaptiveOpponentMix.
    opponent_mix_updates = []       # == eval_updates, kept separate so an
                                     # older history file without this key
                                     # can't desync the two series' lengths
    opponent_mix         = []       # [n_heuristic, n_greedy] active at that point
    # Per-update mean/std pairs (mean+std over exactly the games that fed each
    # update, so charts show one aggregated point per update instead of one raw
    # point per game -- game-to-game noise was dominating the picture).
    update_reward_avg,     update_reward_std     = [], []
    update_length_avg,     update_length_std     = [], []
    update_board_avg,      update_board_std      = [], []
    update_gold_avg,       update_gold_std       = [], []
    update_tier_avg,       update_tier_std       = [], []
    update_cwin_avg,       update_cwin_std       = [], []
    update_cdmg_avg,       update_cdmg_std       = [], []
    update_choice_avg,     update_choice_std     = [], []
    update_branch0_avg,    update_branch0_std    = [], []
    update_sellplace_avg,  update_sellplace_std  = [], []
    update_levelrate_avg,  update_levelrate_std  = [], []
    update_action_rate_avg = {k: [] for k in range(N_ACTION_TYPES)}   # per-update, % of actions of each type
    update_round_bucket_rate_avg = {b: {k: [] for k in range(N_ACTION_TYPES)}
                                     for b in range(N_ROUND_BUCKETS)}  # per-update, % of actions of each type, by round bucket
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
            endturn_golds         = hist.get('endturn_golds', [])
            endturn_tiers         = hist.get('endturn_tiers', [])
            combat_winrates       = hist.get('combat_winrates', [])
            combat_damage         = hist.get('combat_damage', [])
            choice_events         = hist.get('choice_events', [])
            choice_branch0_rates  = hist.get('choice_branch0_rates', [])
            sell_counts            = hist.get('sell_counts', [])
            place_counts           = hist.get('place_counts', [])
            level_counts           = hist.get('level_counts', [])
            total_action_counts    = hist.get('total_action_counts', [])
            sell_place_ratios       = hist.get('sell_place_ratios', [])
            level_rates              = hist.get('level_rates', [])
            _loaded_atgr = hist.get('action_type_game_rates', {})
            action_type_game_rates = {k: _loaded_atgr.get(str(k), []) for k in range(N_ACTION_TYPES)}
            # New key (2026-09-03): .get(..., {}) so an older history file
            # written before round_num existed on Transition just resumes with
            # empty per-bucket series here instead of crashing.
            _loaded_rbagr = hist.get('round_bucket_action_game_rates', {})
            round_bucket_action_game_rates = {
                b: {k: _loaded_rbagr.get(str(b), {}).get(str(k), []) for k in range(N_ACTION_TYPES)}
                for b in range(N_ROUND_BUCKETS)
            }
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
            eval_heur_mean_placement = hist.get('eval_heur_mean_placement', [])
            eval_ref_mean_placement  = hist.get('eval_ref_mean_placement', [])
            eval_gauntlet_placement  = hist.get('eval_gauntlet_placement', [])
            eval_gauntlet_elo        = hist.get('eval_gauntlet_elo', [])
            eval_top1_rate          = hist.get('eval_top1_rate', [])
            eval_top4_rate          = hist.get('eval_top4_rate', [])

            # Left-pad every eval-aligned series to len(eval_updates).
            # These series are appended in lockstep with eval_updates during a
            # run, so they stay aligned -- but a series introduced AFTER a run
            # started (the gauntlet pair, added 2026-09-03) resumes SHORT, and
            # the charts pair them positionally with
            # `zip(eval_updates, series)`, which silently pairs from the LEFT.
            # That put both real gauntlet points ~2350 updates too far left
            # (placement 5.22 measured at update 2400 was drawn at update 50;
            # 4.19 measured at 2600 was drawn at 250) -- the metric was not
            # wrong, its x-axis was. Leading Nones restore the alignment and
            # plot as gaps. Applies to any future late-added eval series too.
            _n_ev = len(eval_updates)
            for _name, _series in (
                ('eval_mean_placement',      eval_mean_placement),
                ('eval_heur_mean_placement', eval_heur_mean_placement),
                ('eval_ref_mean_placement',  eval_ref_mean_placement),
                ('eval_gauntlet_placement',  eval_gauntlet_placement),
                ('eval_gauntlet_elo',        eval_gauntlet_elo),
                ('eval_top1_rate',           eval_top1_rate),
                ('eval_top4_rate',           eval_top4_rate),
            ):
                _missing = _n_ev - len(_series)
                if _missing > 0:
                    _series[:0] = [None] * _missing   # in place: same list object
                    print(f'  [resume] left-padded {_name} with {_missing} None '
                          f'to align with eval_updates ({_n_ev})', flush=True)
            opponent_mix_updates    = hist.get('opponent_mix_updates', [])
            opponent_mix            = hist.get('opponent_mix', [])
            update_reward_avg       = hist.get('update_reward_avg', [])
            update_reward_std       = hist.get('update_reward_std', [])
            update_length_avg       = hist.get('update_length_avg', [])
            update_length_std       = hist.get('update_length_std', [])
            update_board_avg        = hist.get('update_board_avg', [])
            update_gold_avg         = hist.get('update_gold_avg', [])
            update_gold_std         = hist.get('update_gold_std', [])
            update_tier_avg         = hist.get('update_tier_avg', [])
            update_tier_std         = hist.get('update_tier_std', [])
            update_cwin_avg         = hist.get('update_cwin_avg', [])
            update_cwin_std         = hist.get('update_cwin_std', [])
            update_cdmg_avg         = hist.get('update_cdmg_avg', [])
            update_cdmg_std         = hist.get('update_cdmg_std', [])
            update_choice_avg       = hist.get('update_choice_avg', [])
            update_choice_std       = hist.get('update_choice_std', [])
            update_branch0_avg      = hist.get('update_branch0_avg', [])
            update_branch0_std      = hist.get('update_branch0_std', [])
            update_board_std        = hist.get('update_board_std', [])
            update_sellplace_avg    = hist.get('update_sellplace_avg', [])
            update_sellplace_std    = hist.get('update_sellplace_std', [])
            update_levelrate_avg    = hist.get('update_levelrate_avg', [])
            update_levelrate_std    = hist.get('update_levelrate_std', [])
            _loaded_uara = hist.get('update_action_rate_avg', {})
            update_action_rate_avg = {k: _loaded_uara.get(str(k), []) for k in range(N_ACTION_TYPES)}
            _loaded_urbra = hist.get('update_round_bucket_rate_avg', {})
            update_round_bucket_rate_avg = {
                b: {k: _loaded_urbra.get(str(b), {}).get(str(k), []) for k in range(N_ACTION_TYPES)}
                for b in range(N_ROUND_BUCKETS)
            }
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

    # Derived from the policy's own action space so adding an action type
    # (REORDER was added 2026-09-02) can never leave this list stale.
    ACTION_NAMES = [n.upper() for n in ACTION_TYPE_NAMES]

    # N_GAMES defined earlier alongside ANNEAL_STEPS (both needed before
    # _make_ppo_trainer() is called above).
    # N_WORKERS / WORKER_DEVICE retuned 2026-09-01 on contract 49567627:
    # Intel Core i9-14900KF, 24 PHYSICAL cores / 32 threads, single socket
    # (single NUMA node), 6.0GHz boost, 31GB RAM, RTX 4060 Ti 8GB, $0.108/hr.
    #
    # THE KEY LESSON, restated because it is why this block exists: vast.ai
    # reports `cpu_cores=32` for this host, and the naive "physical =
    # cpu_cores / 2" rule would say 16 -- but this is a HYBRID chip (8
    # P-cores WITH hyperthreading + 16 E-cores WITHOUT), so neither number
    # is the physical core count. `lscpu` reports `Core(s) per socket: 24`,
    # which is correct. Always read `lscpu` (Core(s) per socket x
    # Socket(s)); neither `cpu_cores` nor a divide-by-two heuristic is
    # reliable.
    #
    # Measured sweep on this host, WITH PPO UPDATES ENABLED (update_interval
    # == n_workers, i.e. production-equivalent):
    #    16 workers -> 3.13 games/sec
    #    24 workers -> 3.18 games/sec   <- chosen; exactly one worker per
    #                                      physical core
    #    32 workers -> 2.81 games/sec   (past the physical core count; also
    #                                      RAM pressure at ~0.5GB/worker
    #                                      against 31GB)
    #
    # Why "updates ENABLED" is called out prominently: the earlier benchmark
    # for the 90-worker config on the previous host ran with
    # update_interval=10**9, i.e. updates OFF. That made its numbers
    # non-comparable to a real run and is exactly what hid the fact that the
    # box ran ~45% idle in production. Never benchmark this with updates
    # disabled again.
    #
    # Cost/throughput comparison worth keeping: the 48-physical-core Xeon
    # Gold host measured 5.03 games/sec but with updates OFF (so
    # realistically ~3.9 with updates on) at $0.401/hr, versus 3.18
    # games/sec at $0.108/hr here -- roughly 3x better throughput per dollar
    # on this box despite half the physical cores, because its cores are
    # far faster.
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
    #
    # Per-phase instrumentation (new this session): with the
    # weights-broadcast change (tasks now carry a `(path, version)` ref
    # instead of a 13.89MB state_dict), train.py's `phase:` log lines
    # measure dispatch=0.0% and merge=0.0-0.3% of wall-clock -- main-process
    # serialization is no longer a factor. The remaining split is `wait`
    # 68-90% (workers doing useful parallel work) and `update` 9-31%.
    N_WORKERS       = 24
    WORKER_DEVICE   = 'cpu'
    # UPDATE_INTERVAL is deliberately NOT tied to N_WORKERS any more. It was
    # 24 == N_WORKERS, which at the old ~182 steps/game gave ~4,400
    # transitions/update. Real combat cut steps/game to ~86-120, so keeping
    # 24 would have quietly SHRUNK the PPO batch to ~2,100 -- a different
    # (noisier) optimisation regime than the one whose diagnostics were
    # validated (clip_frac 0.12, approx_kl 0.026, explained_var 0.63).
    # 48 restores ~4,100-5,760 transitions/update, i.e. the SAME batch size
    # the previous run actually ran at. Cost: games are collected under up to
    # 2 weight versions instead of 1 -- mild off-policy-ness that PPO's
    # importance clipping exists to handle, and the queue_factor backlog
    # already introduced it anyway.
    UPDATE_INTERVAL = 48     # games per PPO update; see note above
                              # N_WORKERS, so this fires exactly one PPO update
                              # per dispatched batch -- predictable, and at the
                              # ~250 transitions/game measured for a trained
                              # policy under the real seat mix, 24 games/update
                              # gives roughly 6k transitions per update, and
                              # 150000/24 = 6,250 updates across the run. This
                              # is close to the ~6.5k transitions/update of the
                              # older 26-worker configuration, i.e. back in the
                              # PPO batch-size regime that was previously
                              # validated as healthy (explained_var ~0.55-0.64,
                              # clip_frac 0.15-0.2). Keeping UPDATE_INTERVAL ==
                              # N_WORKERS is what keeps the collected batch
                              # maximally on-policy under the rolling-dispatch
                              # scheme in train._train_parallel: games
                              # dispatched after an update use the new
                              # weights, whereas a smaller UPDATE_INTERVAL
                              # would mean many in-flight games were started
                              # under weights several updates stale.
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
    # No fresh eval-worker sweep exists for the current host (the 2026-08-31
    # sweep above was measured on the previous Xeon Gold host at 12/20/28
    # eval workers and does not carry over -- different CPU, different
    # physical core count, no data point at 24 eval workers there).
    # Estimating instead, with the assumption stated plainly: eval games and
    # self-play training games run the identical policy under identical
    # CPU-bound inference with no PPO bookkeeping in the eval path, so 24
    # eval workers should sustain roughly the same 3.18 games/sec measured
    # for 24 self-play workers above -- giving 128 eval games ~= 128/3.18
    # ~= 40s. This is an estimate, not a measurement; re-benchmark once the
    # eval path can be timed directly on this host.
    #
    # Recomputed 2026-09-01 for the N_WORKERS=24 / UPDATE_INTERVAL=24 host:
    # at 3.18 training games/sec, EVAL_EVERY=100 means 100 x 24 = 2,400
    # training games (~755s, ~12.6 min) between evals -> ~40/755 ~= 5%
    # wall-clock overhead (estimated per the assumption above), for a
    # placement standard error of ~0.18 (n=128).
    # The eval-DENSITY tradeoff that hurt on the 90-worker host mostly
    # reverses here: with 6,250 total updates (150000/24), EVAL_EVERY=100
    # yields ~62 eval points across the whole run -- comparable to, in fact
    # slightly higher than, the ~57 points from the older UPDATE_INTERVAL=26
    # configuration (150000/26/100 ~= 57.7), and far denser than the ~16-17
    # points the 90-worker host's config produced. The high-frequency
    # progress signal still comes for free either way: the fixed
    # GreedyPlayAgent/HeuristicAgent seats present in EVERY training game
    # (train_plc - greedy_plc), averaged over thousands of games, so
    # nothing is lost between eval points. This eval is the low-frequency
    # ABSOLUTE anchor that no shaping term can inflate -- ~62 well-spaced,
    # low-noise (SE 0.18) anchor points across the run is plenty to confirm
    # that free signal is tracking something real.
    # Eval budget split across three opponents, TOTAL held at the old
    # EVAL_N_GAMES so eval wall-clock does not grow. greedy keeps the largest
    # share because it is the historical series; the smaller n is more than
    # offset by the fixed EVAL_SEED below, which removes the between-point
    # game-draw variance that used to dominate this metric.
    EVAL_GREEDY_GAMES = 64
    EVAL_HEUR_GAMES   = 32
    EVAL_REF_GAMES    = 32
    EVAL_SEED         = 12345      # FIXED for the whole run -- see the eval block
    # Update at which the frozen reference opponent is snapshotted. Early
    # enough that the policy is already past random flailing, late enough that
    # it is a meaningful bar to measure against for the rest of the run.
    REF_SNAPSHOT_UPDATE = 500
    REF_SNAPSHOT_PATH   = Path('bg_agent_ppo_reference.pt')

    # --- Gauntlet: rating the policy against a SPREAD of its own past -------
    # A single frozen reference is still a fixed bar, and every fixed bar
    # eventually gets cleared -- greedy already has (1.2-1.7, top1 ~0.9 for
    # 1,600 updates, measuring nothing). The gauntlet seats the current policy
    # against GAUNTLET_SIZE DIFFERENT frozen checkpoints spanning the run.
    # Because a BG lobby seats 8, one game is a full 8-way comparison: 28
    # pairwise outcomes per game, so ratings cost O(n) games rather than the
    # O(n^2) matches a 2-player game would need.
    # It does not saturate the way one reference does, because the comparison
    # set rolls forward as new checkpoints are added.
    GAUNTLET_DIR   = Path('gauntlet_refs')
    GAUNTLET_EVERY = 300     # freeze a new gauntlet reference every N updates
    GAUNTLET_SIZE  = 7       # non-eval seats in the lobby
    # Sizing, measured rather than guessed: a gauntlet lobby runs a neural
    # forward pass for ALL 8 seats, where the greedy/heuristic evals run one
    # (the other 7 are scripted). So it costs roughly 8x per game. At 96 games
    # every 50 updates that projected to ~25% of total run wall-clock.
    # 32 games still yields 32 x C(8,2) = 896 pairwise outcomes, which is
    # ample to fit 8 ratings -- the multiplayer lobby is exactly what makes
    # this affordable. Run on its own slower cadence.
    EVAL_GAUNTLET_GAMES = 32
    GAUNTLET_EVAL_EVERY = 200   # updates between gauntlet evals (vs EVAL_EVERY=50)
    EVAL_EVERY      = 50
    EVAL_N_GAMES    = 128
    EVAL_WORKERS    = 24     # CPU workers for evaluate_policy's pool -- eval
                              # workers always run the policy on CPU regardless
                              # of DEVICE (see train.py), so this is
                              # independent of N_WORKERS/training GPU memory;
                              # set to 24 to match this host's physical core
                              # count (see N_WORKERS comment above -- lscpu
                              # shows a single socket x 24 cores, not the
                              # 32-thread cpu_cores figure vast.ai reports).
                              # mean_placement was verified byte-identical
                              # across worker counts (the seed-by-game-index
                              # aggregation is order-independent), so raising
                              # this changes eval speed only, never eval
                              # results.

    # ---- Adaptive opponent mix -------------------------------------------
    # See AdaptiveOpponentMix's class docstring (module level, above) for the
    # full trigger/hysteresis design. Thresholds are the task defaults:
    # switch to the reduced mix below 2.0, back to full above 3.0, each on 2
    # consecutive eval points.
    #
    # Resume: pick up whichever mix was last recorded, so restarting this
    # script mid-run doesn't silently drop back to the full scripted mix for
    # a while after a switch already happened. Streak counters intentionally
    # start at 0 either way -- that just means two fresh consecutive points
    # are required post-resume before the NEXT switch, the safe (conservative)
    # direction to be wrong in.
    _mix = AdaptiveOpponentMix(
        full_mix=(N_HEURISTIC_SLOTS, N_GREEDY_SLOTS),
        reduced_mix=(REDUCED_HEURISTIC_SLOTS, REDUCED_GREEDY_SLOTS),
        low=2.0, high=3.0, streak=2,
        reduced=bool(opponent_mix and
                     list(opponent_mix[-1]) == [REDUCED_HEURISTIC_SLOTS, REDUCED_GREEDY_SLOTS]),
    )

    def opponent_mix_fn():
        # Consulted by _train_parallel._make_task() at task-creation time for
        # EVERY dispatched game -- must stay cheap and side-effect-free.
        return _mix.mix

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

    # scalar_context[94] = ps.gold / 10.0 -- first dim of the 6-dim economy block
    # (own_scalar 24 + all_opponents 64 + lobby 6 = 94), per game_loop.py's
    # _get_observation concatenate order.
    #
    # Unspent gold at END_TURN is the single most direct read on whether the flat
    # gold penalty is doing its job. The whole point of GOLD_PENALTY_SCALE being
    # flat is "in Battlegrounds you almost never want unspent gold" -- and the
    # failure it was introduced to fix (93% of round-13+ turns banking >=5 gold,
    # mean 7.47) was only ever found by instrumenting a checkpoint by hand after
    # the fact. Tracked live, that regression is visible within a few updates.
    #
    # NOTE the /10.0 scaling is nominal, not a cap: ps.max_gold rises above 10
    # via trinkets and Snare Trapper, so this feature can legitimately exceed 1.0
    # and the decoded value below can exceed 10 gold.
    GOLD_SCALAR_IDX = 94

    # card_encoder dim 43 = tavern_tier/7.0 (board context, identical across every
    # token in an encoding call), so any token of a transition carries the tier.
    TIER_TOKEN_IDX = 43

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
            'endturn_golds': endturn_golds,
            'endturn_tiers': endturn_tiers,
            'combat_winrates': combat_winrates,
            'combat_damage': combat_damage,
            'choice_events': choice_events,
            'choice_branch0_rates': choice_branch0_rates,
            'sell_counts': sell_counts,
            'place_counts': place_counts,
            'level_counts': level_counts,
            'total_action_counts': total_action_counts,
            'sell_place_ratios': sell_place_ratios,
            'level_rates': level_rates,
            'action_type_game_rates': action_type_game_rates,
            'round_bucket_action_game_rates': round_bucket_action_game_rates,
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
            'eval_heur_mean_placement': eval_heur_mean_placement,
            'eval_ref_mean_placement': eval_ref_mean_placement,
            'eval_gauntlet_placement': eval_gauntlet_placement,
            'eval_gauntlet_elo': eval_gauntlet_elo,
            'eval_top1_rate': eval_top1_rate,
            'eval_top4_rate': eval_top4_rate,
            'opponent_mix_updates': opponent_mix_updates,
            'opponent_mix': opponent_mix,
            'update_reward_avg': update_reward_avg,
            'update_reward_std': update_reward_std,
            'update_length_avg': update_length_avg,
            'update_length_std': update_length_std,
            'update_board_avg': update_board_avg,
            'update_gold_avg': update_gold_avg,
            'update_gold_std': update_gold_std,
            'update_tier_avg': update_tier_avg,
            'update_tier_std': update_tier_std,
            'update_cwin_avg': update_cwin_avg,
            'update_cwin_std': update_cwin_std,
            'update_cdmg_avg': update_cdmg_avg,
            'update_cdmg_std': update_cdmg_std,
            'update_choice_avg': update_choice_avg,
            'update_choice_std': update_choice_std,
            'update_branch0_avg': update_branch0_avg,
            'update_branch0_std': update_branch0_std,
            'update_board_std': update_board_std,
            'update_sellplace_avg': update_sellplace_avg,
            'update_sellplace_std': update_sellplace_std,
            'update_levelrate_avg': update_levelrate_avg,
            'update_levelrate_std': update_levelrate_std,
            'update_action_rate_avg': update_action_rate_avg,
            'update_round_bucket_rate_avg': update_round_bucket_rate_avg,
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
        fig, axes = plt.subplots(2, 7, figsize=(42, 8))

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
                           'goldenrod', 'teal', 'slategray', 'deeppink', 'saddlebrown']
        _action_colors = _action_colors[:N_ACTION_TYPES]
        _lens = [len(update_action_rate_avg[k]) for k in range(N_ACTION_TYPES)]
        if all(_lens):
            n = min(_lens)
            xs = list(range(n))
            series = [[100 * v for v in update_action_rate_avg[k][:n]] for k in range(N_ACTION_TYPES)]
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
            ax.plot(eval_updates, eval_mean_placement, color='darkorchid',
                    marker='o', ms=3, label=f'vs greedy ({EVAL_GREEDY_GAMES}g)')
            if eval_heur_mean_placement:
                n = min(len(eval_updates), len(eval_heur_mean_placement))
                ax.plot(eval_updates[:n], eval_heur_mean_placement[:n], color='darkorange',
                        marker='s', ms=3, label=f'vs heuristic ({EVAL_HEUR_GAMES}g)')
            # The reference series is None until the snapshot exists; plot only
            # the points that are real rather than substituting a number.
            _ref_xy = [(u, v) for u, v in zip(eval_updates, eval_ref_mean_placement) if v is not None]
            if _ref_xy:
                ax.plot([u for u, _ in _ref_xy], [v for _, v in _ref_xy], color='seagreen',
                        marker='^', ms=3, label=f'vs frozen self ({EVAL_REF_GAMES}g)')
            _g_xy = [(u, v) for u, v in zip(eval_updates, eval_gauntlet_placement) if v is not None]
            if _g_xy:
                ax.plot([u for u, _ in _g_xy], [v for _, v in _g_xy], color='mediumvioletred',
                        marker='D', ms=3, label=f'vs {GAUNTLET_SIZE} past selves')
            ax.axhline(4.5, color='gray', lw=0.8, ls='--')  # random-play expectation, 8 players
            ax.invert_yaxis()
            ax.legend(fontsize=7, loc='best')
            ax.set_xlabel('PPO update'); ax.set_ylabel('Mean placement (inverted, up=better)')
            ax.set_title(f'Eval vs Fixed Opponents (greedy last={eval_mean_placement[-1]:.2f})')

        # Elo gets its OWN axes: it is unbounded rating points, not a 1-8
        # placement. Two different scales on one axis is the chart mistake
        # that makes both unreadable, so they stay separate.
        ax = axes[0][6]
        _e_xy = [(u, v) for u, v in zip(eval_updates, eval_gauntlet_elo) if v is not None]
        if _e_xy:
            ax.plot([u for u, _ in _e_xy], [v for _, v in _e_xy],
                    color='darkslateblue', marker='o', ms=3)
            ax.axhline(0, color='gray', lw=0.8, ls='--')
            ax.set_xlabel('PPO update')
            ax.set_ylabel('Elo vs oldest reference')
            ax.set_title(f'Gauntlet rating (last={_e_xy[-1][1]:+.0f})')
        else:
            ax.set_title('Gauntlet rating (no data yet)')

        # Action mix by round bucket, most recent update -- the "what is the
        # policy doing at round X" view that the flat Action Mix panel above
        # can't show (that one only has a PPO-update x-axis, not a round one).
        # A heatmap over (round bucket x action type) fits both axes without
        # needing yet another 7th row.
        ax = axes[1][6]
        _hm_lens = [len(update_round_bucket_rate_avg[b][k])
                    for b in range(N_ROUND_BUCKETS) for k in range(N_ACTION_TYPES)]
        if _hm_lens and min(_hm_lens) > 0:
            heat = np.array([[100 * update_round_bucket_rate_avg[b][k][-1] for k in range(N_ACTION_TYPES)]
                              for b in range(N_ROUND_BUCKETS)])
            ax.imshow(heat, cmap='viridis', aspect='auto', vmin=0,
                      vmax=max(1.0, float(heat.max())))
            ax.set_xticks(range(N_ACTION_TYPES))
            ax.set_xticklabels(ACTION_NAMES, rotation=90, fontsize=6)
            ax.set_yticks(range(N_ROUND_BUCKETS))
            ax.set_yticklabels(ROUND_BUCKET_LABELS, fontsize=7)
            _hi = float(heat.max())
            for b in range(N_ROUND_BUCKETS):
                for k in range(N_ACTION_TYPES):
                    ax.text(k, b, f'{heat[b, k]:.0f}', ha='center', va='center', fontsize=5,
                            color='white' if heat[b, k] < 0.6 * _hi else 'black')
            ax.set_title(f'Action Mix by Round (% of round-bucket actions, update {ppo_trainer.update_count})')
        else:
            ax.axis('off')

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
            _endturn = [t for t in game_trans if t.type_action == 7]
            endturn_sizes = [t.scalar_context[BOARD_SIZE_SCALAR_IDX] * 7.0 for t in _endturn]
            if endturn_sizes:
                board_sizes.append(float(np.mean(endturn_sizes)))

            # Unspent gold carried into combat -- see GOLD_SCALAR_IDX for why
            # this is the metric that would have caught the gold-hoarding
            # regression live instead of after the fact.
            _gold = [t.scalar_context[GOLD_SCALAR_IDX] * 10.0 for t in _endturn]
            if _gold:
                endturn_golds.append(float(np.mean(_gold)))

            # Tavern tier at end of turn. Read alongside board size, never
            # alone: CLAUDE.md's standing warning is that a naive levelling
            # incentive produces a high-tier, empty-board policy that chases
            # the proxy instead of the objective.
            # Tier lives in the per-card board-context block, so it is only set
            # on tokens for cards that are actually PRESENT -- padding slots are
            # all zero. Reading slot 0 directly returned ~0 whenever the board
            # was empty (common in early rounds), which dragged the average far
            # below the real tier. Max over board AND shop tokens instead: the
            # value is identical across every present token in an encoding call,
            # and the shop is non-empty whenever the board is not.
            _tiers = []
            for t in _endturn:
                _bt = getattr(t, "board_tokens", None)
                _st = getattr(t, "shop_tokens", None)
                _v = 0.0
                if _bt is not None and len(_bt):
                    _v = max(_v, float(np.max(np.asarray(_bt)[:, TIER_TOKEN_IDX])))
                if _st is not None and len(_st):
                    _v = max(_v, float(np.max(np.asarray(_st)[:, TIER_TOKEN_IDX])))
                if _v > 0:
                    _tiers.append(_v * 7.0)
            if _tiers:
                endturn_tiers.append(float(np.mean(_tiers)))

            # Combat outcomes, aggregated in the worker (see train.py's summary).
            _cs = summary.get('combat') or {}
            if _cs.get('n'):
                combat_winrates.append(_cs['wins'] / _cs['n'])
                combat_damage.append(_cs['dmg_taken'] / _cs['n'])

            # Real-choice usage. choice_branch0_rates is a COLLAPSE DETECTOR:
            # if the policy always takes the same Choose One branch regardless
            # of state, this pins at 0.0 or 1.0. That is exactly the pathology
            # ACTIVATE showed when it shared SELL's scorer (~1.000 probability
            # on one fixed slot across wildly different states), and it was only
            # caught by hand-instrumenting a checkpoint. A flat line here means
            # the choose_option head is not discriminating.
            _n_choice = sum(1 for t in game_trans if t.type_action in (10, 11))
            choice_events.append(_n_choice)
            _opts = [t.ptr_action for t in game_trans if t.type_action == 11]
            if _opts:
                choice_branch0_rates.append(
                    sum(1 for p in _opts if p == PTR_SHOP_OFF) / len(_opts)
                )

            # Per-game action tallies for the sell:place, level-up, and action-mix panels.
            type_counts_this_game = [0] * N_ACTION_TYPES
            for t in game_trans:
                type_counts_this_game[t.type_action] += 1
            n_sell, n_place, n_level = type_counts_this_game[1], type_counts_this_game[2], type_counts_this_game[5]
            sell_counts.append(n_sell)
            place_counts.append(n_place)
            level_counts.append(n_level)
            total_action_counts.append(len(game_trans))
            sell_place_ratios.append(n_sell / n_place if n_place > 0 else float('nan'))
            level_rates.append(n_level / len(game_trans) if game_trans else float('nan'))
            for k in range(N_ACTION_TYPES):
                action_type_game_rates[k].append(
                    type_counts_this_game[k] / len(game_trans) if game_trans else float('nan')
                )

            # Same action-mix tally, but split by which round-bucket each action
            # was taken in -- e.g. does the policy front-load REROLL early and
            # shift to SELL/PLACE churn late, independent of the flat mix above.
            bucket_counts_this_game = [[0] * N_ACTION_TYPES for _ in range(N_ROUND_BUCKETS)]
            for t in game_trans:
                b = _round_bucket(getattr(t, 'round_num', None))
                if b is not None:
                    bucket_counts_this_game[b][t.type_action] += 1
            for b in range(N_ROUND_BUCKETS):
                bucket_total = sum(bucket_counts_this_game[b])
                for k in range(N_ACTION_TYPES):
                    round_bucket_action_game_rates[b][k].append(
                        bucket_counts_this_game[b][k] / bucket_total if bucket_total else float('nan')
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
        # Top-3 actions per round bucket, averaged over the last 30 games --
        # narrower window than the cumulative/recent~200 lines above since this
        # is meant to answer "what's the policy doing RIGHT NOW at round X",
        # not track a long-run trend (that's what the dashboard heatmap is for).
        _rb_window = 30
        _rb_parts = []
        for b in range(N_ROUND_BUCKETS):
            _rates = []
            for k in range(N_ACTION_TYPES):
                vals = [v for v in round_bucket_action_game_rates[b][k][-_rb_window:]
                        if v is not None and not np.isnan(v)]
                _rates.append(float(np.mean(vals)) if vals else float('nan'))
            _top = sorted(range(N_ACTION_TYPES), key=lambda k: -_rates[k] if not np.isnan(_rates[k]) else 1)[:3]
            _seg = ' '.join(f'{ACTION_NAMES[k]}={100*_rates[k]:.0f}%'
                             for k in _top if not np.isnan(_rates[k]))
            if _seg:
                _rb_parts.append(f'{ROUND_BUCKET_LABELS[b]}[{_seg}]')
        if _rb_parts:
            print(f'  by-round (top3, last{_rb_window}g): ' + ' '.join(_rb_parts), flush=True)
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
        _append_update_stat(endturn_golds, update_gold_avg, update_gold_std)
        _append_update_stat(endturn_tiers, update_tier_avg, update_tier_std)
        _append_update_stat(combat_winrates, update_cwin_avg, update_cwin_std)
        _append_update_stat(combat_damage, update_cdmg_avg, update_cdmg_std)
        _append_update_stat(choice_events, update_choice_avg, update_choice_std)
        _append_update_stat(choice_branch0_rates, update_branch0_avg, update_branch0_std)
        _append_update_stat(sell_place_ratios, update_sellplace_avg, update_sellplace_std)
        _append_update_stat(level_rates, update_levelrate_avg, update_levelrate_std)
        for k in range(N_ACTION_TYPES):
            _append_update_avg(action_type_game_rates[k], update_action_rate_avg[k])
        for b in range(N_ROUND_BUCKETS):
            for k in range(N_ACTION_TYPES):
                _append_update_avg(round_bucket_action_game_rates[b][k], update_round_bucket_rate_avg[b][k])
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
        # Freeze the reference opponent exactly once, at REF_SNAPSHOT_UPDATE.
        # Written before the eval block below so the eval at that same update
        # already has it available.
        # Freeze a gauntlet reference on a schedule. Seeded from the existing
        # single reference so a RESUMED run starts with a non-empty set rather
        # than a gauntlet that only becomes meaningful hundreds of updates in.
        GAUNTLET_DIR.mkdir(exist_ok=True)
        if not any(GAUNTLET_DIR.glob('ref_u*.pt')) and REF_SNAPSHOT_PATH.exists():
            import shutil as _sh
            _sh.copy2(REF_SNAPSHOT_PATH, GAUNTLET_DIR / f'ref_u{REF_SNAPSHOT_UPDATE}.pt')
            print(f'  Gauntlet seeded from {REF_SNAPSHOT_PATH} '
                  f'-> ref_u{REF_SNAPSHOT_UPDATE}.pt', flush=True)
        if update_count % GAUNTLET_EVERY == 0:
            _gp = GAUNTLET_DIR / f'ref_u{update_count}.pt'
            if not _gp.exists():
                torch.save({k: v.detach().cpu().clone()
                            for k, v in policy_train.state_dict().items()}, str(_gp))
                print(f'  Gauntlet reference frozen at update {update_count}', flush=True)

        if update_count >= REF_SNAPSHOT_UPDATE and not REF_SNAPSHOT_PATH.exists():
            torch.save({k: v.detach().cpu().clone()
                        for k, v in policy_train.state_dict().items()},
                       str(REF_SNAPSHOT_PATH))
            print(f'  Reference opponent frozen at update {update_count} '
                  f'-> {REF_SNAPSHOT_PATH}', flush=True)

        if update_count % EVAL_EVERY == 0:
            # FIXED eval seed, not ppo_trainer.total_steps. A seed that moves
            # every eval makes each point sample a DIFFERENT set of games, so
            # consecutive points differ by game-draw noise on top of any real
            # policy change -- which is most of why the previous run's eval
            # oscillated in a 1.9-2.6 band with no trend. The policy never
            # trains on eval games, so a fixed seed cannot be overfitted to;
            # it just removes the between-point variance and lets a change in
            # the number mean "the policy changed".
            eval_result = evaluate_policy(
                policy_train, card_defs_train,
                n_games=EVAL_GREEDY_GAMES, opponent='greedy',
                device=DEVICE, seed=EVAL_SEED,
                n_workers=EVAL_WORKERS,
            )
            eval_heur = evaluate_policy(
                policy_train, card_defs_train,
                n_games=EVAL_HEUR_GAMES, opponent='heuristic',
                device=DEVICE, seed=EVAL_SEED,
                n_workers=EVAL_WORKERS,
            )
            # Reference eval: vs 7 frozen copies of an early self. greedy and
            # heuristic are FIXED bars that a good policy eventually clears and
            # then stops discriminating (the last run pinned ~2.1 vs greedy from
            # update 1300 to 5766 -- 78% of the run with no readable signal).
            # A frozen former self keeps discriminating past that point.
            ref_mean = None
            if REF_SNAPSHOT_PATH.exists():
                ref_res = evaluate_policy(
                    policy_train, card_defs_train,
                    n_games=EVAL_REF_GAMES, opponent='reference',
                    device=DEVICE, seed=EVAL_SEED,
                    n_workers=EVAL_WORKERS, ref_path=str(REF_SNAPSHOT_PATH),
                )
                ref_mean = ref_res['mean_placement']

            # Gauntlet: seat the policy against a SPREAD of its own past.
            # Picking GAUNTLET_SIZE evenly-spaced checkpoints (always keeping
            # the oldest and newest) means the comparison set spans the whole
            # run instead of clustering at one point in it.
            g_place, g_elo = None, None
            _refs = sorted(GAUNTLET_DIR.glob('ref_u*.pt'),
                           key=lambda p: int(p.stem.split('_u')[1]))
            if _refs and update_count % GAUNTLET_EVAL_EVERY == 0:
                if len(_refs) > GAUNTLET_SIZE:
                    _idx = [round(i * (len(_refs) - 1) / (GAUNTLET_SIZE - 1))
                            for i in range(GAUNTLET_SIZE)]
                    _sel = [_refs[i] for i in sorted(set(_idx))]
                else:
                    _sel = _refs
                try:
                    _gres = evaluate_policy(
                        policy_train, card_defs_train,
                        n_games=EVAL_GAUNTLET_GAMES, opponent='gauntlet',
                        device=DEVICE, seed=EVAL_SEED, n_workers=EVAL_WORKERS,
                        ref_path=[str(p) for p in _sel],
                    )
                    g_place = _gres['mean_placement']
                    g_elo   = (_gres.get('elo') or {}).get('current')
                    print(f'  GAUNTLET @ update {update_count}: '
                          f'placement={g_place:.2f} vs {len(_sel)} past selves '
                          f'({_sel[0].stem}..{_sel[-1].stem}) '
                          f'elo={"n/a" if g_elo is None else f"{g_elo:+.0f}"}', flush=True)
                except Exception as _e:
                    # An eval failure must never take down a multi-hour run.
                    print(f'  GAUNTLET eval failed ({type(_e).__name__}: {_e})', flush=True)

            eval_gauntlet_placement.append(g_place)
            eval_gauntlet_elo.append(g_elo)
            eval_updates.append(update_count)
            eval_mean_placement.append(eval_result['mean_placement'])
            eval_top1_rate.append(eval_result['top1_rate'])
            eval_top4_rate.append(eval_result['top4_rate'])
            eval_heur_mean_placement.append(eval_heur['mean_placement'])
            # None (not 0.0) before the reference snapshot exists -- 0.0 would
            # plot as a spectacular result rather than as missing data.
            eval_ref_mean_placement.append(ref_mean)
            print(f'  EVAL @ update {update_count}: '
                  f'greedy={eval_result["mean_placement"]:.2f} '
                  f'(top1={eval_result["top1_rate"]:.2f} top4={eval_result["top4_rate"]:.2f}) | '
                  f'heuristic={eval_heur["mean_placement"]:.2f} | '
                  f'reference={"n/a" if ref_mean is None else f"{ref_mean:.2f}"}', flush=True)

            # Adaptive opponent mix: trigger ONLY off this fixed-opponent
            # greedy eval (never in-game placement vs the co-evolving
            # SnapshotPool seats -- see AdaptiveOpponentMix's class
            # docstring). Records the ACTIVE mix at every eval point (whether
            # or not this point caused a switch) so opponent_mix_updates/
            # opponent_mix share eval_updates' x-axis exactly and can be
            # overlaid on the eval chart; logs loudly only on an actual switch.
            mix_now = _mix.update(update_count, eval_result['mean_placement'])
            opponent_mix_updates.append(update_count)
            opponent_mix.append(list(mix_now))

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
        opponent_mix_fn=opponent_mix_fn,
    )
    print(f'\nDone -- steps={ppo_trainer.total_steps:,}, updates={ppo_trainer.update_count}', flush=True)
