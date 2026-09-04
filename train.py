#!/usr/bin/env python3
"""
train.py — Entry point for self-play PPO training of the BG agent.

Usage:
    python train.py [--games 500] [--workers 1] [--checkpoint bg_agent_ppo.pt]
                    [--load-bc bg_policy.pt] [--device cpu] [--seed 42]
                    [--update-interval 10] [--no-firestone] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import re
import sys
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# -------------------------------------------------------------------------
# Module-level logger (configured after argument parsing)
# -------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Import project components
# -------------------------------------------------------------------------
from agent.card_encoder import CardEncoder
from agent.policy import (make_policy,
                          BGPolicyNetwork, N_ACTION_TYPES, POINTER_DIM,
                          build_type_mask, build_pointer_mask,
                          PTR_SHOP_OFF, PTR_BOARD_OFF, PTR_HAND_OFF)
from agent.ppo import PPOConfig, PPOTrainer
from env.game_loop import BattlegroundsGame, GameResult, expected_tier_for_round
from env.matchmaker import Matchmaker
from env.player_state import PlayerState, minion_stats
from env.tavern_pool import TavernPool
from symbolic.board_computer import SymbolicBoardComputer
from symbolic.firestone_client import FirestoneClient

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
CARD_DEFS_PATH = Path(__file__).parent / "bg_card_definitions.json"
PIPELINE_SCRIPT = Path(__file__).parent / "bg_card_pipeline.py"
N_PLAYERS         = 8
N_TRAIN_PLAYERS   = 2   # player slots per game that use the current policy and collect transitions
N_HEURISTIC_SLOTS = 2   # opponent slots permanently assigned to HeuristicAgent
N_GREEDY_SLOTS    = 2   # opponent slots permanently assigned to GreedyPlayAgent
SNAPSHOT_EVERY    = 10  # rolling snapshot every N PPO updates
MILESTONE_EVERY   = 50  # protected milestone snapshot every N PPO updates

# Reduced-scripted opponent mix, switched to once the honest fixed-opponent
# eval (evaluate_policy vs greedy — see run_fresh_training.py's adaptive-mix
# trigger) shows the policy has already crushed both scripted baselines.  At
# that point 4 of 6 opponent slots being bots the agent beats almost always
# is wasted compute: reallocate most of them to sampled SnapshotPool
# opponents, which co-evolve with the policy and so cannot saturate the way a
# fixed script can.
#
# Deliberately NOT (0, 0): exactly one GreedyPlayAgent is kept because it is
# the only agent in the whole opponent pool that reliably builds a WIDE board
# (measured 6.75-7.0 minions) rather than a narrow one -- the archetype that
# should punish the narrow ~4-minion carry board the agent currently wins
# with. Pure self-play is auto-curricular in DIFFICULTY (snapshots track
# whatever the policy can currently handle) but not in STRATEGIC DIVERSITY --
# every snapshot plays the same style as the current policy, so dropping
# every scripted seat would remove the only non-self-play behavior left in
# the pool, not just the weakest opponents.
REDUCED_HEURISTIC_SLOTS = 0
REDUCED_GREEDY_SLOTS    = 1

# SnapshotPool tuning (see class docstring for the full mechanism). Measured
# on the run that ended at update 5,766: 115 milestones had accumulated
# against a capacity-20 rolling buffer -- with the old UNIFORM sample over
# _snapshots + _milestones, that's 115/135 = 85% of self-play opponent draws
# coming from an ever-growing pile of ancient checkpoints, made worse every
# single update as more milestones piled up and the rolling buffer's 20-slot
# share of the mix kept shrinking. Eval placement (vs a FIXED greedy
# baseline, so this isn't just "the pool got harder together") flatlined
# from update ~1300 onward for the remaining 107k games -- a training
# population that gets progressively weaker relative to the learner is one
# of the mechanisms that can produce exactly that shape.
# MILESTONE_CAPACITY caps the milestone list and THINS it (never FIFO-evicts)
# so it keeps spanning the whole run instead of growing without bound.
# P_RECENT biases sampling toward the small, always-fresh rolling buffer so
# most self-play opponents track current skill, while a deliberate minority
# still come from milestones for long-run behavioral diversity.
MILESTONE_CAPACITY = 10
P_RECENT           = 0.75

# Board-shape potential weight (see BattlegroundsGame.shape_stats_weight /
# _board_potential): fixed at 1.0 -- the deterministic, quality-weighted stats+
# keyword+synergy potential is the board-shape signal, full stop, no MC
# win-probability blend. This used to anneal from 0.25 down to 0 over 250k steps
# (as early-training-only scaffolding); a full training run showed reward and
# placement-vs-baseline plateauing then regressing right around when that anneal
# finished (~update 106 of 312), with sell:place ratio climbing toward 1.0 and
# board size shrinking in the same window -- consistent with the policy farming
# noise in the 30-trial MC estimate once the deterministic anchor faded out. See
# CONTEXT.md (2026-08-31) for the full analysis before re-introducing a MC blend.
BOARD_SHAPE_STATS_WEIGHT = 1.0


# -------------------------------------------------------------------------
# Snapshot pool for historical self-play
# -------------------------------------------------------------------------

def _default_broadcast_dir() -> Path:
    """Preferred location for files broadcast to worker subprocesses.

    /dev/shm is a RAM-backed tmpfs when writable -- writing/reading through
    it costs no real disk I/O, just a memcpy -- falling back to the platform
    tempdir otherwise. Shared by the current-policy weights broadcast
    (_train_parallel's _write_weights) and SnapshotPool's per-snapshot files
    below, distinguished only by filename prefix, so both mechanisms use the
    identical "write once, let every worker read it in parallel" pattern.
    """
    import tempfile
    shm_dir = Path("/dev/shm")
    if shm_dir.exists() and os.access(shm_dir, os.W_OK):
        return shm_dir
    return Path(tempfile.gettempdir())


class SnapshotPool:
    """Rolling buffer of past policy snapshots for historical self-play.

    Each game pairs N_TRAIN_PLAYERS current-policy agents against
    (N_PLAYERS - N_TRAIN_PLAYERS) agents frozen at a past checkpoint.
    This breaks the echo-chamber that forms when all players share the same
    evolving policy.

    Two snapshot tiers:
      - Rolling  : recent snapshots, FIFO-evicted once `capacity` is exceeded.
      - Milestone: long-lived snapshots added every MILESTONE_EVERY updates,
                   preserved for long-term behavioral diversity (e.g.
                   early-training styles) but THINNED (never simply
                   FIFO-evicted) once `milestone_capacity` is exceeded -- see
                   _thin_milestones. FIFO-evicting milestones would defeat
                   their purpose (losing early-run diversity first, keeping
                   only whatever's most recent -- exactly what the rolling
                   buffer already provides); thinning instead drops roughly
                   every other OLDER entry so the survivors still span the
                   full run, always keeping the very oldest one.

    Sampling is RECENCY-WEIGHTED, not uniform (see MILESTONE_CAPACITY /
    P_RECENT module comment for the 115-milestones-vs-20-rolling measurement
    that motivated this): with probability `p_recent` a sample is drawn from
    the rolling buffer, weighted linearly toward its newest entries, and
    with probability `1 - p_recent` from the milestone list (uniformly --
    milestones are already a curated, run-spanning set, so there's no
    "recent" bias to apply within it). Falls back gracefully to whichever
    list is non-empty if the other is empty.

    Content-addressed file store: `add()` writes the state_dict to disk
    EXACTLY ONCE (atomically -- see _write_snapshot_file), and both the
    rolling and milestone entries for a milestone update reference that same
    file via a refcount, so a snapshot's file is only unlinked once nothing
    references it any more (evicted from the rolling buffer AND thinned out
    of / never added to the milestone list). `sample()` / `sample_n()`
    return `(path_str, snapshot_id, tag)` references instead of raw
    state_dicts -- workers `torch.load` the file themselves (see
    _load_snapshot_policy) -- so this pool never pickles a ~13.89MB
    state_dict into a task; only a path string and a small int cross the
    process boundary.

    Usage::

        pool = SnapshotPool(capacity=20)
        pool.add(policy.state_dict(), update_count=10)                    # rolling
        pool.add(policy.state_dict(), update_count=50, is_milestone=True) # protected
        opp_refs = pool.sample_n(5)   # five independent (path, snap_id, tag) draws
        pool.cleanup()                # at run end: unlink every live snapshot file

    *tag* is a short label like ``"snapshot_u10"`` or ``"milestone_u50"``
    identifying which PPO update the snapshot was frozen at — used to
    attribute wins/losses back to a specific point in training (see
    ``_append_agent_stats``). This format is unchanged from the previous
    (raw state_dict) implementation.
    """

    def __init__(
        self,
        capacity: int = 20,
        milestone_capacity: int = MILESTONE_CAPACITY,
        p_recent: float = P_RECENT,
        weights_dir: Optional[Path] = None,
        file_prefix: str = "bg_snapshot",
    ) -> None:
        self.capacity           = capacity
        self.milestone_capacity = milestone_capacity
        self.p_recent           = p_recent
        self._weights_dir = Path(weights_dir) if weights_dir is not None else _default_broadcast_dir()
        self._file_prefix = file_prefix
        self._snapshots:  List[tuple] = []   # rolling,   evictable: (snap_id, path_str, tag)
        self._milestones: List[tuple] = []   # protected, thinned:   (snap_id, path_str, tag)
        self._next_id   = 0
        self._refcount: Dict[int, int] = {}   # snap_id -> # of lists referencing its file
        self._paths:    Dict[int, str] = {}   # snap_id -> path_str, for referenced (live) files

    def _snapshot_path(self, snap_id: int) -> Path:
        return self._weights_dir / f"{self._file_prefix}_s{snap_id}.pt"

    def _write_snapshot_file(self, state_dict: dict) -> tuple:
        """Write *state_dict* to a fresh, immutable, versioned file and
        return (snap_id, path_str).

        Atomic write: save to a `.tmp` sibling then os.replace() onto the
        real name -- same pattern as _train_parallel's _write_weights, so a
        worker can never torch.load a partially written file.
        """
        snap_id  = self._next_id
        self._next_id += 1
        path     = self._snapshot_path(snap_id)
        tmp_path = path.with_name(path.name + ".tmp")
        cpu_sd   = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        torch.save(cpu_sd, str(tmp_path))
        os.replace(str(tmp_path), str(path))
        path_str = str(path)
        self._paths[snap_id] = path_str
        return snap_id, path_str

    def _release(self, entry: tuple) -> None:
        """Decrement the refcount backing *entry*'s file; unlink it once
        nothing (rolling or milestone) references it any more."""
        snap_id = entry[0]
        self._refcount[snap_id] = self._refcount.get(snap_id, 1) - 1
        if self._refcount[snap_id] <= 0:
            path_str = self._paths.pop(snap_id, None)
            self._refcount.pop(snap_id, None)
            if path_str is not None:
                try:
                    Path(path_str).unlink()
                except OSError:
                    pass  # already gone / never existed -- nothing to reclaim

    def add(
        self,
        state_dict: dict,
        *,
        is_milestone: bool = False,
        update_count: Optional[int] = None,
    ) -> None:
        """Write *state_dict* to a new snapshot file (once) and append it to
        the rolling buffer.

        If *is_milestone* is True, also append the SAME file reference to the
        protected milestone list (refcounted -- see class docstring), then
        thin the milestone list if it has grown past `milestone_capacity`.
        """
        tag = f"{'milestone' if is_milestone else 'snapshot'}_u{update_count}"
        snap_id, path_str = self._write_snapshot_file(state_dict)
        entry = (snap_id, path_str, tag, update_count if update_count is not None else self._next_id)

        refs = 1
        self._snapshots.append(entry)
        if len(self._snapshots) > self.capacity:
            self._release(self._snapshots.pop(0))
        if is_milestone:
            self._milestones.append(entry)
            refs += 1
            self._thin_milestones()
        self._refcount[snap_id] = self._refcount.get(snap_id, 0) + refs

    def _thin_milestones(self) -> None:
        """Cap the milestone list by THINNING, never FIFO-eviction.

        The goal is a retained set that stays evenly spread across the WHOLE
        run, because that is the only thing the milestone tier provides that
        the rolling buffer does not (the rolling buffer already covers
        "recent" far better). FIFO-eviction would drop the oldest first,
        which is exactly backwards.

        Removes one entry at a time, always the most REDUNDANT interior one:
        the entry whose neighbours are closest together in update-count, i.e.
        whose removal widens the largest gap least. The first and last
        entries are never candidates, so the run's full span is preserved.

        An earlier version halved the "older" slice (`older[0::2]`) on each
        overflow. That looks like even thinning but is not: repeated halving
        collapses toward the front and the tail, and measured over a
        6,000-update replay it retained `u50, u5200, u5450, ... u6000` --
        the oldest, then a cluster at the very end, with a 5,150-update hole
        in the middle. Spacing has to be computed, not approximated by
        slicing.
        """
        while len(self._milestones) > self.milestone_capacity and len(self._milestones) > 2:
            # Candidate i (interior only) scored by the gap its removal
            # creates: update[i+1] - update[i-1]. Smallest score = most
            # redundant = safest to drop.
            best_i, best_gap = None, None
            for i in range(1, len(self._milestones) - 1):
                gap = self._milestones[i + 1][3] - self._milestones[i - 1][3]
                if best_gap is None or gap < best_gap:
                    best_i, best_gap = i, gap
            if best_i is None:
                break
            self._release(self._milestones.pop(best_i))

    def _weighted_recent_choice(self, pool: List[tuple]) -> tuple:
        """Pick one entry from *pool*, weighting linearly toward the end
        (newest) of the list -- e.g. for n=4 the weights are [1, 2, 3, 4],
        so the newest entry is 4x as likely as the oldest."""
        n = len(pool)
        if n == 1:
            return pool[0]
        return random.choices(pool, weights=range(1, n + 1), k=1)[0]

    def sample(self) -> Optional[tuple]:
        """Return one recency-weighted (path_str, snapshot_id, tag) reference,
        or None if the pool is empty (see class docstring for the sampling
        rule)."""
        has_rolling   = bool(self._snapshots)
        has_milestone = bool(self._milestones)
        if not has_rolling and not has_milestone:
            return None
        if has_rolling and has_milestone:
            use_rolling = random.random() < self.p_recent
        else:
            use_rolling = has_rolling
        if use_rolling:
            snap_id, path_str, tag = self._weighted_recent_choice(self._snapshots)[:3]
        else:
            snap_id, path_str, tag = random.choice(self._milestones)[:3]
        return path_str, snap_id, tag

    def sample_n(self, n: int) -> List[Optional[tuple]]:
        """Return *n* independently sampled (path_str, snapshot_id, tag)
        references (with replacement), or a list of Nones if the pool is
        empty."""
        return [self.sample() for _ in range(n)]

    def cleanup(self) -> None:
        """Unlink every snapshot file this pool still holds a live reference
        to. Call once at run end (mirrors _train_parallel's weights-file
        `finally:` cleanup) -- failures are swallowed, same rationale as
        that cleanup: a leftover file in /dev/shm must never crash a
        training run on its way out, but must not be left behind either."""
        for path_str in list(self._paths.values()):
            try:
                Path(path_str).unlink()
            except OSError:
                pass
        self._paths.clear()
        self._refcount.clear()

    def __len__(self) -> int:
        return len(self._snapshots) + len(self._milestones)


# -------------------------------------------------------------------------
# Policy-wrapping agent
# -------------------------------------------------------------------------

class PPOAgent:
    """Thin wrapper that calls BGPolicyNetwork.get_action for a single player."""

    def __init__(
        self,
        policy: BGPolicyNetwork,
        ppo_trainer: PPOTrainer,
        player_id: int,
        device: str = "cpu",
        deterministic: bool = False,
        game_uid: Any = None,
    ) -> None:
        self.policy       = policy
        self.trainer      = ppo_trainer
        self.player_id    = player_id
        self.device       = device
        self.deterministic = deterministic
        self._last_obs: Optional[dict] = None
        self._cached_type_mask: Optional[np.ndarray] = None
        self._cached_ptr_mask:  Optional[np.ndarray] = None
        # Identifies which (game, player) trajectory this agent's transitions
        # belong to, so RolloutBuffer.compute_advantages never bootstraps GAE
        # across a different player's or game's transitions that happen to
        # land next to this one in a shared buffer (multiple training players
        # share one PPOTrainer per game, and multiple games get merged into
        # one buffer per update). game_uid should be unique per game — the
        # per-game seed is a natural choice; falls back to a fresh uuid so
        # this is still safe (if less reproducible) when seed is None.
        self.traj_id = (game_uid if game_uid is not None else uuid.uuid4().hex, player_id)

    def get_action(self, obs: dict) -> tuple:
        """Select an action given an observation dict.

        Returns (type_idx, ptr_idx) where ptr_idx is -1 for non-pointer types.

        Caches the action masks computed from the *current* player state so
        that record_transition can use them without re-reading ps (which will
        have changed after step_shopping mutates the live object).
        """
        self._last_obs = obs
        ps  = obs["player_state"]
        dev = torch.device(self.device)

        # Compute and CACHE masks now, before the state is mutated by step_shopping
        self._cached_type_mask = build_type_mask(ps).numpy()

        t_mask_t = torch.from_numpy(self._cached_type_mask).unsqueeze(0).to(dev)
        p_mask_t = build_pointer_mask(ps, -1).unsqueeze(0).to(dev)  # full occupancy fallback

        board_t  = torch.tensor(obs["board_tokens"][None],   dtype=torch.float32, device=dev)
        shop_t   = torch.tensor(obs["shop_tokens"][None],    dtype=torch.float32, device=dev)
        hand_t   = torch.tensor(obs["hand_tokens"][None],    dtype=torch.float32, device=dev)
        scalar_t = torch.tensor(obs["scalar_context"][None], dtype=torch.float32, device=dev)
        opp_np   = obs.get("opp_tokens")
        opp_t    = torch.tensor(opp_np[None], dtype=torch.float32, device=dev) if opp_np is not None else None

        # ptr_mask_fn closes over `ps` (captured pre-mutation, above) so
        # get_action can build the exact type-specific mask itself, AFTER
        # sampling the type, and hand back the mask it actually used -- see
        # get_action's docstring and CONTEXT.md (2026-08-31/09-01) for why
        # sampling under one mask and storing a separately-recomputed one is
        # the bug this closure exists to make impossible.
        type_idx, ptr_idx, _log_prob, _value, used_ptr_mask = self.policy.get_action(
            board_t, shop_t, hand_t, scalar_t,
            type_mask=t_mask_t, pointer_mask=p_mask_t,
            deterministic=self.deterministic, opp_tokens=opp_t,
            ptr_mask_fn=lambda t_idx: build_pointer_mask(ps, t_idx),
        )
        # Cache the mask get_action actually used (also pre-mutation) --
        # stored mask == sampled mask by construction. Do NOT reintroduce a
        # separate build_pointer_mask(ps, type_idx) recomputation here; that
        # is precisely the bug described above.
        self._cached_ptr_mask = used_ptr_mask.cpu().numpy()
        return type_idx, ptr_idx

    def record_transition(
        self,
        obs: dict,
        type_action: int,
        ptr_action:  int,
        reward: float,
        done:   bool,
        is_bootstrap: bool = False,
    ) -> None:
        """Push a completed transition into the PPO rollout buffer.

        is_bootstrap=True marks a row that is NOT a real decision -- the game
        loop's terminal transition, which reuses END_TURN as a no-op carrier for
        the final placement reward. See agent/ppo.py Transition.is_bootstrap.

        Uses masks cached in get_action (pre-mutation) rather than
        recomputing from obs['player_state'] which is a live reference
        and will reflect the post-action state by the time this is called.
        """
        if obs is None:
            return
        # Fall back to building masks from obs if get_action hasn't been called yet
        # (e.g. for terminal transitions delivered to players eliminated mid-game).
        if self._cached_type_mask is None:
            ps = obs.get("player_state")
            self._cached_type_mask = build_type_mask(ps).numpy() if ps is not None else np.ones(N_ACTION_TYPES, dtype=bool)
        if self._cached_ptr_mask is None:
            ps = obs.get("player_state")
            self._cached_ptr_mask = build_pointer_mask(ps, type_action).numpy() if ps is not None else np.ones(POINTER_DIM, dtype=bool)
        type_mask = self._cached_type_mask
        ptr_mask  = self._cached_ptr_mask
        _ps = obs.get("player_state")
        self.trainer.collect_transition(
            board_tokens   = obs["board_tokens"],
            shop_tokens    = obs["shop_tokens"],
            hand_tokens    = obs["hand_tokens"],
            scalar_context = obs["scalar_context"],
            type_action    = type_action,
            ptr_action     = ptr_action,
            type_mask      = type_mask,
            pointer_mask   = ptr_mask,
            reward         = reward,
            done           = done,
            opp_tokens     = obs.get("opp_tokens"),
            traj_id        = self.traj_id,
            round_num      = getattr(_ps, "round_num", None),
            is_bootstrap   = is_bootstrap,
        )

    def record_transition_precomputed(
        self,
        obs:         dict,
        type_action: int,
        ptr_action:  int,
        reward:      float,
        done:        bool,
        log_prob:    float,
        value:       float,
        type_mask:   np.ndarray,
        ptr_mask:    np.ndarray,
    ) -> None:
        """Store a transition using pre-computed log_prob and value.

        Skips the evaluate_actions() forward pass — called from the batched
        shopping loop where log_prob/value come from get_action_batch().
        """
        if obs is None:
            return
        _ps = obs.get("player_state")
        self.trainer.store_transition(
            board_tokens   = obs["board_tokens"],
            shop_tokens    = obs["shop_tokens"],
            hand_tokens    = obs["hand_tokens"],
            scalar_context = obs["scalar_context"],
            type_action    = type_action,
            ptr_action     = ptr_action,
            type_mask      = type_mask,
            pointer_mask   = ptr_mask,
            reward         = reward,
            done           = done,
            log_prob       = log_prob,
            value          = value,
            opp_tokens     = obs.get("opp_tokens"),
            traj_id        = self.traj_id,
            round_num      = getattr(_ps, "round_num", None),
        )


# -------------------------------------------------------------------------
# Static (historical) agent — acts but never records transitions
# -------------------------------------------------------------------------

class StaticAgent:
    """Wraps a frozen policy snapshot; acts in the game but collects no data.

    Provides the same get_action / record_transition* interface as PPOAgent
    so game_loop.py can call it without type-checking.  Transition methods
    are intentional no-ops: historical opponents only provide diverse
    opposition, they don't contribute to the PPO buffer.
    """

    def __init__(
        self,
        policy: BGPolicyNetwork,
        player_id: int,
        device: str = "cpu",
    ) -> None:
        self.policy    = policy
        self.player_id = player_id
        self.device    = device
        self._cached_type_mask: Optional[np.ndarray] = None
        self._cached_ptr_mask:  Optional[np.ndarray] = None

    def get_action(self, obs: dict) -> tuple:
        ps  = obs["player_state"]
        dev = torch.device(self.device)

        self._cached_type_mask = build_type_mask(ps).numpy()
        t_mask_t = torch.from_numpy(self._cached_type_mask).unsqueeze(0).to(dev)
        p_mask_t = build_pointer_mask(ps, -1).unsqueeze(0).to(dev)  # full occupancy fallback

        board_t  = torch.tensor(obs["board_tokens"][None],   dtype=torch.float32, device=dev)
        shop_t   = torch.tensor(obs["shop_tokens"][None],    dtype=torch.float32, device=dev)
        hand_t   = torch.tensor(obs["hand_tokens"][None],    dtype=torch.float32, device=dev)
        scalar_t = torch.tensor(obs["scalar_context"][None], dtype=torch.float32, device=dev)
        opp_np   = obs.get("opp_tokens")
        opp_t    = torch.tensor(opp_np[None], dtype=torch.float32, device=dev) if opp_np is not None else None

        # ptr_mask_fn closes over `ps` (pre-mutation) -- see PPOAgent.get_action
        # for why the mask used to sample must be the one that gets cached.
        with torch.no_grad():
            type_idx, ptr_idx, _, _, used_ptr_mask = self.policy.get_action(
                board_t, shop_t, hand_t, scalar_t,
                type_mask=t_mask_t, pointer_mask=p_mask_t, opp_tokens=opp_t,
                ptr_mask_fn=lambda t_idx: build_pointer_mask(ps, t_idx),
            )
        # Cache the mask get_action actually used -- do NOT reintroduce a
        # separate build_pointer_mask(ps, type_idx) recomputation here.
        self._cached_ptr_mask = used_ptr_mask.cpu().numpy()
        return type_idx, ptr_idx

    def record_transition(self, *_a, **_kw) -> None:  # no-op
        pass

    def record_transition_precomputed(self, *_a, **_kw) -> None:  # no-op
        pass


# ---------------------------------------------------------------------------
# Curve-based leveling for the scripted baselines (HeuristicAgent,
# GreedyPlayAgent).
# ---------------------------------------------------------------------------
#
# Both agents used to have broken or capped leveling (Greedy had none at all;
# Heuristic hard-capped at tier 4), which made them a much weaker ceiling for
# "the agent beats the baseline" than real Battlegrounds opponents -- see the
# task description / CONTEXT.md for the measured impact (damage plateaus
# ~7 instead of ~15-25, games run ~20-22 rounds instead of ~12-18). Both
# agents now level toward the same on-curve target used by the tier-shape
# reward potential (env.game_loop.expected_tier_for_round) so there is
# exactly one definition of "on curve" in the codebase.
#
# Naive "level whenever affordable and behind curve" is itself a failure
# mode: it can pour every turn's gold into tavern tier and field a
# near-empty board, which would make these baselines WORSE than the old
# tier-1/tier-4-capped versions, not better. SCRIPTED_MIN_BOARD_TO_LEVEL
# guards against that by requiring a minimum board presence before either
# agent will divert gold to LEVEL_UP instead of buying/placing -- 3 is a
# light bar (well under the 7-slot cap) chosen so leveling can still start
# early (round 2-3, once the first couple of buys have landed) without ever
# competing with an empty board for gold.
SCRIPTED_MIN_BOARD_TO_LEVEL = 3


def _effective_power(m) -> int:
    """Attack + health for one minion, INCLUDING accumulated buffs.

    Base stats alone are actively misleading when comparing minions: a minion
    sitting on +30/+30 of permanent buffs still reads as its base 1/1, so a
    base-stats ranking happily nominates the most valuable minion on the board
    as the "weakest" one to sell.

    Mirrors the effective-stat sum used by symbolic.board_computer._board_power
    (base + perm_* + game_*), and goes through env.player_state.minion_stats so
    it tolerates both minion-dict shapes rather than re-inlining a .get() --
    reading "attack"/"health" off a raw card-def dict silently returns 0 (see
    CONTEXT.md 2026-09-01).
    """
    d = m if isinstance(m, dict) else getattr(m, "__dict__", {})
    atk, hp = minion_stats(d)
    atk += d.get("perm_atk_bonus", 0) + d.get("game_atk_bonus", 0)
    hp += d.get("perm_hp_bonus", 0) + d.get("game_hp_bonus", 0)
    return atk + hp


def _scripted_choice_action(ps):
    """Resolve a pending PendingChoice for a scripted (non-learning) agent.

    Returns ``(type, pointer)`` when ps.choice_pending is set, else None.

    Scripted agents exist to be a stable, beatable baseline, so these rules are
    deliberately simple but not deliberately bad -- a baseline that throws away
    every Tyrael and every Blood Gem would flatter the learned policy's
    win rate for the wrong reason.

    Target rules:
      set_stats ("set another minion's stats to 50/50") targets the WEAKEST
      minion, because it OVERWRITES stats -- pointing it at the biggest minion
      is a downgrade. Everything else (buffs, Blood Gems, consume, magnetise)
      targets the STRONGEST, concentrating stats the way scripted play should.

    Option rule: always branch 0. The Choose One branches are close enough in
    value that a fixed pick is an honest baseline, and it keeps these agents
    deterministic.
    """
    choice = getattr(ps, "choice_pending", None)
    if choice is None:
        return None

    if getattr(choice, "kind", "target") == "option":
        return 11, PTR_SHOP_OFF + 0

    targets = [i for i in getattr(choice, "targets", []) if 0 <= i < len(ps.board)]
    if not targets:
        # No legal target left. Point at slot 0 anyway: step_shopping leaves the
        # choice pending, and _force_resolve_choices clears it at turn end --
        # much better than returning None and having the caller fall through to
        # an action the mask forbids.
        return 10, PTR_BOARD_OFF + 0

    def _power(i):
        m = ps.board[i]
        return m.effective_attack() + m.effective_health()

    pick = min(targets, key=_power) if choice.effect == "set_stats" else max(targets, key=_power)
    return 10, PTR_BOARD_OFF + pick


def _scripted_should_level(ps) -> bool:
    """Shared LEVEL_UP guard for the scripted baselines.

    Level only when:
      1. Not already at the tier cap (6).
      2. Actually BEHIND expected_tier_for_round(ps.round_num) -- leveling
         further than the round's rough curve pays nothing extra for the
         PPO tier-shape potential either (see _tier_potential's min(1.0, ..)
         cap), so there's no reason for a "competent" scripted baseline to
         over-level at the expense of board development.
      3. The board already has at least SCRIPTED_MIN_BOARD_TO_LEVEL minions
         -- prevents the empty-board failure mode described above by making
         board development take priority whenever the board is thin.
    """
    if ps.tavern_tier >= 6:
        return False
    if ps.tavern_tier >= expected_tier_for_round(ps.round_num):
        return False
    return len(ps.board) >= SCRIPTED_MIN_BOARD_TO_LEVEL


class EvalStaticAgent(StaticAgent):
    """Frozen-reference opponent for evaluate_policy: deterministic, and it
    opts out of batching.

    Two properties matter here, and both were learned the hard way:

    1. `supports_batching = False`. EvalAgent's deterministic argmax is only
       honoured on game_loop's sequential per-player path, and that path is
       chosen only when EVERY seat opts out of batching (see EvalAgent's
       docstring). Plain StaticAgent does not set it, so seating one opposite
       an EvalAgent would silently switch the game to the batched path and
       destroy the determinism the eval depends on.

    2. Deterministic action selection. StaticAgent SAMPLES from the policy
       distribution, which is right for self-play (diverse opposition) and
       wrong for a measurement: measured over 4 repeats of a 16-game
       reference eval at a fixed seed, sampling opponents produced mean
       placements of 5.75 / 6.13 / 6.25 / 6.31 -- a ~0.6 spread of pure noise
       on the metric that is supposed to be the un-gameable progress signal.
       The whole point of pinning per-game seeds is that a change in this
       number means the POLICY changed; a sampling opponent breaks that.
    """

    supports_batching = False

    def get_action(self, obs: dict) -> tuple:
        ps  = obs["player_state"]
        dev = torch.device(self.device)
        self._cached_type_mask = build_type_mask(ps).numpy()
        t_mask_t = torch.from_numpy(self._cached_type_mask).unsqueeze(0).to(dev)
        p_mask_t = build_pointer_mask(ps, -1).unsqueeze(0).to(dev)

        board_t  = torch.tensor(obs["board_tokens"][None],  dtype=torch.float32, device=dev)
        shop_t   = torch.tensor(obs["shop_tokens"][None],   dtype=torch.float32, device=dev)
        hand_t   = torch.tensor(obs["hand_tokens"][None],   dtype=torch.float32, device=dev)
        scalar_t = torch.tensor(obs["scalar_context"][None], dtype=torch.float32, device=dev)
        opp_np   = obs.get("opponent_tokens")
        opp_t    = (torch.tensor(opp_np[None], dtype=torch.float32, device=dev)
                    if opp_np is not None else None)

        with torch.no_grad():
            type_idx, ptr_idx, _, _, used_ptr_mask = self.policy.get_action(
                board_t, shop_t, hand_t, scalar_t,
                type_mask=t_mask_t, pointer_mask=p_mask_t, opp_tokens=opp_t,
                ptr_mask_fn=lambda t_idx: build_pointer_mask(ps, t_idx),
                deterministic=True,          # <-- the difference from StaticAgent
            )
        self._cached_ptr_mask = used_ptr_mask.cpu().numpy()
        return type_idx, ptr_idx


class HeuristicAgent:
    """Leveling-focused scripted opponent for population diversity anchoring.

    Provides the same get_action / record_transition* interface as StaticAgent
    so game_loop.py requires zero changes.  Uses no policy network — pure
    rule-based logic — so it is cheap and always picklable.

    Setting supports_batching = False opts this agent out of the batched
    shopping phase (game_loop._agents_support_batching checks this flag),
    causing the game to fall back to the sequential path.  The sequential path
    calls get_action() per player per step, which is exactly what this class
    implements.

    Priority order each step:
      1. Level up  — curve-based: affordable, below tier 6, behind
                     expected_tier_for_round(round_num), and the board has
                     at least SCRIPTED_MIN_BOARD_TO_LEVEL minions already
                     (see _scripted_should_level -- guards against pouring
                     all gold into tier at the expense of an empty board)
      2. Buy       — highest-tier minion available in the shop
      3. Place     — any card sitting in hand onto the board
      4. Sell      — weakest board minion when board is full and a buy is possible
      5. End turn
    """

    supports_batching = False  # forces sequential shopping path in game_loop

    def __init__(self, player_id: int) -> None:
        self.player_id = player_id

    def get_action(self, obs: dict) -> tuple:
        ps        = obs["player_state"]
        type_mask = build_type_mask(ps)

        def valid(t: int) -> bool:
            return bool(type_mask[t].item())

        # Trinket offer / discover in progress: the valid pointer range here is
        # into ps.discover_pending or the trinket offer list, NOT ps.shop (which
        # step_shopping ignores entirely while one of these is pending, and which
        # this class's own buy logic below reads) -- picking based on ps.shop
        # produces an out-of-range pointer almost every time (discover/trinket
        # offers are ~3 items; ps.shop is up to 7), which step_shopping silently
        # no-ops on rather than erroring, permanently stalling this agent for the
        # rest of the turn (capped at max_actions=30) since neither state clears
        # itself and END_TURN doesn't escape a pending discover. Index 0 is
        # always valid here when pending (both are non-empty by construction).
        # A pending choice pauses shopping and is the ONLY legal action; see
        # _scripted_choice_action.
        _ca = _scripted_choice_action(ps)
        if _ca is not None:
            return _ca

        if ps.trinket_offer_pending or ps.discover_pending:
            return 0, PTR_SHOP_OFF + 0

        # 1. Level up (type 5) — mask already verifies gold >= cost;
        #    _scripted_should_level adds the curve + board-size guard
        if valid(5) and _scripted_should_level(ps):
            return 5, 0

        # 2. Place any card from hand onto the board (type 2).
        #    Placing comes BEFORE buying: this agent used to buy first, which
        #    meant that once the board filled it kept buying cards it could
        #    never play, clogging its hand (measured: BUY 38% of actions vs
        #    PLACE 16%) and paying the per-card hand penalty every turn.
        if valid(2):
            for i, m in enumerate(ps.hand):
                if m is not None and getattr(m, "card_id", ""):
                    return 2, PTR_HAND_OFF + i

        # 3. Sell the weakest board minion (type 1) -- but ONLY when the shop
        #    actually offers something stronger.
        #
        #    This branch used to sit AFTER the buy branch while itself
        #    requiring valid(0), so the buy above always returned first and
        #    this was unreachable dead code: measured 0 sell events across 10
        #    games. The agent therefore never made room for an upgrade once
        #    full. GreedyPlayAgent checks sell before buy, which is one reason
        #    it beat this agent by ~0.9 placement head-to-head despite the same
        #    tavern tier. Ordering it before the buy is what makes the rule
        #    real.
        #
        #    The comparison also ranks minions by EFFECTIVE power, not base
        #    attack+health: a minion sitting on +30/+30 of accumulated buffs
        #    still reads as its base 1/1, so a base-stats ranking would
        #    nominate the best minion on the board as the "weakest" to sell.
        if valid(1) and len(ps.board) >= 7 and valid(0):
            worst_idx, worst_power = -1, float("inf")
            for i, m in enumerate(ps.board):
                if m is not None and getattr(m, "card_id", ""):
                    power = _effective_power(m)
                    if power < worst_power:
                        worst_power, worst_idx = power, i
            best_shop = max(
                (_effective_power(m) for m in ps.shop
                 if m is not None and getattr(m, "card_id", "")),
                default=-1,
            )
            # Strictly better only: swapping for an equal minion is a pure
            # loss once the gold cost is counted.
            if worst_idx >= 0 and best_shop > worst_power:
                return 1, PTR_BOARD_OFF + worst_idx

        # 4. Buy the highest-tier shop minion (type 0)
        if valid(0):
            best_idx, best_tier = -1, -1
            for i, m in enumerate(ps.shop):
                if m is not None and getattr(m, "card_id", ""):
                    t = getattr(m, "tier", 1)
                    if t > best_tier:
                        best_tier, best_idx = t, i
            if best_idx >= 0:
                return 0, PTR_SHOP_OFF + best_idx

        # 5. End turn
        return 7, 0

    def record_transition(self, *_a, **_kw) -> None:  # no-op
        pass

    def record_transition_precomputed(self, *_a, **_kw) -> None:  # no-op
        pass


class GreedyPlayAgent:
    """Naive scripted opponent: buys and plays everything, never sells —
    except to make room for a strictly higher-tier minion in the shop — and
    now levels its tavern on the same curve any real player would.

    Leveling was added because a scripted opponent that sits at tavern tier
    1 for the entire game is not a "naive" baseline, it's a BROKEN one: with
    real minion stats, combat damage in this sim scales with tavern tier
    (see env/game_loop.py), so a permanently-tier-1 opponent caps the whole
    population's late-game damage and makes "the agent beats the baseline"
    a much weaker claim than it looks (see CLAUDE.md task notes / CONTEXT.md
    for the measured before/after). Leveling is curve-based and guarded by
    _scripted_should_level (see the block above HeuristicAgent) so it can't
    regress into the opposite failure mode of dumping every turn's gold into
    tier and fielding an empty board — everything else about this agent's
    naive, un-optimized identity (buy first-affordable, sell only to
    upgrade) is unchanged.

    Provides the same get_action / record_transition* interface as
    HeuristicAgent so game_loop.py requires zero changes.  Uses no policy
    network — pure rule-based logic — so it is cheap and always picklable.

    Priority order each step:
      1. Place — any card sitting in hand onto the board (board has room)
      2. Level — curve-based, guarded by _scripted_should_level (see above)
      3. Sell  — the lowest-tier board minion, but ONLY when the board is
                 full (7/7) AND the shop currently offers a minion with a
                 strictly higher tier than that weakest-tier board minion
      4. Buy   — the first affordable minion in the shop (left to right)
      5. End turn
    """

    supports_batching = False  # forces sequential shopping path in game_loop

    def __init__(self, player_id: int) -> None:
        self.player_id = player_id

    def get_action(self, obs: dict) -> tuple:
        ps        = obs["player_state"]
        type_mask = build_type_mask(ps)

        def valid(t: int) -> bool:
            return bool(type_mask[t].item())

        # Trinket offer / discover in progress -- see HeuristicAgent.get_action
        # for why this must be handled before the normal ps.shop-based logic.
        # A pending choice pauses shopping and is the ONLY legal action; see
        # _scripted_choice_action.
        _ca = _scripted_choice_action(ps)
        if _ca is not None:
            return _ca

        if ps.trinket_offer_pending or ps.discover_pending:
            return 0, PTR_SHOP_OFF + 0

        # 1. Place any card from hand onto the board (type 2)
        if valid(2):
            for i, m in enumerate(ps.hand):
                if m is not None and getattr(m, "card_id", ""):
                    return 2, PTR_HAND_OFF + i

        # 2. Level up (type 5) — mask already verifies gold >= cost;
        #    _scripted_should_level adds the curve + board-size guard
        if valid(5) and _scripted_should_level(ps):
            return 5, 0

        # 3. Sell the lowest-tier board minion, only to make room for a
        #    strictly higher-tier minion currently sitting in the shop
        if valid(1) and len(ps.board) >= 7:
            worst_idx, worst_tier = -1, float("inf")
            for i, m in enumerate(ps.board):
                if m is not None and getattr(m, "card_id", ""):
                    tier = getattr(m, "tier", 1)
                    if tier < worst_tier:
                        worst_tier, worst_idx = tier, i
            if worst_idx >= 0:
                shop_has_upgrade = any(
                    m is not None and getattr(m, "card_id", "")
                    and getattr(m, "tier", 1) > worst_tier
                    for m in ps.shop
                )
                if shop_has_upgrade:
                    return 1, PTR_BOARD_OFF + worst_idx

        # 4. Buy the first affordable minion in the shop (type 0)
        if valid(0):
            for i, m in enumerate(ps.shop):
                if m is not None and getattr(m, "card_id", ""):
                    return 0, PTR_SHOP_OFF + i

        # 5. End turn
        return 7, 0

    def record_transition(self, *_a, **_kw) -> None:  # no-op
        pass

    def record_transition_precomputed(self, *_a, **_kw) -> None:  # no-op
        pass


class EvalAgent:
    """Wraps a policy for read-only, deterministic evaluation.

    Used only by evaluate_policy(): selects the argmax action under the
    masked action distribution (BGPolicyNetwork.get_action(...,
    deterministic=True)) instead of sampling, so the eval metric reflects
    the policy's learned mode rather than exploration noise.

    record_transition* are no-ops and this class never references a
    PPOTrainer or a rollout buffer -- evaluate_policy cannot disturb any
    training state through this agent, even when called mid-training-run.

    supports_batching = False forces BattlegroundsGame's sequential
    per-player action-selection path (see
    BattlegroundsGame._agents_support_batching), which is what makes the
    deterministic flag take effect at all: the batched shopping path calls
    policy.get_action_batch() directly and always samples, with no
    deterministic option and no per-agent hook to change that.
    """

    supports_batching = False

    def __init__(self, policy: BGPolicyNetwork, player_id: int, device: str = "cpu") -> None:
        self.policy    = policy
        self.player_id = player_id
        self.device    = device

    def get_action(self, obs: dict) -> tuple:
        ps  = obs["player_state"]
        dev = torch.device(self.device)

        t_mask   = build_type_mask(ps)
        t_mask_t = t_mask.unsqueeze(0).to(dev)
        p_mask_t = build_pointer_mask(ps, -1).unsqueeze(0).to(dev)  # full occupancy fallback

        board_t  = torch.tensor(obs["board_tokens"][None],   dtype=torch.float32, device=dev)
        shop_t   = torch.tensor(obs["shop_tokens"][None],    dtype=torch.float32, device=dev)
        hand_t   = torch.tensor(obs["hand_tokens"][None],    dtype=torch.float32, device=dev)
        scalar_t = torch.tensor(obs["scalar_context"][None], dtype=torch.float32, device=dev)
        opp_np   = obs.get("opp_tokens")
        opp_t    = torch.tensor(opp_np[None], dtype=torch.float32, device=dev) if opp_np is not None else None

        # ptr_mask_fn ensures the deterministic argmax pointer is chosen
        # under the correct type-specific mask (not the type-agnostic
        # occupancy fallback) -- same fix as PPOAgent/StaticAgent above.
        with torch.no_grad():
            type_idx, ptr_idx, _, _, _used_ptr_mask = self.policy.get_action(
                board_t, shop_t, hand_t, scalar_t,
                type_mask=t_mask_t, pointer_mask=p_mask_t,
                deterministic=True, opp_tokens=opp_t,
                ptr_mask_fn=lambda t_idx: build_pointer_mask(ps, t_idx),
            )
        return type_idx, ptr_idx

    def record_transition(self, *_a, **_kw) -> None:  # no-op — eval collects no data
        pass

    def record_transition_precomputed(self, *_a, **_kw) -> None:  # no-op
        pass


# -------------------------------------------------------------------------
# Card definitions loader
# -------------------------------------------------------------------------

def load_card_defs(path: Path) -> Dict[str, dict]:
    """Load bg_card_definitions.json, running the pipeline script if absent."""
    if not path.exists():
        logger.warning(
            "Card definitions not found at %s. "
            "Run `python bg_card_pipeline.py --output bg_card_definitions.json` first.",
            path,
        )
        # Return a minimal empty dict so the rest of the pipeline can proceed
        # (TavernPool / CardEncoder gracefully degrade on unknown cards).
        return {}

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    # Extract trinkets list before unwrapping cards envelope
    trinkets_raw = []
    if isinstance(raw, dict) and "trinkets" in raw:
        trinkets_raw = raw["trinkets"] if isinstance(raw["trinkets"], list) else []

    # Unwrap {"version": ..., "cards": {...}} envelope if present
    if isinstance(raw, dict) and "cards" in raw and isinstance(raw["cards"], dict):
        defs = raw["cards"]
    elif isinstance(raw, dict) and "cards" in raw and isinstance(raw["cards"], list):
        defs = {d.get("card_id", d.get("id", str(i))): d
                for i, d in enumerate(raw["cards"])}
    elif isinstance(raw, list):
        defs = {d.get("card_id", d.get("id", str(i))): d for i, d in enumerate(raw)}
    else:
        defs = raw  # type: ignore[assignment]

    # Normalise: if still a list, key by card_id
    if isinstance(defs, list):
        defs = {d.get("card_id", d.get("id", str(i))): d for i, d in enumerate(defs)}

    # Merge trinkets into card_defs so TrinketHandler can find them by card_id
    for t in trinkets_raw:
        cid = t.get("card_id")
        if cid:
            defs[cid] = t

    logger.info(
        "Loaded %d card definitions (%d trinkets) from %s",
        len(defs), len(trinkets_raw), path,
    )
    return defs


# -------------------------------------------------------------------------
# Component factory
# -------------------------------------------------------------------------

def build_components(
    card_defs: Dict[str, dict],
    use_firestone: bool,
    device: str,
    seed: Optional[int],
) -> dict:
    """Instantiate all pipeline components and return them in a dict."""
    tavern_pool = TavernPool(card_defs, seed=seed)
    matchmaker  = Matchmaker(n_players=N_PLAYERS, seed=seed)
    board_comp  = SymbolicBoardComputer(card_defs)
    firestone   = FirestoneClient(
        firestone_path=None,
        mock_mode=(not use_firestone),
    )
    encoder = CardEncoder(card_defs)

    policy = make_policy().to(device)

    ppo_config = PPOConfig(device=device)
    ppo_trainer = PPOTrainer(policy, ppo_config)

    return {
        "card_defs":    card_defs,
        "tavern_pool":  tavern_pool,
        "matchmaker":   matchmaker,
        "board_comp":   board_comp,
        "firestone":    firestone,
        "encoder":      encoder,
        "policy":       policy,
        "ppo_trainer":  ppo_trainer,
        "ppo_config":   ppo_config,
    }


# -------------------------------------------------------------------------
# Per-game wrapper
# -------------------------------------------------------------------------

def run_one_game(
    components: dict,
    game_idx: int,
    seed: Optional[int],
) -> GameResult:
    """Create a BattlegroundsGame with shared-policy agents and run it."""
    policy     = components["policy"]
    trainer    = components["ppo_trainer"]
    device     = components["ppo_config"].device
    card_defs  = components["card_defs"]

    # All 8 players share the same policy (self-play) and the same PPOTrainer
    # buffer, which may also persist across multiple run_one_game calls in a
    # serial training loop -- game_idx makes each call's trajectories distinct.
    agents: List[PPOAgent] = [
        PPOAgent(policy, trainer, player_id=pid, device=device, game_uid=game_idx)
        for pid in range(N_PLAYERS)
    ]

    game = BattlegroundsGame(
        card_defs       = card_defs,
        agents          = agents,
        board_computer  = components["board_comp"],
        firestone_client= components["firestone"],
        matchmaker      = components["matchmaker"],
        tavern_pool     = components["tavern_pool"],
        n_players       = N_PLAYERS,
        seed            = (seed + game_idx) if seed is not None else None,
        batched         = True,
    )

    result = game.run_game()
    return result


# -------------------------------------------------------------------------
# Parallel worker  (module-level — required for Windows multiprocessing spawn)
# -------------------------------------------------------------------------

# Per-process cache populated by _worker_init — avoids re-pickling card_defs
# on every single game call (card_defs is ~1 MB and never changes).
_W_CARD_DEFS: dict = {}
_W_DEVICE: str = "cpu"

# One-entry cache for the broadcast policy weights file (see _train_parallel's
# weights_ref / sd_version / _write_weights). Deliberately just ONE entry, not
# an LRU: with UPDATE_INTERVAL == N_WORKERS each worker handles ~1 game per
# weight version, so the hit rate is near zero regardless of cache size — this
# exists only to make an occasional same-version repeat free. The actual
# throughput win is elsewhere: the MAIN process now writes the ~13.89MB
# state_dict to disk ONCE per PPO update instead of pickling a copy into every
# one of the N_WORKERS tasks it dispatches.
_W_SD_VERSION: Optional[int] = None
_W_SD_CACHE: Optional[dict] = None

# Bounded per-worker LRU of loaded SNAPSHOT opponent networks, keyed by the
# pool's stable snapshot_id. Unlike the one-entry current-policy cache above,
# this one genuinely pays off: SnapshotPool holds a small set of ids (capacity
# 20 rolling + MILESTONE_CAPACITY milestones) and recency-weighted sampling
# draws the same handful of ids over and over, so a worker sees repeats
# constantly. Bounded at 8 because each BGPolicyNetwork is ~3.47M params and a
# worker already carries ~500MB RSS; keyed by snapshot_id rather than the old
# `id(sd)` because a memory address is only stable within one task, whereas the
# whole point here is reuse ACROSS tasks.
_W_SNAP_CACHE_MAX = 8
_W_SNAP_CACHE: "OrderedDict[int, Any]" = OrderedDict()


def _load_snapshot_policy(path_str: str, snapshot_id: int, device: str):
    """Return the snapshot network for *snapshot_id*, loading it from
    *path_str* on a cache miss.

    Raises on a failed load. A snapshot file can legitimately disappear (its
    entry was evicted from the rolling buffer or thinned out of the milestone
    list while a queue_factor-backlogged task still referenced it), and the
    only safe response is to fail loudly: silently substituting different
    weights would mean training against an opponent that is not the one the
    run believes it faced, and nothing downstream could ever detect it. This
    mirrors the current-policy weights path, which raises for the same reason.
    """
    import torch as _torch
    global _W_SNAP_CACHE
    cached = _W_SNAP_CACHE.get(snapshot_id)
    if cached is not None:
        _W_SNAP_CACHE.move_to_end(snapshot_id)
        return cached
    try:
        sd = _torch.load(path_str, map_location="cpu")
    except (FileNotFoundError, OSError, RuntimeError, EOFError) as exc:
        raise RuntimeError(
            f"worker could not load snapshot {snapshot_id} from {path_str}: {exc}"
        ) from exc
    pol = make_policy().to(device)
    pol.load_state_dict(sd)
    pol.eval()
    _W_SNAP_CACHE[snapshot_id] = pol
    _W_SNAP_CACHE.move_to_end(snapshot_id)
    while len(_W_SNAP_CACHE) > _W_SNAP_CACHE_MAX:
        _W_SNAP_CACHE.popitem(last=False)
    return pol


_W_EVAL_CACHE: "OrderedDict[str, Any]" = OrderedDict()
# 9 = the 7 gauntlet references + the policy under test + slack. This was 4
# ("one eval needs at most two distinct networks"), which is true for the
# greedy/heuristic/reference evals but WRONG for the gauntlet, where every
# seat holds a different checkpoint -- at 4 the LRU thrashed and reloaded
# networks from disk on essentially every game.
_W_EVAL_CACHE_MAX = 9


def _load_eval_policy(path_str: str, device: str):
    """Load (and cache) an eval-side network from a weights file.

    Keyed on the path, which is safe because every file evaluate_policy writes
    is immutable for the life of the call that wrote it. Bounded at 4 because
    one eval needs at most two distinct networks (the policy under test and the
    frozen reference); the slack just absorbs back-to-back eval calls.
    """
    import torch as _torch
    global _W_EVAL_CACHE
    cached = _W_EVAL_CACHE.get(path_str)
    if cached is not None:
        _W_EVAL_CACHE.move_to_end(path_str)
        return cached
    try:
        sd = _torch.load(path_str, map_location="cpu")
    except (FileNotFoundError, OSError, RuntimeError, EOFError) as exc:
        raise RuntimeError(f"eval worker could not load weights {path_str}: {exc}") from exc
    pol = make_policy().to(device)
    pol.load_state_dict(sd)
    pol.eval()
    _W_EVAL_CACHE[path_str] = pol
    _W_EVAL_CACHE.move_to_end(path_str)
    while len(_W_EVAL_CACHE) > _W_EVAL_CACHE_MAX:
        _W_EVAL_CACHE.popitem(last=False)
    return pol


def _worker_init(card_defs: dict, device: str) -> None:
    """Pool initializer: runs once per worker process on Windows spawn."""
    import torch as _torch
    global _W_CARD_DEFS, _W_DEVICE, _W_SD_VERSION, _W_SD_CACHE, _W_SNAP_CACHE
    _W_CARD_DEFS  = card_defs
    _W_DEVICE     = device
    _W_SD_VERSION = None
    _W_SD_CACHE   = None
    _W_SNAP_CACHE = OrderedDict()
    _W_EVAL_CACHE.clear()
    # Prevent PyTorch from spawning multiple internal threads per worker.
    # With N workers each using 1 thread, total = N threads = one per core.
    _torch.set_num_threads(1)


def _worker_run_game(task: tuple) -> tuple:
    """Run one self-play game in a subprocess.

    N_TRAIN_PLAYERS slots use the current policy and collect PPO transitions.
    The remaining slots use per-slot opponent entries from opp_sds:
      - (dict, tag) : frozen historical BGPolicyNetwork snapshot → StaticAgent,
                      tagged e.g. "snapshot_u12" / "milestone_u50"
      - "heuristic" : HeuristicAgent (no network, leveling-focused)
      - "greedy"    : GreedyPlayAgent (no network, buys/plays everything,
                      only sells to make room for a higher-tier minion)
      - None   : promote to PPOAgent (warm-up fallback when pool is empty)

    Parameters (unpacked from *task*)
    ---------------------------------
    weights_ref  : (path: str, version: int)     — location of the current
                                                    policy.state_dict(), broadcast to
                                                    a shared file by the main process
                                                    instead of being embedded in the
                                                    task (see _train_parallel's
                                                    weights-broadcast comment). Loaded
                                                    via torch.load and cached
                                                    per-process, one entry, keyed by
                                                    version — see _W_SD_VERSION /
                                                    _W_SD_CACHE above.
    opp_sds      : List[tuple | str | None]      — one entry per opponent slot
    seed         : int | None                    — per-game RNG seed
    stats_weight : float                         — BattlegroundsGame.shape_stats_weight
                                                    for this game (see BOARD_SHAPE_STATS_WEIGHT)

    card_defs and device are read from the per-process globals set by
    _worker_init, so they are NOT re-pickled on every call.

    Returns
    -------
    (transitions, summary_dict)
      transitions  : List[Transition]   — only from training-agent slots
      summary_dict : {"placements": dict, "final_rewards": dict, "n_rounds": int,
                       "agent_labels": {player_id: label}} — label identifies which
                       agent occupied that seat ("train_current", "heuristic",
                       "greedy", "snapshot_uN", "milestone_uN") for win-rate tracking.
    """
    import random as _random
    import numpy as _np
    import torch as _torch

    # `task` arrives pre-pickled: the main process serializes it explicitly so
    # the cost is attributable to a phase instead of hiding inside the pool's
    # feeder thread (see _dispatch_one). Tolerate a raw tuple too, so any
    # caller that still passes an object keeps working.
    if isinstance(task, (bytes, bytearray)):
        task = pickle.loads(task)
    weights_ref, opp_sds, seed, stats_weight = task
    weights_path, weights_version = weights_ref
    card_defs = _W_CARD_DEFS
    device    = _W_DEVICE

    # Load the broadcast weights file, reusing the one-entry cache when this
    # task's version matches what's already loaded (see _W_SD_VERSION /
    # _W_SD_CACHE and the module-level comment above them).
    global _W_SD_VERSION, _W_SD_CACHE
    if _W_SD_VERSION != weights_version:
        try:
            _W_SD_CACHE = _torch.load(weights_path, map_location="cpu")
        except (FileNotFoundError, OSError, RuntimeError, EOFError) as exc:
            # A queued task (the queue_factor backlog in _train_parallel) can
            # sit long enough that its weights_ref version falls off the
            # main process's 3-version retention window before a worker gets
            # to it. Silently training on whatever happens to be on disk
            # would be far worse than crashing, so raise clearly -- the
            # caller's existing worker-error path rebuilds the pool and
            # re-dispatches the lost games.
            raise RuntimeError(
                f"Worker failed to load policy weights version {weights_version} "
                f"from {weights_path!r} — likely reclaimed by the main process's "
                f"weights-file retention window before this task was picked up. "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        _W_SD_VERSION = weights_version
    current_sd = _W_SD_CACHE

    if seed is not None:
        _random.seed(seed)
        _np.random.seed(seed)
        _torch.manual_seed(seed)

    tavern_pool = TavernPool(card_defs, seed=seed)
    matchmaker  = Matchmaker(n_players=N_PLAYERS, seed=seed)
    board_comp  = SymbolicBoardComputer(card_defs)
    firestone   = FirestoneClient(firestone_path=None, mock_mode=True)

    # Current policy — used by training agents, records transitions
    current_policy = make_policy().to(device)
    current_policy.load_state_dict(current_sd)

    ppo_config  = PPOConfig(device=device)
    ppo_trainer = PPOTrainer(current_policy, ppo_config)

    # Randomise which player slots are training agents each game so the
    # training agent sees the full range of table positions over time.
    # If all opp_sds entries are None (warm-up), promote all slots to training.
    all_none = all(e is None for e in opp_sds)
    if all_none:
        train_pids = set(range(N_PLAYERS))
    else:
        train_pids = set(_random.sample(range(N_PLAYERS), N_TRAIN_PLAYERS))

    opp_pids = [pid for pid in range(N_PLAYERS) if pid not in train_pids]

    agents: List[Any] = [None] * N_PLAYERS
    agent_labels: Dict[int, str] = {}
    for pid in train_pids:
        agents[pid] = PPOAgent(current_policy, ppo_trainer, player_id=pid, device=device, game_uid=seed)
        agent_labels[pid] = "train_current"

    # Build opponent agents. Snapshot entries are now (path, snapshot_id, tag)
    # REFERENCES rather than raw state_dicts (see SnapshotPool): the weights
    # live in one file the worker reads itself, so a task carries a path string
    # instead of ~13.89MB per snapshot slot. Deduplication is handled by the
    # process-wide _W_SNAP_CACHE keyed on snapshot_id, which also means two
    # slots drawing the SAME snapshot in one game share one loaded network.
    for slot_i, pid in enumerate(opp_pids):
        entry = opp_sds[slot_i]
        if entry == "heuristic":
            agents[pid] = HeuristicAgent(player_id=pid)
            agent_labels[pid] = "heuristic"
        elif entry == "greedy":
            agents[pid] = GreedyPlayAgent(player_id=pid)
            agent_labels[pid] = "greedy"
        elif entry is None:
            # Pool still empty for this slot — promote to training agent
            agents[pid] = PPOAgent(current_policy, ppo_trainer, player_id=pid, device=device, game_uid=seed)
            agent_labels[pid] = "train_current"
        else:
            snap_path, snap_id, tag = entry
            pol = _load_snapshot_policy(snap_path, snap_id, device)
            agents[pid] = StaticAgent(pol, player_id=pid, device=device)
            agent_labels[pid] = tag

    game = BattlegroundsGame(
        card_defs          = card_defs,
        agents             = agents,
        board_computer     = board_comp,
        firestone_client   = firestone,
        matchmaker         = matchmaker,
        tavern_pool        = tavern_pool,
        n_players          = N_PLAYERS,
        seed               = seed,
        batched            = True,
        shape_stats_weight = stats_weight,
    )
    result = game.run_game()

    # ── Combat diagnostics for the training seats ────────────────────────────
    # Placement is the objective, but it is a coarse, end-of-game number: a
    # policy can improve its board a lot and still place 5th because of one bad
    # matchup. Combat win rate and damage taken are the per-round signal that
    # moves first, so they show a change several hundred updates before mean
    # placement does. Aggregated here, in the worker, because result.round_history
    # is large and should not cross the process boundary.
    _train_pids = {pid for pid, lbl in agent_labels.items() if lbl == "train_current"}
    _cs = {"n": 0, "wins": 0, "ties": 0, "dmg_taken": 0.0, "win_prob": 0.0, "ghosts": 0}
    for _rs in result.round_history:
        for _c in _rs.get("combats", []):
            if _c.get("player_id") not in _train_pids:
                continue
            _cs["n"] += 1
            if _c.get("result") == "win":
                _cs["wins"] += 1
            elif _c.get("result") == "tie":
                _cs["ties"] += 1
            _cs["dmg_taken"] += float(_c.get("damage_taken", 0) or 0)
            _cs["win_prob"]  += float(_c.get("win_prob", 0.0) or 0.0)
            _cs["ghosts"]    += 1 if _c.get("is_ghost") else 0

    summary = {
        "placements":    result.placements,
        "final_rewards": result.final_rewards,
        "n_rounds":      result.n_rounds,
        "agent_labels":  agent_labels,
        "combat":        _cs,
    }
    return ppo_trainer.buffer.transitions, summary


def _worker_run_eval_game(task: tuple) -> tuple:
    """Run ONE evaluate_policy() game in a subprocess.

    Mirrors the body of evaluate_policy's sequential per-game loop exactly:
    one deterministic EvalAgent on the eval seat, GreedyPlayAgent/HeuristicAgent
    on the other seven. Collects no PPO transitions, builds no PPOTrainer, and
    touches no trainer state -- this is a read-only measurement, same as the
    sequential path.

    Parameters (unpacked from *task*)
    ---------------------------------
    game_idx  : int                  — 0-based index into the requested
                                        n_games sequence. Determines
                                        eval_pid = game_idx % N_PLAYERS (same
                                        seat-rotation rule as the sequential
                                        path) and is echoed back in the return
                                        value so the caller can aggregate by
                                        sorted game index rather than
                                        completion order.
    policy_sd : dict                 — policy.state_dict() snapshot (CPU
                                        tensors; see evaluate_policy)
    opponent  : {"greedy","heuristic"}
    game_seed : int | None           — per-game seed, already derived by the
                                        caller as (base_seed + game_idx) --
                                        NEVER derived from worker scheduling
                                        order, so results are identical
                                        regardless of n_workers.

    card_defs is read from the per-process global set by _worker_init.
    Device is always "cpu" for the eval pool (see evaluate_policy) --
    each CUDA context costs real VRAM and eval is latency- not
    throughput-bound, so N cheap CPU workers beat a handful of CUDA ones.

    Returns
    -------
    (game_idx, placement, n_rounds)
      placement : the eval seat's final placement (1=winner .. 8=last)
      n_rounds  : rounds the game lasted (cheap to report, not currently
                  aggregated by evaluate_policy but handy for diagnostics)
    """
    card_defs = _W_CARD_DEFS
    device    = _W_DEVICE

    game_idx, policy_ref, opponent, game_seed, ref_path = task
    eval_pid = game_idx % N_PLAYERS

    board_comp = SymbolicBoardComputer(card_defs)
    firestone  = FirestoneClient(firestone_path=None, mock_mode=True)

    # policy_ref is a PATH, not a state_dict. evaluate_policy used to pickle a
    # full ~13.89MB state_dict into every one of its n_games tasks (~1.8GB per
    # eval at n_games=128), all through the pool's single feeder thread; it now
    # writes the weights once and passes the filename. Cached per worker
    # process keyed on the path, since a worker handles several eval games and
    # the file is immutable for the life of one evaluate_policy call.
    policy = _load_eval_policy(policy_ref, device)

    tavern_pool = TavernPool(card_defs, seed=game_seed)
    matchmaker  = Matchmaker(n_players=N_PLAYERS, seed=game_seed)

    agents: List[Any] = [None] * N_PLAYERS
    seat_refs: Dict[int, str] = {}   # gauntlet only: seat -> reference path
    for pid in range(N_PLAYERS):
        if pid == eval_pid:
            # supports_batching = False forces BattlegroundsGame's sequential
            # per-player action path for every seat this game -- the ONLY
            # path that honours EvalAgent's deterministic=True argmax. Do
            # not swap this for a batching-capable agent. See EvalAgent's
            # docstring for the full explanation.
            agents[pid] = EvalAgent(policy, player_id=pid, device=device)
        elif opponent == "greedy":
            agents[pid] = GreedyPlayAgent(player_id=pid)
        elif opponent == "reference":
            # Seven copies of a FROZEN earlier self. Unlike greedy/heuristic
            # this opponent does not saturate: it stays a fixed target while
            # the policy keeps improving past the point where the scripted
            # baselines stop discriminating.
            agents[pid] = EvalStaticAgent(
                _load_eval_policy(ref_path, device), player_id=pid, device=device
            )
        elif opponent == "gauntlet":
            # Seven DIFFERENT frozen checkpoints spanning the run, one per
            # seat. A Battlegrounds lobby seats 8, so ONE game is already a
            # full 8-way comparison -- the game does the round-robin for
            # free. Comparing n agents pairwise would otherwise cost O(n^2)
            # matches; here a single game yields 28 pairwise outcomes, which
            # is what makes fitting ratings affordable at eval cadence.
            # The seat->reference assignment is rotated by game_idx so no
            # reference is permanently advantaged by its seat position.
            _others = [p for p in range(N_PLAYERS) if p != eval_pid]
            _ref    = ref_path[(_others.index(pid) + game_idx) % len(ref_path)]
            agents[pid] = EvalStaticAgent(
                _load_eval_policy(_ref, device), player_id=pid, device=device
            )
            seat_refs[pid] = _ref
        else:
            agents[pid] = HeuristicAgent(player_id=pid)

    game = BattlegroundsGame(
        card_defs         = card_defs,
        agents            = agents,
        board_computer    = board_comp,
        firestone_client  = firestone,
        matchmaker        = matchmaker,
        tavern_pool       = tavern_pool,
        n_players         = N_PLAYERS,
        seed              = game_seed,
        batched           = True,  # moot: every seat's agent sets
                                    # supports_batching=False, forcing the
                                    # sequential path regardless.
    )
    result = game.run_game()
    # 4th element is None for every non-gauntlet opponent, so the cheap eval
    # paths pay nothing for it. For the gauntlet it carries exactly what
    # rating fitting needs: every seat's placement, and which reference sat
    # in each seat.
    extra = None
    if opponent == "gauntlet":
        extra = {
            "eval_pid":   eval_pid,
            "placements": {p: result.placements[p] for p in range(N_PLAYERS)},
            "seat_refs":  seat_refs,
        }
    return game_idx, result.placements[eval_pid], result.n_rounds, extra


# -------------------------------------------------------------------------
# Logging helpers
# -------------------------------------------------------------------------

def _append_agent_stats(
    path: Path,
    game_idx: int,
    total_steps: int,
    summary: dict,
) -> None:
    """Append one JSONL row per player for a finished game.

    Written continuously to *path* (default data/agent_stats.jsonl) so that
    per-agent-type win-rate can be tracked across training sessions/kernel
    restarts, not just within one in-memory run. Row fields:
      game        : game index local to the calling training run (not
                    globally unique across restarts — do not use as an x-axis)
      total_steps : cumulative PPO transitions collected so far (monotonic
                    across checkpoint resumes — the right x-axis for "over time")
      timestamp   : unix time the row was written
      pid         : player_id (0-7)
      label       : agent identity ("train_current", "heuristic", "greedy",
                    "snapshot_uN", "milestone_uN")
      placement   : final placement (1=winner .. 8=last)
      reward      : final accumulated reward for that player
    """
    labels     = summary.get("agent_labels", {})
    placements = summary["placements"]
    rewards    = summary["final_rewards"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for pid, placement in placements.items():
            fh.write(json.dumps({
                "game":        game_idx,
                "total_steps": total_steps,
                "timestamp":   time.time(),
                "pid":         pid,
                "label":       labels.get(pid, "unknown"),
                "placement":   placement,
                "reward":      rewards.get(pid, 0.0),
            }) + "\n")


def log_game_stats(game_idx: int, result: GameResult, elapsed: float) -> None:
    """Print a one-line summary for the finished game."""
    mean_reward = np.mean(list(result.final_rewards.values()))
    winner_id   = min(result.placements, key=result.placements.get)
    logger.info(
        "Game %4d | rounds=%2d | winner=P%d | mean_reward=%+.3f | %.1fs",
        game_idx, result.n_rounds, winner_id, mean_reward, elapsed,
    )


def log_update_metrics(update_idx: int, metrics: dict) -> None:
    """Print a one-line summary after a PPO update."""
    logger.info(
        "PPO update #%d | policy_loss=%.4f | value_loss=%.4f | entropy=%.4f | total=%.4f",
        update_idx,
        metrics.get("policy_loss", 0.0),
        metrics.get("value_loss", 0.0),
        metrics.get("entropy", 0.0),
        metrics.get("total_loss", 0.0),
    )


# -------------------------------------------------------------------------
# Parallel training loop
# -------------------------------------------------------------------------

def _train_parallel(
    n_games: int,
    policy: BGPolicyNetwork,
    ppo_trainer: PPOTrainer,
    card_defs: dict,
    *,
    n_workers: int = 1,
    update_interval: int = 10,
    checkpoint_interval: int = 100,
    checkpoint_path: Optional[str] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
    on_batch: Optional[Any] = None,
    on_update: Optional[Any] = None,
    batch_timeout: int = 300,
    queue_factor: float = 2.0,
    stats_path: Optional[str] = "data/agent_stats.jsonl",
    opponent_mix_fn: Optional[Callable[[], Tuple[int, int]]] = None,
) -> None:
    """Run self-play games in parallel using ProcessPoolExecutor.

    Games are dispatched with a ROLLING WINDOW, sized to an IN-FLIGHT TARGET
    of ceil(n_workers * queue_factor) rather than exactly n_workers: that
    many games are kept submitted to the pool (running OR queued inside it)
    at all times, and the instant one finishes the window is topped back up
    immediately (see the comment block at the top of the dispatch loop below
    for why this replaced the old pool.map()-per-cohort approach, and the
    comment further down for why the window is now overshot beyond
    n_workers rather than sized exactly to it). Each worker reads the
    current policy weights from a versioned file broadcast onto disk (see
    the weights-broadcast comment further down) rather than receiving them
    embedded in its task, runs one game, and returns its collected
    transitions. The main process merges all transitions into
    ppo_trainer.buffer and runs PPO updates at the normal interval.

    Parameters
    ----------
    n_games            : total games to play
    policy             : the policy network being trained
    ppo_trainer        : PPOTrainer instance
    card_defs          : card definitions dict
    n_workers          : number of parallel worker processes (max games in
                         flight at once)
    update_interval    : trigger a PPO update every this many games
    checkpoint_interval: save a checkpoint every this many games
    checkpoint_path    : path for automatic checkpoint saves (None = skip)
    seed               : base RNG seed (None = non-deterministic)
    device             : torch device string for the main process
    on_batch(game_idx, summaries, transitions, elapsed)
                       : optional callback fired every time n_workers games
                         have completed, plus once more at the end for any
                         final partial group. Because dispatch is now a
                         rolling window rather than a lock-step cohort, games
                         are grouped in COMPLETION order, not dispatch order
                         -- a fast game dispatched later can finish before a
                         slow game dispatched earlier, so group membership no
                         longer lines up with dispatch batches the way it did
                         under pool.map.
                         *game_idx* is the total games completed so far.
                         *summaries* is a list of per-game summary dicts.
                         *transitions* is a list of per-game Transition lists
                         (already added to the PPO buffer).
                         *elapsed* is the wall-clock time in seconds since the
                         previous on_batch call (i.e. how long this group of
                         completions took).
    on_update(metrics, update_count)
                       : optional callback fired after every PPO update.
    batch_timeout      : per-game in-flight timeout in seconds (default 300)
                         -- a game still running this long after being
                         dispatched is treated as hung; see the stall
                         detection comment below. With queue_factor > 1 the
                         age check is scaled by queue_factor before being
                         compared against this value, since a queued (not
                         yet started) game's age includes queue wait time,
                         not just run time -- see that comment for why.
    queue_factor       : how large a backlog to keep submitted to the pool,
                         as a multiple of n_workers (default 2.0). The
                         in-flight target is ceil(n_workers * queue_factor)
                         games submitted at once (running or merely queued
                         inside the pool), instead of exactly n_workers.
                         queue_factor=1.0 reproduces the exact old
                         exactly-n_workers-in-flight behavior. See the
                         dispatch-loop comment below for why a backlog is
                         needed at all (short version: it keeps every worker
                         fed straight through a blocking ppo_trainer.update()
                         call, which a window sized to exactly n_workers
                         cannot do).
    stats_path         : JSONL file appended with per-player agent-identity /
                         placement rows for every finished game (used to track
                         which agent types — training policy, heuristic, greedy,
                         historical snapshots — win most often over time).
                         Pass None to disable.
    opponent_mix_fn    : optional zero-argument callable returning
                         (n_heuristic, n_greedy) — the number of
                         HeuristicAgent / GreedyPlayAgent opponent slots to
                         use for the NEXT dispatched game. Consulted inside
                         _make_task at task-creation time, so a change takes
                         effect on the next game submitted to the pool, not
                         retroactively on games already in flight. The
                         remaining opponent slots (N_PLAYERS - N_TRAIN_PLAYERS
                         - n_heuristic - n_greedy) are filled from
                         snapshot_pool, exactly as when this is None. Default
                         None reproduces today's exact fixed behavior
                         (N_HEURISTIC_SLOTS, N_GREEDY_SLOTS) with zero
                         per-task overhead. Callers own their own hysteresis /
                         hold-out logic (see run_fresh_training.py's
                         adaptive-mix trigger) — this function only ever reads
                         whatever (n_heuristic, n_greedy) the callable returns
                         at each task-creation instant.
    """
    import math
    import multiprocessing
    import tempfile
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    mp_context = multiprocessing.get_context("spawn")

    # CONTINUE the global update count across a resume rather than restarting
    # at 0. Everything downstream is indexed by this number -- eval_updates,
    # opponent_mix_updates, gauntlet reference filenames (ref_u<N>.pt, whose
    # <N> the Elo anchor parses to find the OLDEST reference), and the
    # SNAPSHOT_EVERY/MILESTONE_EVERY cadences. Restarting at 0 on a resumed
    # run appended eval points at 50,100,... AFTER existing entries ending at
    # ~2350, which silently corrupts every history series' x-axis and makes
    # the charts read as jumping backwards in time.
    update_count = int(getattr(ppo_trainer, "update_count", 0) or 0)
    game_idx     = 0   # games RESOLVED so far: completed OR lost to error/timeout
    dispatched   = 0   # games SUBMITTED so far

    # IN-FLIGHT TARGET: how many games we keep submitted to the pool (running
    # OR merely queued inside it) at once. queue_factor==1.0 collapses this
    # to exactly n_workers, reproducing the old window size precisely. The
    # min(..., n_games) clamp handles short runs where there aren't even
    # n_workers*queue_factor games to dispatch in total; the steady-state
    # clamp against games remaining to dispatch is handled naturally by the
    # `dispatched < n_games` guards everywhere this target is used below, so
    # it doesn't need to be re-clamped dynamically here.
    in_flight_target = min(math.ceil(n_workers * queue_factor), n_games)

    snapshot_pool  = SnapshotPool(capacity=20)
    stats_file     = Path(stats_path) if stats_path else None

    # Opponent slot composition per game:
    #   n_heuristic slots always use HeuristicAgent (leveling anchor)
    #   n_greedy slots always use GreedyPlayAgent (naive buy/play anchor)
    #   remaining slots sample independently from the snapshot pool
    #
    # (n_heuristic, n_greedy) default to (N_HEURISTIC_SLOTS, N_GREEDY_SLOTS)
    # and are re-read from opponent_mix_fn() on every _make_task() call below
    # when one is supplied -- NOT computed once here. This used to be a
    # single n_policy_slots computed once outside the dispatch loop, which
    # made it impossible for a caller to change the mix mid-run (see
    # opponent_mix_fn's docstring above). N_OPP_SLOTS itself stays fixed: it
    # is just "how many non-training seats a game has" (6), unaffected by
    # which agent types fill them.
    N_OPP_SLOTS = N_PLAYERS - N_TRAIN_PLAYERS   # 6

    def _make_pool() -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp_context,
            initializer=_worker_init,
            initargs=(card_defs, device),
        )

    pool = _make_pool()

    # -----------------------------------------------------------------
    # WEIGHTS BROADCAST: write the policy snapshot to a shared file instead
    # of embedding it in every task.
    #
    # Every task used to carry a full ~13.89MB policy state_dict. With
    # ProcessPoolExecutor that means pickling it in the MAIN process's
    # single feeder thread for EVERY dispatched game -- at
    # UPDATE_INTERVAL == n_workers that's ~n_workers unicast copies (~1.26GB
    # at n_workers=90) per PPO update, all serialized through one largely
    # GIL-bound thread (and unpickled result-side the same way). A
    # per-worker CACHE would not fix this: UPDATE_INTERVAL == N_WORKERS
    # means each worker only handles ~1 game per weight version, so the hit
    # rate is near zero regardless of cache size. The actual fix is to stop
    # unicasting altogether -- write the weights ONCE to a file and let
    # every worker read that file in parallel, each in its own process,
    # entirely off the main process's serial thread.
    #
    # /dev/shm is preferred when available: it's a RAM-backed tmpfs, so
    # writing/reading through it costs no real disk I/O, just a memcpy.
    # -----------------------------------------------------------------
    _shm_dir = Path("/dev/shm")
    if _shm_dir.exists() and os.access(_shm_dir, os.W_OK):
        weights_dir = _shm_dir
    else:
        weights_dir = Path(tempfile.gettempdir())
    _weights_pid = os.getpid()   # so concurrent runs never collide on filenames

    def _weights_path(version: int) -> Path:
        return weights_dir / f"bg_weights_{_weights_pid}_v{version}.pt"

    sd_version = 0                 # monotonic, incremented on every reclone
    written_versions: set = set()  # versions with a live file on disk this run

    def _write_weights(state_dict: dict) -> tuple:
        """Save *state_dict* to a fresh versioned file, atomically, and
        return the (path_str, version) weights_ref to embed in tasks.

        Atomic write: save to a `.tmp` sibling then os.replace() onto the
        real name -- os.replace is a single filesystem rename, so a worker
        can never observe (and torch.load) a partially written file.
        """
        nonlocal sd_version
        sd_version += 1
        version = sd_version
        path     = _weights_path(version)
        tmp_path = path.with_name(path.name + ".tmp")
        torch.save(state_dict, str(tmp_path))
        os.replace(str(tmp_path), str(path))
        written_versions.add(version)

        # Retention: queued tasks (the queue_factor backlog described in the
        # dispatch-loop comment below) may have been built against an OLDER
        # version and are still waiting their turn inside the pool, so the
        # previous file can't be deleted the instant a newer one lands --
        # some in-flight task can legitimately still reference it. Keep the
        # 3 most recent files and reclaim anything older; versions are
        # sequential integers, so "older than the 3 most recent" is simply
        # version - 3.
        stale_version = version - 3
        if stale_version in written_versions:
            try:
                _weights_path(stale_version).unlink()
                written_versions.discard(stale_version)
            except OSError:
                pass  # leave it tracked; the run-end cleanup in finally: retries

        return str(path), version

    # Snapshot weights once; only re-clone (and rewrite the broadcast file)
    # after each PPO update. Under the old pool.map cohort loop this reclone
    # was checked once per batch; with rolling dispatch there is no "start
    # of batch" moment, so it's checked at each TASK-CREATION time instead
    # (inside _make_task below) -- a dispatched game simply uses whatever
    # snapshot was current at the instant it was submitted, which is the
    # natural per-task analogue of the old per-cohort check and is fine and
    # intended.
    sd = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    weights_ref = _write_weights(sd)
    sd_stale = False

    dispatch_counter = 0  # per-dispatch seed offset, monotonic across the whole run
    in_flight: Dict[Any, float] = {}   # Future -> submit timestamp (for stall detection)

    # Per-phase wall-clock accumulators (see the phase-report block inside
    # the on_batch flush below). Aggregate CPU-idle numbers told us the box
    # was running ~45% idle at 90 workers but not WHICH phase was
    # serializing it; these buckets let a run ATTRIBUTE the idle time
    # instead of inferring it. The single most useful signal from them is
    # whether t_dispatch + t_merge (main-process serial work) is large
    # relative to t_wait -- that's the shape a main-process bottleneck takes.
    t_wait     = 0.0   # blocked inside concurrent.futures.wait(...)
    t_dispatch = 0.0   # inside _dispatch_one (task construction + pool.submit)
    t_merge    = 0.0   # per completed game: fut.result(), buffer merge, stats, log line
    t_update   = 0.0   # inside ppo_trainer.update() and the snapshot/logging after it
    t_serialize = 0.0  # inside pickle.dumps(task) -- a SUBSET of t_dispatch
    ser_bytes   = 0    # total task-payload bytes pickled
    ser_count   = 0    # tasks pickled (so mean bytes/task is reportable)

    def _make_task() -> tuple:
        nonlocal sd, sd_stale, weights_ref, dispatch_counter
        if sd_stale:
            sd = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            weights_ref = _write_weights(sd)
            sd_stale = False
        # Re-read the opponent mix at TASK-CREATION time (not once outside
        # the loop) so a caller's opponent_mix_fn can change the mix mid-run
        # and have it take effect on the very next dispatched game -- see
        # opponent_mix_fn's docstring above. Default None reproduces the old
        # fixed (N_HEURISTIC_SLOTS, N_GREEDY_SLOTS) mix exactly.
        if opponent_mix_fn is not None:
            n_heuristic, n_greedy = opponent_mix_fn()
        else:
            n_heuristic, n_greedy = N_HEURISTIC_SLOTS, N_GREEDY_SLOTS
        # Defensive clamp: never let a misbehaving opponent_mix_fn push the
        # snapshot-slot count negative or the fixed-slot count past the
        # number of opponent seats a game actually has (N_OPP_SLOTS) -- either
        # would desync opp_sds's length from opp_pids in _worker_run_game.
        n_heuristic = max(0, min(n_heuristic, N_OPP_SLOTS))
        n_greedy    = max(0, min(n_greedy, N_OPP_SLOTS - n_heuristic))
        n_policy_slots = max(0, N_OPP_SLOTS - n_heuristic - n_greedy)
        policy_sds = snapshot_pool.sample_n(n_policy_slots)
        opp_sds    = (policy_sds
                      + ["heuristic"] * n_heuristic
                      + ["greedy"] * n_greedy)
        seed_value = (seed + ppo_trainer.total_steps + dispatch_counter) if seed is not None else None
        dispatch_counter += 1
        # weights_ref replaces the raw state_dict here -- (path, version) is
        # a couple hundred bytes to pickle instead of ~13.89MB. opp_sds is
        # likewise references now, not raw snapshot state_dicts.
        return (weights_ref, opp_sds, seed_value, BOARD_SHAPE_STATS_WEIGHT)

    def _dispatch_one() -> None:
        nonlocal dispatched, t_dispatch, t_serialize, ser_bytes, ser_count
        _t0 = time.perf_counter()
        task = _make_task()
        # WHY the task is pickled HERE instead of being handed to submit() as
        # an object: t_dispatch used to wrap only pool.submit(), which returns
        # the instant the work item is appended to the pending-work queue. The
        # actual pickling happens LATER, in ProcessPoolExecutor's single
        # GIL-bound queue-feeder thread, so its cost showed up as workers
        # idling inside t_wait and was invisible to t_dispatch BY
        # CONSTRUCTION. A previous session read `dispatch=0.0% merge=0.0-0.3%`
        # off that instrumentation and concluded main-process serialization
        # was "no longer measurable" -- a false negative, not a finding, and
        # at the time each task still carried ~27.8MB of snapshot state_dicts.
        # Pickling explicitly moves the cost into a phase that can be
        # attributed, and yields bytes/task for free.
        _t1 = time.perf_counter()
        payload = pickle.dumps(task, protocol=pickle.HIGHEST_PROTOCOL)
        t_serialize += time.perf_counter() - _t1
        ser_bytes += len(payload)
        ser_count += 1
        fut = pool.submit(_worker_run_game, payload)
        in_flight[fut] = time.time()
        dispatched += 1
        t_dispatch += time.perf_counter() - _t0

    try:
        # -----------------------------------------------------------------
        # WHY rolling dispatch instead of pool.map:
        #
        # pool.map(..., timeout=...) dispatches exactly n_workers games as a
        # single cohort and blocks until ALL of them return -- every worker
        # that finishes early just sits idle until the slowest game in the
        # cohort (the straggler) completes. Game length varies a lot round
        # to round (15-25 rounds, std ~3.4), so a cohort's wall-clock is set
        # by its straggler, not its average. Measured on the training host
        # (32 dedicated cores) this produced self-play throughput that was
        # essentially FLAT as worker count rose -- the signature of a
        # synchronization barrier eating the parallelism gains:
        #
        #     N_WORKERS=16 -> 1.94 games/sec
        #     N_WORKERS=20 -> 1.96 games/sec
        #     N_WORKERS=26 -> 2.01 games/sec
        #     N_WORKERS=30 -> 2.19 games/sec
        #
        # with only ~72-80% CPU utilization during self-play (vmstat) --
        # i.e. 20-28% of the box sat idle waiting on stragglers every single
        # cohort. The fix: keep up to n_workers games in flight AT ALL TIMES
        # and, the instant any one finishes, immediately submit the next
        # queued game into that freed slot -- no worker ever waits on
        # another worker.
        # -----------------------------------------------------------------
        # WHY OVERSHOOT the window past n_workers (queue_factor > 1):
        #
        # A window sized to exactly n_workers keeps every worker fed ONLY
        # while this main loop is actively polling and resubmitting. But
        # ppo_trainer.update() below is a long BLOCKING call in this same
        # process (~90 games x ~250 transitions per update) -- while it
        # runs, nothing harvests completed futures and nothing gets
        # resubmitted, so every worker finishes its current game and then
        # sits idle waiting for the main process to come back. Measured on
        # the 48-physical-core host: workers busy ~44 cores, main process
        # ~3.3 cores -> only ~47/92 cores in use (46% idle), with the GPU
        # busy (i.e. inside an update) in ~23% of samples -- and batch
        # wall-times for the same 90 games swung 3.2s / 11.5s / 15.3s, the
        # long ones being exactly the groups that straddled an update.
        #
        # ProcessPoolExecutor queues submitted-but-not-yet-running tasks
        # internally and hands each worker its next queued task the instant
        # it finishes one, with NO involvement from this (stalled) main
        # process. So instead of keeping exactly n_workers games in flight,
        # we keep `in_flight_target` = ceil(n_workers * queue_factor) games
        # submitted at once -- the extra (queue_factor - 1) * n_workers
        # games sit queued inside the pool as a backlog that keeps workers
        # busy straight through a main-process stall. queue_factor=1.0
        # disables the backlog and reproduces the old exactly-n_workers
        # window exactly.
        #
        # TRADEOFF -- weight staleness: each task is built by _make_task()
        # with whatever policy snapshot `sd` (and its broadcast weights_ref)
        # is current AT SUBMIT time (see the sd/sd_stale comment above), not
        # at run time. A game sitting in the backlog queue is therefore
        # dispatched with a weights_ref that can go stale while it waits --
        # this is exactly why _write_weights above keeps the last 3 versions
        # on disk instead of deleting the previous one immediately: a queued
        # task's weights_ref can still point at a version that is no longer
        # the newest by the time a worker actually reads it. With
        # queue_factor=2.0 and UPDATE_INTERVAL == n_workers (one PPO update
        # per ~one window's worth of completions),
        # the backlog is about one window deep, so a queued game is at most
        # roughly one PPO update stale by the time it actually runs. That's
        # an ordinary amount of off-policyness for PPO -- the importance
        # ratio in the loss already accounts for the behavior policy having
        # drifted from the current one -- but it IS a real (small) increase
        # in staleness traded for keeping the box busy through update()
        # stalls. Going much higher than queue_factor=2.0 would only make
        # this worse without further throughput benefit: the backlog only
        # needs to be deep enough to cover one update's wall-clock duration,
        # not more.
        # -----------------------------------------------------------------
        for _ in range(in_flight_target):
            _dispatch_one()

        last_batch_t       = time.time()
        group_summaries:    List[dict] = []
        group_transitions:  List[list] = []
        # Discard priming-loop dispatch cost from the first phase report --
        # the report's job is to describe steady-state per-interval
        # behavior, not the one-time startup dispatch above.
        t_wait = t_dispatch = t_merge = t_update = 0.0

        while game_idx < n_games or in_flight:
            if not in_flight:
                # Nothing in flight and nothing left to resolve/dispatch --
                # only reachable if n_games == 0 (loop condition below would
                # otherwise keep it alive). Defensive break to avoid spinning.
                break

            _t0 = time.perf_counter()
            done, _ = wait(list(in_flight.keys()), timeout=1.0,
                            return_when=FIRST_COMPLETED)
            t_wait += time.perf_counter() - _t0

            broken = False
            for fut in done:
                in_flight.pop(fut, None)
                _t0 = time.perf_counter()
                try:
                    transitions, summary = fut.result()
                except Exception as exc:
                    t_merge += time.perf_counter() - _t0
                    # A crashed/killed worker generally takes the whole pool
                    # down with it (BrokenProcessPool) -- every OTHER future
                    # still tracked as in-flight (whether still running or
                    # merely not yet iterated to in this `done` loop) is
                    # collateral damage from the same broken pool, so treat
                    # all of it as lost rather than trying to salvage
                    # individual results out of a pool we're about to
                    # discard. Mirrors the old code's behavior of discarding
                    # its whole cohort on any worker error.
                    logger.warning("Worker error (%s: %s) — rebuilding pool", type(exc).__name__, exc)
                    # `lost` is the actual size of the in-flight dict (plus
                    # the one future that just failed and was already popped
                    # above) -- with queue_factor > 1 this can be up to
                    # in_flight_target, not just n_workers. Re-prime the
                    # fresh pool back up to in_flight_target (not just to
                    # `lost` games), same as the initial priming loop.
                    lost = 1 + len(in_flight)
                    in_flight.clear()
                    pool.shutdown(wait=False)
                    pool = _make_pool()
                    game_idx += lost
                    while dispatched < n_games and len(in_flight) < in_flight_target:
                        _dispatch_one()
                    broken = True
                    break
                t_merge += time.perf_counter() - _t0

                # A slot just freed up -- top the window back up to
                # in_flight_target before doing any of the bookkeeping below
                # (buffer merge, stats file I/O, possible PPO
                # update/checkpoint), so workers never sit idle waiting on
                # the main process. With queue_factor==1.0 this pops exactly
                # once (in_flight_target == n_workers, and one slot just
                # freed), reproducing the old single-dispatch behavior. With
                # queue_factor > 1 it's usually also one dispatch per
                # completion in steady state -- but a `while`, not an `if`,
                # because after a stall/rebuild the window can be more than
                # one slot short and needs multiple submissions to refill.
                while dispatched < n_games and len(in_flight) < in_flight_target:
                    _dispatch_one()

                prev_game_idx = game_idx
                game_idx += 1
                g = game_idx

                _t0 = time.perf_counter()
                for t in transitions:
                    ppo_trainer.buffer.add(t)
                    ppo_trainer.total_steps += 1
                group_summaries.append(summary)
                group_transitions.append(transitions)

                if stats_file is not None:
                    _append_agent_stats(stats_file, g, ppo_trainer.total_steps, summary)

                winner_id   = min(summary["placements"], key=summary["placements"].get)
                mean_reward = float(np.mean(list(summary["final_rewards"].values())))
                logger.info(
                    "Game %4d | rounds=%2d | winner=P%d | mean_reward=%+.3f",
                    g, summary["n_rounds"], winner_id, mean_reward,
                )
                t_merge += time.perf_counter() - _t0

                # PPO update if we crossed an update_interval boundary.
                # Evaluated once per completed game now (rather than once
                # per cohort under the old pool.map loop) -- the boundary
                # arithmetic is unchanged, just checked at finer grain.
                if (game_idx // update_interval) > (prev_game_idx // update_interval):
                    if len(ppo_trainer.buffer) > 0:
                        _t0 = time.perf_counter()
                        metrics = ppo_trainer.update(last_value=0.0)
                        update_count += 1
                        sd_stale = True   # weights changed — reclone (and rewrite the broadcast file) before next dispatch
                        if update_count % SNAPSHOT_EVERY == 0:
                            is_milestone = (update_count % MILESTONE_EVERY == 0)
                            snapshot_pool.add(policy.state_dict(), is_milestone=is_milestone,
                                              update_count=update_count)
                            if is_milestone:
                                logger.info(
                                    "Milestone snapshot added (update=%d, milestones=%d)",
                                    update_count, len(snapshot_pool._milestones),
                                )
                            else:
                                logger.info("Rolling snapshot added (pool size=%d)", len(snapshot_pool))
                        log_update_metrics(update_count, metrics)

                        nan_count = sum(
                            int(torch.isnan(p).any().item())
                            for p in policy.parameters()
                        )
                        logger.info("NaN params after update: %d", nan_count)

                        if on_update is not None:
                            on_update(metrics, update_count)
                        t_update += time.perf_counter() - _t0

                # Checkpoint if we crossed a checkpoint_interval boundary
                if checkpoint_path:
                    if (game_idx // checkpoint_interval) > (prev_game_idx // checkpoint_interval):
                        ppo_trainer.save_checkpoint(
                            checkpoint_path, extra={"game": game_idx}
                        )
                        logger.info("Checkpoint saved at game %d → %s", game_idx, checkpoint_path)

                # Fire on_batch every n_workers completed games, in
                # completion order (see docstring). The final partial group
                # (if any) is flushed once the loop below exits.
                if len(group_summaries) >= n_workers:
                    flush_t = time.time()
                    wall = flush_t - last_batch_t
                    if on_batch is not None:
                        on_batch(game_idx, group_summaries, group_transitions, wall)
                    # Per-phase report: aggregate CPU-idle numbers say the
                    # box is idle but not WHICH phase is serializing it, so
                    # this line attributes the interval's wall-clock to the
                    # four buckets above. Watch t_dispatch + t_merge
                    # (main-process serial work) relative to t_wait -- large
                    # dispatch/merge time relative to wait time is the
                    # signature of a main-process bottleneck.
                    if wall > 0:
                        logger.info(
                            "phase: wait=%.1f%% dispatch=%.1f%% (serialize=%.1f%%) merge=%.1f%% "
                            "update=%.1f%% (%.1fs wall) | %.0f B/task, %.1f MB total",
                            100.0 * t_wait / wall, 100.0 * t_dispatch / wall,
                            100.0 * t_serialize / wall,
                            100.0 * t_merge / wall, 100.0 * t_update / wall, wall,
                            (ser_bytes / ser_count) if ser_count else 0.0,
                            ser_bytes / 1e6,
                        )
                    last_batch_t = flush_t
                    group_summaries   = []
                    group_transitions = []
                    t_wait = t_dispatch = t_merge = t_update = 0.0

            if broken:
                continue

            # ---------------------------------------------------------
            # Stall detection.
            #
            # The old pool.map(timeout=batch_timeout) failed its whole
            # cohort the instant any single game in it hung. A naive
            # `wait(..., return_when=FIRST_COMPLETED)` rolling loop has no
            # equivalent: as long as SOME other worker keeps finishing, the
            # loop keeps making progress and would never notice one stuck
            # game -- that hung slot would silently leak out of the window
            # forever, permanently shrinking effective parallelism by one
            # worker per hang, with no warning ever logged. So we track a
            # submit timestamp per future and explicitly age-check every
            # future still in flight on every poll, independent of whether
            # anything else completed this iteration.
            #
            # With queue_factor > 1, a future's tracked "submit" timestamp
            # can now include real QUEUE wait time, not just run time: a
            # backlog of up to (queue_factor - 1) * n_workers games can sit
            # queued inside the pool behind whatever's currently running
            # before a worker ever picks them up. Age-checking that queued
            # time against a threshold sized for RUN time alone would let a
            # perfectly healthy queued game spuriously trip the timeout. So
            # the threshold is scaled by queue_factor: a task can reasonably
            # wait up to about (queue_factor - 1) game-durations queued
            # before it even starts, on top of batch_timeout to run. This is
            # deliberately conservative -- the check exists to catch a
            # worker that's permanently wedged, not to enforce a tight SLA,
            # so erring toward a larger allowance here just delays detecting
            # a real hang slightly; it doesn't mask one.
            # ---------------------------------------------------------
            now = time.time()
            effective_timeout = batch_timeout * queue_factor
            stale = [f for f, t0 in in_flight.items() if now - t0 > effective_timeout]
            if stale:
                logger.warning(
                    "%d game(s) timed out after %.0fs — rebuilding pool",
                    len(stale), effective_timeout,
                )
                # `lost` is the actual size of the in-flight dict (may be up
                # to in_flight_target under a backlog, not just n_workers).
                lost = len(in_flight)
                in_flight.clear()
                pool.shutdown(wait=False)
                pool = _make_pool()
                game_idx += lost
                # Re-prime the fresh pool back up to in_flight_target, same
                # as the initial priming loop and the error-rebuild path.
                while dispatched < n_games and len(in_flight) < in_flight_target:
                    _dispatch_one()

        # Flush any leftover partial group -- reached whether the last
        # group completed exactly on an n_workers boundary (nothing left to
        # flush) or the run ended mid-group (successfully or via a lost
        # game pushing game_idx to n_games).
        if group_summaries:
            wall = time.time() - last_batch_t
            if on_batch is not None:
                on_batch(game_idx, group_summaries, group_transitions, wall)
            if wall > 0:
                logger.info(
                    "phase: wait=%.1f%% dispatch=%.1f%% (serialize=%.1f%%) merge=%.1f%% "
                    "update=%.1f%% (%.1fs wall) | %.0f B/task, %.1f MB total",
                    100.0 * t_wait / wall, 100.0 * t_dispatch / wall,
                    100.0 * t_serialize / wall,
                    100.0 * t_merge / wall, 100.0 * t_update / wall, wall,
                    (ser_bytes / ser_count) if ser_count else 0.0,
                    ser_bytes / 1e6,
                )
    finally:
        pool.shutdown(wait=True)
        # Unlink every weights file this run created. A leftover file in
        # /dev/shm (or the tempdir fallback) must never crash a training
        # run, so failures here are swallowed -- retention above already
        # keeps this set small (at most ~3-4 entries at any time), this is
        # just final teardown once no worker can possibly need them anymore.
        for _v in list(written_versions):
            try:
                _weights_path(_v).unlink()
            except OSError:
                pass
        # Same teardown for the snapshot pool's own files, which live in the
        # same directory under a different prefix. Refcounted eviction already
        # reclaims most of them during the run; this catches whatever is still
        # referenced at exit. Without it a run leaks up to
        # (capacity + MILESTONE_CAPACITY) x ~13.89MB of RAM-backed tmpfs.
        snapshot_pool.cleanup()


# -------------------------------------------------------------------------
# Honest evaluation: fixed scripted opponent, deterministic policy
# -------------------------------------------------------------------------

def _evaluate_policy_sequential(
    policy: BGPolicyNetwork,
    card_defs: Dict[str, dict],
    n_games: int,
    opponent: str,
    device: str,
    seed: Optional[int],
    ref_path: Optional[str] = None,
) -> List[int]:
    """Single-process implementation of the evaluate_policy game loop.

    This is the original (pre-parallel) evaluate_policy body, extracted so it
    can serve double duty: the n_workers<=1 path, and the fallback used by
    evaluate_policy when parallel pool construction fails (see
    _evaluate_policy_parallel / evaluate_policy).

    Returns the raw list of eval-seat placements, one per game, in game-index
    order (index g -> eval_pid = g % N_PLAYERS, game_seed = seed + g).
    """
    # Stateless, safe to share across all n_games below (see build_components /
    # _worker_run_game — SymbolicBoardComputer only ever reads card_defs, and
    # mock_mode=True FirestoneClient holds no per-game state either).
    board_comp = SymbolicBoardComputer(card_defs)
    firestone  = FirestoneClient(firestone_path=None, mock_mode=True)

    placements: List[int] = []
    extras: List[dict] = []          # gauntlet only; see the parallel path
    for g in range(n_games):
        game_seed = (seed + g) if seed is not None else None
        # Rotate which seat the policy occupies so no single table
        # position is systematically favoured/disfavoured over n_games.
        eval_pid = g % N_PLAYERS

        # Fresh per game — both are stateful (draw history / pairing
        # history), exactly like _worker_run_game builds a fresh pair
        # for every game rather than reusing one across a batch.
        tavern_pool = TavernPool(card_defs, seed=game_seed)
        matchmaker  = Matchmaker(n_players=N_PLAYERS, seed=game_seed)

        agents: List[Any] = [None] * N_PLAYERS
        seat_refs: Dict[int, str] = {}
        for pid in range(N_PLAYERS):
            if pid == eval_pid:
                agents[pid] = EvalAgent(policy, player_id=pid, device=device)
            elif opponent == "greedy":
                agents[pid] = GreedyPlayAgent(player_id=pid)
            elif opponent == "reference":
                # Must match the parallel path's seating exactly -- this is
                # the fallback when the pool fails, and a fallback that
                # measures something different is worse than no fallback.
                agents[pid] = EvalStaticAgent(
                    _load_eval_policy(ref_path, device), player_id=pid, device=device
                )
            elif opponent == "gauntlet":
                # Same rotation rule as the parallel path -- see there. This
                # loop's game index is `g`, not `game_idx`.
                _others = [p for p in range(N_PLAYERS) if p != eval_pid]
                _ref    = ref_path[(_others.index(pid) + g) % len(ref_path)]
                agents[pid] = EvalStaticAgent(
                    _load_eval_policy(_ref, device), player_id=pid, device=device
                )
                seat_refs[pid] = _ref
            else:
                agents[pid] = HeuristicAgent(player_id=pid)

        game = BattlegroundsGame(
            card_defs         = card_defs,
            agents            = agents,
            board_computer    = board_comp,
            firestone_client  = firestone,
            matchmaker        = matchmaker,
            tavern_pool       = tavern_pool,
            n_players         = N_PLAYERS,
            seed              = game_seed,
            batched           = True,  # moot here: GreedyPlayAgent/
                                        # HeuristicAgent set
                                        # supports_batching=False, which
                                        # forces the sequential path for
                                        # every seat including EvalAgent's
                                        # — see _agents_support_batching.
        )
        result = game.run_game()
        placements.append(result.placements[eval_pid])
        if opponent == "gauntlet":
            extras.append({
                "eval_pid":   eval_pid,
                "placements": {p: result.placements[p] for p in range(N_PLAYERS)},
                "seat_refs":  dict(seat_refs),
            })
    return placements, extras


def _evaluate_policy_parallel(
    policy: BGPolicyNetwork,
    card_defs: Dict[str, dict],
    n_games: int,
    opponent: str,
    seed: Optional[int],
    n_workers: int,
    ref_path: Optional[str] = None,
) -> List[int]:
    """Parallel implementation of the evaluate_policy game loop.

    Dispatches n_games independent single-game tasks across a
    ProcessPoolExecutor of up to n_workers CPU worker processes (see
    _worker_run_eval_game), mirroring the ProcessPoolExecutor/spawn-context/
    _worker_init machinery _train_parallel already uses for training games.

    Determinism vs. n_workers: each task's seed is derived as (seed + g) from
    the game index g alone, exactly like the sequential path — NEVER from
    worker scheduling/completion order. pool.map already returns results in
    input order, but results are additionally sorted by the game index each
    result carries before placements are extracted, as a defensive guarantee
    that this function returns bit-identical output to
    _evaluate_policy_sequential for the same (policy, seed) regardless of
    n_workers or however results happen to arrive from the pool.

    Workers always run the policy on CPU regardless of the training
    process's device: each CUDA context costs ~0.3-0.6GB of VRAM, the target
    GPU has only 8GB, and eval is latency-bound (many small forward passes)
    rather than throughput-bound, so CPU is the right choice for many
    concurrent eval workers. This is why _worker_init is called with
    device="cpu" below no matter what the caller's `device` argument was —
    that argument only affects the sequential path (n_workers<=1, or the
    fallback below if pool construction fails).

    Raises on any pool-construction failure (caller — evaluate_policy — is
    responsible for catching and falling back to the sequential path).
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    mp_context = multiprocessing.get_context("spawn")

    # CPU tensors only — never pickle a live CUDA module across a spawn
    # boundary (see _train_parallel's identical `sd` snapshot pattern).
    policy_sd = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

    # Broadcast the weights through a file rather than pickling them into each
    # task: at n_games=128 the old form pushed ~1.8GB (128 x ~13.89MB) through
    # the pool's single feeder thread per eval. Same atomic .tmp + os.replace
    # pattern as _train_parallel._write_weights, so no worker can observe a
    # half-written file.
    _bdir = _default_broadcast_dir()
    _tag  = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    eval_weights_path = _bdir / f"bg_eval_{_tag}.pt"
    _tmp = eval_weights_path.with_name(eval_weights_path.name + ".tmp")
    torch.save(policy_sd, str(_tmp))
    os.replace(str(_tmp), str(eval_weights_path))

    tasks = [
        (
            g,
            str(eval_weights_path),
            opponent,
            (seed + g) if seed is not None else None,
            ref_path,
        )
        for g in range(n_games)
    ]

    pool = ProcessPoolExecutor(
        max_workers=max(1, min(n_workers, n_games)),
        mp_context=mp_context,
        initializer=_worker_init,
        initargs=(card_defs, "cpu"),
    )
    try:
        results = list(pool.map(_worker_run_eval_game, tasks))
    finally:
        pool.shutdown(wait=True)
        try:
            eval_weights_path.unlink()
        except OSError:
            pass

    # Aggregate by game index, not completion/arrival order (see docstring).
    results.sort(key=lambda r: r[0])
    # The worker returns a 4-tuple; the 4th element is the gauntlet's per-seat
    # detail (None otherwise). Returned alongside the placements so the caller
    # can fit ratings without re-running any games.
    placements = [placement for _g, placement, _n, _x in results]
    extras     = [x for _g, _p, _n, x in results if x is not None]
    return placements, extras


def fit_gauntlet_elo(extras, n_iter: int = 200, prior: float = 1.0, anchor: str = None):
    """Fit Bradley-Terry ratings (reported on the Elo scale) from gauntlet games.

    A Battlegrounds lobby ranks 8 players at once, so each game yields
    C(8,2) = 28 pairwise outcomes "i placed above j" -- which is why fitting
    ratings here is cheap: n agents cost O(n) GAMES, not O(n^2) matches.

    Why ratings at all: once the policy beats every fixed bar, in-game
    placement is pinned at 4.5 by construction and a single frozen reference
    saturates at 1.0. A rating fitted over a SPREAD of past selves keeps
    moving as long as improvement is real, because the comparison set spans
    the whole run rather than sitting at one point in it.

    Solver: the MM / Zermelo iteration
        p_i <- (W_i + prior) / sum_j [ n_ij / (p_i + p_j) ]
    which is the standard algorithm for Bradley-Terry and converges
    monotonically with no step size to tune. An earlier version of this used
    plain gradient ascent with a fixed learning rate in Elo units; on a
    synthetic set with known strengths it diverged to +-2000 and scrambled the
    ordering outright, because the gradient scales with the number of
    comparisons while the step size did not. `prior` adds a fraction of a
    virtual win and loss against an average opponent, which keeps an
    undefeated agent's rating finite instead of running off to infinity.

    Ratings are identified only up to a constant, so the anchor (default: the
    oldest reference, the weakest and most stable member) is pinned at 0 and
    everything else is relative to it. Without an anchor the scale drifts and
    successive evals are not comparable.

    CAVEAT when reading the number: Elo assumes TRANSITIVITY. Self-play
    readily produces cyclic strength (A beats B beats C beats A) that a single
    scalar cannot represent -- a rising rating alongside a FLAT gauntlet
    placement is the signature of that. See Balduzzi et al. 2018,
    "Re-evaluating Evaluation", for the Nash-averaging treatment of the
    cyclic case.

    Returns {"current", "per_ref", "n_pairs", "anchor"} or None.
    """
    import math
    CUR = "__current__"
    wins: Dict[str, float] = {}
    n_ij: Dict[tuple, float] = {}
    players = set()
    for e in extras:
        pl, refs, epid = e["placements"], e["seat_refs"], e["eval_pid"]
        ident = {p: (CUR if p == epid else refs.get(p)) for p in pl}
        seats = sorted(pl)
        for x in range(len(seats)):
            for y in range(x + 1, len(seats)):
                a, b = seats[x], seats[y]
                ia, ib = ident.get(a), ident.get(b)
                if ia is None or ib is None or ia == ib:
                    continue      # same checkpoint on both seats: no information
                players.add(ia); players.add(ib)
                key = (ia, ib) if ia < ib else (ib, ia)
                n_ij[key] = n_ij.get(key, 0.0) + 1.0
                win = ia if pl[a] < pl[b] else ib
                wins[win] = wins.get(win, 0.0) + 1.0
    if not n_ij:
        return None
    players = sorted(players)
    if anchor is None:
        refs_only = [p for p in players if p != CUR]
        # Anchor on the OLDEST reference by update number. Using min() on the
        # raw path strings sorts lexicographically -- "ref_u1200" < "ref_u300"
        # -- which silently picked the wrong anchor, and worse, would CHANGE
        # the anchor as new references were added, making successive Elo
        # values incomparable. That defeats the entire purpose of anchoring.
        def _age(path):
            m = re.search(r"_u(\d+)", str(path))
            return int(m.group(1)) if m else float("inf")
        anchor = min(refs_only, key=lambda p: (_age(p), str(p))) if refs_only else CUR

    opponents: Dict[str, list] = {p: [] for p in players}
    for (i, j), c in n_ij.items():
        opponents[i].append((j, c))
        opponents[j].append((i, c))

    p = {q: 1.0 for q in players}
    for _ in range(n_iter):
        new_p = {}
        for q in players:
            denom = sum(c / (p[q] + p[o]) for o, c in opponents[q])
            # prior: one virtual win and one virtual loss vs an average (p=1)
            # opponent, so an undefeated agent stays finite.
            denom += 2.0 * prior / (p[q] + 1.0)
            new_p[q] = (wins.get(q, 0.0) + prior) / denom if denom > 0 else p[q]
        gm = math.exp(sum(math.log(max(v, 1e-12)) for v in new_p.values()) / len(new_p))
        p = {q: v / gm for q, v in new_p.items()}

    def elo(q):
        return 400.0 * math.log10(max(p[q], 1e-12) / max(p[anchor], 1e-12))
    return {
        "current": round(elo(CUR), 1) if CUR in p else None,
        "per_ref": {q: round(elo(q), 1) for q in players if q != CUR},
        "n_pairs": int(sum(n_ij.values())),
        "anchor":  anchor,
    }


def evaluate_policy(
    policy: BGPolicyNetwork,
    card_defs: Dict[str, dict],
    n_games: int = 32,
    opponent: str = "greedy",
    device: str = "cpu",
    seed: Optional[int] = None,
    n_workers: int = 1,
    ref_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Measure the current policy's raw skill against a FIXED scripted opponent.

    This is the only progress signal in this module that a shaping term or a
    co-evolving opponent pool cannot inflate:

    - Mean training reward includes potential-based shaping (board-shape,
      tier-shape -- see game_loop.py / CLAUDE.md) that can rise even while
      the policy's actual in-game skill doesn't improve at all (this is
      exactly what happened in the 312-update/731k-step run analysed
      2026-08-31: reward improved, placement never left 4.58).
    - train_placements / heuristic_placements / greedy_placements (tracked
      in run_fresh_training.py's on_batch/on_update callbacks) are measured
      each game against whichever mix of SnapshotPool snapshots / heuristic /
      greedy seats that game happened to draw. The snapshot pool co-evolves
      with the training policy, so if the whole pool improves together those
      numbers can plateau around ~4.5 forever -- they can't tell "no one got
      better" apart from "everyone got better together."

    Seating exactly one deterministic policy seat against seven seats of a
    scripted opponent that never changes turns "placement" back into an
    absolute measurement: a drop in mean_placement here can only mean the
    policy itself improved.

    Determinism: actions are selected via argmax over the masked action
    distribution (BGPolicyNetwork.get_action(..., deterministic=True), the
    same flag PPOAgent already exposes for its own `deterministic` mode --
    see EvalAgent) rather than sampled, so this measures the policy's
    learned mode, not exploration noise.

    Isolation: this function collects no PPO transitions, runs no optimizer
    step, and never references a PPOTrainer, so it cannot disturb the
    rollout buffer or any trainer counters (total_steps, update_count) even
    when called mid-training-run. All game/tavern/matchmaker randomness for
    the games played here comes from *seed* alone (TavernPool/Matchmaker/
    BattlegroundsGame all take their own local `random.Random` instances,
    per-game, exactly as the parallel training workers already do -- see
    _worker_run_game). As a belt-and-braces guard against any incidental use
    of the *global* random/np.random/torch RNGs deeper in the symbolic/env
    stack, this function also snapshots and restores all three global RNG
    states around the whole call, so it is safe to call between PPO updates
    without perturbing anything the training loop's own global-RNG usage
    depends on (e.g. SnapshotPool.sample_n in _train_parallel draws from the
    global `random` module).

    Parameters
    ----------
    policy : BGPolicyNetwork
        Network to evaluate. Not mutated by this call (only .eval()/no_grad
        forward passes are run against it; its train()/eval() mode flag is
        restored to whatever it was on entry).
    card_defs : dict
        Card definitions dict, same shape used elsewhere in this module
        (see load_card_defs).
    n_games : int
        Number of games to play (default 32).
    opponent : {"greedy", "heuristic"}
        Which scripted agent fills the other 7 seats each game.
    device : str
        Torch device for the policy's eval forward passes. Only affects the
        sequential path (n_workers<=1, or the fallback if parallel pool
        construction fails) — parallel workers always run on CPU regardless
        of this argument; see _evaluate_policy_parallel.
    seed : int, optional
        Base seed for this call. None = non-deterministic (system entropy).
    n_workers : int
        Number of parallel CPU worker processes to evaluate games with
        (default 1 = sequential, in-process, honouring `device`). With
        n_workers > 1, games are dispatched across a ProcessPoolExecutor
        (see _evaluate_policy_parallel / _worker_run_eval_game), each worker
        running the policy on CPU. Per-game seeds are derived from (seed,
        game index) alone, so results are identical to the sequential path
        for the same (policy, seed) regardless of n_workers — see the
        n_workers=1 vs n_workers=8 equality check exercised in this module's
        verification. If pool construction itself fails, this function logs
        a warning and falls back to the sequential path rather than
        propagating.

    Returns
    -------
    dict with keys: mean_placement, top1_rate, top4_rate, n_games, opponent.
    On ANY exception during evaluation (including a parallel pool failure
    that also fails to fall back), this function logs a warning and returns
    this same dict shape with every metric set to float("nan") instead of
    propagating — a failed eval must never take down a multi-hour training
    run. Callers should treat NaN metrics as "eval unavailable this round"
    rather than a real measurement.
    """
    if opponent not in ("greedy", "heuristic", "reference", "gauntlet"):
        raise ValueError(
            f"evaluate_policy: unknown opponent {opponent!r}, expected "
            "'greedy', 'heuristic', 'reference' or 'gauntlet'"
        )
    if opponent == "reference" and not ref_path:
        raise ValueError("evaluate_policy: opponent='reference' requires ref_path")
    if opponent == "gauntlet":
        if not ref_path or isinstance(ref_path, str):
            raise ValueError(
                "evaluate_policy: opponent='gauntlet' requires ref_path to be a "
                "LIST of checkpoint paths (one per non-eval seat), not a single path"
            )

    was_training  = policy.training
    _random_state = random.getstate()
    _np_state     = np.random.get_state()
    _torch_state  = torch.get_rng_state()
    try:
        try:
            if n_workers > 1:
                try:
                    placements, extras = _evaluate_policy_parallel(
                        policy, card_defs, n_games, opponent, seed, n_workers,
                        ref_path=ref_path,
                    )
                except Exception as exc:
                    logger.warning(
                        "evaluate_policy: parallel pool failed (%s: %s) -- "
                        "falling back to sequential eval",
                        type(exc).__name__, exc,
                    )
                    placements, extras = _evaluate_policy_sequential(
                        policy, card_defs, n_games, opponent, device, seed,
                        ref_path=ref_path,
                    )
            else:
                placements, extras = _evaluate_policy_sequential(
                    policy, card_defs, n_games, opponent, device, seed,
                    ref_path=ref_path,
                )

            placements_arr = np.asarray(placements, dtype=np.float64)
            _elo = fit_gauntlet_elo(extras) if opponent == "gauntlet" and extras else None
            return {
                "mean_placement": float(placements_arr.mean()),
                "top1_rate":       float((placements_arr == 1).mean()),
                "top4_rate":       float((placements_arr <= 4).mean()),
                "n_games":         n_games,
                "opponent":        opponent,
                "elo":             _elo,
            }
        except Exception as exc:
            # Belt-and-braces: even the sequential fallback above raised (or
            # n_workers<=1 raised directly). A failed eval must never kill a
            # multi-hour training job -- log and hand back NaN metrics in the
            # same shape a real result would have, instead of propagating.
            logger.warning(
                "evaluate_policy: eval failed (%s: %s) -- returning NaN metrics",
                type(exc).__name__, exc,
            )
            return {
                "mean_placement": float("nan"),
                "top1_rate":       float("nan"),
                "top4_rate":       float("nan"),
                "n_games":         n_games,
                "opponent":        opponent,
            }
    finally:
        random.setstate(_random_state)
        np.random.set_state(_np_state)
        torch.set_rng_state(_torch_state)
        policy.train(was_training)


# -------------------------------------------------------------------------
# Main training loop
# -------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # Seed everything
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Load card definitions
    card_defs = load_card_defs(CARD_DEFS_PATH)

    # Build all components
    components = build_components(
        card_defs    = card_defs,
        use_firestone= not args.no_firestone,
        device       = args.device,
        seed         = args.seed,
    )
    policy      = components["policy"]
    ppo_trainer = components["ppo_trainer"]

    # Optional BC warm-start (v2 takes priority over legacy v1)
    if args.load_bc_v2:
        bc_path = Path(args.load_bc_v2)
        if bc_path.exists():
            logger.info("Warm-starting from BC v2 checkpoint: %s", bc_path)
            policy.load_bc_v2_weights(str(bc_path))
        else:
            logger.warning("BC v2 checkpoint not found: %s — skipping warm-start", bc_path)

    # Optional: resume from PPO checkpoint
    if args.checkpoint and Path(args.checkpoint).exists():
        logger.info("Resuming from checkpoint: %s", args.checkpoint)
        ppo_trainer.load_checkpoint(args.checkpoint)

    n_games = 2 if args.dry_run else args.games
    if args.dry_run:
        logger.info("Dry-run mode: running 2 games then exiting.")

    update_interval     = args.update_interval
    checkpoint_interval = 100
    update_count        = 0

    n_workers = max(1, args.workers)
    logger.info(
        "Starting training: %d games, update_interval=%d, device=%s, firestone=%s, workers=%d",
        n_games, update_interval, args.device, not args.no_firestone, n_workers,
    )

    stats_file = Path(args.stats_path) if args.stats_path else None

    if n_workers > 1:
        _train_parallel(
            n_games, policy, ppo_trainer, card_defs,
            n_workers=n_workers,
            update_interval=update_interval,
            checkpoint_interval=checkpoint_interval,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
            device=args.device,
            stats_path=args.stats_path,
        )
    else:
        for game_idx in range(1, n_games + 1):
            t0 = time.time()

            result = run_one_game(components, game_idx, args.seed)

            elapsed = time.time() - t0
            log_game_stats(game_idx, result, elapsed)

            if stats_file is not None:
                # Single-process path is pure self-play: every seat is the
                # current training policy.
                summary = {
                    "placements":    result.placements,
                    "final_rewards": result.final_rewards,
                    "agent_labels":  {pid: "train_current" for pid in result.placements},
                }
                _append_agent_stats(stats_file, game_idx, ppo_trainer.total_steps, summary)

            # PPO update every update_interval games
            if game_idx % update_interval == 0 and len(ppo_trainer.buffer) > 0:
                metrics = ppo_trainer.update(last_value=0.0)
                update_count += 1
                log_update_metrics(update_count, metrics)

            # Checkpoint every checkpoint_interval games
            if (
                args.checkpoint
                and game_idx % checkpoint_interval == 0
            ):
                ppo_trainer.save_checkpoint(
                    args.checkpoint,
                    extra={"game": game_idx},
                )
                logger.info("Checkpoint saved at game %d → %s", game_idx, args.checkpoint)

    # Final checkpoint
    if args.checkpoint:
        ppo_trainer.save_checkpoint(
            args.checkpoint,
            extra={"game": n_games},
        )
        logger.info("Final checkpoint saved → %s", args.checkpoint)

    logger.info(
        "Training complete. Total PPO updates: %d, total steps: %d",
        ppo_trainer.update_count, ppo_trainer.total_steps,
    )


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Self-play PPO training for Hearthstone Battlegrounds agent."
    )
    p.add_argument(
        "--games", type=int, default=500,
        help="Number of self-play games to run (default: 500).",
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel game workers (default: 1). Use 4–16 on a multi-core CPU.",
    )
    p.add_argument(
        "--checkpoint", type=str, default="bg_agent_ppo.pt",
        help="Path to save/load PPO checkpoint (default: bg_agent_ppo.pt).",
    )
    p.add_argument(
        "--load-bc", type=str, default=None,
        dest="load_bc",
        help="Path to BC v1 checkpoint for warm-start (legacy).",
    )
    p.add_argument(
        "--load-bc-v2", type=str, default=None,
        dest="load_bc_v2",
        help="Path to BC v2 checkpoint (bc_v2.pt) for structured warm-start.",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device: 'cpu' or 'cuda' (default: cpu).",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Global RNG seed for reproducibility.",
    )
    p.add_argument(
        "--update-interval", type=int, default=10,
        dest="update_interval",
        help="Run a PPO update every N games (default: 10).",
    )
    p.add_argument(
        "--no-firestone", action="store_true",
        dest="no_firestone",
        help="Disable Firestone subprocess; use heuristic combat estimator.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        dest="dry_run",
        help="Run 2 games and exit (for testing the pipeline end-to-end).",
    )
    p.add_argument(
        "--stats-path", type=str, default="data/agent_stats.jsonl",
        dest="stats_path",
        help="JSONL path logging per-game per-player agent identity and "
             "placement, for tracking win rate by agent type over time. "
             "Pass '' to disable.",
    )
    p.add_argument(
        "--log-level", type=str, default="INFO",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    train(args)


if __name__ == "__main__":
    main()
