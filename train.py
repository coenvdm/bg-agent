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
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from agent.policy import (BGPolicyNetwork, N_ACTION_TYPES, POINTER_DIM,
                          build_type_mask, build_pointer_mask,
                          PTR_SHOP_OFF, PTR_BOARD_OFF, PTR_HAND_OFF)
from agent.ppo import PPOConfig, PPOTrainer
from env.game_loop import BattlegroundsGame, GameResult
from env.matchmaker import Matchmaker
from env.player_state import PlayerState
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
N_HEURISTIC_SLOTS = 1   # opponent slots permanently assigned to HeuristicAgent
SNAPSHOT_EVERY    = 10  # rolling snapshot every N PPO updates
MILESTONE_EVERY   = 50  # protected milestone snapshot every N PPO updates


# -------------------------------------------------------------------------
# Snapshot pool for historical self-play
# -------------------------------------------------------------------------

class SnapshotPool:
    """Rolling buffer of past policy state_dicts for historical self-play.

    Each game pairs N_TRAIN_PLAYERS current-policy agents against
    (N_PLAYERS - N_TRAIN_PLAYERS) agents frozen at a past checkpoint.
    This breaks the echo-chamber that forms when all players share the same
    evolving policy.

    Two snapshot tiers:
      - Rolling  : recent snapshots, evicted when capacity is exceeded.
      - Milestone: protected snapshots that are never evicted; preserve
                   long-term behavioral diversity (e.g. early-training styles).

    Usage::

        pool = SnapshotPool(capacity=20)
        pool.add(policy.state_dict())                    # rolling
        pool.add(policy.state_dict(), is_milestone=True) # protected
        opp_sds = pool.sample_n(5)   # five independent draws
    """

    def __init__(self, capacity: int = 20) -> None:
        self.capacity    = capacity
        self._snapshots:  List[dict] = []   # rolling, evictable
        self._milestones: List[dict] = []   # protected, never evicted

    def add(self, state_dict: dict, *, is_milestone: bool = False) -> None:
        """Append a CPU clone of *state_dict* to the rolling buffer.

        If *is_milestone* is True, also add to the protected milestone list.
        """
        snap = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        self._snapshots.append(snap)
        if len(self._snapshots) > self.capacity:
            self._snapshots.pop(0)
        if is_milestone:
            self._milestones.append(
                {k: v.detach().cpu().clone() for k, v in state_dict.items()}
            )

    def sample(self) -> Optional[dict]:
        """Return a uniformly sampled snapshot, or None if the pool is empty."""
        pool = self._snapshots + self._milestones
        return random.choice(pool) if pool else None

    def sample_n(self, n: int) -> List[Optional[dict]]:
        """Return *n* independently sampled snapshots (with replacement)."""
        pool = self._snapshots + self._milestones
        if not pool:
            return [None] * n
        return [random.choice(pool) for _ in range(n)]

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
    ) -> None:
        self.policy       = policy
        self.trainer      = ppo_trainer
        self.player_id    = player_id
        self.device       = device
        self.deterministic = deterministic
        self._last_obs: Optional[dict] = None
        self._cached_type_mask: Optional[np.ndarray] = None
        self._cached_ptr_mask:  Optional[np.ndarray] = None

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
        p_mask_t = build_pointer_mask(ps, -1).unsqueeze(0).to(dev)  # full occupancy

        board_t  = torch.tensor(obs["board_tokens"][None],   dtype=torch.float32, device=dev)
        shop_t   = torch.tensor(obs["shop_tokens"][None],    dtype=torch.float32, device=dev)
        hand_t   = torch.tensor(obs["hand_tokens"][None],    dtype=torch.float32, device=dev)
        scalar_t = torch.tensor(obs["scalar_context"][None], dtype=torch.float32, device=dev)
        opp_np   = obs.get("opp_tokens")
        opp_t    = torch.tensor(opp_np[None], dtype=torch.float32, device=dev) if opp_np is not None else None

        type_idx, ptr_idx, _log_prob, _value = self.policy.get_action(
            board_t, shop_t, hand_t, scalar_t,
            type_mask=t_mask_t, pointer_mask=p_mask_t,
            deterministic=self.deterministic, opp_tokens=opp_t,
        )
        # Cache the pointer mask for the chosen type (also pre-mutation)
        self._cached_ptr_mask = build_pointer_mask(ps, type_idx).numpy()
        return type_idx, ptr_idx

    def record_transition(
        self,
        obs: dict,
        type_action: int,
        ptr_action:  int,
        reward: float,
        done:   bool,
    ) -> None:
        """Push a completed transition into the PPO rollout buffer.

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
        p_mask_t = build_pointer_mask(ps, -1).unsqueeze(0).to(dev)

        board_t  = torch.tensor(obs["board_tokens"][None],   dtype=torch.float32, device=dev)
        shop_t   = torch.tensor(obs["shop_tokens"][None],    dtype=torch.float32, device=dev)
        hand_t   = torch.tensor(obs["hand_tokens"][None],    dtype=torch.float32, device=dev)
        scalar_t = torch.tensor(obs["scalar_context"][None], dtype=torch.float32, device=dev)
        opp_np   = obs.get("opp_tokens")
        opp_t    = torch.tensor(opp_np[None], dtype=torch.float32, device=dev) if opp_np is not None else None

        with torch.no_grad():
            type_idx, ptr_idx, _, _ = self.policy.get_action(
                board_t, shop_t, hand_t, scalar_t,
                type_mask=t_mask_t, pointer_mask=p_mask_t, opp_tokens=opp_t,
            )
        self._cached_ptr_mask = build_pointer_mask(ps, type_idx).numpy()
        return type_idx, ptr_idx

    def record_transition(self, *_a, **_kw) -> None:  # no-op
        pass

    def record_transition_precomputed(self, *_a, **_kw) -> None:  # no-op
        pass


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
      1. Level up  — if affordable and currently below tier 4
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

        # 1. Level up (type 5) — mask already verifies gold >= cost
        if ps.tavern_tier < 4 and valid(5):
            return 5, 0

        # 2. Buy highest-tier shop minion (type 0)
        if valid(0):
            best_idx, best_tier = -1, -1
            for i, m in enumerate(ps.shop):
                if m is not None and getattr(m, "card_id", ""):
                    t = getattr(m, "tier", 1)
                    if t > best_tier:
                        best_tier, best_idx = t, i
            if best_idx >= 0:
                return 0, PTR_SHOP_OFF + best_idx

        # 3. Place any card from hand onto the board (type 2)
        if valid(2):
            for i, m in enumerate(ps.hand):
                if m is not None and getattr(m, "card_id", ""):
                    return 2, PTR_HAND_OFF + i

        # 4. Sell weakest board minion if board is full and a buy is possible (type 1)
        if valid(1) and len(ps.board) >= 7 and valid(0):
            worst_idx, worst_power = -1, float("inf")
            for i, m in enumerate(ps.board):
                if m is not None and getattr(m, "card_id", ""):
                    power = getattr(m, "attack", 0) + getattr(m, "health", 0)
                    if power < worst_power:
                        worst_power, worst_idx = power, i
            if worst_idx >= 0:
                return 1, PTR_BOARD_OFF + worst_idx

        # 5. End turn
        return 7, 0

    def record_transition(self, *_a, **_kw) -> None:  # no-op
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
        defs = json.load(fh)

    # Unwrap {"version": ..., "cards": {...}} envelope if present
    if isinstance(defs, dict) and "cards" in defs and isinstance(defs["cards"], dict):
        defs = defs["cards"]
    elif isinstance(defs, dict) and "cards" in defs and isinstance(defs["cards"], list):
        defs = {d.get("card_id", d.get("id", str(i))): d for i, d in enumerate(defs["cards"])}

    # Normalise: if the JSON is a list, key by card_id
    if isinstance(defs, list):
        defs = {d.get("card_id", d.get("id", str(i))): d for i, d in enumerate(defs)}

    logger.info("Loaded %d card definitions from %s", len(defs), path)
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

    policy = BGPolicyNetwork(
        card_dim=44,
        d_model=256,
        nhead=8,
        num_layers=4,
        scalar_dim=98,
        dropout=0.1,
    ).to(device)

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

    # All 8 players share the same policy (self-play)
    agents: List[PPOAgent] = [
        PPOAgent(policy, trainer, player_id=pid, device=device)
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


def _worker_init(card_defs: dict, device: str) -> None:
    """Pool initializer: runs once per worker process on Windows spawn."""
    import torch as _torch
    global _W_CARD_DEFS, _W_DEVICE
    _W_CARD_DEFS = card_defs
    _W_DEVICE    = device
    # Prevent PyTorch from spawning multiple internal threads per worker.
    # With N workers each using 1 thread, total = N threads = one per core.
    _torch.set_num_threads(1)


def _worker_run_game(task: tuple) -> tuple:
    """Run one self-play game in a subprocess.

    N_TRAIN_PLAYERS slots use the current policy and collect PPO transitions.
    The remaining slots use per-slot opponent entries from opp_sds:
      - dict   : frozen historical BGPolicyNetwork snapshot → StaticAgent
      - "heuristic" : HeuristicAgent (no network, leveling-focused)
      - None   : promote to PPOAgent (warm-up fallback when pool is empty)

    Parameters (unpacked from *task*)
    ---------------------------------
    current_sd : dict                    — current policy.state_dict() snapshot
    opp_sds    : List[dict | str | None] — one entry per opponent slot
    seed       : int | None              — per-game RNG seed

    card_defs and device are read from the per-process globals set by
    _worker_init, so they are NOT re-pickled on every call.

    Returns
    -------
    (transitions, summary_dict)
      transitions  : List[Transition]   — only from training-agent slots
      summary_dict : {"placements": dict, "final_rewards": dict, "n_rounds": int}
    """
    import random as _random
    import numpy as _np
    import torch as _torch

    current_sd, opp_sds, seed = task
    card_defs = _W_CARD_DEFS
    device    = _W_DEVICE

    if seed is not None:
        _random.seed(seed)
        _np.random.seed(seed)
        _torch.manual_seed(seed)

    tavern_pool = TavernPool(card_defs, seed=seed)
    matchmaker  = Matchmaker(n_players=N_PLAYERS, seed=seed)
    board_comp  = SymbolicBoardComputer(card_defs)
    firestone   = FirestoneClient(firestone_path=None, mock_mode=True)

    # Current policy — used by training agents, records transitions
    current_policy = BGPolicyNetwork(
        card_dim=44, d_model=256, nhead=8, num_layers=4,
        scalar_dim=98, dropout=0.1,
    ).to(device)
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
    for pid in train_pids:
        agents[pid] = PPOAgent(current_policy, ppo_trainer, player_id=pid, device=device)

    # Build opponent agents; deduplicate policy networks by state_dict identity
    # to avoid loading the same weights into multiple BGPolicyNetwork instances.
    _policy_cache: Dict[int, Any] = {}
    for slot_i, pid in enumerate(opp_pids):
        entry = opp_sds[slot_i]
        if entry == "heuristic":
            agents[pid] = HeuristicAgent(player_id=pid)
        elif entry is None:
            # Pool still empty for this slot — promote to training agent
            agents[pid] = PPOAgent(current_policy, ppo_trainer, player_id=pid, device=device)
        else:
            sd_id = id(entry)
            if sd_id not in _policy_cache:
                pol = BGPolicyNetwork(
                    card_dim=44, d_model=256, nhead=8, num_layers=4,
                    scalar_dim=98, dropout=0.1,
                ).to(device)
                pol.load_state_dict(entry)
                pol.eval()
                _policy_cache[sd_id] = pol
            agents[pid] = StaticAgent(_policy_cache[sd_id], player_id=pid, device=device)

    game = BattlegroundsGame(
        card_defs        = card_defs,
        agents           = agents,
        board_computer   = board_comp,
        firestone_client = firestone,
        matchmaker       = matchmaker,
        tavern_pool      = tavern_pool,
        n_players        = N_PLAYERS,
        seed             = seed,
        batched          = True,
    )
    result = game.run_game()

    summary = {
        "placements":    result.placements,
        "final_rewards": result.final_rewards,
        "n_rounds":      result.n_rounds,
    }
    return ppo_trainer.buffer.transitions, summary


# -------------------------------------------------------------------------
# Logging helpers
# -------------------------------------------------------------------------

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
) -> None:
    """Run self-play games in parallel using ProcessPoolExecutor.

    Games are dispatched in batches of *n_workers*.  Each worker receives a
    frozen copy of the current policy weights, runs one game, and returns its
    collected transitions.  The main process merges all transitions into
    ppo_trainer.buffer and runs PPO updates at the normal interval.

    Parameters
    ----------
    n_games            : total games to play
    policy             : the policy network being trained
    ppo_trainer        : PPOTrainer instance
    card_defs          : card definitions dict
    n_workers          : number of parallel worker processes
    update_interval    : trigger a PPO update every this many games
    checkpoint_interval: save a checkpoint every this many games
    checkpoint_path    : path for automatic checkpoint saves (None = skip)
    seed               : base RNG seed (None = non-deterministic)
    device             : torch device string for the main process
    on_batch(game_idx, summaries, transitions, elapsed)
                       : optional callback fired after every batch of games.
                         *game_idx* is the total games completed so far.
                         *summaries* is a list of per-game summary dicts.
                         *transitions* is a list of per-game Transition lists
                         (already added to the PPO buffer).
                         *elapsed* is the batch wall-clock time in seconds.
    on_update(metrics, update_count)
                       : optional callback fired after every PPO update.
    batch_timeout      : per-batch timeout in seconds (default 300)
    """
    from concurrent.futures import ProcessPoolExecutor

    update_count = 0
    game_idx     = 0

    snapshot_pool  = SnapshotPool(capacity=20)

    # Opponent slot composition per game:
    #   N_HEURISTIC_SLOTS slots always use HeuristicAgent (leveling anchor)
    #   remaining slots sample independently from the snapshot pool
    N_OPP_SLOTS    = N_PLAYERS - N_TRAIN_PLAYERS          # 6
    n_policy_slots = N_OPP_SLOTS - N_HEURISTIC_SLOTS      # 5

    pool = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(card_defs, device),
    )
    try:
        # Snapshot weights once; only re-clone after each PPO update
        sd = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        sd_stale = False

        while game_idx < n_games:
            batch_n = min(n_workers, n_games - game_idx)

            if sd_stale:
                sd = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
                sd_stale = False

            # Build per-slot opponent list: 5 independent policy snapshots + 1 heuristic.
            policy_sds = snapshot_pool.sample_n(n_policy_slots)
            opp_sds    = policy_sds + ["heuristic"] * N_HEURISTIC_SLOTS

            # Use total_steps as seed offset so each re-run gets fresh seeds
            seed_base = ppo_trainer.total_steps
            tasks = [
                (
                    sd,
                    opp_sds,
                    (seed + seed_base + i) if seed is not None else None,
                )
                for i in range(batch_n)
            ]

            t0 = time.time()
            try:
                worker_results = list(pool.map(_worker_run_game, tasks,
                                               timeout=batch_timeout))
            except TimeoutError:
                logger.warning("Batch timed out after %ds — skipping", batch_timeout)
                game_idx += batch_n
                continue
            except Exception as exc:
                logger.warning("Worker error (%s: %s) — rebuilding pool", type(exc).__name__, exc)
                pool.shutdown(wait=False)
                pool = ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_worker_init,
                    initargs=(card_defs, device),
                )
                game_idx += batch_n
                continue
            batch_elapsed = time.time() - t0

            # Merge transitions into buffer; collect per-game data for callbacks
            batch_summaries:    List[dict] = []
            batch_transitions:  List[list] = []
            for i, (transitions, summary) in enumerate(worker_results):
                g = game_idx + i + 1
                for t in transitions:
                    ppo_trainer.buffer.add(t)
                    ppo_trainer.total_steps += 1
                batch_summaries.append(summary)
                batch_transitions.append(transitions)

                winner_id   = min(summary["placements"], key=summary["placements"].get)
                mean_reward = float(np.mean(list(summary["final_rewards"].values())))
                logger.info(
                    "Game %4d | rounds=%2d | winner=P%d | mean_reward=%+.3f | (batch %.1fs)",
                    g, summary["n_rounds"], winner_id, mean_reward, batch_elapsed,
                )

            prev_game_idx = game_idx
            game_idx     += batch_n

            if on_batch is not None:
                on_batch(game_idx, batch_summaries, batch_transitions, batch_elapsed)

            # PPO update if we crossed an update_interval boundary
            if (game_idx // update_interval) > (prev_game_idx // update_interval):
                if len(ppo_trainer.buffer) > 0:
                    metrics = ppo_trainer.update(last_value=0.0)
                    update_count += 1
                    sd_stale = True   # weights changed — reclone before next batch
                    if update_count % SNAPSHOT_EVERY == 0:
                        is_milestone = (update_count % MILESTONE_EVERY == 0)
                        snapshot_pool.add(policy.state_dict(), is_milestone=is_milestone)
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

            # Checkpoint if we crossed a checkpoint_interval boundary
            if checkpoint_path:
                if (game_idx // checkpoint_interval) > (prev_game_idx // checkpoint_interval):
                    ppo_trainer.save_checkpoint(
                        checkpoint_path, extra={"game": game_idx}
                    )
                    logger.info("Checkpoint saved at game %d → %s", game_idx, checkpoint_path)
    finally:
        pool.shutdown(wait=True)


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

    if n_workers > 1:
        _train_parallel(
            n_games, policy, ppo_trainer, card_defs,
            n_workers=n_workers,
            update_interval=update_interval,
            checkpoint_interval=checkpoint_interval,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
            device=args.device,
        )
    else:
        for game_idx in range(1, n_games + 1):
            t0 = time.time()

            result = run_one_game(components, game_idx, args.seed)

            elapsed = time.time() - t0
            log_game_stats(game_idx, result, elapsed)

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
