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
                          build_type_mask, build_pointer_mask)
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
N_PLAYERS = 8


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

    Receives a frozen policy snapshot, rebuilds all components locally, runs
    the game, and returns the collected transitions (already contain value +
    log_prob computed with the behaviour policy, so main can merge them
    directly into the PPO buffer without another forward pass).

    Parameters (unpacked from *task*)
    ---------------------------------
    state_dict : dict           — policy.state_dict() snapshot
    seed       : int | None     — per-game RNG seed

    card_defs and device are read from the per-process globals set by
    _worker_init, so they are NOT re-pickled on every call.

    Returns
    -------
    (transitions, summary_dict)
      transitions  : List[Transition]
      summary_dict : {"placements": dict, "final_rewards": dict, "n_rounds": int}
    """
    import random as _random
    import numpy as _np
    import torch as _torch

    state_dict, seed = task
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

    policy = BGPolicyNetwork(
        card_dim=44, d_model=256, nhead=8, num_layers=4,
        scalar_dim=98, dropout=0.1,
    ).to(device)
    policy.load_state_dict(state_dict)

    ppo_config  = PPOConfig(device=device)
    ppo_trainer = PPOTrainer(policy, ppo_config)

    agents = [
        PPOAgent(policy, ppo_trainer, player_id=pid, device=device)
        for pid in range(N_PLAYERS)
    ]

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
    args,
    n_games: int,
    policy: BGPolicyNetwork,
    ppo_trainer: PPOTrainer,
    card_defs: dict,
    update_interval: int,
    checkpoint_interval: int,
) -> None:
    """Run self-play games in parallel using ProcessPoolExecutor.

    Games are dispatched in batches of *args.workers*.  Each worker receives
    a frozen copy of the current policy weights, runs one game, and returns
    its collected transitions.  The main process merges all transitions into
    ppo_trainer.buffer and runs PPO updates at the normal interval.
    """
    from concurrent.futures import ProcessPoolExecutor

    n_workers    = args.workers
    update_count = 0
    game_idx     = 0

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(card_defs, args.device),
    ) as pool:
        # Snapshot weights once; only re-clone after each PPO update
        sd = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        sd_stale = False

        while game_idx < n_games:
            batch_n = min(n_workers, n_games - game_idx)

            if sd_stale:
                sd = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
                sd_stale = False

            # Use total_steps as seed offset so each re-run gets fresh seeds
            seed_base = ppo_trainer.total_steps
            tasks = [
                (
                    sd,
                    (args.seed + seed_base + i) if args.seed is not None else None,
                )
                for i in range(batch_n)
            ]

            t0 = time.time()
            worker_results = list(pool.map(_worker_run_game, tasks))
            batch_elapsed  = time.time() - t0

            # Merge transitions and log each game
            for i, (transitions, summary) in enumerate(worker_results):
                g = game_idx + i + 1
                for t in transitions:
                    ppo_trainer.buffer.add(t)
                    ppo_trainer.total_steps += 1

                winner_id   = min(summary["placements"], key=summary["placements"].get)
                mean_reward = float(np.mean(list(summary["final_rewards"].values())))
                logger.info(
                    "Game %4d | rounds=%2d | winner=P%d | mean_reward=%+.3f | (batch %.1fs)",
                    g, summary["n_rounds"], winner_id, mean_reward, batch_elapsed,
                )

            prev_game_idx = game_idx
            game_idx     += batch_n

            # PPO update if we crossed an update_interval boundary
            if (game_idx // update_interval) > (prev_game_idx // update_interval):
                if len(ppo_trainer.buffer) > 0:
                    metrics = ppo_trainer.update(last_value=0.0)
                    update_count += 1
                    sd_stale = True   # weights changed — reclone before next batch
                    log_update_metrics(update_count, metrics)

                    nan_count = sum(
                        int(torch.isnan(p).any().item())
                        for p in policy.parameters()
                    )
                    logger.info("NaN params after update: %d", nan_count)

            # Checkpoint if we crossed a checkpoint_interval boundary
            if args.checkpoint:
                if (game_idx // checkpoint_interval) > (prev_game_idx // checkpoint_interval):
                    ppo_trainer.save_checkpoint(
                        args.checkpoint, extra={"game": game_idx}
                    )
                    logger.info("Checkpoint saved at game %d → %s", game_idx, args.checkpoint)


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
            args, n_games, policy, ppo_trainer, card_defs,
            update_interval, checkpoint_interval,
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
