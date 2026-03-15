"""
PPO training loop with action masking for the BG agent.

Uses Generalised Advantage Estimation (GAE) and the clipped PPO objective.
Action masking sets invalid action log-probs to -inf before any computation.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from agent.policy import BGPolicyNetwork, N_ACTIONS

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class PPOConfig:
    """Hyperparameters for the PPO trainer."""

    lr: float = 3e-4
    gamma: float = 0.99        # discount factor
    gae_lambda: float = 0.95   # GAE λ
    clip_eps: float = 0.2      # PPO clip epsilon
    value_coef: float = 0.5    # value loss coefficient
    entropy_coef: float = 0.01 # entropy bonus coefficient
    max_grad_norm: float = 0.5 # gradient clipping norm
    n_epochs: int = 4          # PPO update epochs per rollout
    batch_size: int = 64
    device: str = "cpu"


# ------------------------------------------------------------------
# Transition
# ------------------------------------------------------------------

@dataclass
class Transition:
    """Single (state, action, reward, done) transition for PPO rollout."""

    board_tokens:   np.ndarray   # [7,  44]
    shop_tokens:    np.ndarray   # [7,  44]
    hand_tokens:    np.ndarray   # [10, 44]
    opp_tokens:     np.ndarray   # [7,  44] last seen opponent board
    scalar_context: np.ndarray   # [38]
    action_mask:    np.ndarray   # [N_ACTIONS] bool
    action:         int
    reward:         float
    done:           bool
    value:          float        # stored for GAE bootstrap
    log_prob:       float        # stored for importance-sampling ratio


# ------------------------------------------------------------------
# Rollout buffer
# ------------------------------------------------------------------

class RolloutBuffer:
    """Fixed-capacity buffer that accumulates Transitions for PPO updates."""

    def __init__(self, capacity: int = 2048) -> None:
        self.capacity = capacity
        self.transitions: List[Transition] = []

    def add(self, t: Transition) -> None:
        """Append a transition.  Does not enforce capacity (caller manages)."""
        self.transitions.append(t)

    def is_full(self) -> bool:
        return len(self.transitions) >= self.capacity

    def clear(self) -> None:
        self.transitions = []

    def __len__(self) -> int:
        return len(self.transitions)

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_advantages(
        self,
        gamma: float,
        gae_lambda: float,
        last_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and discounted returns.

        Parameters
        ----------
        last_value:
            Bootstrap value of the state *after* the final transition
            (0.0 when the episode ended, V(s_T) when truncated).

        Returns
        -------
        advantages : np.ndarray [N]
        returns    : np.ndarray [N]   (advantages + values, used as value targets)
        """
        n = len(self.transitions)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            tr = self.transitions[t]
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 0.0 if tr.done else 1.0
            else:
                next_value = self.transitions[t + 1].value
                next_non_terminal = 0.0 if tr.done else 1.0

            delta = tr.reward + gamma * next_value * next_non_terminal - tr.value
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array([t.value for t in self.transitions], dtype=np.float32)
        return advantages, returns

    # ------------------------------------------------------------------
    # Tensor conversion
    # ------------------------------------------------------------------

    def to_tensors(self, device: str) -> Dict[str, torch.Tensor]:
        """Convert all stored transitions to batched tensors on *device*.

        Returns a dict with keys:
          board_tokens, shop_tokens, hand_tokens, opp_tokens, scalar_context,
          action_mask, actions, rewards, dones, values, log_probs
        """
        dev = torch.device(device)
        board   = np.stack([t.board_tokens   for t in self.transitions])
        shop    = np.stack([t.shop_tokens    for t in self.transitions])
        hand    = np.stack([t.hand_tokens    for t in self.transitions])
        opp     = np.stack([t.opp_tokens     for t in self.transitions])
        scalar  = np.stack([t.scalar_context for t in self.transitions])
        mask    = np.stack([t.action_mask    for t in self.transitions])
        actions = np.array([t.action   for t in self.transitions], dtype=np.int64)
        rewards = np.array([t.reward   for t in self.transitions], dtype=np.float32)
        dones   = np.array([t.done     for t in self.transitions], dtype=np.float32)
        values  = np.array([t.value    for t in self.transitions], dtype=np.float32)
        logprobs= np.array([t.log_prob for t in self.transitions], dtype=np.float32)

        return {
            "board_tokens":   torch.tensor(board,    dtype=torch.float32, device=dev),
            "shop_tokens":    torch.tensor(shop,     dtype=torch.float32, device=dev),
            "hand_tokens":    torch.tensor(hand,     dtype=torch.float32, device=dev),
            "opp_tokens":     torch.tensor(opp,      dtype=torch.float32, device=dev),
            "scalar_context": torch.tensor(scalar,   dtype=torch.float32, device=dev),
            "action_mask":    torch.tensor(mask,     dtype=torch.bool,    device=dev),
            "actions":        torch.tensor(actions,  dtype=torch.long,    device=dev),
            "rewards":        torch.tensor(rewards,  dtype=torch.float32, device=dev),
            "dones":          torch.tensor(dones,    dtype=torch.float32, device=dev),
            "values":         torch.tensor(values,   dtype=torch.float32, device=dev),
            "log_probs":      torch.tensor(logprobs, dtype=torch.float32, device=dev),
        }


# ------------------------------------------------------------------
# PPO Trainer
# ------------------------------------------------------------------

class PPOTrainer:
    """Orchestrates PPO data collection and policy updates.

    Parameters
    ----------
    policy:
        The BGPolicyNetwork to train.
    config:
        PPOConfig hyperparameters.
    """

    def __init__(self, policy: BGPolicyNetwork, config: PPOConfig) -> None:
        self.policy  = policy
        self.config  = config
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)
        self.buffer  = RolloutBuffer()
        self.total_steps  = 0
        self.update_count = 0
        self.metrics: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss":  [],
            "entropy":     [],
            "total_loss":  [],
        }

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def collect_transition(
        self,
        board_tokens:   np.ndarray,
        shop_tokens:    np.ndarray,
        hand_tokens:    np.ndarray,
        scalar_context: np.ndarray,
        action_mask:    np.ndarray,
        action:         int,
        reward:         float,
        done:           bool,
        opp_tokens:     Optional[np.ndarray] = None,
    ) -> None:
        """Build a Transition (computing value/log_prob from policy) and add it.

        Runs a single forward pass in eval mode to obtain the stored value
        and log_prob for importance-ratio computation during updates.
        """
        dev = torch.device(self.config.device)

        board_t  = torch.tensor(board_tokens[None],   dtype=torch.float32, device=dev)
        shop_t   = torch.tensor(shop_tokens[None],    dtype=torch.float32, device=dev)
        hand_t   = torch.tensor(hand_tokens[None],    dtype=torch.float32, device=dev)
        scalar_t = torch.tensor(scalar_context[None], dtype=torch.float32, device=dev)
        mask_t   = torch.tensor(action_mask[None],    dtype=torch.bool,    device=dev)
        action_t = torch.tensor([action],             dtype=torch.long,    device=dev)

        # Opponent board — zero-fill if not yet observed (round 1)
        if opp_tokens is None:
            opp_tokens = np.zeros((7, 44), dtype=np.float32)
        opp_t = torch.tensor(opp_tokens[None], dtype=torch.float32, device=dev)

        self.policy.eval()
        with torch.no_grad():
            log_probs, values, _ = self.policy.evaluate_actions(
                board_t, shop_t, hand_t, scalar_t, action_t, mask_t, opp_t,
            )
        value_f    = float(values.squeeze().item())
        log_prob_f = float(log_probs.squeeze().item())

        t = Transition(
            board_tokens=board_tokens,
            shop_tokens=shop_tokens,
            hand_tokens=hand_tokens,
            opp_tokens=opp_tokens,
            scalar_context=scalar_context,
            action_mask=action_mask,
            action=action,
            reward=reward,
            done=done,
            value=value_f,
            log_prob=log_prob_f,
        )
        self.buffer.add(t)
        self.total_steps += 1

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """Run a PPO update on the current rollout buffer.

        Parameters
        ----------
        last_value:
            Bootstrap value for GAE (0 if episode ended, V(s_T) if truncated).

        Returns
        -------
        Dict of average loss metrics for this update batch.

        Algorithm
        ---------
        1. Compute GAE advantages (normalised) and discounted returns.
        2. For n_epochs:
           a. Shuffle transitions into mini-batches of size batch_size.
           b. For each mini-batch:
              - evaluate_actions → new_log_probs, new_values, entropy
              - ratio = exp(new_log_probs - old_log_probs)
              - surr1 = ratio * adv
              - surr2 = clip(ratio, 1±ε) * adv
              - policy_loss = -mean(min(surr1, surr2))
              - value_loss  = 0.5 * mean((returns - new_values)^2)
              - entropy_loss = -mean(entropy)
              - total = policy_loss + value_coef*value_loss + entropy_coef*entropy_loss
              - backward + grad_clip + optimizer step
        3. Clear buffer, increment update_count.
        """
        if len(self.buffer) == 0:
            logger.warning("PPOTrainer.update called on empty buffer — skipping.")
            return {}

        cfg = self.config
        adv_np, ret_np = self.buffer.compute_advantages(
            cfg.gamma, cfg.gae_lambda, last_value
        )

        # Normalise advantages
        adv_mean = adv_np.mean()
        adv_std  = adv_np.std() + 1e-8
        adv_np   = (adv_np - adv_mean) / adv_std

        data  = self.buffer.to_tensors(cfg.device)
        dev   = torch.device(cfg.device)
        adv_t = torch.tensor(adv_np, dtype=torch.float32, device=dev)
        ret_t = torch.tensor(ret_np, dtype=torch.float32, device=dev)

        n = len(self.buffer)
        indices = list(range(n))

        epoch_metrics: Dict[str, List[float]] = {
            "policy_loss": [], "value_loss": [], "entropy": [], "total_loss": []
        }

        self.policy.train()
        for _ in range(cfg.n_epochs):
            random.shuffle(indices)
            for start in range(0, n, cfg.batch_size):
                batch_idx = indices[start: start + cfg.batch_size]
                if not batch_idx:
                    continue

                idx_t = torch.tensor(batch_idx, dtype=torch.long, device=dev)
                b_board   = data["board_tokens"][idx_t]
                b_shop    = data["shop_tokens"][idx_t]
                b_hand    = data["hand_tokens"][idx_t]
                b_opp     = data["opp_tokens"][idx_t]
                b_scalar  = data["scalar_context"][idx_t]
                b_mask    = data["action_mask"][idx_t]
                b_actions = data["actions"][idx_t]
                b_old_lp  = data["log_probs"][idx_t]
                b_adv     = adv_t[idx_t]
                b_ret     = ret_t[idx_t]

                new_log_probs, new_values, entropy = self.policy.evaluate_actions(
                    b_board, b_shop, b_hand, b_scalar, b_actions, b_mask, b_opp,
                )
                new_values = new_values.squeeze(-1)  # [B]

                # Importance-sampling ratio
                ratio = torch.exp(new_log_probs - b_old_lp)

                # Clipped surrogate objective
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (unclipped)
                value_loss = 0.5 * (b_ret - new_values).pow(2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                total_loss = (
                    policy_loss
                    + cfg.value_coef  * value_loss
                    + cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                epoch_metrics["policy_loss"].append(float(policy_loss.item()))
                epoch_metrics["value_loss"].append(float(value_loss.item()))
                epoch_metrics["entropy"].append(float(-entropy_loss.item()))
                epoch_metrics["total_loss"].append(float(total_loss.item()))

        self.buffer.clear()
        self.update_count += 1

        # Aggregate
        avg = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_metrics.items()}
        for k, v in avg.items():
            self.metrics[k].append(v)

        return avg

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, extra: Optional[dict] = None) -> None:
        """Save policy weights, optimizer state, and training counters."""
        payload = {
            "model_state_dict":     self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps":          self.total_steps,
            "update_count":         self.update_count,
            "config":               self.config.__dict__,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        logger.info("Checkpoint saved to %s (steps=%d)", path, self.total_steps)

    def load_checkpoint(self, path: str) -> None:
        """Load policy weights and optimizer state from a checkpoint file."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps  = checkpoint.get("total_steps", 0)
        self.update_count = checkpoint.get("update_count", 0)
        logger.info(
            "Loaded checkpoint from %s (steps=%d, updates=%d)",
            path, self.total_steps, self.update_count,
        )
