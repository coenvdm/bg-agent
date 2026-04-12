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

from agent.policy import BGPolicyNetwork, N_ACTION_TYPES, POINTER_DIM

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
    entropy_coef: float = 0.05 # entropy bonus coefficient
    max_grad_norm: float = 0.5 # gradient clipping norm
    n_epochs: int = 4          # PPO update epochs per rollout (KL early-stop may cut short)
    target_kl: float = 0.02    # KL divergence threshold for early stopping epochs
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
    scalar_context: np.ndarray   # [98]
    type_mask:      np.ndarray   # [8]  bool — valid action types
    pointer_mask:   np.ndarray   # [24] bool — valid pointer slots (zone+occupancy)
    type_action:    int          # 0-7
    ptr_action:     int          # 0-23 or -1 for non-pointer types
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
          type_mask, pointer_mask, type_actions, ptr_actions,
          rewards, dones, values, log_probs
        """
        dev = torch.device(device)
        board    = np.stack([t.board_tokens   for t in self.transitions])
        shop     = np.stack([t.shop_tokens    for t in self.transitions])
        hand     = np.stack([t.hand_tokens    for t in self.transitions])
        opp      = np.stack([t.opp_tokens     for t in self.transitions])
        scalar   = np.stack([t.scalar_context for t in self.transitions])
        t_mask   = np.stack([t.type_mask      for t in self.transitions])
        p_mask   = np.stack([t.pointer_mask   for t in self.transitions])
        t_acts   = np.array([t.type_action    for t in self.transitions], dtype=np.int64)
        p_acts   = np.array([t.ptr_action     for t in self.transitions], dtype=np.int64)
        rewards  = np.array([t.reward         for t in self.transitions], dtype=np.float32)
        dones    = np.array([t.done           for t in self.transitions], dtype=np.float32)
        values   = np.array([t.value          for t in self.transitions], dtype=np.float32)
        logprobs = np.array([t.log_prob       for t in self.transitions], dtype=np.float32)

        return {
            "board_tokens":   torch.tensor(board,    dtype=torch.float32, device=dev),
            "shop_tokens":    torch.tensor(shop,     dtype=torch.float32, device=dev),
            "hand_tokens":    torch.tensor(hand,     dtype=torch.float32, device=dev),
            "opp_tokens":     torch.tensor(opp,      dtype=torch.float32, device=dev),
            "scalar_context": torch.tensor(scalar,   dtype=torch.float32, device=dev),
            "type_mask":      torch.tensor(t_mask,   dtype=torch.bool,    device=dev),
            "pointer_mask":   torch.tensor(p_mask,   dtype=torch.bool,    device=dev),
            "type_actions":   torch.tensor(t_acts,   dtype=torch.long,    device=dev),
            "ptr_actions":    torch.tensor(p_acts,   dtype=torch.long,    device=dev),
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
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr, weight_decay=1e-4)
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
        type_action:    int,
        ptr_action:     int,
        type_mask:      np.ndarray,     # [8]  bool
        pointer_mask:   np.ndarray,     # [24] bool — zone+occupancy for this type
        reward:         float,
        done:           bool,
        opp_tokens:     Optional[np.ndarray] = None,
    ) -> None:
        """Build a Transition (computing value/log_prob from policy) and add it.

        Runs a single forward pass in eval mode to obtain the stored value
        and log_prob for importance-ratio computation during updates.
        """
        dev = torch.device(self.config.device)

        board_t    = torch.tensor(board_tokens[None],   dtype=torch.float32, device=dev)
        shop_t     = torch.tensor(shop_tokens[None],    dtype=torch.float32, device=dev)
        hand_t     = torch.tensor(hand_tokens[None],    dtype=torch.float32, device=dev)
        scalar_t   = torch.tensor(scalar_context[None], dtype=torch.float32, device=dev)
        t_mask_t   = torch.tensor(type_mask[None],      dtype=torch.bool,    device=dev)
        p_mask_t   = torch.tensor(pointer_mask[None],   dtype=torch.bool,    device=dev)
        t_action_t = torch.tensor([type_action],        dtype=torch.long,    device=dev)
        p_action_t = torch.tensor([ptr_action],         dtype=torch.long,    device=dev)

        if opp_tokens is None:
            opp_tokens = np.zeros((7, 44), dtype=np.float32)
        opp_t = torch.tensor(opp_tokens[None], dtype=torch.float32, device=dev)

        self.policy.eval()
        with torch.no_grad():
            log_probs, values, _ = self.policy.evaluate_actions(
                board_t, shop_t, hand_t, scalar_t,
                t_action_t, p_action_t, t_mask_t, p_mask_t, opp_t,
            )
        value_f    = float(values.squeeze().item())
        log_prob_f = float(log_probs.squeeze().item())

        t = Transition(
            board_tokens=board_tokens,
            shop_tokens=shop_tokens,
            hand_tokens=hand_tokens,
            opp_tokens=opp_tokens,
            scalar_context=scalar_context,
            type_mask=type_mask,
            pointer_mask=pointer_mask,
            type_action=type_action,
            ptr_action=ptr_action,
            reward=reward,
            done=done,
            value=value_f,
            log_prob=log_prob_f,
        )
        self.buffer.add(t)
        self.total_steps += 1

    def store_transition(
        self,
        board_tokens:   np.ndarray,
        shop_tokens:    np.ndarray,
        hand_tokens:    np.ndarray,
        scalar_context: np.ndarray,
        type_action:    int,
        ptr_action:     int,
        type_mask:      np.ndarray,
        pointer_mask:   np.ndarray,
        reward:         float,
        done:           bool,
        log_prob:       float,
        value:          float,
        opp_tokens:     Optional[np.ndarray] = None,
    ) -> None:
        """Store a transition with pre-computed log_prob and value.

        Skips the evaluate_actions() forward pass — use this when log_prob and
        value were already computed by get_action_batch() to avoid redundant
        inference.
        """
        if opp_tokens is None:
            opp_tokens = np.zeros((7, 44), dtype=np.float32)
        t = Transition(
            board_tokens=board_tokens,
            shop_tokens=shop_tokens,
            hand_tokens=hand_tokens,
            opp_tokens=opp_tokens,
            scalar_context=scalar_context,
            type_mask=type_mask,
            pointer_mask=pointer_mask,
            type_action=type_action,
            ptr_action=ptr_action,
            reward=reward,
            done=done,
            value=value,
            log_prob=log_prob,
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

        # Clip returns before value-function fitting to prevent loss spikes when
        # the reward distribution shifts (e.g. new shaping terms added mid-run).
        ret_np = np.clip(ret_np, -10.0, 10.0)

        # Normalise advantages
        adv_mean = adv_np.mean()
        adv_std  = adv_np.std() + 1e-8
        adv_np   = (adv_np - adv_mean) / adv_std

        # Normalise returns so value-function targets stay on a consistent unit
        # scale across updates (same idea as advantage normalisation).
        ret_mean = ret_np.mean()
        ret_std  = ret_np.std() + 1e-8
        ret_np   = (ret_np - ret_mean) / ret_std

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
        for epoch_i in range(cfg.n_epochs):
            epoch_kl = 0.0
            random.shuffle(indices)
            for start in range(0, n, cfg.batch_size):
                batch_idx = indices[start: start + cfg.batch_size]
                if not batch_idx:
                    continue

                idx_t = torch.tensor(batch_idx, dtype=torch.long, device=dev)
                b_board    = data["board_tokens"][idx_t]
                b_shop     = data["shop_tokens"][idx_t]
                b_hand     = data["hand_tokens"][idx_t]
                b_opp      = data["opp_tokens"][idx_t]
                b_scalar   = data["scalar_context"][idx_t]
                b_t_mask   = data["type_mask"][idx_t]
                b_p_mask   = data["pointer_mask"][idx_t]
                b_t_acts   = data["type_actions"][idx_t]
                b_p_acts   = data["ptr_actions"][idx_t]
                b_old_lp   = data["log_probs"][idx_t]
                b_adv      = adv_t[idx_t]
                b_ret      = ret_t[idx_t]

                # Evaluate in eval mode (dropout off) so new_log_probs are
                # directly comparable to old_log_probs, which were also computed
                # in eval mode during collection.  Gradients still flow normally
                # through an eval-mode forward pass.
                self.policy.eval()
                new_log_probs, new_values, entropy = self.policy.evaluate_actions(
                    b_board, b_shop, b_hand, b_scalar,
                    b_t_acts, b_p_acts, b_t_mask, b_p_mask, b_opp,
                )
                self.policy.train()
                new_values = new_values.squeeze(-1)  # [B]

                # Skip only on true NaN (not -inf: ratio=exp(-inf)=0 is handled fine
                # by the clipped surrogate and does not propagate NaN to the loss)
                if torch.isnan(new_log_probs).any() or torch.isnan(new_values).any():
                    logger.warning("NaN detected in evaluate_actions — skipping mini-batch")
                    continue

                # Importance-sampling ratio — clamp log_probs to avoid exp(+inf)
                # when old_log_prob is -inf (stale near-zero-prob transitions)
                new_log_probs_c = new_log_probs.clamp(min=-20.0)
                b_old_lp_c      = b_old_lp.clamp(min=-20.0)
                ratio = torch.exp(new_log_probs_c - b_old_lp_c)

                # KL early stopping: approximate KL(old||new) = mean(old_lp - new_lp).
                # If the policy has drifted too far from the data, stop updating —
                # further gradient steps would be on stale importance weights.
                with torch.no_grad():
                    approx_kl = (b_old_lp_c - new_log_probs_c).mean().item()
                epoch_kl = max(epoch_kl, approx_kl)
                if approx_kl > cfg.target_kl:
                    break  # stop this epoch early

                # Clipped surrogate objective
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss — plain MSE (no value clipping).
                value_loss = 0.5 * (new_values - b_ret).pow(2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                total_loss = (
                    policy_loss
                    + cfg.value_coef  * value_loss
                    + cfg.entropy_coef * entropy_loss
                )

                if not torch.isfinite(total_loss):
                    logger.warning("Abnormal loss %.3e — skipping mini-batch", total_loss.item())
                    continue

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                epoch_metrics["policy_loss"].append(float(policy_loss.item()))
                epoch_metrics["value_loss"].append(float(value_loss.item()))
                epoch_metrics["entropy"].append(float(-entropy_loss.item()))
                epoch_metrics["total_loss"].append(float(total_loss.item()))

            if epoch_kl > cfg.target_kl:
                logger.debug("KL early stop at epoch %d (kl=%.4f)", epoch_i, epoch_kl)
                break  # stop remaining epochs too

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
        """Load policy weights and optimizer state from a checkpoint file.

        If the checkpoint was saved with an incompatible architecture (e.g. an
        older policy_head layout), logs a warning and skips loading rather than
        crashing — training will start from scratch / BC init instead.
        """
        checkpoint = torch.load(path, map_location=self.config.device)
        try:
            self.policy.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as exc:
            logger.warning(
                "Checkpoint at '%s' is incompatible with the current architecture "
                "and will be ignored: %s", path, exc
            )
            return
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps  = checkpoint.get("total_steps", 0)
        self.update_count = checkpoint.get("update_count", 0)
        logger.info(
            "Loaded checkpoint from %s (steps=%d, updates=%d)",
            path, self.total_steps, self.update_count,
        )
