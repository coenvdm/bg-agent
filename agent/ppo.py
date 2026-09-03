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
from typing import Any, Dict, List, Optional, Tuple

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

    lr: float = 2.5e-4
    lr_final: float = 5e-5     # lr linearly anneals lr -> lr_final over anneal_steps
    gamma: float = 0.997       # discount factor — long episodes (~120 steps) need high gamma
    gae_lambda: float = 0.95   # GAE λ
    clip_eps: float = 0.2      # PPO clip epsilon
    value_coef: float = 0.5    # value loss coefficient
    entropy_coef: float = 0.015        # entropy bonus coefficient (start of anneal)
    entropy_coef_final: float = 0.004  # entropy bonus coefficient (end of anneal)
    anneal_steps: int = 4_000_000  # total_steps at which lr/entropy reach their *_final values
    max_grad_norm: float = 0.5 # gradient clipping norm
    n_epochs: int = 4          # PPO update epochs per rollout (KL early-stop may cut epochs
                                # short BETWEEN epochs only — epoch 0 always runs in full)
    target_kl: float = 0.03    # KL divergence threshold for early stopping between epochs
    batch_size: int = 256
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
    scalar_context: np.ndarray   # [100]
    type_mask:      np.ndarray   # [N_ACTION_TYPES]  bool — valid action types
    pointer_mask:   np.ndarray   # [24] bool — valid pointer slots (zone+occupancy)
    type_action:    int          # 0-8
    ptr_action:     int          # 0-23 or -1 for non-pointer types
    reward:         float
    done:           bool
    value:          float        # stored for GAE bootstrap
    log_prob:       float        # stored for importance-sampling ratio
    traj_id:        Any = None   # identifies which (game, player) trajectory this
                                  # transition belongs to — see compute_advantages.
                                  # None means "caller didn't tag it", which is only
                                  # safe when the whole buffer really is one trajectory.
    round_num:      Optional[int] = None  # ps.round_num at the time of this action —
                                  # metadata only, like traj_id: not consumed by the
                                  # network, just carried along so training scripts can
                                  # break the action mix down by game round.


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

        The buffer is not one trajectory — it's however many (game, player)
        rollouts got merged into it (multiple training players per game, and
        multiple games per update batch). Transitions are grouped by
        traj_id and GAE is run independently within each group, using the
        group's own chronological order, so the recursion never bootstraps
        off a neighbouring transition that happens to belong to a different
        player or game. (Previously this walked the flat buffer as if
        transitions[t+1] were always "what happens after t", which silently
        pulled in an unrelated trajectory's value at every player/game
        boundary — see CONTEXT.md 2026-08-31 for the diagnosis.)

        Parameters
        ----------
        last_value:
            Bootstrap value for a genuinely truncated rollout — applied only
            to whichever trajectory contains the chronologically-last
            transition in the buffer, and only if that transition isn't
            already done=True (0.0 when the episode ended, V(s_T) when
            truncated mid-episode).

        Returns
        -------
        advantages : np.ndarray [N]
        returns    : np.ndarray [N]   (advantages + values, used as value targets)
        """
        n = len(self.transitions)
        advantages = np.zeros(n, dtype=np.float32)

        groups: Dict[Any, List[int]] = {}
        for i, tr in enumerate(self.transitions):
            groups.setdefault(tr.traj_id, []).append(i)

        for idxs in groups.values():
            last_gae = 0.0
            for k in reversed(range(len(idxs))):
                i  = idxs[k]
                tr = self.transitions[i]
                if k == len(idxs) - 1:
                    # Tail of this trajectory's segment in the buffer. Only the
                    # segment containing the buffer's actual last transition
                    # gets the caller's last_value; every other segment's tail
                    # should already be done=True (elimination/game-end), which
                    # zeroes next_non_terminal regardless of next_value.
                    next_value = last_value if i == n - 1 else 0.0
                    next_non_terminal = 0.0 if tr.done else 1.0
                else:
                    next_value = self.transitions[idxs[k + 1]].value
                    next_non_terminal = 0.0 if tr.done else 1.0

                delta = tr.reward + gamma * next_value * next_non_terminal - tr.value
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
                advantages[i] = last_gae

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

    # Number of raw-return samples that must be folded into ret_rms before
    # ret_std trusts its own estimate over the fresh-start default of 1.0.
    _RET_RMS_MIN_COUNT: float = 10.0

    def __init__(self, policy: BGPolicyNetwork, config: PPOConfig) -> None:
        self.policy  = policy
        self.config  = config
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr, weight_decay=1e-4)
        self.buffer  = RolloutBuffer()
        self.total_steps  = 0
        self.update_count = 0

        # ── FIX 3: persistent running RETURN scale (not per-batch return
        # normalisation) — see ret_std / _update_ret_rms / update() for the
        # full convention. This is a running second moment of raw returns
        # about ZERO, deliberately never mean-centred: the value head's
        # zero point must stay pinned to raw reward units so GAE deltas
        # (which mix stored values with raw rewards) stay on one consistent
        # scale for the whole run, not a different offset/scale every update.
        self.ret_rms_var: float = 1.0
        self.ret_rms_count: float = 0.0

        self.metrics: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss":  [],
            "entropy":     [],
            "total_loss":  [],
        }

    # ------------------------------------------------------------------
    # Return-scale (ret_std) bookkeeping — FIX 3
    # ------------------------------------------------------------------

    @property
    def ret_std(self) -> float:
        """Current running scale of the raw return distribution.

        This is an RMS (root of a second moment about zero), NOT a
        mean-centred std — no mean is ever subtracted, so a value of 1.0
        (raw units) stays a fixed, stable reference point across the whole
        run rather than drifting with every update like the old per-batch
        return normalisation did.

        Floored at 1e-4 to avoid dividing by ~0, and pinned to 1.0 until at
        least `_RET_RMS_MIN_COUNT` return samples have been folded in, so
        early updates (where the running estimate is itself noisy) don't
        divide value targets by a near-arbitrary number.

        Used to convert between the value head's native output units
        ("per current ret_std at the time of that forward pass") and RAW
        reward units: collect_transition/store_transition multiply the
        network's raw output by this before storing into the buffer, and
        update() divides GAE-derived raw returns by this to form value
        targets. Any external caller computing `last_value` for update()
        directly from get_action/get_action_batch must apply the same
        multiplication first.
        """
        if self.ret_rms_count < self._RET_RMS_MIN_COUNT:
            return 1.0
        return max(1e-4, float(np.sqrt(max(self.ret_rms_var, 0.0))))

    def _update_ret_rms(self, returns: np.ndarray) -> None:
        """Fold this batch's raw returns into the running second-moment estimate.

        Combines the running E[return^2] with this batch's E[return^2] via a
        count-weighted average (a Welford-style parallel combination, minus
        the mean-tracking half — see the ret_std docstring for why no mean is
        subtracted). Must be called with RAW-scale returns, before those
        returns are divided by ret_std to form value targets.
        """
        batch_count = returns.size
        if batch_count == 0:
            return
        batch_second_moment = float(np.mean(np.square(returns)))
        total_count = self.ret_rms_count + batch_count
        self.ret_rms_var = (
            self.ret_rms_var * self.ret_rms_count + batch_second_moment * batch_count
        ) / total_count
        self.ret_rms_count = total_count

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
        type_mask:      np.ndarray,     # [N_ACTION_TYPES]  bool
        pointer_mask:   np.ndarray,     # [24] bool — zone+occupancy for this type
        reward:         float,
        done:           bool,
        opp_tokens:     Optional[np.ndarray] = None,
        traj_id:        Any = None,
        round_num:      Optional[int] = None,
    ) -> None:
        """Build a Transition (computing value/log_prob from policy) and add it.

        Runs a single forward pass in eval mode to obtain the stored value
        and log_prob for importance-ratio computation during updates.

        traj_id should uniquely identify the (game, player) this transition
        came from whenever more than one trajectory may share this buffer
        (e.g. multiple training players per game, or multiple games merged
        before an update) — see RolloutBuffer.compute_advantages.
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
        # FIX 3: the value head's raw output is in units of the CURRENT
        # ret_std, not raw reward units. Multiply back to raw units here so
        # everything stored in the buffer — and the GAE deltas in
        # compute_advantages, which mix tr.value with tr.reward — stays on
        # one consistent scale for the whole run. See PPOTrainer.ret_std.
        value_f    = float(values.squeeze().item()) * self.ret_std
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
            traj_id=traj_id,
            round_num=round_num,
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
        traj_id:        Any = None,
        round_num:      Optional[int] = None,
    ) -> None:
        """Store a transition with pre-computed log_prob and value.

        Skips the evaluate_actions() forward pass — use this when log_prob and
        value were already computed by get_action_batch() to avoid redundant
        inference. See collect_transition for traj_id semantics.

        `value` must be the RAW network value-head output — exactly what
        get_action / get_action_batch return, i.e. NOT yet rescaled. This
        method applies the same ret_std -> raw-reward-units conversion that
        collect_transition applies internally (see FIX 3 / PPOTrainer.ret_std),
        so Transition.value is on a consistent raw-reward scale no matter
        which of the two entry points populated it.
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
            value=value * self.ret_std,
            log_prob=log_prob,
            traj_id=traj_id,
            round_num=round_num,
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
            Bootstrap value for GAE, in RAW reward units (0 if episode ended,
            V(s_T) if truncated) — the same convention as Transition.value
            (see collect_transition / ret_std). If this was obtained directly
            from get_action / get_action_batch, multiply it by `self.ret_std`
            first: those methods return the value head's native units, not
            raw reward units.

        Returns
        -------
        Dict of average metrics for this update batch:
        policy_loss, value_loss, entropy, total_loss, n_minibatches,
        approx_kl, clip_frac, explained_var, lr, entropy_coef, ret_std.

        Algorithm
        ---------
        1. Compute GAE advantages and RAW-scale discounted returns.
        2. Update the persistent running return-scale (ret_std) from this
           batch's raw returns, then divide returns by it to form value
           targets — see the FIX 3 comment below for why this replaced
           per-batch return mean/std normalisation.
        3. Normalise advantages (mean 0 / std 1) — this one stays per-batch
           and scale-free; it has no persistent-units meaning to preserve.
        4. Linearly anneal lr and entropy_coef from cfg.*/cfg.*_final based
           on self.total_steps / cfg.anneal_steps.
        5. For n_epochs (KL is checked only BETWEEN epochs — epoch 0 always
           runs to completion, so an update can never apply zero gradient
           steps purely because of the KL guard):
           a. Shuffle transitions into mini-batches of size batch_size.
           b. For each mini-batch:
              - evaluate_actions → new_log_probs, new_values, entropy
              - ratio = exp(new_log_probs - old_log_probs)
              - surr1 = ratio * adv
              - surr2 = clip(ratio, 1±ε) * adv
              - policy_loss = -mean(min(surr1, surr2))
              - value_loss  = 0.5 * mean((value_targets - new_values)^2)
              - entropy_loss = -mean(entropy)
              - total = policy_loss + value_coef*value_loss + entropy_coef*entropy_loss
              - backward + grad_clip + optimizer step
        6. Clear buffer, increment update_count.
        """
        if len(self.buffer) == 0:
            logger.warning("PPOTrainer.update called on empty buffer — skipping.")
            return {}

        cfg = self.config

        # ── FIX 4: linear lr / entropy_coef annealing ───────────────────────
        # progress saturates at 1.0 once total_steps reaches anneal_steps and
        # stays there, so lr/entropy_coef hold at their *_final values for the
        # rest of training instead of extrapolating past the schedule.
        progress = min(1.0, self.total_steps / cfg.anneal_steps)
        lr = cfg.lr + (cfg.lr_final - cfg.lr) * progress
        entropy_coef = cfg.entropy_coef + (cfg.entropy_coef_final - cfg.entropy_coef) * progress
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        adv_np, ret_np = self.buffer.compute_advantages(
            cfg.gamma, cfg.gae_lambda, last_value
        )

        # ── FIX 3: persistent running return scale, not per-batch return
        # normalisation ─────────────────────────────────────────────────────
        # The OLD code re-centred and re-scaled `ret_np` to mean 0 / std 1 on
        # EVERY update — a different offset and scale each time — while
        # compute_advantages' `delta = reward + gamma*next_value - value` mixed
        # those normalised values with RAW rewards. The baseline was never on
        # the same scale as what it was meant to baseline, so advantages were
        # close to noise and the value function chased a moving target.
        #
        # Fix: track ONE running scale (`ret_std`, no mean-subtraction so the
        # zero point stays pinned to raw reward units — see the ret_std
        # docstring) across the whole run. Transition.value is now stored in
        # RAW reward units (collect_transition/store_transition multiply the
        # network's raw output by ret_std before storing). Here we only
        # divide by ret_std to form the value head's *targets* for this batch.
        self._update_ret_rms(ret_np)
        running_std = self.ret_std
        ret_target_np = ret_np / running_std

        # Normalise advantages — standard, scale-free PPO practice; unlike
        # returns this has no persistent "raw units" meaning to preserve.
        adv_mean = adv_np.mean()
        adv_std  = adv_np.std() + 1e-8
        adv_np   = (adv_np - adv_mean) / adv_std

        data  = self.buffer.to_tensors(cfg.device)
        dev   = torch.device(cfg.device)
        adv_t = torch.tensor(adv_np, dtype=torch.float32, device=dev)
        ret_t = torch.tensor(ret_target_np, dtype=torch.float32, device=dev)

        # explained_var is computed once, on RAW-scale returns/values for the
        # whole batch (not the ret_std-scaled value targets) — it answers
        # "how good is the value function in the units the game actually
        # cares about," which the ret_std scaling would otherwise obscure.
        values_np = data["values"].cpu().numpy()
        var_ret = float(np.var(ret_np))
        if var_ret < 1e-8:
            explained_var = float("nan")
        else:
            explained_var = float(1.0 - np.var(ret_np - values_np) / var_ret)

        n = len(self.buffer)
        indices = list(range(n))

        epoch_metrics: Dict[str, List[float]] = {
            "policy_loss": [], "value_loss": [], "entropy": [], "total_loss": [],
            "approx_kl": [], "clip_frac": [],
        }

        self.policy.train()
        for epoch_i in range(cfg.n_epochs):
            epoch_kls: List[float] = []
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
                logratio = new_log_probs_c - b_old_lp_c
                ratio = logratio.exp()

                # KL estimator — Schulman's k3 (http://joschu.net/blog/kl-approx.html):
                # approx_kl = E[(ratio - 1) - logratio], a low-variance, non-negative
                # estimator of KL(old||new). The old signed estimator
                # mean(old_lp - new_lp) was noisy enough that, combined with only
                # ~3 minibatches/update and a mid-epoch break, a single unlucky
                # minibatch could trip early stopping and discard the rest of the
                # epoch's gradient steps (measured: 8/264 updates applied zero
                # gradient steps in a 312-update run).
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()
                epoch_kls.append(float(approx_kl.item()))

                # Clipped surrogate objective
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss — plain MSE against ret_std-scaled targets (no value clipping).
                value_loss = 0.5 * (new_values - b_ret).pow(2).mean()

                # Entropy bonus — entropy_coef is this update's annealed value (FIX 4)
                entropy_loss = -entropy.mean()

                total_loss = (
                    policy_loss
                    + cfg.value_coef  * value_loss
                    + entropy_coef * entropy_loss
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
                epoch_metrics["approx_kl"].append(float(approx_kl.item()))
                epoch_metrics["clip_frac"].append(float(clip_frac.item()))

            # KL is checked only BETWEEN epochs now, never mid-epoch (FIX 2).
            # The old code broke out of the minibatch loop the instant one
            # minibatch's signed KL estimate exceeded target_kl; with only a
            # few minibatches per update that could discard an entire epoch's
            # remaining gradient steps on one noisy sample, or even abort the
            # whole update before the first epoch finished. Epoch 0 always
            # runs to completion regardless of KL, so this guard alone can
            # never make an update apply zero gradient steps.
            if epoch_kls:
                mean_epoch_kl = float(np.mean(epoch_kls))
                if mean_epoch_kl > cfg.target_kl:
                    logger.debug(
                        "KL early stop after epoch %d (kl=%.4f)", epoch_i, mean_epoch_kl
                    )
                    break  # stop remaining epochs

        self.buffer.clear()
        self.update_count += 1

        # Aggregate. If every mini-batch got skipped this update (the NaN /
        # abnormal-loss guard fired on all of them), no gradient step happened
        # at all: this batch of collected data was discarded with zero
        # training on it. Report NaN rather than 0.0 for that case -- 0.0
        # previously looked like an (impossibly) perfect loss instead of "no
        # update happened", which silently hid how often this was occurring.
        # (KL can no longer cause this on its own — see the epoch-boundary
        # comment above.)
        n_minibatches = len(epoch_metrics["total_loss"])
        if n_minibatches == 0:
            logger.warning(
                "PPO update %d: every mini-batch skipped (NaN guard) -- "
                "buffer discarded with zero gradient steps applied.",
                self.update_count,
            )
        avg = {k: float(np.mean(v)) if v else float("nan") for k, v in epoch_metrics.items()}
        avg["n_minibatches"]  = n_minibatches
        avg["explained_var"]  = explained_var
        avg["lr"]             = lr
        avg["entropy_coef"]   = entropy_coef
        avg["ret_std"]        = running_std
        for k, v in avg.items():
            self.metrics.setdefault(k, []).append(v)

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
            # FIX 3: persist the running return-scale so a resumed run keeps
            # a consistent value-target scale instead of resetting to 1.0 and
            # re-estimating from scratch (which would look like a value-loss
            # spike right after every resume).
            "ret_rms_var":          self.ret_rms_var,
            "ret_rms_count":        self.ret_rms_count,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        logger.info("Checkpoint saved to %s (steps=%d)", path, self.total_steps)

    def load_checkpoint(self, path: str) -> bool:
        """Load policy weights and optimizer state from a checkpoint file.

        If the checkpoint was saved with an incompatible architecture (e.g. an
        older policy_head layout, or a type_head sized for a different action
        space — such as any checkpoint saved before the "activate" action type
        was added), logs a warning and skips loading rather than crashing —
        training will start from scratch / BC init instead.

        Returns True if weights were actually loaded, False if skipped due to
        incompatibility — callers should not report a checkpoint as "resumed"
        without checking this, since total_steps/update_count are also left
        untouched (at their pre-call values) when this returns False.
        """
        checkpoint = torch.load(path, map_location=self.config.device)
        model_state = checkpoint["model_state_dict"]

        # ------------------------------------------------------------------
        # MIGRATION: seed `activate_scorer` from `sell_scorer` for
        # pre-activate_scorer checkpoints.
        #
        # `BGPolicyNetwork` gained a new `self.activate_scorer` head (the
        # ACTIVATE(board_idx) action, action type 8 — see the Action Space
        # section of CLAUDE.md) built identically to the pre-existing
        # `self.sell_scorer`. Every checkpoint saved before that change —
        # including the live run's `bg_agent_ppo.pt` at ~3167 updates /
        # 28.8M steps — has a `model_state_dict` with no `activate_scorer.*`
        # keys at all. Loaded with plain `strict=True`, that raises
        # RuntimeError for the missing keys, which the `except RuntimeError`
        # below would treat as "incompatible checkpoint" and discard —
        # silently restarting training from scratch and losing those 28.8M
        # steps. That must not happen.
        #
        # Instead: if the *current* model has `activate_scorer.*` parameters
        # that the checkpoint lacks, and the checkpoint has the
        # corresponding `sell_scorer.*` tensors to seed from, synthesise the
        # missing keys by cloning sell_scorer's weights into activate_scorer
        # (derived by string-replacing the prefix, not a hardcoded layer
        # shape — this keeps the migration correct even if the head's
        # internal layout changes later). Both heads score the same board
        # tokens with the same shape by construction, so this makes the
        # migrated state_dict bit-identical in behaviour to the pre-change
        # checkpoint: ACTIVATE and SELL logits coincide at the instant of
        # resume, and the two heads only differentiate as training proceeds
        # from there. `strict=True` is kept on the actual load below — after
        # this migration all keys should be present, and strictness is what
        # catches any OTHER genuine incompatibility.
        #
        # This block only fires when there is something safe to seed from;
        # if `sell_scorer.*` isn't present either, it falls through and lets
        # the strict load raise, hitting the existing incompatible-checkpoint
        # path rather than inventing weights from nothing.
        #
        # Safe to delete once no pre-activate_scorer checkpoints remain in
        # use anywhere (i.e. every checkpoint anyone might resume from was
        # saved after this migration existed).
        # ------------------------------------------------------------------
        current_keys = self.policy.state_dict().keys()
        missing_activate_keys = sorted(
            k for k in current_keys
            if k.startswith("activate_scorer.") and k not in model_state
        )
        if missing_activate_keys:
            seeded = {}
            for key in missing_activate_keys:
                source_key = key.replace("activate_scorer", "sell_scorer", 1)
                if source_key in model_state:
                    seeded[key] = model_state[source_key].clone()
            if seeded:
                # Don't mutate checkpoint["model_state_dict"] in place — work
                # on a shallow copy so `checkpoint` still reflects the file
                # on disk if anything downstream inspects it.
                model_state = dict(model_state)
                model_state.update(seeded)
                logger.info(
                    "Checkpoint at '%s' predates the activate_scorer head; "
                    "seeded %d key(s) from sell_scorer so behaviour is "
                    "preserved on resume: %s",
                    path, len(seeded), sorted(seeded.keys()),
                )
            # else: nothing to seed from (checkpoint doesn't have
            # sell_scorer.* either) — fall through to the strict load below,
            # which will raise and be handled by the incompatible-checkpoint
            # path rather than us guessing weights.

        try:
            self.policy.load_state_dict(model_state, strict=True)
        except RuntimeError as exc:
            logger.warning(
                "Checkpoint at '%s' is incompatible with the current architecture "
                "and will be ignored: %s", path, exc
            )
            return False
        if "optimizer_state_dict" in checkpoint:
            # AdamW is built over policy.parameters(); adding activate_scorer
            # changes that parameter list's size/shape, so loading an old
            # optimizer_state_dict here can raise (typically ValueError:
            # "loaded state dict contains a parameter group that doesn't
            # match the size of the optimizer's group"). This must never
            # abort the resume: losing Adam's per-parameter moment estimates
            # costs at most a few dozen updates of noisier gradient steps
            # while they re-accumulate from zero, but failing the resume
            # outright costs the entire 28.8M steps of training this
            # checkpoint represents. Catch broadly and keep going with
            # freshly-initialised optimizer state — weights, total_steps,
            # and update_count below are unaffected either way.
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as exc:
                logger.warning(
                    "Optimizer state in checkpoint '%s' is incompatible with "
                    "the current parameter set (%s: %s). Adam moment "
                    "estimates reset for all parameters; expect a brief "
                    "transient while they re-accumulate. Model weights, "
                    "total_steps, and update_count were still loaded "
                    "successfully.", path, type(exc).__name__, exc,
                )
        self.total_steps  = checkpoint.get("total_steps", 0)
        self.update_count = checkpoint.get("update_count", 0)
        # FIX 3: restore the running return-scale; tolerate older checkpoints
        # saved before this existed by falling back to the fresh-start values
        # (ret_std then reads as 1.0 until enough new samples accumulate).
        self.ret_rms_var   = checkpoint.get("ret_rms_var", 1.0)
        self.ret_rms_count = checkpoint.get("ret_rms_count", 0.0)
        logger.info(
            "Loaded checkpoint from %s (steps=%d, updates=%d)",
            path, self.total_steps, self.update_count,
        )
        return True
