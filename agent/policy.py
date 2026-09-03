"""
BGPolicyNetwork — Two-headed Transformer policy + value network.

Architecture:
  - Project 44-dim card tokens to d_model
  - Zone embedding: 0=board, 1=shop, 2=hand, 3=opponent_board
  - CLS token prepended
  - Single Transformer encoder over [CLS + 7 board + 7 shop + 10 hand + 7 opp] = 32 tokens
  - type_head:    CLS + scalar_context → 9 action-type logits
  - pointer_head: per-token scorers (sell/buy/place/activate) acting directly on
                  Transformer outputs for board/shop/hand tokens → 24 card-pointer
                  logits, returned as TWO parallel [B, 24] tensors (pointer_logits
                  and activate_pointer_logits) that agree everywhere except the
                  board slice -- see forward()'s docstring for why ACTIVATE needs
                  a scorer separate from SELL/REORDER's.
  - value_head:   CLS + scalar_context → scalar value

Action types (10; 0-7 match the original BGPolicyV2 BC model, 8-9 added later):
  0 buy        → pointer: shop  slot  [PTR_SHOP_OFF  + i,  i in 0-6]
  1 sell       → pointer: board slot  [PTR_BOARD_OFF + i,  i in 0-6]
  2 place      → pointer: hand  slot  [PTR_HAND_OFF  + i,  i in 0-9]
  3 reroll     → no pointer
  4 freeze     → no pointer
  5 level_up   → no pointer
  6 hero_power → no pointer
  7 end_turn   → no pointer
  8 activate   → pointer: board slot  [PTR_BOARD_OFF + i,  i in 0-6] (minion's
                 own "Activate (N)" ability; shares the board ZONE with SELL
                 since the target is always the activating minion itself, but
                 is scored by its own activate_scorer, not sell_scorer -- see
                 forward()'s docstring)
  9 reorder    → pointer: board slot  [PTR_BOARD_OFF + i,  i in 1-6] (move that
                 minion to the FRONT of the board). Slot 0 is never valid, so
                 every REORDER is a real state change and can never be used as
                 a free no-op to stall the discount (the exact exploit ACTIVATE
                 enabled before its mask was fixed -- see CONTEXT.md
                 2026-09-01). Move-to-front composes: any permutation of n
                 minions is reachable in at most n-1 of these, so a single
                 board-slot pointer is fully expressive and no second
                 ("to where?") pointer head is needed.

Pointer layout (24, matching BGPolicyV2 BC model):
  [0-6]   shop  slots
  [7-13]  board slots
  [14-23] hand  slots

scalar_context layout (94 dims):
  [0:24]  own board features (SymbolicBoardComputer.to_scalar_vector)
  [24:88] all-opponent features: 8 × 8 dims, indexed by player_id (own slot zeroed)
          each 8-dim block: tier/7, health/40, armor/10, board_size/7,
          dominant_tribe_count/7, is_synergistic, rounds_since_seen/10, health_delta/40
  [88:94] lobby-wide features: num_alive/8, mean_opp_tier/7, mean_opp_health/40,
          num_synergistic_boards/7, health_rank/8, tier_rank/8
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Input dimensions ──────────────────────────────────────────────────────────
CARD_DIM   = 44
SCALAR_DIM = 100  # 24 own-board + 64 all-opponents (8×8, own slot zeroed) + 6 lobby + 6 economy

# ── Action type space (0-7 match BGPolicyV2; 8=activate, 9=reorder added later) ──
N_ACTION_TYPES    = 10
ACTION_TYPE_NAMES = ["buy", "sell", "place", "reroll", "freeze",
                     "level_up", "hero_power", "end_turn", "activate", "reorder"]

# Types that require a card pointer; all others use ptr_idx = -1
TYPES_WITH_POINTER = frozenset({0, 1, 2, 8, 9})  # buy, sell, place, activate, reorder

# Max REORDER actions per shopping turn. 6 is exactly what a full 7-minion
# board needs to reach ANY permutation via move-to-front (n-1), so the budget
# never blocks a reachable arrangement -- it only stops unbounded no-cost
# stalling (see build_type_mask's reorder branch).
REORDER_BUDGET_PER_TURN = 6

# ── Pointer space (matches BGPolicyV2) ────────────────────────────────────────
SHOP_ZONE_SIZE  = 7
BOARD_ZONE_SIZE = 7
HAND_ZONE_SIZE  = 10
POINTER_DIM     = SHOP_ZONE_SIZE + BOARD_ZONE_SIZE + HAND_ZONE_SIZE  # 24

PTR_SHOP_OFF  = 0                                  # buy  target: shop  slot i  → 0-6
PTR_BOARD_OFF = SHOP_ZONE_SIZE                     # sell target: board slot i  → 7-13
PTR_HAND_OFF  = SHOP_ZONE_SIZE + BOARD_ZONE_SIZE   # place target: hand slot i  → 14-23

# Per-type zone slice: (start_idx, size) used to restrict pointer after type is sampled
_ZONE_SLICE = {
    0: (PTR_SHOP_OFF,  SHOP_ZONE_SIZE),
    1: (PTR_BOARD_OFF, BOARD_ZONE_SIZE),
    2: (PTR_HAND_OFF,  HAND_ZONE_SIZE),
    8: (PTR_BOARD_OFF, BOARD_ZONE_SIZE),  # activate → board zone (same ZONE as sell,
                                           # but scored by activate_scorer -- see forward())
    9: (PTR_BOARD_OFF, BOARD_ZONE_SIZE),  # reorder  → board zone (minion to move)
}


# ── Mask/logit sanitisation helpers ────────────────────────────────────────────
#
# A batch row whose mask is entirely False (a genuine state-inconsistency, or a
# non-pointer action type sharing the pointer mask) makes `masked_fill(..., -inf)`
# produce an all "-inf" row. softmax over an all "-inf" row is NaN, and NaN
# survives multiplication by zero (`NaN * 0.0 == NaN` under IEEE-754) — so a
# later `* needs_ptr.float()` guard does NOT neutralise it; it poisons every
# downstream reduction (log_prob, entropy, loss, gradient) for the WHOLE batch,
# not just the offending row. Confirmed root cause: 8/264 PPO updates in a
# 312-update run applied zero gradient steps because one NaN row triggered the
# update's NaN guard to discard the entire minibatch. These two helpers give
# every row a well-defined distribution before any Categorical is built.

def _sanitize_mask(mask: torch.Tensor) -> torch.Tensor:
    """Replace any row that is entirely False with an all-True row.

    Used at every masked_fill site so a fully-masked row degrades to "treat
    everything as valid" (uniform, harmless once combined with downstream
    zero-contribution guards) rather than producing -inf everywhere. Real
    masking is untouched for every row with >=1 valid entry.
    """
    empty_rows = ~mask.any(dim=-1)
    if empty_rows.any():
        mask = mask.clone()
        mask[empty_rows] = True
    return mask


def _safe_categorical(logits: torch.Tensor) -> torch.distributions.Categorical:
    """Build a Categorical, defusing any row that is entirely -inf first.

    Defence-in-depth for the action-sampling call sites (get_action,
    get_action_batch): even though forward() now sanitises masks before its
    own masked_fill, this guards against an all -inf row reaching Categorical
    from any other path (e.g. a locally-combined zone/occupancy mask) without
    needing the original mask on hand at the call site.
    """
    bad_rows = torch.isneginf(logits).all(dim=-1)
    if bad_rows.any():
        logits = logits.clone()
        logits[bad_rows] = 0.0   # uniform fallback distribution
    return torch.distributions.Categorical(logits=logits)


# ── Network ───────────────────────────────────────────────────────────────────

class BGPolicyNetwork(nn.Module):
    """Two-headed Transformer policy + value network for Hearthstone Battlegrounds.

    Produces separate action-type logits and card-pointer logits, matching the
    BGPolicyV2 BC model's factored output structure so that BC weights transfer
    directly via load_bc_v2_weights().

    Parameters
    ----------
    card_dim:
        Dimensionality of input card feature vectors (default: 44).
    d_model:
        Internal Transformer dimension.  Must be 128 to load BC v2 weights.
    nhead:
        Number of attention heads.
    num_layers:
        Number of Transformer encoder layers.
    scalar_dim:
        Dimensionality of the scalar context vector (default: 38).
    dropout:
        Dropout probability applied in the Transformer and heads.
    """

    def __init__(
        self,
        card_dim:   int = CARD_DIM,
        d_model:    int = 256,
        nhead:      int = 8,
        num_layers: int = 4,
        scalar_dim: int = SCALAR_DIM,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Card token projection: 44 → d_model
        self.card_proj = nn.Linear(card_dim, d_model)

        # Zone type embedding: 0=board, 1=shop, 2=hand, 3=opponent_board
        self.zone_embed = nn.Embedding(4, d_model)

        # Slot position embedding: shared across zones, 10 positions (hand is widest)
        self.slot_pos_embed = nn.Embedding(10, d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer encoder (batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Scalar context projection: scalar_dim → d_model
        self.scalar_proj = nn.Linear(scalar_dim, d_model)

        # ── Type head: [CLS ‖ scalar] → 8 action-type logits ─────────────────
        self.type_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, N_ACTION_TYPES),
        )

        # ── Per-token pointer scorers (replaces global pointer_head MLP) ──────
        # Each scorer maps a token's Transformer output directly to a scalar score.
        # Token layout after forward: 0=CLS, 1-7=board, 8-14=shop, 15-24=hand, 25-31=opp
        self.sell_scorer  = nn.Linear(d_model, 1)   # scores board tokens → sell logits
        self.buy_scorer   = nn.Linear(d_model, 1)   # scores shop  tokens → buy   logits
        self.place_scorer = nn.Linear(d_model, 1)   # scores hand  tokens → place logits

        # activate_scorer looks at the SAME board tokens as sell_scorer (type 8
        # ACTIVATE, like type 1 SELL, targets a minion on the board) but scores
        # them through its own Linear instead of reusing sell_scorer's output.
        # "Worth selling?" and "worth activating?" are different questions about
        # the same minion, so one number cannot answer both -- before this, the
        # ACTIVATE pointer WAS sell_logits verbatim (both types shared
        # _ZONE_SLICE's board slice, and forward() only ever computed one score
        # per board token), and the network learned to put ~1.000 probability on
        # a single fixed board slot for ACTIVATE across many different game
        # states, since it had no way to express an activate-specific
        # preference. Constructed identically to sell_scorer (same shape, same
        # default init) so it starts as a true drop-in before training
        # differentiates the two.
        #
        # REORDER (type 9) deliberately keeps sharing sell_scorer's board score
        # -- that pairing is intentional (see _ZONE_SLICE) and is NOT changed
        # here; only ACTIVATE gets a dedicated scorer.
        self.activate_scorer = nn.Linear(d_model, 1)  # scores board tokens → activate logits

        # Value head: [CLS ‖ scalar] → scalar value
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        # Zero-init the value head output layer so initial estimates start near 0.
        # Without this, Xavier init can produce outputs of ±50+ which dwarf the
        # clipped return targets ([-10, 10]) and make value_loss >> 50 from step 1.
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        board_tokens:   torch.Tensor,                    # [B, 7,  44]
        shop_tokens:    torch.Tensor,                    # [B, 7,  44]
        hand_tokens:    torch.Tensor,                    # [B, 10, 44]
        scalar_context: torch.Tensor,                    # [B, 38]
        type_mask:    Optional[torch.Tensor] = None,     # [B, 8]  True=valid
        pointer_mask: Optional[torch.Tensor] = None,     # [B, 24] True=valid
        opp_tokens:   Optional[torch.Tensor] = None,     # [B, 7,  44]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        type_logits             : [B, 8]
        pointer_logits          : [B, 24] -- buy/sell/place scores (REORDER, type 9,
                                   reads this tensor too: it intentionally shares
                                   sell_scorer's board slice, see _ZONE_SLICE)
        activate_pointer_logits : [B, 24] -- identical to pointer_logits EXCEPT the
                                   board slice (PTR_BOARD_OFF : PTR_BOARD_OFF +
                                   BOARD_ZONE_SIZE) holds activate_scorer's output
                                   instead of sell_logits. Callers must read the
                                   pointer for an ACTIVATE (type 8) action from
                                   THIS tensor, and from `pointer_logits` for every
                                   other pointer type -- never mix the two for a
                                   single transition (see get_action's used_ptr_mask
                                   docstring for why sampling- and evaluation-time
                                   distributions must match exactly).
        value                   : [B, 1]
        """
        B      = board_tokens.shape[0]
        device = board_tokens.device

        # Project card tokens to d_model
        board_emb = self.card_proj(board_tokens)   # [B, 7,  d_model]
        shop_emb  = self.card_proj(shop_tokens)    # [B, 7,  d_model]
        hand_emb  = self.card_proj(hand_tokens)    # [B, 10, d_model]

        if opp_tokens is not None:
            opp_emb = self.card_proj(opp_tokens)
        else:
            opp_emb = torch.zeros(B, 7, self.d_model, device=device)

        # Slot positional encoding: shared table, applied per zone independently
        # so slot 0 in the shop and slot 0 on the board share the same "first slot" signal.
        def _add_slot_pos(emb: torch.Tensor) -> torch.Tensor:
            n = emb.shape[1]
            pos_ids = torch.arange(n, device=device)            # [n]
            return emb + self.slot_pos_embed(pos_ids).unsqueeze(0)  # broadcast over B

        board_emb = _add_slot_pos(board_emb)
        shop_emb  = _add_slot_pos(shop_emb)
        hand_emb  = _add_slot_pos(hand_emb)
        opp_emb   = _add_slot_pos(opp_emb)

        # Zone embeddings over 31 card tokens: board:7, shop:7, hand:10, opp:7
        # Token layout (after CLS prepend): 0=CLS, 1-7=board, 8-14=shop, 15-24=hand, 25-31=opp
        zone_ids = torch.zeros(B, 31, dtype=torch.long, device=device)
        zone_ids[:, :7]    = 0   # board
        zone_ids[:, 7:14]  = 1   # shop
        zone_ids[:, 14:24] = 2   # hand (10 slots)
        zone_ids[:, 24:]   = 3   # opponent board
        zone_emb = self.zone_embed(zone_ids)   # [B, 31, d_model]

        tokens = torch.cat([board_emb, shop_emb, hand_emb, opp_emb], dim=1)  # [B, 31, d_model]
        tokens = tokens + zone_emb

        # Prepend CLS token → 32 tokens total
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)   # [B, 32, d_model]
        tokens = self.transformer(tokens)          # [B, 32, d_model]
        cls_out = tokens[:, 0, :]                  # [B, d_model]

        # Scalar context
        scalar_emb = self.scalar_proj(scalar_context)         # [B, d_model]
        fused      = torch.cat([cls_out, scalar_emb], dim=-1) # [B, 2*d_model]

        # Type head and value head use the global CLS+scalar representation
        type_logits = self.type_head(fused)   # [B, 8]
        value       = self.value_head(fused)  # [B, 1]

        # Per-token pointer scoring: each scorer acts directly on the token's
        # Transformer output, which has already attended to all other tokens.
        # Indices: 1-7=board, 8-14=shop, 15-24=hand  (0=CLS, 25-31=opp unused for pointers)
        sell_logits     = self.sell_scorer(tokens[:, 1:8, :]).squeeze(-1)     # [B, 7]  board→sell
        buy_logits      = self.buy_scorer(tokens[:, 8:15, :]).squeeze(-1)     # [B, 7]  shop→buy
        place_logits    = self.place_scorer(tokens[:, 15:25, :]).squeeze(-1)  # [B, 10] hand→place
        activate_logits = self.activate_scorer(tokens[:, 1:8, :]).squeeze(-1) # [B, 7]  board→activate
        pointer_logits = torch.cat([buy_logits, sell_logits, place_logits], dim=-1)  # [B, 24]
        # activate_pointer_logits mirrors pointer_logits layout-for-layout, with
        # only the board slice swapped for activate_logits. Built via a fresh
        # torch.cat (never by mutating pointer_logits in place) so the two
        # tensors are fully independent -- neither aliases the other's storage
        # or autograd graph.
        activate_pointer_logits = torch.cat(
            [buy_logits, activate_logits, place_logits], dim=-1
        )  # [B, 24]

        if type_mask is not None:
            # Sanitise before masked_fill: an all-False row here would fill to
            # all -inf, and softmax(all -inf) = NaN (see module note above).
            type_mask = _sanitize_mask(type_mask)
            type_logits = type_logits.masked_fill(~type_mask, float("-inf"))
        if pointer_mask is not None:
            # Same guard for the pointer head — a non-pointer-type row or a
            # state-inconsistent occupancy mask must never reach Categorical
            # as an all -inf row. Both pointer_logits and activate_pointer_logits
            # get the IDENTICAL mask and treatment: whichever one a given
            # transition's action type ends up reading from must be equally
            # NaN-safe, and a pointer slot that's invalid for SELL/BUY/PLACE is
            # invalid for ACTIVATE too (both key off the same board/shop/hand
            # occupancy).
            pointer_mask = _sanitize_mask(pointer_mask)
            pointer_logits = pointer_logits.masked_fill(~pointer_mask, float("-inf"))
            activate_pointer_logits = activate_pointer_logits.masked_fill(
                ~pointer_mask, float("-inf")
            )

        return type_logits, pointer_logits, activate_pointer_logits, value

    # ── Action sampling ───────────────────────────────────────────────────────

    def get_action(
        self,
        board_tokens:   torch.Tensor,
        shop_tokens:    torch.Tensor,
        hand_tokens:    torch.Tensor,
        scalar_context: torch.Tensor,
        type_mask:    Optional[torch.Tensor] = None,   # [B, 8]
        pointer_mask: Optional[torch.Tensor] = None,   # [B, 24] full occupancy
        deterministic: bool = False,
        opp_tokens:   Optional[torch.Tensor] = None,
        ptr_mask_fn: Optional[Callable[[int], torch.Tensor]] = None,
    ) -> Tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Two-step action sampling.

        Step 1: sample action type from type_logits.
        Step 2: if the type requires a card pointer (buy/sell/place/activate),
                restrict pointer_logits to the type's zone intersected with
                ptr_mask_fn's (or pointer_mask's) mask, then sample the
                pointer slot.

        Parameters
        ----------
        pointer_mask:
            Full [24] (or [1, 24]) occupancy mask marking all non-empty slots
            across all zones, used as a fallback when `ptr_mask_fn` is not
            given. NOTE: this is typically the type-AGNOSTIC occupancy mask
            (e.g. build_pointer_mask(ps, -1)), intersected here only with the
            sampled type's zone bits -- it does NOT enforce a type's finer
            validity rules (e.g. ACTIVATE's cost/afford/not-yet-activated
            rule is stricter than "board slot is occupied"). Prefer
            `ptr_mask_fn` whenever the caller can supply it.
        ptr_mask_fn : optional callable (sampled_type_idx) -> [POINTER_DIM]
            bool tensor. When given, this is called AFTER the type is
            sampled and its result is used instead of `pointer_mask` for
            the pointer sampling -- mirroring get_action_batch()'s
            "sample type, then build the type-specific mask" order, without
            this module needing to know anything about player-state
            internals (the caller closes over its own state and calls e.g.
            build_pointer_mask(state, sampled_type_idx)).

        Returns
        -------
        type_idx  : int  (0-7)
        ptr_idx   : int  (0-23) or -1 for non-pointer types
        log_prob  : scalar tensor  — log p(type) + log p(ptr | type)
        value     : scalar tensor  [1]
        used_ptr_mask : [POINTER_DIM] bool tensor -- the pointer mask
            actually used to sample (and compute log_prob for) the pointer,
            i.e. zone-restricted and, for pointer types, further restricted
            by ptr_mask_fn/pointer_mask exactly as applied below. Comes back
            all-True for non-pointer types (mask is irrelevant there,
            matching build_pointer_mask's own convention for non-pointer
            types).

            Callers MUST store this returned mask (not recompute a mask
            separately) as the transition's `pointer_mask`. Recomputing a
            mask after the fact -- even a "more correct" one -- can silently
            diverge from the mask actually used to sample/score the action.
            That divergence is exactly the bug this return value exists to
            make structurally impossible: this sequential path used to
            sample the pointer under a type-agnostic occupancy mask but
            store a separately-recomputed type-specific mask, so PPO's
            importance ratio at update time compared log-probs computed
            under two different distributions even when the policy had not
            changed -- see CONTEXT.md (2026-08-31/09-01) for the measured
            impact (88.6% no-op rate on ACTIVATE, 22.2% of all actions, on
            the real training seat mix). Mirrors get_action_batch's
            `used_ptr_masks` return value.
        """
        self.eval()
        with torch.no_grad():
            type_logits, ptr_logits, activate_ptr_logits, value = self.forward(
                board_tokens, shop_tokens, hand_tokens,
                scalar_context, type_mask, None, opp_tokens,
                # pass pointer_mask=None here; zone restriction applied below
            )
            t_logits_1d = type_logits.squeeze(0)   # [8]
            dev = t_logits_1d.device

            # _safe_categorical: defensive fallback in case an all -inf row
            # ever reaches here (see module note above forward()).
            t_dist = _safe_categorical(t_logits_1d)
            type_tensor = t_logits_1d.argmax() if deterministic else t_dist.sample()
            type_idx    = int(type_tensor.item())
            log_prob    = t_dist.log_prob(type_tensor)

            ptr_idx = -1
            used_ptr_mask = torch.ones(POINTER_DIM, dtype=torch.bool, device=dev)
            if type_idx in TYPES_WITH_POINTER:
                # ACTIVATE (type 8) reads its pointer score from
                # activate_pointer_logits, not ptr_logits -- they are two
                # different distributions (see forward()'s docstring on why
                # "worth selling" and "worth activating" need separate scores
                # for the same board slot). REORDER (type 9) intentionally
                # keeps reading ptr_logits/sell_scorer's score, unchanged.
                p_logits_1d = (activate_ptr_logits if type_idx == 8 else ptr_logits).squeeze(0)  # [24]

                # Restrict pointer to this type's zone
                start, size = _ZONE_SLICE[type_idx]
                zone_bits   = torch.zeros(POINTER_DIM, dtype=torch.bool, device=dev)
                zone_bits[start:start + size] = True

                if ptr_mask_fn is not None:
                    row_mask = ptr_mask_fn(type_idx).to(dev)
                    combined = zone_bits & row_mask
                    if not combined.any():
                        combined = zone_bits  # fallback: zone only (state inconsistency guard)
                elif pointer_mask is not None:
                    combined = zone_bits & pointer_mask.squeeze(0)
                    if not combined.any():
                        combined = zone_bits  # fallback: zone only (state inconsistency guard)
                else:
                    combined = zone_bits

                masked_ptr = p_logits_1d.masked_fill(~combined, float("-inf"))
                # Same defensive fallback as above for the pointer distribution.
                p_dist     = _safe_categorical(masked_ptr)
                ptr_tensor = masked_ptr.argmax() if deterministic else p_dist.sample()
                ptr_idx    = int(ptr_tensor.item())
                log_prob   = log_prob + p_dist.log_prob(ptr_tensor)
                used_ptr_mask = combined

        return type_idx, ptr_idx, log_prob, value.squeeze(0), used_ptr_mask

    def get_action_batch(
        self,
        board_tokens:   torch.Tensor,
        shop_tokens:    torch.Tensor,
        hand_tokens:    torch.Tensor,
        scalar_context: torch.Tensor,
        type_mask:    Optional[torch.Tensor] = None,
        pointer_mask: Optional[torch.Tensor] = None,
        opp_tokens:   Optional[torch.Tensor] = None,
        deterministic: bool = False,
        ptr_mask_fn: Optional[Callable[[int, int], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched action sampling for B players in a single forward pass.

        Compared to calling get_action() B times, this runs the Transformer
        once for the whole batch, giving ~3–5× speedup on CPU via better BLAS
        utilisation (weights loaded into cache once for all B samples).

        Parameters
        ----------
        board_tokens, shop_tokens, hand_tokens, scalar_context:
            Batched inputs [B, *, card_dim / scalar_dim].
        type_mask : [B, 8] bool — valid action types per player.
        pointer_mask : [B, 24] bool — full occupancy mask per player, used as a
            fallback when `ptr_mask_fn` is not given. NOTE: this is typically
            the type-AGNOSTIC occupancy mask (e.g. build_pointer_mask(ps, -1)),
            intersected here only with the sampled type's zone bits — it does
            NOT enforce a type's finer validity rules (e.g. ACTIVATE's
            cost/afford/not-yet-activated rule is stricter than "board slot is
            occupied"). Prefer `ptr_mask_fn` whenever the caller can supply it.
        opp_tokens : [B, 7, card_dim] optional.
        ptr_mask_fn : optional callable (batch_index, sampled_type_idx) -> [POINTER_DIM]
            bool tensor. When given, this is called AFTER the type is sampled
            for each row and its result is used instead of `pointer_mask` for
            that row's pointer sampling — mirroring get_action()'s two-step
            "sample type, then build the type-specific mask" order, but without
            this module needing to know anything about player-state internals
            (the caller closes over its own state and calls e.g.
            build_pointer_mask(state, sampled_type_idx)).

        Returns
        -------
        type_actions     : [B] int64
        ptr_actions      : [B] int64  (-1 for non-pointer types)
        log_probs        : [B] float32
        values           : [B] float32
        used_ptr_masks   : [B, POINTER_DIM] bool — the pointer mask actually
            used to sample (and compute log_prob for) each row's pointer, i.e.
            zone-restricted and, for pointer types, further restricted by
            ptr_mask_fn/pointer_mask exactly as applied above. Non-pointer-type
            rows come back all-True (mask is irrelevant there, matching
            build_pointer_mask's own convention for non-pointer types).

            Callers MUST store this returned mask (not recompute a mask
            separately) as the transition's `pointer_mask`. Recomputing a mask
            after the fact — even a "more correct" one — can silently diverge
            from the mask actually used to sample/score the action. That
            divergence is exactly the bug this return value exists to make
            structurally impossible: the batched shopping path used to sample
            the pointer under a type-agnostic occupancy mask but store a
            separately-recomputed type-specific mask, so PPO's importance
            ratio at update time compared log-probs computed under two
            different distributions even when the policy had not changed —
            see CONTEXT.md (2026-08-31/09-01) for the measured impact (up to
            ~92% no-op rate on ACTIVATE, ratio ≈ exp(log(1/1) - log(1/15))
            hard-clipped every update for masked-out pointer types).
        """
        self.eval()
        with torch.no_grad():
            type_logits, ptr_logits, activate_ptr_logits, values = self.forward(
                board_tokens, shop_tokens, hand_tokens,
                scalar_context, type_mask, None, opp_tokens,
            )
            # type_logits: [B, 8], ptr_logits: [B, 24], activate_ptr_logits: [B, 24], values: [B, 1]
            # _safe_categorical: defensive fallback for any all -inf row (see
            # module note above forward()).
            t_dist = _safe_categorical(type_logits)
            type_actions = type_logits.argmax(dim=-1) if deterministic else t_dist.sample()
            log_probs    = t_dist.log_prob(type_actions)  # [B]

            B   = board_tokens.shape[0]
            dev = type_logits.device
            ptr_actions    = torch.full((B,), -1, dtype=torch.long, device=dev)
            used_ptr_masks = torch.ones((B, POINTER_DIM), dtype=torch.bool, device=dev)

            for i in range(B):
                t_idx = int(type_actions[i].item())
                if t_idx in TYPES_WITH_POINTER:
                    start, size = _ZONE_SLICE[t_idx]
                    zone_bits = torch.zeros(POINTER_DIM, dtype=torch.bool, device=dev)
                    zone_bits[start:start + size] = True

                    if ptr_mask_fn is not None:
                        row_mask = ptr_mask_fn(i, t_idx).to(dev)
                        combined = zone_bits & row_mask
                        if not combined.any():
                            combined = zone_bits  # fallback: zone only (state inconsistency guard)
                    elif pointer_mask is not None:
                        occ = zone_bits & pointer_mask[i]
                        combined = occ if occ.any() else zone_bits
                    else:
                        combined = zone_bits

                    # ACTIVATE (type 8) rows score their pointer from
                    # activate_ptr_logits; every other pointer type (including
                    # REORDER, which intentionally still shares sell_scorer's
                    # score) reads ptr_logits, matching get_action()'s
                    # single-row logic and forward()'s docstring.
                    row_ptr_logits = activate_ptr_logits[i] if t_idx == 8 else ptr_logits[i]
                    masked_ptr = row_ptr_logits.masked_fill(~combined, float("-inf"))
                    # Defensive fallback, same reasoning as get_action() above.
                    p_dist     = _safe_categorical(masked_ptr)
                    ptr_actions[i] = masked_ptr.argmax() if deterministic else p_dist.sample()
                    log_probs[i]   = log_probs[i] + p_dist.log_prob(ptr_actions[i])
                    used_ptr_masks[i] = combined

        return type_actions, ptr_actions, log_probs, values.squeeze(-1), used_ptr_masks

    # ── PPO evaluation ────────────────────────────────────────────────────────

    def evaluate_actions(
        self,
        board_tokens:   torch.Tensor,
        shop_tokens:    torch.Tensor,
        hand_tokens:    torch.Tensor,
        scalar_context: torch.Tensor,
        type_actions:   torch.Tensor,             # [B] int64
        ptr_actions:    torch.Tensor,             # [B] int64, -1 for non-pointer types
        type_mask:    Optional[torch.Tensor] = None,    # [B, 8]
        pointer_mask: Optional[torch.Tensor] = None,    # [B, 24] zone+occupancy specific
        opp_tokens:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate joint log-probs and entropy for a batch of stored actions.

        For pointer types, log_prob = log p(type) + log p(ptr | type).
        For non-pointer types, log_prob = log p(type).

        The pointer_mask stored in each Transition should be the zone+occupancy
        mask that was active when the pointer was sampled (from build_pointer_mask).

        Returns
        -------
        log_probs : [B]
        values    : [B, 1]
        entropy   : [B]  — H(type) + H(ptr) for pointer types, H(type) otherwise
        """
        type_logits, ptr_logits, activate_ptr_logits, values = self.forward(
            board_tokens, shop_tokens, hand_tokens,
            scalar_context, type_mask, pointer_mask, opp_tokens,
        )

        t_dist     = torch.distributions.Categorical(logits=type_logits)
        log_probs  = t_dist.log_prob(type_actions)   # [B]
        entropy    = t_dist.entropy()                # [B]

        # Add pointer contribution for buy/sell/place transitions
        needs_ptr = torch.zeros(
            type_actions.shape[0], dtype=torch.bool, device=type_actions.device
        )
        for t_idx in TYPES_WITH_POINTER:
            needs_ptr = needs_ptr | (type_actions == t_idx)

        if needs_ptr.any():
            # Rows in this batch differ in action type, so the per-row pointer
            # logits must be assembled BEFORE a single Categorical is built for
            # the whole batch: an ACTIVATE (type 8) row was sampled from
            # activate_ptr_logits (see get_action/get_action_batch above), and
            # every other pointer-type row (BUY/SELL/PLACE/REORDER) was sampled
            # from ptr_logits. Building one p_dist straight from ptr_logits
            # alone -- the old behaviour, before activate_ptr_logits existed --
            # would re-score ACTIVATE transitions under a DIFFERENT distribution
            # than the one they were sampled under, corrupting PPO's importance
            # ratio for those rows even when the policy hasn't changed. This is
            # exactly the class of sampling/evaluation-mismatch bug documented
            # on get_action's `used_ptr_mask` return value above (88.6% no-op
            # rate on ACTIVATE, 22.2% of all actions, from a prior instance of
            # this same failure mode with masks instead of scorers).
            ptr_logits_sel = torch.where(
                (type_actions == 8).unsqueeze(-1),
                activate_ptr_logits,
                ptr_logits,
            )
            # _safe_categorical: forward() already sanitises pointer_mask so
            # neither tensor should contain an all -inf row, but this is cheap
            # defence-in-depth against the NaN failure mode described above
            # forward() (an all -inf row -> Categorical -> NaN log_prob/entropy).
            p_dist = _safe_categorical(ptr_logits_sel)
            # Clamp ptr_actions to [0, POINTER_DIM-1] for rows where ptr == -1;
            # those rows are masked out by needs_ptr anyway.
            safe_ptr = ptr_actions.clamp(min=0)
            ptr_lp = p_dist.log_prob(safe_ptr)        # [B]
            ptr_ent = p_dist.entropy()                # [B]
            # torch.where, NOT `* needs_ptr.float()`: if ptr_lp/ptr_ent were
            # ever NaN for a non-pointer row (NaN * 0.0 == NaN, it does NOT
            # zero out), the multiply would silently poison log_probs/entropy
            # for the WHOLE batch once summed/meaned downstream. torch.where
            # instead substitutes an exact 0.0 for those rows regardless of
            # what the pointer head produced for them.
            log_probs = log_probs + torch.where(needs_ptr, ptr_lp, torch.zeros_like(ptr_lp))
            entropy   = entropy   + torch.where(needs_ptr, ptr_ent, torch.zeros_like(ptr_ent))

        return log_probs, values, entropy

    # ── BC warm-start ─────────────────────────────────────────────────────────

    def load_bc_v2_weights(self, bc_path: str) -> None:
        """Warm-start from a BGPolicyV2 BC checkpoint (bc_v2.pt).

        NOTE: This method is a no-op for d_model != 128. The architecture was
        upgraded to d_model=256 with a per-token pointer head; the old BC weight
        shapes are incompatible. To re-enable warm-start, retrain the BC model
        using BGPolicyNetwork directly with the new architecture.

        Legacy transfers (only applied when d_model == 128):
        1. BC type_head [8, 128]   → PPO type_head[-1]    [8, 128]
        2. BC pointer_head [24, 128] → PPO pointer_head[-1] [24, 128]  (removed)
        3. BC shared.4 [128, 128]  → scalar half of type_head[0].weight

        Requires d_model=128 (the BC hidden size).
        """
        if self.d_model != 128:
            logger.warning(
                "load_bc_v2_weights: skipped — network uses d_model=%d but BC "
                "checkpoint requires d_model=128. Retrain BC with the new "
                "architecture to re-enable warm-start.", self.d_model
            )
            return
        try:
            ckpt = torch.load(bc_path, map_location="cpu")
            sd   = ckpt["state_dict"]

            # 1. type_head output layer
            th_w = sd["type_head.weight"]   # [8, 128]
            th_b = sd["type_head.bias"]     # [8]
            assert self.type_head[-1].weight.shape == th_w.shape, (
                f"type_head shape mismatch: {self.type_head[-1].weight.shape} vs {th_w.shape}"
                " — was the network built with d_model=128?"
            )
            self.type_head[-1].weight.data.copy_(th_w)
            self.type_head[-1].bias.data.copy_(th_b)
            logger.info("load_bc_v2_weights: copied type_head [8, 128]")

            # 2. pointer_head output layer
            ph_w = sd["pointer_head.weight"]  # [24, 128]
            ph_b = sd["pointer_head.bias"]    # [24]
            assert self.pointer_head[-1].weight.shape == ph_w.shape, (
                f"pointer_head shape mismatch: {self.pointer_head[-1].weight.shape} vs {ph_w.shape}"
            )
            self.pointer_head[-1].weight.data.copy_(ph_w)
            self.pointer_head[-1].bias.data.copy_(ph_b)
            logger.info("load_bc_v2_weights: copied pointer_head [24, 128]")

            # 3. BC trunk layer 4 (shared.4: Linear 128→128) → scalar half of
            #    each head's first linear (Linear 256→128; cols d_model: = scalar half)
            trunk_w = sd["shared.4.weight"]   # [128, 128]
            trunk_b = sd["shared.4.bias"]     # [128]
            for name, head in (("type_head", self.type_head),
                               ("pointer_head", self.pointer_head)):
                w = head[0].weight.data.clone()     # [128, 256]
                w[:, self.d_model:] = trunk_w        # overwrite scalar half
                head[0].weight.data.copy_(w)
                head[0].bias.data.copy_(trunk_b)
            logger.info(
                "load_bc_v2_weights: seeded type_head[0] and pointer_head[0] "
                "scalar half from BC shared.4"
            )

        except Exception as exc:
            logger.warning("load_bc_v2_weights: failed to load '%s': %s", bc_path, exc)


# ── Action mask builders ──────────────────────────────────────────────────────

def _activatable_board_slots(board, gold: int) -> List[bool]:
    """Return, per board slot (up to BOARD_ZONE_SIZE), whether Activate is usable now.

    A slot is activatable when its minion has activate_cost > 0, hasn't been
    activated yet this turn, and the player can afford the Gold cost.
    """
    out = []
    for slot in board[:BOARD_ZONE_SIZE]:
        if not _slot_occupied(slot):
            out.append(False)
            continue
        cost = getattr(slot, "activate_cost", 0) if not isinstance(slot, dict) else slot.get("activate_cost", 0)
        used = getattr(slot, "activated_this_turn", False) if not isinstance(slot, dict) else slot.get("activated_this_turn", False)
        out.append(bool(cost) and cost > 0 and not used and gold >= cost)
    return out


def build_type_mask(player_state) -> torch.Tensor:
    """Build a [10] boolean mask of valid action types from a player state.

    Accepts either a dataclass/object with attribute access or a plain dict.

    Masking rules
    -------------
    0 buy        : shop non-empty AND gold >= 3 AND len(hand) < 10
    1 sell       : board non-empty
    2 place      : hand non-empty AND len(board) < 7
    3 reroll     : gold >= 1
    4 freeze     : shop not already frozen this turn
    5 level_up   : tavern_tier < 6 AND gold >= level_cost
    6 hero_power : not used this turn AND charges > 0 AND gold >= cost AND hero active
    7 end_turn   : always valid
    8 activate   : any board minion has an unused, affordable Activate (N) ability
    9 reorder    : >= 2 minions on board AND this turn's reorder budget not spent
    """
    if isinstance(player_state, dict):
        gold        = player_state.get("gold", 0)
        shop        = player_state.get("shop", [])
        board       = player_state.get("board", [])
        hand        = player_state.get("hand", [])
        tavern_tier = player_state.get("tavern_tier", 1)
        level_cost  = player_state.get("level_cost", 5)
    else:
        gold        = getattr(player_state, "gold", 0)
        shop        = getattr(player_state, "shop", [])
        board       = getattr(player_state, "board", [])
        hand        = getattr(player_state, "hand", [])
        tavern_tier = getattr(player_state, "tavern_tier", 1)
        level_cost  = getattr(player_state, "level_cost", 5)

    n_shop  = sum(1 for s in shop  if _slot_occupied(s))
    n_board = sum(1 for s in board if _slot_occupied(s))
    n_hand  = sum(1 for s in hand  if _slot_occupied(s))

    _buy_cost    = getattr(player_state, "buy_cost", 3)
    _reroll_cost = getattr(player_state, "reroll_cost", 1)
    _buy_discount = getattr(player_state, "buy_discount", 0)
    _first_free  = getattr(player_state, "first_buy_free", False)
    _eff_buy_cost = 0 if _first_free else max(0, _buy_cost - _buy_discount)

    # Trinket offer pending: only BUY (select) or END_TURN (decline) are valid
    _trinket_pending = getattr(player_state, "trinket_offer_pending", False)
    if _trinket_pending:
        mask = torch.zeros(N_ACTION_TYPES, dtype=torch.bool)
        mask[0] = True  # BUY → select trinket from shop slots 0-2
        mask[7] = True  # END_TURN → decline offer
        return mask

    # When discover is pending, only BUY (pointer into shop zone = discover slot) is valid
    _discover = getattr(player_state, "discover_pending", [])
    if _discover:
        mask = torch.zeros(N_ACTION_TYPES, dtype=torch.bool)
        mask[0] = True  # BUY only — pointer selects among the 3 discover options
        return mask

    mask = torch.zeros(N_ACTION_TYPES, dtype=torch.bool)
    if n_shop  > 0 and gold >= _eff_buy_cost and n_hand < 10: mask[0] = True  # buy
    if n_board > 0:                                            mask[1] = True  # sell
    if n_hand  > 0 and n_board < 7:                           mask[2] = True  # place
    _free_refreshes = getattr(player_state, "_free_refreshes", 0)
    if gold >= _reroll_cost or _free_refreshes > 0:            mask[3] = True  # reroll
    _already_frozen = getattr(player_state, "frozen", False)
    if not _already_frozen:                                    mask[4] = True  # freeze (once per turn)
    if tavern_tier < 6 and gold >= level_cost:    mask[5] = True  # level_up
    # hero_power: valid when not used, has charges, enough gold, and hero is active
    _hp_used    = getattr(player_state, "hero_power_used", False)
    _hp_charges = getattr(player_state, "hero_power_charges", -1)
    _hp_cost    = getattr(player_state, "hero_power_cost", 0)
    _hero_id    = getattr(player_state, "hero_card_id", "")
    _hp_active  = _hero_id not in ("", "TB_BaconShop_HERO_00")
    mask[6] = bool(
        not _hp_used
        and (_hp_charges == -1 or _hp_charges > 0)
        and gold >= _hp_cost
        and _hp_active
    )  # hero_power
    mask[7] = True                                                  # end_turn
    mask[8] = any(_activatable_board_slots(board, gold))            # activate
    # reorder: needs something to reorder (>=2 minions, since slot 0 is never a
    # valid target) and budget left this turn. The budget exists because a
    # state-changing-but-costless action is otherwise a stalling device: each
    # one advances a gamma=0.997 step, so spamming them discounts the pending
    # END_TURN penalties. ACTIVATE demonstrated exactly this failure mode.
    _reorders_left = getattr(player_state, "reorders_left", REORDER_BUDGET_PER_TURN)
    mask[9] = bool(n_board >= 2 and _reorders_left > 0)              # reorder
    return mask


def build_pointer_mask(player_state, type_idx: int) -> torch.Tensor:
    """Build a [24] boolean mask of valid pointer slots.

    When type_idx is in TYPES_WITH_POINTER (0/1/2), only the relevant zone
    is enabled and only occupied slots are marked True.

    When type_idx is not in TYPES_WITH_POINTER, all slots are set to True
    (the pointer distribution is irrelevant and won't be sampled).

    When type_idx == -1, returns the full occupancy mask across all zones
    (used when the type is not yet known, e.g. in get_action).
    """
    if isinstance(player_state, dict):
        shop  = player_state.get("shop",  [])
        board = player_state.get("board", [])
        hand  = player_state.get("hand",  [])
    else:
        shop  = getattr(player_state, "shop",  [])
        board = getattr(player_state, "board", [])
        hand  = getattr(player_state, "hand",  [])

    _trinket_pending = getattr(player_state, "trinket_offer_pending", False)
    _discover        = getattr(player_state, "discover_pending", [])

    mask = torch.zeros(POINTER_DIM, dtype=torch.bool)

    if type_idx == 0:          # buy → shop zone (trinket offer / discover / normal)
        if _trinket_pending:
            # Trinket offer: up to 3 choices encoded in shop slots 0-2
            for i in range(3):
                mask[PTR_SHOP_OFF + i] = True
            return mask
        if _discover:
            # Discover in progress: only indices 0..len-1 in the shop zone are valid
            for i in range(min(len(_discover), SHOP_ZONE_SIZE)):
                mask[PTR_SHOP_OFF + i] = True
            return mask
        for i, slot in enumerate(shop[:SHOP_ZONE_SIZE]):
            if _slot_occupied(slot):
                mask[PTR_SHOP_OFF + i] = True
        if not mask.any():
            mask[PTR_SHOP_OFF:PTR_SHOP_OFF + SHOP_ZONE_SIZE] = True  # fallback
    elif type_idx == 1:        # sell → board zone
        for i, slot in enumerate(board[:BOARD_ZONE_SIZE]):
            if _slot_occupied(slot):
                mask[PTR_BOARD_OFF + i] = True
        if not mask.any():
            mask[PTR_BOARD_OFF:PTR_BOARD_OFF + BOARD_ZONE_SIZE] = True
    elif type_idx == 2:        # place → hand zone
        for i, slot in enumerate(hand[:HAND_ZONE_SIZE]):
            if _slot_occupied(slot):
                mask[PTR_HAND_OFF + i] = True
        if not mask.any():
            mask[PTR_HAND_OFF:PTR_HAND_OFF + HAND_ZONE_SIZE] = True
    elif type_idx == 8:        # activate → board zone, only currently-activatable minions
        gold = player_state.get("gold", 0) if isinstance(player_state, dict) else getattr(player_state, "gold", 0)
        for i, activatable in enumerate(_activatable_board_slots(board, gold)):
            if activatable:
                mask[PTR_BOARD_OFF + i] = True
        if not mask.any():
            mask[PTR_BOARD_OFF:PTR_BOARD_OFF + BOARD_ZONE_SIZE] = True
    elif type_idx == 9:        # reorder → board zone, EXCLUDING slot 0
        # Slot 0 is excluded on purpose: moving the front minion to the front
        # is a no-op, and a costless no-op is a discount-stalling exploit.
        for i, slot in enumerate(board[:BOARD_ZONE_SIZE]):
            if i > 0 and _slot_occupied(slot):
                mask[PTR_BOARD_OFF + i] = True
        if not mask.any():
            # No legal target (0 or 1 minions). build_type_mask already
            # forbids type 9 here; this only guards against an all-False row,
            # which would make softmax NaN and poison the whole batch.
            mask[PTR_BOARD_OFF + 1:PTR_BOARD_OFF + BOARD_ZONE_SIZE] = True
    elif type_idx == -1:       # full occupancy mask, all zones
        # Shop portion must respect trinket/discover restriction the same way
        # the type_idx==0 branch above does -- this mask is what get_action()
        # uses for its first, type-agnostic forward pass (before type_idx is
        # even sampled), so leaving it as raw ps.shop occupancy here let the
        # policy sample an out-of-range shop pointer during a trinket/discover
        # offer (step_shopping ignores ps.shop entirely while either is
        # pending, silently no-op'ing on anything outside the actual offer's
        # range) -- degrading real training data, not just scripted opponents.
        if _trinket_pending:
            for i in range(3):
                mask[PTR_SHOP_OFF + i] = True
        elif _discover:
            for i in range(min(len(_discover), SHOP_ZONE_SIZE)):
                mask[PTR_SHOP_OFF + i] = True
        else:
            for i, slot in enumerate(shop[:SHOP_ZONE_SIZE]):
                if _slot_occupied(slot):
                    mask[PTR_SHOP_OFF + i] = True
        for i, slot in enumerate(board[:BOARD_ZONE_SIZE]):
            if _slot_occupied(slot):
                mask[PTR_BOARD_OFF + i] = True
        for i, slot in enumerate(hand[:HAND_ZONE_SIZE]):
            if _slot_occupied(slot):
                mask[PTR_HAND_OFF + i] = True
    else:
        mask[:] = True   # non-pointer type; mask is irrelevant

    return mask


def build_type_mask_batch(player_states) -> torch.Tensor:
    """Stack build_type_mask() for each player state → [B, N_ACTION_TYPES] bool tensor."""
    return torch.stack([build_type_mask(ps) for ps in player_states])


def build_pointer_mask_batch(player_states, type_indices) -> torch.Tensor:
    """Stack build_pointer_mask() per player → [B, 24] bool tensor.

    Parameters
    ----------
    player_states : list of B player state objects
    type_indices  : [B] int tensor or list of ints — sampled type per player
    """
    return torch.stack([
        build_pointer_mask(ps, int(t))
        for ps, t in zip(player_states, type_indices)
    ])


def _slot_occupied(slot) -> bool:
    """Return True if the given shop/board/hand slot contains a real minion."""
    if slot is None:
        return False
    if isinstance(slot, dict):
        return bool(slot.get("card_id", ""))
    card_id = getattr(slot, "card_id", None)
    return bool(card_id)
