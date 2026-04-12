"""
BGPolicyNetwork — Two-headed Transformer policy + value network.

Architecture:
  - Project 44-dim card tokens to d_model
  - Zone embedding: 0=board, 1=shop, 2=hand, 3=opponent_board
  - CLS token prepended
  - Single Transformer encoder over [CLS + 7 board + 7 shop + 10 hand + 7 opp] = 32 tokens
  - type_head:    CLS + scalar_context → 8 action-type logits
  - pointer_head: per-token scorers (sell/buy/place) acting directly on Transformer
                  outputs for board/shop/hand tokens → 24 card-pointer logits
  - value_head:   CLS + scalar_context → scalar value

Action types (8, matching BGPolicyV2 BC model):
  0 buy        → pointer: shop  slot  [PTR_SHOP_OFF  + i,  i in 0-6]
  1 sell       → pointer: board slot  [PTR_BOARD_OFF + i,  i in 0-6]
  2 place      → pointer: hand  slot  [PTR_HAND_OFF  + i,  i in 0-9]
  3 reroll     → no pointer
  4 freeze     → no pointer
  5 level_up   → no pointer
  6 hero_power → no pointer
  7 end_turn   → no pointer

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
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Input dimensions ──────────────────────────────────────────────────────────
CARD_DIM   = 44
SCALAR_DIM = 98   # 24 own-board + 64 all-opponents (8×8, own slot zeroed) + 6 lobby + 4 economy

# ── Action type space (matches BGPolicyV2) ────────────────────────────────────
N_ACTION_TYPES    = 8
ACTION_TYPE_NAMES = ["buy", "sell", "place", "reroll", "freeze",
                     "level_up", "hero_power", "end_turn"]

# Types that require a card pointer; all others use ptr_idx = -1
TYPES_WITH_POINTER = frozenset({0, 1, 2})  # buy, sell, place

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
}


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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        type_logits    : [B, 8]
        pointer_logits : [B, 24]
        value          : [B, 1]
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
        sell_logits  = self.sell_scorer(tokens[:, 1:8, :]).squeeze(-1)    # [B, 7]  board→sell
        buy_logits   = self.buy_scorer(tokens[:, 8:15, :]).squeeze(-1)    # [B, 7]  shop→buy
        place_logits = self.place_scorer(tokens[:, 15:25, :]).squeeze(-1) # [B, 10] hand→place
        pointer_logits = torch.cat([buy_logits, sell_logits, place_logits], dim=-1)  # [B, 24]

        if type_mask is not None:
            type_logits = type_logits.masked_fill(~type_mask, float("-inf"))
        if pointer_mask is not None:
            pointer_logits = pointer_logits.masked_fill(~pointer_mask, float("-inf"))

        return type_logits, pointer_logits, value

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
    ) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
        """Two-step action sampling.

        Step 1: sample action type from type_logits.
        Step 2: if the type requires a card pointer (buy/sell/place), restrict
                pointer_logits to the type's zone intersected with pointer_mask,
                then sample the pointer slot.

        Parameters
        ----------
        pointer_mask:
            Full [B, 24] occupancy mask marking all non-empty slots across all
            zones.  get_action will further restrict this to the relevant zone
            based on the sampled type.  If None, only zone restriction applies.

        Returns
        -------
        type_idx  : int  (0-7)
        ptr_idx   : int  (0-23) or -1 for non-pointer types
        log_prob  : scalar tensor  — log p(type) + log p(ptr | type)
        value     : scalar tensor  [1]
        """
        self.eval()
        with torch.no_grad():
            type_logits, ptr_logits, value = self.forward(
                board_tokens, shop_tokens, hand_tokens,
                scalar_context, type_mask, None, opp_tokens,
                # pass pointer_mask=None here; zone restriction applied below
            )
            t_logits_1d = type_logits.squeeze(0)   # [8]
            p_logits_1d = ptr_logits.squeeze(0)    # [24]

            t_dist = torch.distributions.Categorical(logits=t_logits_1d)
            type_tensor = t_logits_1d.argmax() if deterministic else t_dist.sample()
            type_idx    = int(type_tensor.item())
            log_prob    = t_dist.log_prob(type_tensor)

            ptr_idx = -1
            if type_idx in TYPES_WITH_POINTER:
                # Restrict pointer to this type's zone
                start, size = _ZONE_SLICE[type_idx]
                zone_bits   = torch.zeros(POINTER_DIM, dtype=torch.bool, device=t_logits_1d.device)
                zone_bits[start:start + size] = True

                combined = zone_bits
                if pointer_mask is not None:
                    combined = zone_bits & pointer_mask.squeeze(0)
                    if not combined.any():
                        combined = zone_bits  # fallback: zone only (state inconsistency guard)

                masked_ptr = p_logits_1d.masked_fill(~combined, float("-inf"))
                p_dist     = torch.distributions.Categorical(logits=masked_ptr)
                ptr_tensor = masked_ptr.argmax() if deterministic else p_dist.sample()
                ptr_idx    = int(ptr_tensor.item())
                log_prob   = log_prob + p_dist.log_prob(ptr_tensor)

        return type_idx, ptr_idx, log_prob, value.squeeze(0)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched action sampling for B players in a single forward pass.

        Compared to calling get_action() B times, this runs the Transformer
        once for the whole batch, giving ~3–5× speedup on CPU via better BLAS
        utilisation (weights loaded into cache once for all B samples).

        Parameters
        ----------
        board_tokens, shop_tokens, hand_tokens, scalar_context:
            Batched inputs [B, *, card_dim / scalar_dim].
        type_mask : [B, 8] bool — valid action types per player.
        pointer_mask : [B, 24] bool — full occupancy mask per player.
        opp_tokens : [B, 7, card_dim] optional.

        Returns
        -------
        type_actions : [B] int64
        ptr_actions  : [B] int64  (-1 for non-pointer types)
        log_probs    : [B] float32
        values       : [B] float32
        """
        self.eval()
        with torch.no_grad():
            type_logits, ptr_logits, values = self.forward(
                board_tokens, shop_tokens, hand_tokens,
                scalar_context, type_mask, None, opp_tokens,
            )
            # type_logits: [B, 8], ptr_logits: [B, 24], values: [B, 1]
            t_dist = torch.distributions.Categorical(logits=type_logits)
            type_actions = type_logits.argmax(dim=-1) if deterministic else t_dist.sample()
            log_probs    = t_dist.log_prob(type_actions)  # [B]

            B   = board_tokens.shape[0]
            dev = type_logits.device
            ptr_actions = torch.full((B,), -1, dtype=torch.long, device=dev)

            for i in range(B):
                t_idx = int(type_actions[i].item())
                if t_idx in TYPES_WITH_POINTER:
                    start, size = _ZONE_SLICE[t_idx]
                    zone_bits = torch.zeros(POINTER_DIM, dtype=torch.bool, device=dev)
                    zone_bits[start:start + size] = True
                    combined = zone_bits
                    if pointer_mask is not None:
                        occ = zone_bits & pointer_mask[i]
                        combined = occ if occ.any() else zone_bits
                    masked_ptr = ptr_logits[i].masked_fill(~combined, float("-inf"))
                    p_dist     = torch.distributions.Categorical(logits=masked_ptr)
                    ptr_actions[i] = masked_ptr.argmax() if deterministic else p_dist.sample()
                    log_probs[i]   = log_probs[i] + p_dist.log_prob(ptr_actions[i])

        return type_actions, ptr_actions, log_probs, values.squeeze(-1)

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
        type_logits, ptr_logits, values = self.forward(
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
            p_dist = torch.distributions.Categorical(logits=ptr_logits)
            # Clamp ptr_actions to [0, POINTER_DIM-1] for rows where ptr == -1;
            # those rows are masked out by needs_ptr anyway.
            safe_ptr = ptr_actions.clamp(min=0)
            ptr_lp = p_dist.log_prob(safe_ptr)        # [B]
            ptr_ent = p_dist.entropy()                # [B]
            log_probs = log_probs + ptr_lp  * needs_ptr.float()
            entropy   = entropy   + ptr_ent * needs_ptr.float()

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

def build_type_mask(player_state) -> torch.Tensor:
    """Build a [8] boolean mask of valid action types from a player state.

    Accepts either a dataclass/object with attribute access or a plain dict.

    Masking rules
    -------------
    0 buy        : shop non-empty AND gold >= 3 AND len(hand) < 10
    1 sell       : board non-empty
    2 place      : hand non-empty AND len(board) < 7
    3 reroll     : gold >= 1
    4 freeze     : always valid
    5 level_up   : tavern_tier < 6 AND gold >= level_cost
    6 hero_power : always valid
    7 end_turn   : always valid
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
    mask[4] = True                                                  # freeze
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

    _discover = getattr(player_state, "discover_pending", [])

    mask = torch.zeros(POINTER_DIM, dtype=torch.bool)

    if type_idx == 0:          # buy → shop zone (or discover zone)
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
    elif type_idx == -1:       # full occupancy mask, all zones
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
    """Stack build_type_mask() for each player state → [B, 8] bool tensor."""
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
