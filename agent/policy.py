"""
BGPolicyNetwork — Transformer-based policy + value network for BG agent.

Architecture:
  - Project 44-dim card tokens to d_model
  - Zone embedding: 0=board, 1=shop, 2=hand, 3=opponent_board
  - CLS token prepended
  - Single Transformer encoder over [CLS + 7 board + 7 shop + 10 hand + 7 opp] = 32 tokens
  - Policy head: CLS + scalar_context → action logits
  - Value head: CLS + scalar_context → scalar value
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

CARD_DIM = 44
SCALAR_DIM = 30  # 24 own-board features + 6 opponent features

# PLAY action layout:  play_h{hand_idx}_p{board_pos}
#   hand_idx  : 0-9   (up to 10 hand slots)
#   board_pos : 0-6   (insert before slot j; j == len(board) appends)
#   index     : PLAY_START + hand_idx * PLAY_BOARD_POS + board_pos
PLAY_HAND_SLOTS = 10
PLAY_BOARD_POS  = 7
PLAY_START      = 14
PLAY_END        = PLAY_START + PLAY_HAND_SLOTS * PLAY_BOARD_POS - 1  # 83

# Fixed actions after the play block
LEVEL_UP_IDX   = PLAY_END + 1   # 84
FREEZE_IDX     = PLAY_END + 2   # 85
REFRESH_IDX    = PLAY_END + 3   # 86
HERO_POWER_IDX = PLAY_END + 4   # 87
END_TURN_IDX   = PLAY_END + 5   # 88
SWAP_START     = PLAY_END + 6   # 89  (swap_01 .. swap_56 → 89-94)

N_ACTIONS = SWAP_START + 6  # 95

ACTION_NAMES = (
    [f"buy_{i}"  for i in range(7)]   +  # 0-6
    [f"sell_{i}" for i in range(7)]   +  # 7-13
    [
        f"play_h{h}_p{p}"
        for h in range(PLAY_HAND_SLOTS)
        for p in range(PLAY_BOARD_POS)
    ]                                  +  # 14-83
    ["level_up", "freeze", "refresh", "hero_power", "end_turn"]  +  # 84-88
    [f"swap_{i}{i+1}" for i in range(6)]  # 89-94
)


class BGPolicyNetwork(nn.Module):
    """Transformer policy + value network for Hearthstone Battlegrounds.

    Accepts per-zone card token tensors and a scalar board-feature context
    vector, encodes them with a Transformer, and produces action logits and
    a state value estimate.

    Parameters
    ----------
    card_dim:
        Dimensionality of input card feature vectors (default: 44).
    d_model:
        Internal Transformer dimension.
    nhead:
        Number of attention heads.
    num_layers:
        Number of Transformer encoder layers.
    scalar_dim:
        Dimensionality of the scalar context vector (default: 24).
    n_actions:
        Number of output actions (default: 26).
    dropout:
        Dropout probability applied in the Transformer and heads.
    """

    def __init__(
        self,
        card_dim: int = CARD_DIM,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        scalar_dim: int = SCALAR_DIM,
        n_actions: int = N_ACTIONS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions

        # Card token projection: 44 → d_model
        self.card_proj = nn.Linear(card_dim, d_model)

        # Zone type embedding: 0=board, 1=shop, 2=hand, 3=opponent_board
        self.zone_embed = nn.Embedding(4, d_model)

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

        # Scalar context projection: 24 → d_model
        self.scalar_proj = nn.Linear(scalar_dim, d_model)

        # Policy head: [CLS‖scalar] → action logits
        self.policy_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_actions),
        )

        # Value head: [CLS‖scalar] → scalar value
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise CLS token with truncated normal; linears with Xavier."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        board_tokens: torch.Tensor,             # [B, 7,  44]
        shop_tokens: torch.Tensor,              # [B, 7,  44]
        hand_tokens: torch.Tensor,              # [B, 10, 44]
        scalar_context: torch.Tensor,           # [B, 30]
        action_mask: Optional[torch.Tensor] = None,  # [B, N_ACTIONS] True=valid
        opp_tokens: Optional[torch.Tensor] = None,   # [B, 7, 44] last seen opponent board
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        action_logits : [B, N_ACTIONS]
        value         : [B, 1]
        """
        B = board_tokens.shape[0]
        device = board_tokens.device

        # Project card tokens to d_model
        board_emb = self.card_proj(board_tokens)   # [B, 7,  d_model]
        shop_emb  = self.card_proj(shop_tokens)    # [B, 7,  d_model]
        hand_emb  = self.card_proj(hand_tokens)    # [B, 10, d_model]

        if opp_tokens is not None:
            opp_emb = self.card_proj(opp_tokens)   # [B, 7, d_model]
        else:
            # Fall back to zeros for the opponent zone (safe before first combat)
            opp_emb = torch.zeros(B, 7, self.d_model, device=device)

        # Zone embeddings: board:7, shop:7, hand:10, opp:7 → 31 tokens total
        n_card_tokens = 31
        zone_ids = torch.zeros(B, n_card_tokens, dtype=torch.long, device=device)
        zone_ids[:, :7]   = 0   # board
        zone_ids[:, 7:14] = 1   # shop
        zone_ids[:, 14:24] = 2  # hand (10 slots)
        zone_ids[:, 24:]  = 3   # opponent board
        zone_emb = self.zone_embed(zone_ids)  # [B, 31, d_model]

        # Concatenate zones and add zone embeddings
        tokens = torch.cat([board_emb, shop_emb, hand_emb, opp_emb], dim=1)  # [B, 31, d_model]
        tokens = tokens + zone_emb

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)    # [B, 1,  d_model]
        tokens = torch.cat([cls, tokens], dim=1)  # [B, 32, d_model]

        # Transformer
        tokens = self.transformer(tokens)  # [B, 32, d_model]
        cls_out = tokens[:, 0, :]          # [B, d_model]

        # Scalar context
        scalar_emb = self.scalar_proj(scalar_context)        # [B, d_model]
        fused = torch.cat([cls_out, scalar_emb], dim=-1)     # [B, 2*d_model]

        # Policy and value heads
        logits = self.policy_head(fused)   # [B, N_ACTIONS]
        value  = self.value_head(fused)    # [B, 1]

        # Mask invalid actions with -inf so softmax/log_softmax treats them as 0
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))

        return logits, value

    # ------------------------------------------------------------------
    # Action sampling / evaluation
    # ------------------------------------------------------------------

    def get_action(
        self,
        board_tokens: torch.Tensor,
        shop_tokens: torch.Tensor,
        hand_tokens: torch.Tensor,
        scalar_context: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        opp_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample (or argmax) an action given the current state.

        Parameters
        ----------
        deterministic:
            If True, take the argmax of the logits rather than sampling.

        Returns
        -------
        action_idx : int
        log_prob   : scalar tensor
        value      : scalar tensor [1]
        """
        self.eval()
        with torch.no_grad():
            logits, value = self.forward(
                board_tokens, shop_tokens, hand_tokens,
                scalar_context, action_mask, opp_tokens,
            )
            # logits: [1, N_ACTIONS] or [N_ACTIONS] — squeeze batch dim
            logits_1d = logits.squeeze(0)
            dist = torch.distributions.Categorical(logits=logits_1d)
            if deterministic:
                action = logits_1d.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), log_prob, value.squeeze(0)

    def evaluate_actions(
        self,
        board_tokens: torch.Tensor,
        shop_tokens: torch.Tensor,
        hand_tokens: torch.Tensor,
        scalar_context: torch.Tensor,
        actions: torch.Tensor,                          # [B]
        action_mask: Optional[torch.Tensor] = None,
        opp_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probs and entropy for a batch of taken actions.

        Used during PPO update to compute the importance-sampling ratio.

        Returns
        -------
        log_probs : [B]
        values    : [B, 1]
        entropy   : [B]
        """
        logits, values = self.forward(
            board_tokens, shop_tokens, hand_tokens,
            scalar_context, action_mask, opp_tokens,
        )
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values, entropy

    # ------------------------------------------------------------------
    # Warm-start from BC checkpoint
    # ------------------------------------------------------------------

    def load_bc_weights(self, bc_model_path: str) -> None:
        """Try to warm-start from a behavioural-cloning checkpoint.

        The BC model stored at *bc_model_path* may have a different
        architecture (e.g. an MLP with 181-dim input).  We load only layers
        whose parameter tensors exactly match in shape and log a warning for
        every mismatch.  Never raises — silently skips incompatible weights.
        """
        try:
            checkpoint = torch.load(bc_model_path, map_location="cpu")
            # Accept both raw state_dicts and {"model_state_dict": ...} wrappers
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            else:
                state_dict = checkpoint

            own_state = self.state_dict()
            loaded = 0
            skipped = 0
            for name, param in state_dict.items():
                if name in own_state and own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    loaded += 1
                else:
                    skipped += 1
                    logger.warning(
                        "load_bc_weights: skipping '%s' (shape mismatch or not found)", name
                    )

            self.load_state_dict(own_state)
            logger.info(
                "load_bc_weights: loaded %d params, skipped %d from '%s'",
                loaded, skipped, bc_model_path,
            )
        except Exception as exc:
            logger.warning("load_bc_weights: failed to load '%s': %s", bc_model_path, exc)


# ------------------------------------------------------------------
# Action mask builder
# ------------------------------------------------------------------

def build_action_mask(player_state) -> torch.Tensor:
    """Build a boolean action mask from a player state.

    Accepts either a dataclass/object with attribute access or a plain dict.

    Masking rules
    -------------
    buy_i         (0-6):       shop[i] occupied AND gold >= 3 AND len(hand) < 10
    sell_i        (7-13):      board[i] occupied
    play_h{h}_p{p}(14-83):    hand[h] occupied AND len(board) < 7 AND p <= len(board)
                               index = 14 + h*7 + p
    level_up      (84):        tavern_tier < 6 AND gold >= level_cost
    freeze        (85):        always valid
    refresh       (86):        gold >= 1
    hero_power    (87):        always valid
    end_turn      (88):        always valid
    swap_ij       (89-94):     board[i] and board[i+1] both occupied (i = action-89)

    Returns
    -------
    torch.BoolTensor of shape [N_ACTIONS], True = action is valid.
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

    mask = torch.zeros(N_ACTIONS, dtype=torch.bool)

    # buy_i (0-6)
    can_buy = gold >= 3 and len(hand) < 10
    for i in range(7):
        if can_buy and i < len(shop) and _slot_occupied(shop[i]):
            mask[i] = True

    # sell_i (7-13)
    for i in range(7):
        if i < len(board) and _slot_occupied(board[i]):
            mask[7 + i] = True

    # play_h{h}_p{p} (14-83)
    # Valid insert positions: 0 .. len(board)  (len(board) == append to end)
    # Clipped to PLAY_BOARD_POS-1 (6) so the action exists in the table.
    board_size = len(board)
    if board_size < 7:
        for h in range(PLAY_HAND_SLOTS):
            if h < len(hand) and _slot_occupied(hand[h]):
                for p in range(min(board_size + 1, PLAY_BOARD_POS)):
                    mask[PLAY_START + h * PLAY_BOARD_POS + p] = True

    # level_up (84)
    if tavern_tier < 6 and gold >= level_cost:
        mask[LEVEL_UP_IDX] = True

    # freeze (85): always valid
    mask[FREEZE_IDX] = True

    # refresh (86)
    if gold >= 1:
        mask[REFRESH_IDX] = True

    # hero_power (87): always valid
    mask[HERO_POWER_IDX] = True

    # end_turn (88): always valid
    mask[END_TURN_IDX] = True

    # swap_ij (89-94)
    for i in range(6):
        if i + 1 < len(board) and _slot_occupied(board[i]) and _slot_occupied(board[i + 1]):
            mask[SWAP_START + i] = True

    return mask


def _slot_occupied(slot) -> bool:
    """Return True if the given shop/board slot contains a real minion."""
    if slot is None:
        return False
    if isinstance(slot, dict):
        return bool(slot.get("card_id", ""))
    # MinionState or similar object
    card_id = getattr(slot, "card_id", None)
    return bool(card_id)
