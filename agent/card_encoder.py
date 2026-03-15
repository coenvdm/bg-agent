"""
CardEncoder — encodes a Hearthstone Battlegrounds minion as a 44-dim feature vector.

Dim layout (must match exactly):
  0   attack / 20.0
  1   health / 20.0
  2   taunt (bool)
  3   divine_shield (bool)
  4   reborn (bool)
  5   venomous (bool)   [also catches "poisonous" from log]
  6   windfury (bool)
  7   golden (bool)
  8   magnetic (bool)
  9   tier / 7.0
 10   zone_pos / 7.0
 11   is_present (1.0 for real card, 0.0 for padding)
     # backward-compatible with existing 11-dim BC encoding above this line
 12   tribe_BEAST
 13   tribe_DEMON
 14   tribe_DRAGON
 15   tribe_ELEMENTAL
 16   tribe_MECH
 17   tribe_MURLOC
 18   tribe_NAGA
 19   tribe_PIRATE
 20   tribe_QUILBOAR
 21   tribe_UNDEAD
     # trigger type one-hot (7 slots)
 22   trigger_battlecry
 23   trigger_deathrattle
 24   trigger_end_of_turn
 25   trigger_start_of_combat
 26   trigger_on_sell
 27   trigger_rally
 28   trigger_other  (avenge, spellcraft, passive, on_buy)
     # effect duration one-hot
 29   duration_permanent
 30   duration_this_combat
 31   duration_this_game
     # special flags from card_def
 32   is_multiplier
 33   is_aura
 34   scales_with_board
     # accumulated permanent buffs
 35   perm_atk_bonus / 20.0
 36   perm_hp_bonus / 20.0
     # avenge / deathrattle summon metadata
 37   avenge_count / 5.0   (0 if not avenge card)
 38   dr_summons_minion    (bool: deathrattle summons a token)
     # board context (same for all cards in the same encoding call)
 39   board_size / 7.0
 40   tribe_synergy_score  (dominant tribe count / 7.0)
 41   total_aura_dependency  (clipped [0, 1])
 42   round_num / 25.0
 43   tavern_tier / 7.0
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np

CARD_FEATURE_DIM: int = 44

TRIBE_LIST: List[str] = [
    "BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECH",
    "MURLOC", "NAGA", "PIRATE", "QUILBOAR", "UNDEAD",
]
TRIBE_IDX: Dict[str, int] = {t: i for i, t in enumerate(TRIBE_LIST)}

# Trigger types with assigned one-hot indices 22-27; index 28 is "other".
TRIGGER_TYPES: List[str] = [
    "battlecry",
    "deathrattle",
    "end_of_turn",
    "start_of_combat",
    "on_sell",
    "rally",
]
TRIGGER_IDX: Dict[str, int] = {t: i for i, t in enumerate(TRIGGER_TYPES)}

# Effect durations with assigned one-hot indices 29-31; "instant" → all zeros.
DURATION_TYPES: List[str] = ["permanent", "this_combat", "this_game"]
DURATION_IDX: Dict[str, int] = {t: i for i, t in enumerate(DURATION_TYPES)}


class CardEncoder:
    """Encodes a single minion (or a full board) into fixed-size feature vectors.

    All encoding is done with numpy only; torch is never imported here.

    Parameters
    ----------
    card_defs:
        The full card-definition dictionary loaded from bg_card_definitions.json.
    """

    def __init__(self, card_defs: Dict[str, dict]):
        self.card_defs = card_defs

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _normalize_card_id(self, card_id: str) -> str:
        """Normalise to match card_defs keys.

        Lowercases, converts spaces and hyphens to underscores, strips
        apostrophes, commas, and dots.
        """
        s = card_id.lower()
        s = re.sub(r"[\s\-]", "_", s)
        s = re.sub(r"[',.]", "", s)
        return s

    def _get_def(self, card_id: str) -> Optional[dict]:
        """Look up card definition, trying the raw key and several normalisations.

        Returns None when the card is not in card_defs (graceful degradation).
        """
        if not card_id:
            return None
        # 1. Exact match
        if card_id in self.card_defs:
            return self.card_defs[card_id]
        # 2. Normalised match
        norm = self._normalize_card_id(card_id)
        if norm in self.card_defs:
            return self.card_defs[norm]
        # 3. Normalised substring match against keys (last resort, pick first hit)
        for key, val in self.card_defs.items():
            if norm and norm in self._normalize_card_id(key):
                return val
            if norm and norm == self._normalize_card_id(val.get("name", "")):
                return val
        return None

    # ------------------------------------------------------------------
    # Single-minion encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        minion: dict,
        card_defs_override: Optional[dict] = None,
        board_size: int = 0,
        dominant_tribe_count: int = 0,
        total_aura_dependency: float = 0.0,
        round_num: int = 1,
        tavern_tier: int = 1,
    ) -> np.ndarray:
        """Encode a single minion.  Returns float32 array of shape (44,).

        Parameters
        ----------
        minion:
            A plain dict whose keys match MinionState field names (or the snap
            dict produced by parse_bg.py).  Unrecognised keys are ignored.
        card_defs_override:
            If provided, use this dict instead of self.card_defs for look-ups.
        board_size, dominant_tribe_count, total_aura_dependency,
        round_num, tavern_tier:
            Board-level context applied identically to all cards encoded in the
            same call (dims 39-43).
        """
        feat = np.zeros(CARD_FEATURE_DIM, dtype=np.float32)

        # ---- Dims 0-11: base stats (always from the live minion dict) ----
        feat[0] = minion.get("attack", 0) / 20.0
        feat[1] = minion.get("health", 0) / 20.0
        feat[2] = float(bool(minion.get("taunt", False)))
        feat[3] = float(bool(minion.get("divine_shield", False)))
        feat[4] = float(bool(minion.get("reborn", False)))
        # "poisonous" is the parser key; "venomous" is the normalised field name
        feat[5] = float(bool(minion.get("venomous", minion.get("poisonous", False))))
        feat[6] = float(bool(minion.get("windfury", False)))
        feat[7] = float(bool(minion.get("golden", False)))
        # dim 8 (magnetic) is filled from card_def below
        feat[9] = minion.get("tier", 1) / 7.0
        feat[10] = minion.get("zone_pos", 0) / 7.0
        feat[11] = 1.0  # is_present — padding slots are left at 0.0

        # ---- Dims 12-43: from card_def (gracefully degrade if unknown) ----
        card_defs_active = card_defs_override if card_defs_override is not None else self.card_defs
        # Temporarily swap for look-up if override provided
        if card_defs_override is not None:
            _orig = self.card_defs
            self.card_defs = card_defs_override
            cdef = self._get_def(minion.get("card_id", ""))
            self.card_defs = _orig
        else:
            cdef = self._get_def(minion.get("card_id", ""))

        if cdef:
            # dim 8: magnetic
            feat[8] = float(bool(
                cdef.get("has_magnetic", False)
                or cdef.get("keywords", {}).get("magnetic", False)
            ))

            # dims 12-21: tribe one-hot (multi-tribe cards set multiple bits)
            for t in cdef.get("tribes", []):
                t_upper = t.upper()
                if t_upper in TRIBE_IDX:
                    feat[12 + TRIBE_IDX[t_upper]] = 1.0

            # dims 22-28: trigger type one-hot
            trigger = cdef.get("trigger_type", "passive")
            if trigger in TRIGGER_IDX:
                feat[22 + TRIGGER_IDX[trigger]] = 1.0
            else:
                # avenge, spellcraft, passive, on_buy → "other"
                feat[28] = 1.0

            # dims 29-31: effect duration one-hot ("instant" → all zeros)
            duration = cdef.get("effect_duration", "instant")
            if duration in DURATION_IDX:
                feat[29 + DURATION_IDX[duration]] = 1.0

            # dims 32-34: special flags
            feat[32] = float(bool(cdef.get("is_multiplier", False)))
            feat[33] = float(bool(cdef.get("is_aura", False)))
            feat[34] = float(bool(cdef.get("scales_with_board", False)))

            # dim 37: avenge counter normalised
            avenge = cdef.get("avenge_count")
            if avenge is not None:
                feat[37] = avenge / 5.0

            # dim 38: deathrattle that summons a token
            raw = cdef.get("raw_text", "")
            feat[38] = float(
                "Deathrattle: Summon" in raw
                or "deathrattle: summon" in raw.lower()
            )

        # ---- Dims 35-36: permanent buff deltas (from MinionState if present) ----
        feat[35] = minion.get("perm_atk_bonus", 0) / 20.0
        feat[36] = minion.get("perm_hp_bonus", 0) / 20.0

        # ---- Dims 39-43: board context ----
        feat[39] = board_size / 7.0
        feat[40] = dominant_tribe_count / 7.0
        feat[41] = min(float(total_aura_dependency), 1.0)
        feat[42] = round_num / 25.0
        feat[43] = tavern_tier / 7.0

        return feat

    # ------------------------------------------------------------------
    # Board encoding (padded to max_slots × 44)
    # ------------------------------------------------------------------

    def encode_board(
        self,
        minions: List[dict],
        board_size: Optional[int] = None,
        dominant_tribe_count: int = 0,
        total_aura_dependency: float = 0.0,
        round_num: int = 1,
        tavern_tier: int = 1,
        max_slots: int = 7,
    ) -> np.ndarray:
        """Encode a list of minions with zero-padding.

        Returns float32 array of shape (max_slots, 44).  Padding rows have
        all zeros (is_present dim 11 remains 0.0 for padding slots).

        Parameters
        ----------
        minions:
            Up to max_slots minion dicts.  Extras beyond max_slots are ignored.
        board_size:
            Number of real minions (defaults to len(minions)).
        dominant_tribe_count:
            Count of the most common tribe on the board (for dim 40).
        total_aura_dependency:
            Summed aura dependency score from SymbolicBoardComputer (dim 41).
        round_num, tavern_tier:
            Game state context shared across the board encoding.
        max_slots:
            Number of rows in the output array (default 7).
        """
        result = np.zeros((max_slots, CARD_FEATURE_DIM), dtype=np.float32)
        if board_size is None:
            board_size = len(minions)

        for i, m in enumerate(minions[:max_slots]):
            # Always use the actual array index as zone_pos so the feature
            # stays correct after board swaps (MinionState.zone_pos is stale
            # once positions shift).
            m_with_pos = dict(m, zone_pos=i + 1) if isinstance(m, dict) else m
            if not isinstance(m_with_pos, dict):
                m_with_pos = {**vars(m_with_pos), "zone_pos": i + 1}
            result[i] = self.encode(
                m_with_pos,
                board_size=board_size,
                dominant_tribe_count=dominant_tribe_count,
                total_aura_dependency=total_aura_dependency,
                round_num=round_num,
                tavern_tier=tavern_tier,
            )

        return result
