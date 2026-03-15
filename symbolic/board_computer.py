"""
SymbolicBoardComputer — deterministic mechanical analysis of a BG board.
Implements the 5 symbolic layer rules from CLAUDE.md.

Rule order: 1 (multipliers) → 4 (tribes) → 2 (auras) → 3 (effect durations) → combat stats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import torch

TRIBE_LIST = [
    "BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECH",
    "MURLOC", "NAGA", "PIRATE", "QUILBOAR", "UNDEAD",
]

# Aura cards whose removal would hurt board power.
# Maps normalised card_id fragment → (atk_bonus_per_trigger, hp_bonus_per_trigger)
# These are used for a rough delta estimate when the card_def marks is_aura=True.
_KNOWN_AURA_CARDS = {
    "twilight_watcher": (1, 3),       # +1/+3 per Dragon attack
    "roaring_recruiter": (3, 1),      # +3/+1 per Dragon attack (on attacker)
    "shore_marauder": (1, 1),         # passive +1/+1 to Pirates + Elementals
    "lord_of_the_ruins": (2, 1),      # +2/+1 after Demon deals damage
    "hardy_orca": (1, 1),             # +1/+1 when this takes damage
    "iridescent_skyblazer": (3, 1),   # +3/+1 when Beast takes damage
}


@dataclass
class BoardFeatures:
    # Rule 1: Multipliers (always detected first)
    brann_active: bool = False
    titus_active: bool = False
    drakkari_active: bool = False

    # Rule 4: Tribal density
    tribe_counts: Dict[str, int] = field(default_factory=dict)   # 10 tribes
    tribe_densities: Dict[str, float] = field(default_factory=dict)
    dominant_tribe: Optional[str] = None
    is_synergistic: bool = False  # any tribe count >= 4

    # Rule 2: Aura dependency
    aura_sources: List[Tuple[str, float]] = field(default_factory=list)  # (card_id, score)
    total_aura_dependency: float = 0.0

    # Rule 3: Effect duration profile
    permanent_effect_count: int = 0
    this_combat_count: int = 0
    this_game_count: int = 0

    # Combat power
    dr_count: int = 0
    dr_summon_count: int = 0
    dr_buff_count: int = 0
    effective_dr_count: int = 0       # dr_count * (2 if titus else 1)
    total_atk: int = 0
    total_hp: int = 0
    divine_shield_count: int = 0
    venomous_count: int = 0
    taunt_count: int = 0
    reborn_count: int = 0
    windfury_count: int = 0
    avg_atk: float = 0.0
    avg_hp: float = 0.0
    board_size: int = 0

    # Filled by firestone_client later
    win_prob: float = 0.5
    expected_damage_dealt: float = 0.0
    expected_damage_taken: float = 0.0

    def to_scalar_vector(self) -> np.ndarray:
        """Return 24-dim numpy scalar summary for policy network input."""
        tribe_dens = [self.tribe_densities.get(t, 0.0) for t in TRIBE_LIST]
        # 6 + 10 + 8 = 24 dims total
        v = [
            self.win_prob,                              # 0
            self.expected_damage_dealt / 40.0,          # 1
            self.expected_damage_taken / 40.0,          # 2
            float(self.brann_active),                   # 3
            float(self.titus_active),                   # 4
            float(self.drakkari_active),                # 5
        ] + tribe_dens + [                              # 6-15
            min(self.total_aura_dependency, 1.0),       # 16
            self.dr_count / 7.0,                        # 17
            float(self.is_synergistic),                 # 18
            self.board_size / 7.0,                      # 19
            self.divine_shield_count / 7.0,             # 20
            self.venomous_count / 7.0,                  # 21
            self.taunt_count / 7.0,                     # 22
            self.reborn_count / 7.0,                    # 23
        ]
        assert len(v) == 24, f"to_scalar_vector: expected 24 dims, got {len(v)}"
        return np.array(v, dtype=np.float32)


def _minion_to_dict(m) -> dict:
    """Normalise a MinionState or dict to a plain dict for uniform processing."""
    if isinstance(m, dict):
        return m
    # MinionState or any object with __dict__
    return m.__dict__ if hasattr(m, "__dict__") else {}


def _board_power(board_dicts: List[dict]) -> float:
    """Sum of effective_attack + effective_health for all minions."""
    total = 0.0
    for m in board_dicts:
        atk = m.get("attack", 0) + m.get("perm_atk_bonus", 0) + m.get("game_atk_bonus", 0)
        hp = m.get("health", 0) + m.get("perm_hp_bonus", 0) + m.get("game_hp_bonus", 0)
        total += atk + hp
    return max(total, 1.0)


class SymbolicBoardComputer:
    """Deterministic symbolic analysis of a Battlegrounds board.

    Follows the 5 rules defined in CLAUDE.md in the prescribed order:
    1. Detect multipliers first.
    2. Compute aura dependency scores.
    3. Tag effect durations.
    4. Compute tribal density.
    (5 implied) Compute combat keyword / DR stats.
    """

    MULTIPLIER_NAMES = {
        "brann_bronzebeard": "brann",
        "titus_rivendare": "titus",
        "drakkari_enchanter": "drakkari",
    }

    def __init__(self, card_defs: Dict[str, dict]):
        self.card_defs = card_defs

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _card_id_from_name(self, card_id: str) -> str:
        """Normalise card_id: lowercase, spaces→underscores, strip punctuation."""
        s = card_id.lower()
        s = s.replace(" ", "_")
        s = re.sub(r"['\",.\-]", "", s)
        return s

    def _get_def(self, card_id: str) -> Optional[dict]:
        """Look up card definition, trying the raw key and several normalisations."""
        if not card_id:
            return None
        # 1. Exact match
        if card_id in self.card_defs:
            return self.card_defs[card_id]
        # 2. Normalised
        norm = self._card_id_from_name(card_id)
        if norm in self.card_defs:
            return self.card_defs[norm]
        # 3. Substring match on normalised keys (last resort)
        for key, val in self.card_defs.items():
            if norm and norm in self._card_id_from_name(key):
                return val
            # Also match by name field
            name_norm = self._card_id_from_name(val.get("name", ""))
            if norm and norm == name_norm:
                return val
        return None

    # ------------------------------------------------------------------
    # Rule 1: Multiplier detection
    # ------------------------------------------------------------------

    def _detect_multipliers(self, board: List[dict]) -> Tuple[bool, bool, bool]:
        """Rule 1: detect Brann, Titus, Drakkari on board.

        Returns (brann_active, titus_active, drakkari_active).
        """
        brann = titus = drakkari = False
        for m in board:
            cid = m.get("card_id", "")
            cdef = self._get_def(cid)

            # Prefer card_def is_multiplier flag
            if cdef and cdef.get("is_multiplier"):
                name_norm = self._card_id_from_name(cdef.get("name", cid))
                if "brann" in name_norm:
                    brann = True
                elif "titus" in name_norm:
                    titus = True
                elif "drakkari" in name_norm:
                    drakkari = True
                continue

            # Fallback: match well-known card_id substrings used by the log parser
            cid_lower = cid.lower()
            if "brann" in cid_lower or "tb_baconups_800" in cid_lower:
                brann = True
            elif "titus" in cid_lower or "tb_baconups_116" in cid_lower:
                titus = True
            elif "drakkari" in cid_lower or "tb_baconups_090" in cid_lower:
                drakkari = True

        return brann, titus, drakkari

    # ------------------------------------------------------------------
    # Rule 4: Tribal density
    # ------------------------------------------------------------------

    def _compute_tribes(
        self, board: List[dict]
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        """Rule 4: tribal density.  Multi-tribe cards count for every tribe.

        Returns (tribe_counts, tribe_densities).
        tribe_densities[tribe] = count / board_size  (0 if board empty).
        """
        counts: Dict[str, int] = {t: 0 for t in TRIBE_LIST}
        n = len(board)

        for m in board:
            cid = m.get("card_id", "")
            cdef = self._get_def(cid)
            tribes_for_card: List[str] = []

            if cdef:
                tribes_for_card = [
                    t.upper() for t in cdef.get("tribes", [])
                    if t.upper() in counts
                ]
            # No def: nothing counted (unrecognised card contributes no tribe)

            for t in tribes_for_card:
                counts[t] += 1

        if n > 0:
            densities = {t: counts[t] / n for t in TRIBE_LIST}
        else:
            densities = {t: 0.0 for t in TRIBE_LIST}

        return counts, densities

    # ------------------------------------------------------------------
    # Rule 2: Aura dependency
    # ------------------------------------------------------------------

    def _compute_auras(
        self, board: List[dict]
    ) -> Tuple[List[Tuple[str, float]], float]:
        """Rule 2: aura_dependency_score = (power_with − power_without) / power_with.

        Returns (aura_sources, total_aura_dependency).
        aura_sources is a list of (card_id, score) for each aura minion.
        total_aura_dependency is the sum of all scores (clipped to 1.0 later by caller).
        """
        aura_sources: List[Tuple[str, float]] = []
        total_dep = 0.0
        power_with = _board_power(board)

        for m in board:
            cid = m.get("card_id", "")
            cdef = self._get_def(cid)
            if not (cdef and cdef.get("is_aura")):
                continue

            # Estimate board power without this aura source
            board_without = [x for x in board if x is not m]
            power_without = _board_power(board_without)

            # The aura source itself contributes its own stats; subtract those
            # so we measure only the *aura* contribution to the rest of the board.
            # We approximate: power_with_aura = power_with, power_without_aura = power_without.
            if power_with > 0:
                score = max(0.0, (power_with - power_without) / power_with)
            else:
                score = 0.0

            aura_sources.append((cid, score))
            total_dep += score

        return aura_sources, total_dep

    # ------------------------------------------------------------------
    # Rule 3: Effect durations
    # ------------------------------------------------------------------

    def _tag_effects(self, board: List[dict]) -> Tuple[int, int, int]:
        """Rule 3: count minions by their effect_duration.

        Returns (permanent_count, this_combat_count, this_game_count).
        """
        perm = combat = game = 0
        for m in board:
            cdef = self._get_def(m.get("card_id", ""))
            if not cdef:
                continue
            dur = cdef.get("effect_duration", "instant")
            if dur == "permanent":
                perm += 1
            elif dur == "this_combat":
                combat += 1
            elif dur == "this_game":
                game += 1
        return perm, combat, game

    # ------------------------------------------------------------------
    # Combat stats helper
    # ------------------------------------------------------------------

    def _compute_combat_stats(self, board: List[dict], titus_active: bool) -> dict:
        """Compute DR counts, keyword counts, and total atk/hp."""
        dr_count = dr_summon = dr_buff = 0
        total_atk = total_hp = 0
        ds_count = ven_count = taunt_count = reborn_count = wf_count = 0

        for m in board:
            # Keywords — prefer live minion data, fall back to card_def
            total_atk += (
                m.get("attack", 0)
                + m.get("perm_atk_bonus", 0)
                + m.get("game_atk_bonus", 0)
            )
            total_hp += (
                m.get("health", 0)
                + m.get("perm_hp_bonus", 0)
                + m.get("game_hp_bonus", 0)
            )
            if m.get("divine_shield"):
                ds_count += 1
            if m.get("venomous") or m.get("poisonous"):
                ven_count += 1
            if m.get("taunt"):
                taunt_count += 1
            if m.get("reborn"):
                reborn_count += 1
            if m.get("windfury"):
                wf_count += 1

            cdef = self._get_def(m.get("card_id", ""))
            if cdef and cdef.get("trigger_type") == "deathrattle":
                dr_count += 1
                raw = cdef.get("raw_text", "")
                raw_lower = raw.lower()
                if "summon" in raw_lower:
                    dr_summon += 1
                else:
                    dr_buff += 1

        effective_dr = dr_count * (2 if titus_active else 1)
        n = len(board)
        return {
            "dr_count": dr_count,
            "dr_summon_count": dr_summon,
            "dr_buff_count": dr_buff,
            "effective_dr_count": effective_dr,
            "total_atk": total_atk,
            "total_hp": total_hp,
            "divine_shield_count": ds_count,
            "venomous_count": ven_count,
            "taunt_count": taunt_count,
            "reborn_count": reborn_count,
            "windfury_count": wf_count,
            "avg_atk": total_atk / n if n else 0.0,
            "avg_hp": total_hp / n if n else 0.0,
            "board_size": n,
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute(
        self,
        board,
        gold: int = 0,
        round_num: int = 1,
        tavern_tier: int = 1,
    ) -> BoardFeatures:
        """Main entry: follow CLAUDE.md rules 1 → 4 → 2 → 3 → combat.

        board can be List[MinionState] or List[dict].
        """
        # Normalise to list of dicts
        board_dicts: List[dict] = [_minion_to_dict(m) for m in board]

        features = BoardFeatures()

        # Rule 1: multipliers first
        brann, titus, drakkari = self._detect_multipliers(board_dicts)
        features.brann_active = brann
        features.titus_active = titus
        features.drakkari_active = drakkari

        # Rule 4: tribal density
        counts, densities = self._compute_tribes(board_dicts)
        features.tribe_counts = counts
        features.tribe_densities = densities
        if counts:
            dominant = max(counts, key=lambda t: counts[t])
            features.dominant_tribe = dominant if counts[dominant] > 0 else None
        features.is_synergistic = any(v >= 4 for v in counts.values())

        # Rule 2: aura dependency
        aura_sources, total_dep = self._compute_auras(board_dicts)
        features.aura_sources = aura_sources
        features.total_aura_dependency = total_dep

        # Rule 3: effect durations
        perm, combat_cnt, game_cnt = self._tag_effects(board_dicts)
        features.permanent_effect_count = perm
        features.this_combat_count = combat_cnt
        features.this_game_count = game_cnt

        # Combat stats
        cs = self._compute_combat_stats(board_dicts, titus)
        features.dr_count = cs["dr_count"]
        features.dr_summon_count = cs["dr_summon_count"]
        features.dr_buff_count = cs["dr_buff_count"]
        features.effective_dr_count = cs["effective_dr_count"]
        features.total_atk = cs["total_atk"]
        features.total_hp = cs["total_hp"]
        features.divine_shield_count = cs["divine_shield_count"]
        features.venomous_count = cs["venomous_count"]
        features.taunt_count = cs["taunt_count"]
        features.reborn_count = cs["reborn_count"]
        features.windfury_count = cs["windfury_count"]
        features.avg_atk = cs["avg_atk"]
        features.avg_hp = cs["avg_hp"]
        features.board_size = cs["board_size"]

        return features

    # ------------------------------------------------------------------
    # Network encoding
    # ------------------------------------------------------------------

    def encode_board_for_network(
        self,
        board,
        features: BoardFeatures,
        gold: int = 0,
        round_num: int = 1,
        tavern_tier: int = 1,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Return (tokens, scalar) tensors for the policy network.

        tokens : FloatTensor [7, 44]  — padded per-card encodings
        scalar : FloatTensor [24]     — board-level summary

        CardEncoder is imported lazily to avoid circular imports at module load.
        torch is also imported lazily.
        """
        import torch  # lazy import — only needed when calling this method

        # Lazy import to avoid circular dependency
        from agent.card_encoder import CardEncoder  # noqa: PLC0415

        encoder = CardEncoder(self.card_defs)
        board_dicts: List[dict] = [_minion_to_dict(m) for m in board]

        dominant_tribe_count = (
            features.tribe_counts.get(features.dominant_tribe, 0)
            if features.dominant_tribe
            else 0
        )

        tokens_np = encoder.encode_board(
            board_dicts,
            board_size=features.board_size,
            dominant_tribe_count=dominant_tribe_count,
            total_aura_dependency=features.total_aura_dependency,
            round_num=round_num,
            tavern_tier=tavern_tier,
            max_slots=7,
        )
        scalar_np = features.to_scalar_vector()

        tokens = torch.tensor(tokens_np, dtype=torch.float32)
        scalar = torch.tensor(scalar_np, dtype=torch.float32)
        return tokens, scalar
