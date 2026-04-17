"""
Per-card buy-phase value estimation for the symbolic layer.

ShopAnalyzer scores each card visible in the tavern shop and returns a ranked
list of ShopCardValue objects.  It intentionally does not import torch or any
neural-network code — it is a pure symbolic/heuristic module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from symbolic.board_computer import BoardFeatures, SymbolicBoardComputer

# Weights for total_value formula
_W_BASE = 0.40
_W_SYNERGY = 0.30
_W_TEMPO = 0.15
_W_SCALING = 0.15

# Recommendation thresholds
_STRONG_BUY_THRESH = 0.50
_CONSIDER_THRESH = 0.30


@dataclass
class ShopCardValue:
    card_id: str
    name: str
    base_power: float       # (atk + hp) normalised + keyword bonuses
    synergy_bonus: float    # tribe synergy with the current board
    tempo_value: float      # immediate impact estimate
    scaling_value: float    # long-term / permanent-effect estimate
    total_value: float      # weighted sum of the above components
    recommendation: str     # "strong_buy" | "consider" | "pass"
    aura_context: str = "neutral"  # "reduces_dependency" | "adds_aura_redundancy" | "neutral"


class ShopAnalyzer:
    """Score each shop card for buying value given the current board state.

    Parameters
    ----------
    card_defs:
        The full card-definition dictionary (from bg_card_definitions.json).
    board_computer:
        A SymbolicBoardComputer instance (used for its _get_def helper).
    """

    def __init__(
        self,
        card_defs: Dict[str, dict],
        board_computer: "SymbolicBoardComputer",
    ):
        self.card_defs = card_defs
        self.board_computer = board_computer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_shop(
        self,
        shop_cards: List[dict],
        board_features: "BoardFeatures",
        gold: int,
        tavern_tier: int,
    ) -> List[ShopCardValue]:
        """Score each shop card and return a sorted list (highest first).

        Parameters
        ----------
        shop_cards:
            List of minion dicts from the current shop (same schema as
            MinionState.__dict__ or _minion_snap).
        board_features:
            Pre-computed BoardFeatures for the player's current warband.
        gold:
            Gold currently available (reserved for future cost gating).
        tavern_tier:
            Current tavern tier (1-7).
        """
        # Detect which multiplier card_ids are already on the board so we can
        # penalise duplicate multipliers in the shop.
        active_multiplier_names: set[str] = set()
        if board_features.brann_active:
            active_multiplier_names.add("brann")
        if board_features.titus_active:
            active_multiplier_names.add("titus")
        if board_features.drakkari_active:
            active_multiplier_names.add("drakkari")

        results: List[ShopCardValue] = []
        aura_dep = getattr(board_features, "total_aura_dependency", 0.0)

        for card in shop_cards:
            cid = card.get("card_id", "")
            cdef = self.board_computer._get_def(cid)
            name = (cdef.get("name") if cdef else None) or card.get("name", cid)

            base_power = self._estimate_card_power(card, cdef)
            synergy_bonus = self._synergy_bonus(card, cdef, board_features, aura_dep)
            tempo_value = self._tempo_value(cdef)
            scaling_value = self._scaling_value(cdef, board_features)

            # Penalty: if a multiplier of this type is already active, buying a
            # second copy is much weaker (golden synergy aside, treat as penalty).
            if cdef and cdef.get("is_multiplier"):
                mult_name = self._multiplier_name(cdef.get("name", ""))
                if mult_name in active_multiplier_names:
                    scaling_value = max(0.0, scaling_value - 0.10)

            total_value = (
                _W_BASE * base_power
                + _W_SYNERGY * synergy_bonus
                + _W_TEMPO * tempo_value
                + _W_SCALING * scaling_value
            )

            # Slight penalty for over-committing to aura strategy
            if cdef and cdef.get("is_aura") and aura_dep > 0.6:
                total_value = max(0.0, total_value - 0.05)

            if total_value > _STRONG_BUY_THRESH:
                recommendation = "strong_buy"
            elif total_value > _CONSIDER_THRESH:
                recommendation = "consider"
            else:
                recommendation = "pass"

            # Aura context label
            is_aura_card = bool(cdef and cdef.get("is_aura"))
            if is_aura_card and aura_dep >= 0.3:
                aura_context = "adds_aura_redundancy"
            elif not is_aura_card and aura_dep >= 0.5:
                aura_context = "reduces_dependency"
            else:
                aura_context = "neutral"

            results.append(
                ShopCardValue(
                    card_id=cid,
                    name=name,
                    base_power=round(base_power, 4),
                    synergy_bonus=round(synergy_bonus, 4),
                    tempo_value=round(tempo_value, 4),
                    scaling_value=round(scaling_value, 4),
                    total_value=round(total_value, 4),
                    recommendation=recommendation,
                    aura_context=aura_context,
                )
            )

        results.sort(key=lambda x: x.total_value, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Component estimators
    # ------------------------------------------------------------------

    def _estimate_card_power(self, card: dict, cdef: Optional[dict]) -> float:
        """Power estimate from atk + hp + keyword bonuses.

        Normalised to [0, ~1]: a vanilla 20/20 would be 2.0 before cap, but in
        practice Battlegrounds cards sit comfortably below 1.0.
        """
        atk = card.get("attack", 0)
        hp = card.get("health", 0)
        power = (atk + hp) / 40.0

        # Keyword bonuses (independent of card_def so they work on live minions)
        if card.get("divine_shield"):
            power += 0.10
        if card.get("venomous") or card.get("poisonous"):
            power += 0.15
        if card.get("reborn"):
            power += 0.08
        if card.get("taunt"):
            power += 0.03
        if card.get("windfury"):
            power += 0.06
        if card.get("golden"):
            power *= 1.8

        return power

    def _synergy_bonus(
        self,
        card: dict,
        cdef: Optional[dict],
        board_features: "BoardFeatures",
        aura_dep: float = 0.0,
    ) -> float:
        """Tribe-synergy bonus: +0.3 if the card shares the dominant tribe
        on a synergistic board (>= 4 of that tribe already out).

        Additionally: multiplier cards get a flat +0.2 synergy bonus because
        they improve every other card on the board.
        """
        bonus = 0.0

        # Multipliers synergise with everything
        if cdef and cdef.get("is_multiplier"):
            bonus += 0.20

        # Tribe synergy
        if board_features.is_synergistic and board_features.dominant_tribe:
            card_tribes: List[str] = []
            if cdef:
                card_tribes = [t.upper() for t in cdef.get("tribes", [])]
            if board_features.dominant_tribe in card_tribes:
                bonus += 0.30

        # Aura source bonus: rewards adding aura redundancy to a fragile board
        if cdef and cdef.get("is_aura"):
            bonus += 0.25 if aura_dep >= 0.3 else 0.15

        # Non-aura cards are more resilient when board has high aura dependency
        elif aura_dep >= 0.5:
            bonus += 0.10

        return min(bonus, 0.50)  # cap to prevent overshooting

    def _tempo_value(self, cdef: Optional[dict]) -> float:
        """Immediate-impact estimate based on trigger type."""
        if not cdef:
            return 0.05
        trigger = cdef.get("trigger_type", "passive")
        return {
            "battlecry": 0.20,
            "start_of_combat": 0.15,
            "on_sell": 0.10,
            "deathrattle": 0.12,
            "avenge": 0.10,
            "rally": 0.12,
            "end_of_turn": 0.10,
            "spellcraft": 0.08,
            "passive": 0.08,
            "on_buy": 0.05,
        }.get(trigger, 0.05)

    def _scaling_value(self, cdef: Optional[dict], board_features: "BoardFeatures" = None) -> float:
        """Long-term scaling estimate based on effect duration and board-scaling."""
        if not cdef:
            return 0.05
        base = {
            "permanent": 0.30,
            "this_game": 0.20,
            "this_combat": 0.10,
            "instant": 0.05,
        }.get(cdef.get("effect_duration", "instant"), 0.05)

        if cdef.get("scales_with_board"):
            base += 0.10

        if cdef.get("is_multiplier"):
            base += 0.15

        if cdef.get("trigger_type") == "battlecry" and cdef.get("effect_duration") == "this_game":
            base += 0.08

        # Tribal synergy card scales better in synergistic boards
        if board_features is not None and board_features.is_synergistic and board_features.dominant_tribe:
            card_tribes = [t.upper() for t in cdef.get("tribes", [])]
            if board_features.dominant_tribe in card_tribes:
                base += 0.05

        return base

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _multiplier_name(card_name: str) -> str:
        """Return a short identifier for a multiplier card."""
        n = card_name.lower()
        if "brann" in n:
            return "brann"
        if "titus" in n:
            return "titus"
        if "drakkari" in n:
            return "drakkari"
        return ""
