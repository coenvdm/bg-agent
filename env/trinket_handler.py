"""TrinketHandler — trinket offering, equipping, and effect application.

Trinkets are offered at fixed rounds:
  Round 4  (and every 4 rounds after): 3 Lesser trinkets offered, player picks 1
  Round 8  (and every 8 rounds after): 3 Greater trinkets offered, player picks 1

Integration points:
  - game_loop calls maybe_offer(ps, round_num) at round setup
  - When ps.trinket_offer_pending is True, BUY(0/1/2) → select(); END_TURN → decline()
  - apply_on_round_start(ps) fires at the start of each buy phase
  - apply_on_combat_end(ps, result) fires after each combat
"""
from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_LESSER_START    = 4
_LESSER_INTERVAL = 4
_GREATER_START   = 8
_GREATER_INTERVAL = 8
MAX_TRINKETS = 2


class TrinketHandler:
    """Manages trinket lifecycle for all players.

    Parameters
    ----------
    card_defs:
        Full card definition mapping (card_id → def dict). Trinkets are
        identified by card_def["type"] == "TRINKET" or presence of
        card_def["trinket_rarity"] in {"lesser", "greater"}.
    rng:
        Optional random.Random for reproducibility.
    """

    def __init__(self, card_defs: Dict[str, dict], rng: Optional[random.Random] = None) -> None:
        self.card_defs = card_defs
        self._rng = rng or random.Random()
        self._lesser_pool: List[str] = []
        self._greater_pool: List[str] = []

        for card_id, cdef in card_defs.items():
            rarity = cdef.get("trinket_rarity", "").lower()
            ctype  = cdef.get("type", "").upper()
            if ctype == "TRINKET" or rarity in ("lesser", "greater"):
                if rarity == "greater":
                    self._greater_pool.append(card_id)
                else:
                    self._lesser_pool.append(card_id)

        # Per-player pending offers: player_id → list of up to 3 card_ids
        self._pending: Dict[int, List[str]] = {}

    # ------------------------------------------------------------------
    # Round offer
    # ------------------------------------------------------------------

    def _is_lesser_round(self, round_num: int) -> bool:
        return round_num >= _LESSER_START and (round_num - _LESSER_START) % _LESSER_INTERVAL == 0

    def _is_greater_round(self, round_num: int) -> bool:
        return round_num >= _GREATER_START and (round_num - _GREATER_START) % _GREATER_INTERVAL == 0

    def maybe_offer(self, ps, round_num: int) -> bool:
        """Offer trinkets to ps if this round triggers an offer. Returns True if offered."""
        if len(ps.equipped_trinkets) >= MAX_TRINKETS:
            return False

        if self._is_greater_round(round_num):
            pool = self._greater_pool
        elif self._is_lesser_round(round_num):
            pool = self._lesser_pool
        else:
            return False

        available = [cid for cid in pool if cid not in ps.equipped_trinkets]
        if not available:
            return False

        offered = self._rng.sample(available, min(3, len(available)))
        self._pending[ps.player_id] = offered
        ps.trinket_offer_pending = True
        return True

    def get_pending_offer(self, player_id: int) -> List[str]:
        """Return current trinket offer card_ids for player_id (may be empty)."""
        return self._pending.get(player_id, [])

    # ------------------------------------------------------------------
    # Selection / decline
    # ------------------------------------------------------------------

    def select(self, ps, choice_idx: int) -> bool:
        """Equip the trinket at choice_idx from the pending offer. Returns True on success."""
        offered = self._pending.get(ps.player_id, [])
        if not offered or not ps.trinket_offer_pending:
            return False
        if not (0 <= choice_idx < len(offered)):
            return False
        chosen = offered[choice_idx]
        ps.equipped_trinkets.append(chosen)
        ps.trinket_offer_pending = False
        self._pending.pop(ps.player_id, None)
        self._apply_on_equip(ps, chosen)
        return True

    def decline(self, ps) -> None:
        """Player declines the trinket offer without equipping."""
        ps.trinket_offer_pending = False
        self._pending.pop(ps.player_id, None)

    # ------------------------------------------------------------------
    # Effect application
    # ------------------------------------------------------------------

    def _apply_on_equip(self, ps, card_id: str) -> None:
        cdef   = self.card_defs.get(card_id, {})
        effect = cdef.get("trinket_effect", {})
        etype  = effect.get("type", "")

        if etype == "gold_per_round":
            ps.hero_extra_gold += int(effect.get("amount", 1))

        elif etype == "armor":
            ps.armor += int(effect.get("amount", 5))

        elif etype == "passive_spell":
            ps.active_spells[effect.get("effect_id", card_id)] = dict(effect)

        elif etype == "stat_buff_all":
            atk = int(effect.get("atk", 0))
            hp  = int(effect.get("hp", 0))
            for m in ps.board:
                m.perm_atk_bonus += atk
                m.perm_hp_bonus  += hp
                m.max_health     += hp

        elif etype:
            logger.debug("Trinket %s: unhandled effect type '%s'", card_id, etype)

    def apply_on_round_start(self, ps) -> None:
        """Fire round-start trinket effects (e.g. gold_per_round bonus)."""
        for card_id in ps.equipped_trinkets:
            cdef   = self.card_defs.get(card_id, {})
            effect = cdef.get("trinket_effect", {})
            if effect.get("type") == "gold_per_round":
                ps.gold = min(ps.max_gold, ps.gold + int(effect.get("amount", 1)))

    def apply_on_combat_end(self, ps, result: str) -> None:
        """Fire combat-end trinket effects (e.g. stat buffs on win)."""
        for card_id in ps.equipped_trinkets:
            cdef   = self.card_defs.get(card_id, {})
            effect = cdef.get("trinket_effect", {})
            if effect.get("type") == "stat_buff_on_win" and result == "win":
                atk = int(effect.get("atk", 0))
                hp  = int(effect.get("hp", 0))
                for m in ps.board:
                    m.perm_atk_bonus += atk
                    m.perm_hp_bonus  += hp
                    m.max_health     += hp
