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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tribe_match(minion, tribe: str) -> bool:
        tribes = getattr(minion, "tribes", None) or []
        if isinstance(tribes, str):
            tribes = [tribes]
        if tribe in [t.upper() for t in tribes]:
            return True
        single = getattr(minion, "tribe", "") or ""
        return single.upper() == tribe

    @staticmethod
    def _buff_minions(minions, atk: int, hp: int, *, permanent: bool = True) -> None:
        for m in minions:
            if permanent:
                m.perm_atk_bonus += atk
                m.perm_hp_bonus  += hp
            else:
                m.attack  += atk
                m.health  += hp
            m.max_health += hp

    def _apply_on_equip(self, ps, card_id: str) -> None:
        cdef   = self.card_defs.get(card_id, {})
        effect = cdef.get("trinket_effect", {})
        etype  = effect.get("type", "")

        if etype == "gold_per_round":
            # Tracked in apply_on_round_start; nothing to do on equip
            pass

        elif etype == "gold_gain":
            amount = int(effect.get("amount", 0))
            ps.gold = min(ps.max_gold, ps.gold + amount)
            mx = int(effect.get("max_gold_increase", 0))
            if mx:
                ps.max_gold = min(20, ps.max_gold + mx)

        elif etype == "max_gold_increase":
            ps.max_gold = min(20, ps.max_gold + int(effect.get("amount", 1)))

        elif etype == "armor":
            ps.armor += int(effect.get("amount", 5))

        elif etype == "passive_spell":
            ps.active_spells[effect.get("effect_id", card_id)] = dict(effect)

        elif etype == "stat_buff_all":
            atk = int(effect.get("atk", 0))
            hp  = int(effect.get("hp", 0))
            self._buff_minions(ps.board, atk, hp)

        elif etype == "stat_buff_tribe":
            tribe = effect.get("tribe", "")
            atk   = int(effect.get("atk", 0))
            hp    = int(effect.get("hp", 0))
            self._buff_minions(
                [m for m in ps.board if self._tribe_match(m, tribe)], atk, hp
            )

        elif etype == "stat_buff_low_tier":
            max_tier = int(effect.get("max_tier", 3))
            atk      = int(effect.get("atk", 0))
            hp       = int(effect.get("hp", 0))
            self._buff_minions(
                [m for m in ps.board if getattr(m, "tier", 1) <= max_tier], atk, hp
            )

        elif etype == "level_cost_reduction":
            ps.level_cost = max(0, ps.level_cost - int(effect.get("amount", 1)))

        elif etype in (
            "start_of_combat", "start_of_combat_buff_all", "start_of_combat_buff_tribe",
            "end_of_turn", "end_of_turn_buff_all", "end_of_turn_buff_leftmost",
            "end_of_turn_buff_tribe", "max_gold_per_round", "level_cost_reduction_per_round",
            "avenge", "combat_trigger", "spellcraft", "discover",
            "tavern_aura", "round_start_effect", "stat_buff_on_win", "complex",
        ):
            # Handled in their respective hooks; nothing to do on equip
            pass

        elif etype:
            logger.debug("Trinket %s: unhandled effect type '%s'", card_id, etype)

    def apply_on_round_start(self, ps) -> None:
        """Fire round-start trinket effects."""
        for card_id in ps.equipped_trinkets:
            cdef   = self.card_defs.get(card_id, {})
            effect = cdef.get("trinket_effect", {})
            etype  = effect.get("type", "")

            if etype == "gold_per_round":
                ps.gold = min(ps.max_gold, ps.gold + int(effect.get("amount", 1)))

            elif etype == "max_gold_per_round":
                ps.max_gold = min(20, ps.max_gold + int(effect.get("amount", 1)))

            elif etype == "level_cost_reduction_per_round":
                ps.level_cost = max(0, ps.level_cost - int(effect.get("amount", 1)))

    def apply_on_round_end(self, ps) -> None:
        """Fire end-of-shopping-turn trinket effects (called at END_TURN / freeze)."""
        for card_id in ps.equipped_trinkets:
            cdef   = self.card_defs.get(card_id, {})
            effect = cdef.get("trinket_effect", {})
            etype  = effect.get("type", "")

            if etype == "end_of_turn_buff_all":
                atk = int(effect.get("atk", 0))
                hp  = int(effect.get("hp", 0))
                self._buff_minions(ps.board, atk, hp)

            elif etype == "end_of_turn_buff_leftmost":
                if ps.board:
                    atk = int(effect.get("atk", 0))
                    hp  = int(effect.get("hp", 0))
                    self._buff_minions([ps.board[0]], atk, hp)

            elif etype == "end_of_turn_buff_tribe":
                tribe = effect.get("tribe", "")
                atk   = int(effect.get("atk", 0))
                hp    = int(effect.get("hp", 0))
                self._buff_minions(
                    [m for m in ps.board if self._tribe_match(m, tribe)], atk, hp
                )

    def apply_on_combat_start(self, ps) -> None:
        """Fire start-of-combat trinket effects (called just before combat sim)."""
        for card_id in ps.equipped_trinkets:
            cdef   = self.card_defs.get(card_id, {})
            effect = cdef.get("trinket_effect", {})
            etype  = effect.get("type", "")

            if etype == "start_of_combat_buff_all":
                atk = int(effect.get("atk", 0))
                hp  = int(effect.get("hp", 0))
                # Combat-only: use health/attack directly (not perm bonus)
                self._buff_minions(ps.board, atk, hp, permanent=False)

            elif etype == "start_of_combat_buff_tribe":
                tribe = effect.get("tribe", "")
                atk   = int(effect.get("atk", 0))
                hp    = int(effect.get("hp", 0))
                self._buff_minions(
                    [m for m in ps.board if self._tribe_match(m, tribe)],
                    atk, hp, permanent=False,
                )

    def apply_on_combat_end(self, ps, result: str) -> None:
        """Fire combat-end trinket effects (e.g. stat buffs on win)."""
        for card_id in ps.equipped_trinkets:
            cdef   = self.card_defs.get(card_id, {})
            effect = cdef.get("trinket_effect", {})
            if effect.get("type") == "stat_buff_on_win" and result == "win":
                atk = int(effect.get("atk", 0))
                hp  = int(effect.get("hp", 0))
                self._buff_minions(ps.board, atk, hp)

            elif effect.get("type") == "gold_per_round" and "self_damage_per_round" in effect:
                # Wax Imprinter: deal self-damage if player can afford the gold next turn
                dmg = int(effect["self_damage_per_round"])
                effective_hp = ps.health + ps.armor
                if effective_hp > dmg:
                    # Damage comes off armor first
                    if ps.armor >= dmg:
                        ps.armor -= dmg
                    else:
                        ps.health -= (dmg - ps.armor)
                        ps.armor = 0
