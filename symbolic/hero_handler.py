"""
Hero power handler for Hearthstone Battlegrounds.

Handles passive hooks (on_sell, on_buy, on_play, on_refresh, on_tavern_upgrade,
on_start_of_round, on_end_turn) and active no-pointer hero power activations
(activate_no_pointer) for Phase 1 and Phase 2 heroes.

Phase 3 targeted heroes and Phase 4 complex heroes are not yet implemented.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from env.player_state import PlayerState, MinionState


# ---------------------------------------------------------------------------
# Shared helpers (mirrors effect_handler.py)
# ---------------------------------------------------------------------------

_TRIBE_LIST = [
    "BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECH",
    "MURLOC", "NAGA", "PIRATE", "QUILBOAR", "UNDEAD",
]


def _minion_tribes(minion: "MinionState", card_defs: dict) -> List[str]:
    cdef = card_defs.get(minion.card_id, {})
    if not cdef:
        name_lower = minion.name.lower().replace(" ", "")
        for cd in card_defs.values():
            if cd.get("name", "").lower().replace(" ", "") == name_lower:
                cdef = cd
                break
    tribes_raw = cdef.get("tribes", [])
    if tribes_raw:
        return [t.upper() for t in tribes_raw if t.upper() in _TRIBE_LIST]
    token_tribe = getattr(minion, "tribe", None)
    if token_tribe:
        return [token_tribe.upper()]
    return []


def _buff(minion: "MinionState", atk: int, hp: int) -> None:
    minion.perm_atk_bonus += atk
    minion.perm_hp_bonus  += hp
    minion.max_health     += hp


def _make_token(name: str, attack: int, health: int, tier: int = 1,
                tribe: Optional[str] = None) -> "MinionState":
    from env.player_state import MinionState
    m = MinionState(name=name, attack=attack, health=health,
                    max_health=health, tier=tier)
    if tribe is not None:
        m.tribe = tribe  # type: ignore[attr-defined]
    return m


# ---------------------------------------------------------------------------
# HeroPowerHandler
# ---------------------------------------------------------------------------

class HeroPowerHandler:
    """Dispatch passive and active (no-pointer) hero powers.

    Parameters
    ----------
    card_defs:
        Mapping card_id → card definition (from bg_card_definitions.json).
    hero_defs:
        Mapping hero_card_id → hero definition (from hero_definitions.json).
    """

    def __init__(self, card_defs: dict, hero_defs: dict) -> None:
        self._card_defs = card_defs
        self._hero_defs = hero_defs
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # Passive hooks  (called by game_loop on the relevant event)
    # ------------------------------------------------------------------

    def on_start_of_round(self, ps: "PlayerState") -> None:
        """Called at the start of each shopping phase (after gold is set)."""
        hid = ps.hero_card_id
        if hid == "TB_BaconShop_HERO_02":        # Cap'n Hoggarr
            ps.gold = min(ps.max_gold, ps.gold + 1)
        elif hid == "TB_BaconShop_HERO_12":      # Millhouse Manastorm
            # Buys cost 2 gold, reroll stays 1 but level costs +1 more
            ps.buy_cost   = 2
            ps.reroll_cost = 1
            # Level cost override: applied when tavern upgrade is processed in step_shopping
            # We tag the state so step_shopping can read it
            ps._millhouse = True  # type: ignore[attr-defined]
        elif hid == "TB_BaconShop_HERO_24":      # Pyramad — increment X if unused
            # hero_power_x was already reset to 4 if used last turn;
            # on_start_of_round fires before any actions, so nothing to do here.
            # The increment happens in on_end_turn when power was NOT used.
            pass
        # Apply pending extra gold (e.g., Gallywix previous turn)
        if ps.hero_extra_gold > 0:
            ps.gold = min(ps.max_gold, ps.gold + ps.hero_extra_gold)
            ps.hero_extra_gold = 0

    def on_sell(self, ps: "PlayerState", minion: "MinionState") -> None:
        """Called after a minion is sold from the board."""
        hid = ps.hero_card_id
        if hid == "TB_BaconShop_HERO_15":        # Trade Prince Gallywix
            ps.hero_extra_gold += 1
        elif hid == "TB_BaconShop_HERO_08":      # Fungalmancer Flurgl
            # After selling a Murloc, add a random Murloc to the Tavern
            if "MURLOC" in _minion_tribes(minion, self._card_defs):
                # Inject a Murloc back into the shop (if there's space)
                from env.tavern_pool import TavernPool  # avoid circular
                # We don't have a reference to the pool here; set a flag instead
                # that game_loop reads and injects a Murloc
                ps._flurgl_murloc_due = True  # type: ignore[attr-defined]

    def on_buy(self, ps: "PlayerState", minion: "MinionState") -> None:
        """Called after a minion is bought (moved from shop to hand)."""
        hid = ps.hero_card_id
        if hid == "TB_BaconShop_HERO_03":        # Cariel Roame
            if minion.divine_shield:
                for m in ps.board:
                    _buff(m, 1, 1)
        elif hid == "TB_BaconShop_HERO_10":      # Kael'thas Sunstrider
            # Every 3rd buy this turn costs 0 gold → rebate 3 gold after the fact
            ps.hero_power_counter += 1
            if ps.hero_power_counter % 3 == 0:
                ps.gold = min(ps.max_gold, ps.gold + ps.buy_cost)
        elif hid == "TB_BaconShop_HERO_14":      # Sneed
            if minion.golden:
                # Get a copy of the golden minion
                from env.player_state import MinionState
                import copy
                copy_m = copy.copy(minion)
                if len(ps.hand) < 10:
                    ps.hand.append(copy_m)
        elif hid == "TB_BaconShop_HERO_11":      # Liadrin
            if minion.divine_shield:
                # Give all friendly minions +1/+1
                for m in ps.board:
                    _buff(m, 1, 1)

    def on_play(self, ps: "PlayerState", minion: "MinionState") -> None:
        """Called after a minion is played from hand to board."""
        hid = ps.hero_card_id
        if hid == "TB_BaconShop_HERO_04":        # Chenvaala
            if "ELEMENTAL" in _minion_tribes(minion, self._card_defs):
                ps.hero_power_counter += 1
                if ps.hero_power_counter % 3 == 0:
                    # Reduce tavern upgrade cost by 3
                    ps.level_cost = max(0, ps.level_cost - 3)
        elif hid == "TB_BaconShop_HERO_05":      # Edwin VanCleef
            # Give a discount on the next buy
            ps.buy_discount = max(ps.buy_discount, 1)

    def on_refresh(self, ps: "PlayerState") -> None:
        """Called after the shop is redrawn (on refresh or round start).

        Note: game_loop should call this AFTER ps.shop is already updated.
        """
        hid = ps.hero_card_id
        if hid == "TB_BaconShop_HERO_06":        # Enhance-o-Mechano
            if ps.board:
                target = self._rng.choice(ps.board)
                kw = self._rng.choice(["taunt", "windfury", "reborn", "divine_shield"])
                setattr(target, kw, True)
        elif hid == "TB_BaconShop_HERO_13":      # Millificent Manastorm
            for m in ps.shop:
                if m is not None and "MECH" in _minion_tribes(m, self._card_defs):
                    m.attack += 2
        elif hid == "TB_BaconShop_HERO_16":      # Ysera
            # Add a Dragon at current tier to the shop (handled in game_loop to
            # avoid TavernPool circular import; set flag)
            ps._ysera_dragon_due = True  # type: ignore[attr-defined]

    def on_tavern_upgrade(self, ps: "PlayerState") -> None:
        """Called immediately after tavern tier is incremented."""
        hid = ps.hero_card_id
        if hid == "TB_BaconShop_HERO_07":        # Forest Warden Omu
            ps.gold = min(ps.max_gold, ps.gold + 2)

    def on_end_turn(self, ps: "PlayerState") -> None:
        """Called when the player hits END_TURN."""
        hid = ps.hero_card_id
        if hid == "TB_BaconShop_HERO_24":        # Pyramad
            if not ps.hero_power_used:
                ps.hero_power_x += 1
        # Reset per-turn counters
        if hid == "TB_BaconShop_HERO_10":        # Kael'thas: reset buy counter
            ps.hero_power_counter = 0
        # Edwin: discount consumed on first buy, reset if unused
        ps.buy_discount = 0
        # Aranna: first_buy_free resets each turn (it's earned mid-game once)
        # (we do NOT reset first_buy_free here — it's permanent once earned)

    # ------------------------------------------------------------------
    # Active no-pointer  (type_action == 6, Phase 2 heroes)
    # ------------------------------------------------------------------

    def activate_no_pointer(
        self, ps: "PlayerState", tavern_pool=None
    ) -> None:
        """Activate this player's active (no-pointer) hero power.

        Expects ps.gold has already been deducted and ps.hero_power_used set
        to True by the caller (step_shopping).
        """
        hid = ps.hero_card_id

        if hid == "TB_BaconShop_HERO_21":        # George the Fallen
            self._hp_george(ps)
        elif hid == "TB_BaconShop_HERO_24":      # Pyramad
            self._hp_pyramad(ps)
        elif hid == "TB_BaconShop_HERO_22":      # Galakrond
            self._hp_galakrond(ps, tavern_pool)
        elif hid == "TB_BaconShop_HERO_23":      # Infinite Toki
            self._hp_infinite_toki(ps, tavern_pool)
        elif hid == "TB_BaconShop_HERO_17":      # Arch-Villain Rafaam
            ps._rafaam_active = True  # type: ignore[attr-defined]
        elif hid == "TB_BaconShop_HERO_19":      # Death Speaker Blackthorn
            self._hp_blackthorn(ps)
        elif hid == "TB_BaconShop_HERO_18":      # Dancin' Deryl
            self._hp_dancin_deryl(ps)
        elif hid == "TB_BaconShop_HERO_20":      # Doctor Holli'dae
            self._hp_doctor_hollidae(ps)
        elif hid == "TB_BaconShop_HERO_25":      # Tess Greymane
            ps._tess_active = True  # type: ignore[attr-defined]
        elif hid == "TB_BaconShop_HERO_26":      # The Great Akazamzarak
            self._hp_akazamzarak(ps)
        elif hid == "TB_BaconShop_HERO_27":      # Varden Dawngrasp
            self._hp_varden(ps)
        elif hid == "TB_BaconShop_HERO_28":      # Yogg-Saron
            self._hp_yogg(ps, tavern_pool)

    # ------------------------------------------------------------------
    # Phase 2 hero power implementations
    # ------------------------------------------------------------------

    def _hp_george(self, ps: "PlayerState") -> None:
        """Give a random friendly minion Divine Shield."""
        if not ps.board:
            return
        without_ds = [m for m in ps.board if not m.divine_shield]
        candidates = without_ds if without_ds else ps.board
        self._rng.choice(candidates).divine_shield = True

    def _hp_pyramad(self, ps: "PlayerState") -> None:
        """Give a random friendly minion +hero_power_x Health. Reset x to 4."""
        if ps.board:
            target = self._rng.choice(ps.board)
            hp_gain = max(4, ps.hero_power_x)
            target.perm_hp_bonus += hp_gain
            target.max_health    += hp_gain
        ps.hero_power_x = 4  # reset

    def _hp_galakrond(self, ps: "PlayerState", tavern_pool) -> None:
        """Replace a random shop minion with one from a higher tier."""
        if not ps.shop or tavern_pool is None:
            return
        # Pick a random shop slot to replace
        idx = self._rng.randrange(len(ps.shop))
        old = ps.shop[idx]
        # Return old card to pool
        if old is not None:
            tavern_pool.return_cards([{"card_id": old.card_id, "name": old.name,
                                       "tier": old.tier}])
        # Draw one from a higher tier (or same tier if at cap)
        higher_tier = min(6, ps.tavern_tier + 1)
        new_cards = tavern_pool.draw(higher_tier, 1)
        if new_cards:
            from env.player_state import MinionState
            d = new_cards[0]
            ps.shop[idx] = MinionState(
                card_id=d.get("card_id", ""),
                name=d.get("name", ""),
                attack=d.get("attack", 0),
                health=d.get("health", 0),
                max_health=d.get("health", 0),
                tier=d.get("tier", higher_tier),
            )

    def _hp_infinite_toki(self, ps: "PlayerState", tavern_pool) -> None:
        """Refresh shop, adding one extra minion from a higher tier."""
        if tavern_pool is None:
            return
        higher_tier = min(6, ps.tavern_tier + 1)
        extra = tavern_pool.draw(higher_tier, 1)
        if extra:
            from env.player_state import MinionState
            d = extra[0]
            extra_minion = MinionState(
                card_id=d.get("card_id", ""),
                name=d.get("name", ""),
                attack=d.get("attack", 0),
                health=d.get("health", 0),
                max_health=d.get("health", 0),
                tier=d.get("tier", higher_tier),
            )
            ps.shop.append(extra_minion)

    def _hp_blackthorn(self, ps: "PlayerState") -> None:
        """Get 2 Blood Gems (+1/+1 spell cards) in hand (twice if Drakkari)."""
        if len(ps.hand) >= 10:
            return
        times = 4 if ps.has_drakkari else 2  # Drakkari doubles end-of-turn
        for _ in range(times):
            if len(ps.hand) < 10:
                gem = _make_token("Blood Gem", attack=0, health=0, tier=1)
                gem.is_spell = True  # type: ignore[attr-defined]
                ps.hand.append(gem)

    def _hp_dancin_deryl(self, ps: "PlayerState") -> None:
        """Give two random minions in hand +1/+1."""
        hand_minions = [m for m in ps.hand if m is not None and not m.is_spell]
        sample = self._rng.sample(hand_minions, min(2, len(hand_minions)))
        for m in sample:
            _buff(m, 1, 1)

    def _hp_doctor_hollidae(self, ps: "PlayerState") -> None:
        """Add a random Tavern Spell to hand."""
        if len(ps.hand) < 10:
            spell = _make_token("Tavern Spell", attack=0, health=0, tier=1)
            spell.is_spell = True  # type: ignore[attr-defined]
            ps.hand.append(spell)

    def _hp_akazamzarak(self, ps: "PlayerState") -> None:
        """Add a Secret to the battlefield (simulate as +1/+1 buff to board)."""
        # Simplified: we don't model Secrets properly yet.
        # Apply a small buff to a random board minion as a placeholder.
        if ps.board:
            _buff(self._rng.choice(ps.board), 1, 1)

    def _hp_varden(self, ps: "PlayerState") -> None:
        """Freeze all Tavern minions and give each +2/+2."""
        ps.frozen = True
        for m in ps.shop:
            if m is not None:
                _buff(m, 2, 2)

    def _hp_yogg(self, ps: "PlayerState", tavern_pool) -> None:
        """Steal a random Tavern minion and give it +1/+1."""
        if not ps.shop or tavern_pool is None or len(ps.hand) >= 10:
            return
        idx = self._rng.randrange(len(ps.shop))
        minion = ps.shop.pop(idx)
        if minion is not None:
            _buff(minion, 1, 1)
            ps.hand.append(minion)
