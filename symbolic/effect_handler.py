"""
Buy-phase effect handler for Hearthstone Battlegrounds.

Handles battlecries (triggered when a minion is played from hand to board)
and sell effects (triggered when a minion is sold from the board).

Multiplier awareness: when ps.has_brann is True, battlecry buff effects
are applied twice (Brann Bronzebeard doubles battlecries).
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from env.player_state import PlayerState, MinionState


# ---------------------------------------------------------------------------
# Token factory
# ---------------------------------------------------------------------------

def _make_token(
    name: str,
    attack: int,
    health: int,
    tier: int = 1,
    tribe: Optional[str] = None,
) -> "MinionState":
    """Create a minimal MinionState token for hand/board injection."""
    from env.player_state import MinionState

    m = MinionState(
        name=name,
        attack=attack,
        health=health,
        max_health=health,
        tier=tier,
    )
    # tribe stored as a plain attribute; MinionState has no tribe field by default
    if tribe is not None:
        m.tribe = tribe  # type: ignore[attr-defined]
    return m


# ---------------------------------------------------------------------------
# Tribe-matching helpers
# ---------------------------------------------------------------------------

_TRIBE_LIST = [
    "BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECH",
    "MURLOC", "NAGA", "PIRATE", "QUILBOAR", "UNDEAD",
]


def _minion_tribes(minion: "MinionState", card_defs: dict) -> List[str]:
    """Return the list of uppercase tribe strings for a MinionState.

    Looks up the card_defs entry first (authoritative), then falls back to
    the minion's own ``tribe`` attribute if present.
    """
    # Try card_defs lookup by card_id
    cdef = card_defs.get(minion.card_id, {})
    if not cdef:
        # Fallback: search by normalised name
        name_lower = minion.name.lower().replace(" ", "")
        for cd in card_defs.values():
            if cd.get("name", "").lower().replace(" ", "") == name_lower:
                cdef = cd
                break

    tribes_raw = cdef.get("tribes", [])
    if tribes_raw:
        return [t.upper() for t in tribes_raw if t.upper() in _TRIBE_LIST]

    # Last resort: token tribe attribute set at creation time
    token_tribe = getattr(minion, "tribe", None)
    if token_tribe:
        return [token_tribe.upper()]

    return []


def _friendly_with_tribe(
    board: List["MinionState"],
    tribe: str,
    exclude: Optional["MinionState"],
    card_defs: dict,
) -> List["MinionState"]:
    """Return all board minions of the given tribe, optionally excluding one."""
    result = []
    for m in board:
        if exclude is not None and m is exclude:
            continue
        if tribe in _minion_tribes(m, card_defs):
            result.append(m)
    return result


# ---------------------------------------------------------------------------
# Card matching helper
# ---------------------------------------------------------------------------

def _match_def(minion: "MinionState", card_defs: dict) -> dict:
    """Look up the card definition for *minion*.

    Tries card_id exact match first, then a normalised name search.
    Returns an empty dict when no match is found so callers can use ``.get()``.
    """
    d = card_defs.get(minion.card_id, {})
    if not d:
        name_lower = minion.name.lower().replace(" ", "")
        for cd in card_defs.values():
            if cd.get("name", "").lower().replace(" ", "") == name_lower:
                return cd
    return d


# ---------------------------------------------------------------------------
# Buff helpers
# ---------------------------------------------------------------------------

def _buff_minion(minion: "MinionState", atk: int, hp: int) -> None:
    """Apply a permanent buff to a single minion in-place."""
    minion.perm_atk_bonus += atk
    minion.perm_hp_bonus  += hp
    minion.max_health     += hp


def _buff_random(
    candidates: List["MinionState"],
    atk: int,
    hp: int,
    rng: random.Random,
    times: int = 1,
) -> None:
    """Buff a random minion from *candidates* (in-place), *times* times.

    Each application picks independently (with replacement allowed) so that
    Brann can apply a single buff twice to the same or different targets.
    """
    if not candidates:
        return
    for _ in range(times):
        target = rng.choice(candidates)
        _buff_minion(target, atk, hp)


def _buff_all(
    candidates: List["MinionState"],
    atk: int,
    hp: int,
    times: int = 1,
) -> None:
    """Buff every minion in *candidates* (in-place), *times* times."""
    for _ in range(times):
        for m in candidates:
            _buff_minion(m, atk, hp)


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

class EffectHandler:
    """Dispatch buy-phase battlecry and sell effects for known cards.

    Unknown cards are silently ignored so the handler degrades gracefully
    when card_defs is empty or missing entries.
    """

    def __init__(self, card_defs: dict, tavern_pool=None) -> None:
        self._card_defs = card_defs
        self._tavern_pool = tavern_pool  # Optional[TavernPool] for discover/draw BCs
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_play(self, ps: "PlayerState", minion: "MinionState") -> None:
        """Fire battlecry for *minion* just played from hand to board."""
        name_key = minion.name.lower().replace(" ", "")

        # Determine how many times battlecry fires (1 normally, 2 with Brann)
        times = 2 if ps.has_brann else 1

        # Dispatch by normalised name (partial matching where noted)
        if "murozond" in name_key:
            self._bc_murozond(ps, minion, times)
        elif "waxridertogwaggle" in name_key:
            self._bc_waxrider_togwaggle(ps, minion, times)
        elif "deflectobot" in name_key:
            self._bc_deflect_o_bot(ps, minion, times)
        elif "masterofrealities" in name_key:
            self._bc_master_of_realities(ps, minion, times)
        elif "recruiter" in name_key and "roaring" not in name_key:
            # "Recruiter" (the generic one that spawns a Recruit token)
            # but NOT "Roaring Recruiter" (an aura card, no battlecry token)
            self._bc_recruiter(ps, minion, times)
        # ── P1-C: "This game" tribe buff battlecries ─────────────────────────
        elif "nerubiandeathswarmer" in name_key:
            self._register_game_buff(ps, "UNDEAD", 1 * times, 0)
        elif "dunedweller" in name_key:
            self._register_game_buff(ps, "ELEMENTAL", 1 * times, 1 * times)
        elif "felemental" in name_key:
            self._register_game_buff(ps, "ALL", 2 * times, 1 * times)
        # ── P1-D: Blood Gem bonus battlecries ────────────────────────────────
        elif "moonbaconjazzer" in name_key:
            ps.blood_gem_hp_bonus += 1 * times
        # ── P2-C: Token/item generator battlecries ───────────────────────────
        elif "razorfengeomancer" in name_key:
            # Add Blood Gem spells to hand (2 per battlecry fire)
            for _ in range(2 * times):
                if len(ps.hand) < 10:
                    gem = _make_token("Blood Gem", attack=0, health=0, tier=1)
                    gem.is_spell = True  # type: ignore[attr-defined]
                    ps.hand.append(gem)
        elif "shellcollector" in name_key:
            ps.gold = min(ps.max_gold, ps.gold + 1 * times)
        elif "briarbackdrummer" in name_key:
            # Add Blood Gem Barrage spell (AoE Blood Gem) to hand
            for _ in range(times):
                if len(ps.hand) < 10:
                    barrage = _make_token("Blood Gem Barrage", attack=0, health=0, tier=1)
                    barrage.is_spell = True  # type: ignore[attr-defined]
                    barrage.is_barrage = True  # type: ignore[attr-defined]
                    ps.hand.append(barrage)
        elif "refreshinganomaly" in name_key:
            ps._free_refreshes = getattr(ps, "_free_refreshes", 0) + 2 * times  # type: ignore[attr-defined]
        elif "taverntempest" in name_key:
            self._bc_draw_tribe(ps, tier=ps.tavern_tier, tribe="ELEMENTAL", count=times)
        elif "archaedas" in name_key:
            self._bc_discover(ps, tier=5)
        # ── P2-D: Discover battlecries ───────────────────────────────────────
        elif "huntingtigershark" in name_key:
            self._bc_discover(ps, tier=4, tribe="BEAST")
        elif "imposingpercussionist" in name_key:
            self._bc_discover(ps, tier=4, tribe="DEMON")
        elif "primalfinlookout" in name_key:
            # Only if controlling another Murloc
            other_murlocs = _friendly_with_tribe(ps.board, "MURLOC", minion, self._card_defs)
            if other_murlocs:
                self._bc_discover(ps, tier=5, tribe="MURLOC")
        elif "rodeo" in name_key and "performer" in name_key:
            # Approximate: draw a random mid-tier minion
            self._bc_draw_tribe(ps, tier=min(3, ps.tavern_tier))
        # ── P2-E: Consume-shop-minion battlecries ────────────────────────────
        elif "pickyeater" in name_key:
            self._bc_consume_shop(ps, minion, self._rng, times=times)
        elif "mindmuck" in name_key:
            # Highest-ATK friendly Demon consumes
            demons = _friendly_with_tribe(ps.board, "DEMON", minion, self._card_defs)
            if demons:
                gainer = max(demons, key=lambda m: m.attack + m.perm_atk_bonus)
                self._bc_consume_shop(ps, gainer, self._rng, times=times)
        elif "furiousdriver" in name_key:
            # Every other friendly Demon consumes once
            demons = _friendly_with_tribe(ps.board, "DEMON", minion, self._card_defs)
            for demon in demons:
                self._bc_consume_shop(ps, demon, self._rng, times=1)
        # Righteous Protector has no battlecry — static keywords only.

    def on_sell(self, ps: "PlayerState", minion: "MinionState") -> None:
        """Fire sell-triggered effects for *minion* just removed from board."""
        name_key = minion.name.lower().replace(" ", "")

        if "sellemental" in name_key:
            self._sell_sellemental(ps)
        elif "goldgrubber" in name_key:
            self._sell_gold_grubber(ps, minion)
        # ── P1-D: Blood Gem bonus on sell/death ──────────────────────────────
        elif "pricklypiper" in name_key:
            ps.blood_gem_atk_bonus += 1
        # ── P2-A: Missing sell effects ────────────────────────────────────────
        elif "fireballer" in name_key:
            _buff_all(ps.board, 1, 0)
        elif "snowballer" in name_key:
            _buff_all(ps.board, 0, 1)
        elif "mintedcorsair" in name_key:
            ps.gold = min(ps.max_gold, ps.gold + 1)
        elif "tad" in name_key and len(name_key) <= 4:
            # Tad: get a random Murloc. Use a flag processed in game_loop.
            ps._tad_due = True  # type: ignore[attr-defined]
        # Pack Leader sell effect: none (its effect is in combat).
        # Yo-Ho-Ho / Shifter Zerus: skip.

    # ------------------------------------------------------------------
    # Battlecry implementations
    # ------------------------------------------------------------------

    def _bc_murozond(
        self, ps: "PlayerState", minion: "MinionState", times: int
    ) -> None:
        """Murozond: give all other friendly minions +1/+1."""
        others = [m for m in ps.board if m is not minion]
        _buff_all(others, 1, 1, times=times)

    def _bc_waxrider_togwaggle(
        self, ps: "PlayerState", minion: "MinionState", times: int
    ) -> None:
        """Waxrider Togwaggle: give a random friendly Dragon +2/+2."""
        dragons = _friendly_with_tribe(ps.board, "DRAGON", minion, self._card_defs)
        _buff_random(dragons, 2, 2, self._rng, times=times)

    def _bc_deflect_o_bot(
        self, ps: "PlayerState", minion: "MinionState", times: int
    ) -> None:
        """Deflect-o-Bot: give a random friendly Mech divine_shield."""
        mechs = _friendly_with_tribe(ps.board, "MECH", minion, self._card_defs)
        if not mechs:
            return
        for _ in range(times):
            target = self._rng.choice(mechs)
            target.divine_shield = True

    def _bc_master_of_realities(
        self, ps: "PlayerState", minion: "MinionState", times: int
    ) -> None:
        """Master of Realities: give a random friendly Elemental +1/+1 if one exists."""
        elementals = _friendly_with_tribe(ps.board, "ELEMENTAL", minion, self._card_defs)
        _buff_random(elementals, 1, 1, self._rng, times=times)

    def _bc_recruiter(
        self, ps: "PlayerState", minion: "MinionState", times: int
    ) -> None:
        """Recruiter: add a 1/1 Recruit token to hand (once per battlecry fire)."""
        for _ in range(times):
            if len(ps.hand) < 10:
                token = _make_token("Recruit", attack=1, health=1, tier=1)
                ps.hand.append(token)

    # ------------------------------------------------------------------
    # Sell effect implementations
    # ------------------------------------------------------------------

    def _sell_sellemental(self, ps: "PlayerState") -> None:
        """Sellemental: add a 1/1 Water Droplet Elemental to hand."""
        if len(ps.hand) < 10:
            token = _make_token(
                "Water Droplet", attack=1, health=1, tier=1, tribe="ELEMENTAL"
            )
            ps.hand.append(token)

    def _sell_gold_grubber(
        self, ps: "PlayerState", sold_minion: "MinionState"
    ) -> None:
        """Gold Grubber: give a random friendly minion +2/+1."""
        candidates = [m for m in ps.board if m is not sold_minion]
        _buff_random(candidates, 2, 1, self._rng, times=1)

    # ------------------------------------------------------------------
    # P1-C: "This game" buff registry
    # ------------------------------------------------------------------

    def _register_game_buff(
        self, ps: "PlayerState", tribe_key: str, atk: int, hp: int
    ) -> None:
        """Register a permanent tribe buff and apply it to current board minions.

        tribe_key: "UNDEAD", "ALL", "ELEMENTAL", "BEAST:beetle", etc.
        """
        cur_atk, cur_hp = ps.game_buffs.get(tribe_key, (0, 0))
        ps.game_buffs[tribe_key] = (cur_atk + atk, cur_hp + hp)
        # Apply the delta immediately to existing board minions
        for m in ps.board:
            if self._matches_tribe_key(m, tribe_key):
                m.game_atk_bonus += atk
                m.game_hp_bonus  += hp
                m.max_health     += hp

    def _matches_tribe_key(self, minion: "MinionState", tribe_key: str) -> bool:
        """Return True if *minion* matches the tribe_key (e.g. "UNDEAD", "ALL")."""
        if tribe_key == "ALL":
            return True
        if ":" in tribe_key:
            # Sub-tribe like "BEAST:beetle" — match by token name substring
            _, token_name = tribe_key.split(":", 1)
            return token_name.lower() in minion.name.lower()
        return tribe_key in _minion_tribes(minion, self._card_defs)

    # ------------------------------------------------------------------
    # P2-B: Discover and pool-draw helpers
    # ------------------------------------------------------------------

    def _dict_to_minion(self, card: dict) -> "MinionState":
        """Convert a card dict from TavernPool into a MinionState."""
        from env.player_state import MinionState
        m = MinionState(
            card_id=card.get("card_id", ""),
            name=card.get("name", ""),
            attack=card.get("attack", 0),
            health=card.get("health", 0),
            max_health=card.get("health", 0),
            tier=card.get("tier", 1),
            golden=bool(card.get("golden", False)),
            divine_shield=bool(card.get("divine_shield", False)),
            taunt=bool(card.get("taunt", False)),
            reborn=bool(card.get("reborn", False)),
            windfury=bool(card.get("windfury", False)),
        )
        return m

    def _bc_discover(
        self, ps: "PlayerState", tier: int, tribe: Optional[str] = None
    ) -> None:
        """Draw 3 cards (optionally filtered to tribe) and store in ps.discover_pending.

        Shopping is paused until the agent picks one via BUY(0/1/2).
        """
        if self._tavern_pool is None:
            return
        candidates = self._tavern_pool.draw(tier, 3)
        if tribe and candidates:
            tribe_upper = tribe.upper()
            tribe_matches = [
                c for c in candidates
                if tribe_upper in [t.upper() for t in (c.get("tribes") or [])]
            ]
            rejects = [c for c in candidates if c not in tribe_matches]
            if rejects:
                self._tavern_pool.return_cards(rejects)
            candidates = tribe_matches if tribe_matches else candidates
        if not candidates:
            return
        ps.discover_pending = [self._dict_to_minion(c) for c in candidates]

    def _bc_draw_tribe(
        self, ps: "PlayerState", tier: int, tribe: Optional[str] = None, count: int = 1
    ) -> None:
        """Draw *count* random cards (optionally filtered to tribe) into hand."""
        if self._tavern_pool is None:
            return
        drawn = self._tavern_pool.draw(tier, count)
        for card in drawn:
            if len(ps.hand) >= 10:
                self._tavern_pool.return_cards([card])
                break
            ps.hand.append(self._dict_to_minion(card))

    # ------------------------------------------------------------------
    # P2-E: Consume-shop helper
    # ------------------------------------------------------------------

    def _bc_consume_shop(
        self,
        ps: "PlayerState",
        gainer: "MinionState",
        rng: random.Random,
        times: int = 1,
    ) -> None:
        """Pop a random card from the shop and add its stats to *gainer*."""
        if not ps.shop:
            return
        for _ in range(times):
            if not ps.shop:
                break
            idx = rng.randrange(len(ps.shop))
            consumed = ps.shop.pop(idx)
            gainer.perm_atk_bonus += consumed.attack
            gainer.perm_hp_bonus  += consumed.health
            gainer.max_health     += consumed.health

    def on_after_combat(self, ps: "PlayerState", dead_card_ids: List[str]) -> None:
        """Called after combat with card IDs of minions that died this combat.

        Handles deathrattle-triggered "this game" buffs that need persistence
        beyond the combat trial (e.g. Anubarak Nerubian King).
        """
        for cid in dead_card_ids:
            cid_lower = cid.lower().replace("_", "").replace(" ", "")
            # Anubarak Nerubian King: +1 Attack to all Undead this game
            if "anubarak" in cid_lower or "nerubianking" in cid_lower:
                self._register_game_buff(ps, "UNDEAD", 1, 0)
