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

    def __init__(self, card_defs: dict) -> None:
        self._card_defs = card_defs
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
        # Righteous Protector has no battlecry — static keywords only.
        # Yo-Ho-Ho / Shifter Zerus / discover-type cards: skip.

    def on_sell(self, ps: "PlayerState", minion: "MinionState") -> None:
        """Fire sell-triggered effects for *minion* just removed from board."""
        name_key = minion.name.lower().replace(" ", "")

        if "sellemental" in name_key:
            self._sell_sellemental(ps)
        elif "goldgrubber" in name_key:
            self._sell_gold_grubber(ps, minion)
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
