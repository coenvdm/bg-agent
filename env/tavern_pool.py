"""Shared card pool for Hearthstone Battlegrounds tavern draws."""

from __future__ import annotations

import random
from typing import Dict, List


# Approximate number of copies of each card in the shared pool, indexed by tier.
# Tier 7 is a special singleton tier (discovered/generated, not drawn normally).
COPIES_PER_TIER: Dict[int, int] = {
    1: 18,
    2: 15,
    3: 13,
    4: 11,
    5: 9,
    6: 7,
    7: 1,
}


class TavernPool:
    """Shared card pool that tracks available cards for all players.

    Parameters
    ----------
    card_defs:
        Mapping of card_id -> card definition dict.  Each definition must
        contain at least a ``"tier"`` key (int 1-7).  All other fields are
        passed through unchanged when cards are returned from draw().
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(self, card_defs: Dict[str, dict], seed: int = None) -> None:
        self._card_defs = card_defs
        self._rng = random.Random(seed)

        # Master list of all card copies (each entry is a card def dict).
        # Built once and replenished by reset().
        self._pool: List[dict] = []
        self._build_pool()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pool(self) -> None:
        """Fill self._pool with COPIES_PER_TIER copies of every card."""
        self._pool = []
        for card_id, card_def in self._card_defs.items():
            tier = card_def.get("tier", 1)
            n_copies = COPIES_PER_TIER.get(tier, 1)
            for _ in range(n_copies):
                # Store a shallow copy so callers can't mutate the master def.
                self._pool.append(dict(card_def))
        self._rng.shuffle(self._pool)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(self, tier: int, count: int) -> List[dict]:
        """Draw *count* cards whose tier is <= *tier*.

        Preference is given to cards at exactly *tier*; lower-tier cards are
        used as fall-backs when the pool for the requested tier is exhausted.
        Returns however many cards are available (may be fewer than *count*).
        """
        if count <= 0:
            return []

        # Partition available cards into exact-tier and lower-tier buckets.
        exact: List[int] = []
        lower: List[int] = []
        for idx, card in enumerate(self._pool):
            card_tier = card.get("tier", 1)
            if card_tier == tier:
                exact.append(idx)
            elif card_tier < tier:
                lower.append(idx)

        # Build the candidate index list: prefer exact tier, then lower.
        candidates = exact + lower
        n = min(count, len(candidates))
        if n == 0:
            return []

        chosen_indices = self._rng.sample(candidates, n)
        # Remove chosen cards from pool (iterate in reverse to preserve indices).
        drawn = []
        for idx in sorted(chosen_indices, reverse=True):
            drawn.append(self._pool.pop(idx))

        # Restore original order (earlier positions drawn first).
        drawn.reverse()
        return drawn

    def return_cards(self, cards: List[dict]) -> None:
        """Return a list of card dicts back to the pool and reshuffle."""
        self._pool.extend(dict(c) for c in cards)
        self._rng.shuffle(self._pool)

    def available_count(self, tier: int) -> int:
        """Count cards available at exactly *tier*."""
        return sum(1 for c in self._pool if c.get("tier", 1) == tier)

    def reset(self) -> None:
        """Refill the pool to its initial state and reshuffle."""
        self._build_pool()
