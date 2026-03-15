"""Matchmaking for 8-player Hearthstone Battlegrounds."""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from env.player_state import PlayerState


class Matchmaker:
    """Pairs alive players for each combat round.

    For an odd number of alive players one player is assigned a *ghost*
    matchup against a randomly-selected dead player's last board state.

    Parameters
    ----------
    n_players:
        Total number of players at game start (default 8).
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(self, n_players: int = 8, seed: int = None) -> None:
        self.n_players = n_players
        self._rng = random.Random(seed)
        self.history: List[List[Tuple[int, int]]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pair_players(
        self,
        players: List[PlayerState],
        round_num: int,
    ) -> List[Tuple[int, int]]:
        """Return combat pairings for this round.

        Only alive players (``player.alive == True``) are paired.  When the
        number of alive players is odd, one player receives a ghost matchup
        represented as ``(player_id, -1)``.

        The algorithm tries to avoid giving a player the same opponent they
        faced in the immediately preceding round.  It falls back to any valid
        pairing if avoidance is not possible.

        Parameters
        ----------
        players:
            Current state of all players (alive and dead).
        round_num:
            The 1-based round number being paired (used for logging / history
            bookkeeping but not for the algorithm itself).

        Returns
        -------
        List of ``(player_id_a, player_id_b)`` pairs.  Ghost matchups are
        ``(player_id, -1)``.
        """
        alive = [p for p in players if p.alive]
        alive_ids = [p.player_id for p in alive]
        self._rng.shuffle(alive_ids)

        last_round_pairs: dict[int, int] = {}
        if self.history:
            for a, b in self.history[-1]:
                if b != -1:
                    last_round_pairs[a] = b
                    last_round_pairs[b] = a

        ghost_player_id: Optional[int] = None
        if len(alive_ids) % 2 == 1:
            # Pick the ghost candidate: prefer someone who faced a ghost last round.
            ghost_candidates = [
                pid for pid in alive_ids if last_round_pairs.get(pid) == -1
            ]
            if not ghost_candidates:
                ghost_candidates = alive_ids
            ghost_player_id = self._rng.choice(ghost_candidates)
            alive_ids.remove(ghost_player_id)

        pairs = self._round_robin_avoid(alive_ids, last_round_pairs)

        if ghost_player_id is not None:
            pairs.append((ghost_player_id, -1))

        self.update_history(pairs)
        return pairs

    def get_ghost(self, dead_players: List[PlayerState]) -> Optional[PlayerState]:
        """Return a randomly chosen dead player's state for a ghost matchup.

        Returns ``None`` if there are no dead players.
        """
        if not dead_players:
            return None
        return self._rng.choice(dead_players)

    def update_history(self, pairs: List[Tuple[int, int]]) -> None:
        """Record a completed set of pairings in the history log."""
        self.history.append(list(pairs))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _round_robin_avoid(
        self,
        ids: List[int],
        last_pairs: dict[int, int],
    ) -> List[Tuple[int, int]]:
        """Greedily pair *ids* while avoiding last-round rematches.

        Falls back to any valid completion when avoidance is impossible.
        Uses a simple greedy matching rather than an exhaustive search to
        keep runtime O(n).
        """
        if not ids:
            return []

        remaining = list(ids)
        pairs: List[Tuple[int, int]] = []

        while len(remaining) >= 2:
            a = remaining.pop(0)
            last_opp = last_pairs.get(a)

            # Try to find a preferred opponent (not the same as last round).
            preferred = [p for p in remaining if p != last_opp]
            if preferred:
                b = self._rng.choice(preferred)
            else:
                # Everyone left was our last opponent — pick anyone.
                b = self._rng.choice(remaining)

            remaining.remove(b)
            pairs.append((a, b))

        # Sanity check: no leftover players (caller ensures even count).
        assert len(remaining) == 0, f"Leftover players after pairing: {remaining}"
        return pairs
