"""Firestone combat simulator client for win probability estimation.

Priority order for simulation backends:
1. Real Firestone subprocess (if *firestone_path* is given and exists).
2. Pure-Python BGCombatSim (default — fast, no external deps).
3. Heuristic power-ratio estimate (last-resort fallback).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

# SimResult and BGCombatSim live in combat_sim; re-export SimResult so that
# callers which import it from here continue to work unchanged.
from symbolic.combat_sim import BGCombatSim, SimResult  # noqa: F401


class FirestoneClient:
    """Wrapper around the Firestone combat simulator subprocess.

    When ``firestone_path`` is ``None`` or points to a non-existent file the
    client uses the pure-Python BGCombatSim as its primary backend.

    Parameters
    ----------
    firestone_path:
        Path to the Firestone simulator executable or entry-point script.
        Pass ``None`` (the default) to use the Python simulator.
    n_trials:
        Number of Monte Carlo trials per simulation call (default 200).
    mock_mode:
        Force the heuristic-only estimator.  Useful for unit tests that need
        deterministic, instant results.
    """

    def __init__(
        self,
        firestone_path: Optional[str] = None,
        n_trials: int = 200,
        mock_mode: bool = False,
    ) -> None:
        self.n_trials = n_trials
        self._bg_sim  = BGCombatSim(n_trials=n_trials)

        if mock_mode:
            self._firestone_path: Optional[Path] = None
            self.mock_mode = True
        elif firestone_path is not None:
            resolved = Path(firestone_path)
            self._firestone_path = resolved if resolved.exists() else None
            self.mock_mode = False
        else:
            self._firestone_path = None
            self.mock_mode = False   # BGCombatSim is the default, not "mock"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(
        self,
        player_board:   List[dict],
        opponent_board: List[dict],
        player_tier:    int = 1,
        opp_tier:       int = 1,
    ) -> SimResult:
        """Estimate win probability for *player_board* against *opponent_board*.

        Parameters
        ----------
        player_board, opponent_board:
            Lists of minion dicts (keys: name, card_id, attack, health, tier,
            divine_shield, venomous/poisonous, reborn, taunt, windfury, golden,
            tribes/tribe, perm_atk_bonus, game_atk_bonus, perm_hp_bonus,
            game_hp_bonus).
        player_tier, opp_tier:
            Tavern tiers — used for win-damage calculation.

        Returns
        -------
        SimResult with win_prob and expected damage figures.
        """
        # Heuristic-only mode (testing / debugging)
        if self.mock_mode:
            return self._heuristic_estimate(player_board, opponent_board)

        # Subprocess Firestone (real simulator, if configured)
        if self._firestone_path is not None:
            try:
                return self._run_firestone(player_board, opponent_board)
            except Exception:
                pass  # fall through to Python sim

        # Python BGCombatSim (primary default)
        try:
            return self._bg_sim.simulate(
                player_board, opponent_board,
                player_tier=player_tier,
                opp_tier=opp_tier,
            )
        except Exception:
            return self._heuristic_estimate(player_board, opponent_board)

    def is_available(self) -> bool:
        """Return True when a real simulator backend (not heuristic) is active."""
        return not self.mock_mode

    # ------------------------------------------------------------------
    # Heuristic estimator (last-resort fallback)
    # ------------------------------------------------------------------

    def _heuristic_estimate(
        self,
        player_board: List[dict],
        opponent_board: List[dict],
    ) -> SimResult:
        """Simple power-based heuristic: total (ATK+HP), golden counts double."""
        def _power(board: List[dict]) -> float:
            total = 0.0
            for m in board:
                atk = m.get("attack", 0)
                hp  = m.get("health", 0)
                total += (atk + hp) * (2.0 if m.get("golden") else 1.0)
            return total

        pp = _power(player_board)
        op = _power(opponent_board)
        win_prob = max(0.05, min(0.95, pp / (pp + op + 1e-9)))

        return SimResult(
            win_prob=win_prob,
            expected_damage_dealt=win_prob * 5.0,
            expected_damage_taken=(1.0 - win_prob) * 5.0,
            trials=0,
        )

    # ------------------------------------------------------------------
    # Subprocess interface (real Firestone, optional)
    # ------------------------------------------------------------------

    def _run_firestone(
        self,
        player_board: List[dict],
        opponent_board: List[dict],
    ) -> SimResult:
        """Run the Firestone simulator as a subprocess and parse its JSON output.

        Expected stdout format::

            {"win_prob": 0.61, "expected_damage_dealt": 3.05,
             "expected_damage_taken": 1.95, "trials": 200}
        """
        payload = {
            "player_board":  player_board,
            "opponent_board": opponent_board,
            "trials":        self.n_trials,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(payload, tmp)
            tmp_path = tmp.name

        try:
            proc = subprocess.run(
                [str(self._firestone_path), "--input", tmp_path,
                 "--trials", str(self.n_trials)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Firestone exited {proc.returncode}: {proc.stderr.strip()}"
                )
            data = json.loads(proc.stdout)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return SimResult(
            win_prob=float(data["win_prob"]),
            expected_damage_dealt=float(data.get("expected_damage_dealt", 0.0)),
            expected_damage_taken=float(data.get("expected_damage_taken", 0.0)),
            trials=int(data.get("trials", self.n_trials)),
        )
