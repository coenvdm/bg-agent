"""Firestone combat simulator client for win probability estimation."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class SimResult:
    """Result of a single combat simulation run."""

    win_prob: float
    expected_damage_dealt: float
    expected_damage_taken: float
    trials: int


class FirestoneClient:
    """Wrapper around the Firestone combat simulator subprocess.

    When ``firestone_path`` is ``None`` or points to a non-existent file the
    client automatically falls back to a fast heuristic estimator so that the
    rest of the pipeline can run without the simulator installed.

    Parameters
    ----------
    firestone_path:
        Path to the Firestone simulator executable or entry-point script.
        Pass ``None`` to force mock/heuristic mode.
    n_trials:
        Number of Monte Carlo trials to request per simulation call.
    mock_mode:
        Override flag to force heuristic mode regardless of ``firestone_path``.
    """

    def __init__(
        self,
        firestone_path: Optional[str] = None,
        n_trials: int = 200,
        mock_mode: bool = False,
    ) -> None:
        self.n_trials = n_trials

        if mock_mode or firestone_path is None:
            self.mock_mode = True
            self._firestone_path: Optional[Path] = None
        else:
            resolved = Path(firestone_path)
            if resolved.exists():
                self._firestone_path = resolved
                self.mock_mode = False
            else:
                self._firestone_path = None
                self.mock_mode = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(
        self,
        player_board: List[dict],
        opponent_board: List[dict],
    ) -> SimResult:
        """Estimate win probability for *player_board* against *opponent_board*.

        Falls back to the heuristic estimator if the Firestone subprocess is
        unavailable or raises an error.

        Parameters
        ----------
        player_board:
            List of minion dicts for the player (same schema as
            parse_bg.py ``_minion_snap`` output).
        opponent_board:
            List of minion dicts for the opponent.

        Returns
        -------
        SimResult with win probability and expected damage figures.
        """
        if self.mock_mode:
            return self._heuristic_estimate(player_board, opponent_board)

        try:
            return self._run_firestone(player_board, opponent_board)
        except Exception:
            # Graceful degradation: fall back to heuristic if subprocess fails.
            return self._heuristic_estimate(player_board, opponent_board)

    def is_available(self) -> bool:
        """Return True when the Firestone subprocess is configured and reachable."""
        return not self.mock_mode

    # ------------------------------------------------------------------
    # Heuristic estimator (mock mode)
    # ------------------------------------------------------------------

    def _heuristic_estimate(
        self,
        player_board: List[dict],
        opponent_board: List[dict],
    ) -> SimResult:
        """Simple power-based heuristic for win probability.

        Computes total (attack + health) power for each side.  The win
        probability is the player's share of total power, clamped to
        [0.05, 0.95] to avoid degenerate certainties.
        """
        def _board_power(board: List[dict]) -> float:
            total = 0.0
            for minion in board:
                atk = minion.get("attack", 0)
                hp = minion.get("health", 0)
                multiplier = 2.0 if minion.get("golden", False) else 1.0
                total += (atk + hp) * multiplier
            return total

        player_power = _board_power(player_board)
        opponent_power = _board_power(opponent_board)

        raw_win_prob = player_power / (player_power + opponent_power + 1e-9)
        win_prob = max(0.05, min(0.95, raw_win_prob))

        return SimResult(
            win_prob=win_prob,
            expected_damage_dealt=win_prob * 5.0,
            expected_damage_taken=(1.0 - win_prob) * 5.0,
            trials=0,
        )

    # ------------------------------------------------------------------
    # Subprocess interface
    # ------------------------------------------------------------------

    def _run_firestone(
        self,
        player_board: List[dict],
        opponent_board: List[dict],
    ) -> SimResult:
        """Run the Firestone simulator as a subprocess and parse its output.

        Writes board state to a temporary JSON file, invokes Firestone with
        ``--input <file> --trials <n>``, and parses the JSON result from
        stdout.

        Expected Firestone stdout format::

            {
                "win_prob": 0.61,
                "expected_damage_dealt": 3.05,
                "expected_damage_taken": 1.95,
                "trials": 200
            }
        """
        payload = {
            "player_board": player_board,
            "opponent_board": opponent_board,
            "trials": self.n_trials,
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            json.dump(payload, tmp)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [str(self._firestone_path), "--input", tmp_path,
                 "--trials", str(self.n_trials)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Firestone exited with code {result.returncode}: "
                    f"{result.stderr.strip()}"
                )
            data = json.loads(result.stdout)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return SimResult(
            win_prob=float(data["win_prob"]),
            expected_damage_dealt=float(data.get("expected_damage_dealt", 0.0)),
            expected_damage_taken=float(data.get("expected_damage_taken", 0.0)),
            trials=int(data.get("trials", self.n_trials)),
        )

    def _serialize_board(self, board: List[dict]) -> str:
        """Serialize a board (list of minion dicts) to a JSON string."""
        return json.dumps(board, ensure_ascii=False)
