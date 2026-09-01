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
from env.player_state import minion_stats


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic-estimator damage constants
#
# Real Battlegrounds combat damage (see symbolic/combat_sim.py CombatSide.
# win_damage, the ground truth this heuristic approximates) is:
#     winner's tavern_tier + sum(tier for m in winner's SURVIVING minions)
# The heuristic below cannot know which minions survive without actually
# simulating combat (it must stay O(n) over the boards, not Monte Carlo), so
# it approximates "surviving minion tier sum" as the pre-combat tier sum of
# the winning board scaled by a "dominance" factor derived from how lopsided
# the win_prob/loss_prob estimate is: a crushing win (win_prob near its
# clamped ceiling) leaves most of the board alive; a narrow win leaves little.
# ─────────────────────────────────────────────────────────────────────────────

# Floor on the fraction of the winning board's pre-combat tier-sum that is
# assumed to "survive" into the damage number, even for the narrowest possible
# win. Real narrow wins in BG almost never wipe the winner's board down to
# nothing — the last attacker(s) that sealed the win are usually still up —
# so 0.0 would understate even a coin-flip win. Kept modest since a narrow
# win legitimately does lose most of the board.
MIN_SURVIVE_FRAC = 0.15

# Ceiling on that same fraction: a maximally dominant win (win_prob/loss_prob
# at the heuristic's own clamp bounds, see win_prob = clamp(0.05, 0.95, ...)
# below) is treated as leaving the entire pre-combat board alive. 1.0 rather
# than something higher keeps damage bounded by tavern_tier + board tier-sum,
# matching the real formula's own ceiling (it can't exceed that either, since
# the real formula only counts survivors, a subset of the pre-combat board).
MAX_SURVIVE_FRAC = 1.0

# win_prob/loss_prob are clamped to [0.05, 0.95] a few lines below (no signal
# past those bounds). This is the corresponding clamp *radius* around the 0.5
# (evenly-matched) midpoint, i.e. 0.95 - 0.5 == 0.5 - 0.05. Dividing by it
# maps the full clamped win_prob/loss_prob range onto survive-fraction
# [MIN_SURVIVE_FRAC, MAX_SURVIVE_FRAC] linearly, with no dead zone at either
# end.
DOMINANCE_RANGE = 0.45


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
            return self._heuristic_estimate(
                player_board, opponent_board,
                player_tier=player_tier, opp_tier=opp_tier,
            )

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
        player_tier: int = 1,
        opp_tier:    int = 1,
    ) -> SimResult:
        """Simple power-based heuristic: total effective (ATK+HP), golden
        counts double.

        Effective stats (base + perm_*/game_* bonuses) mirror
        symbolic.board_computer._board_power -- this used to only look at
        base attack/health via "attack"/"health" keys, so (a) it ignored
        every permanent/this-game buff on the board, and (b) prior to the
        base_atk/base_hp normalisation fix (see env/player_state.py:
        minion_stats), it silently read 0 for every minion, making this
        heuristic's win_prob a constant 0.05 regardless of board strength.
        This heuristic backend is what mock_mode=True (used by the actual
        parallel self-play training workers, see train.py._worker_run_game)
        resolves every combat with -- see CONTEXT.md 2026-09-01.

        Damage: real Battlegrounds damage is
            winner's tavern_tier + sum(tier for m in winner's SURVIVING minions)
        (see symbolic/combat_sim.py CombatSide.win_damage, the ground truth
        this approximates). This heuristic doesn't simulate combat so it
        can't know exactly which minions survive; it approximates the
        surviving-tier-sum as the winning board's full pre-combat tier sum
        scaled by a "dominance" factor in [MIN_SURVIVE_FRAC, MAX_SURVIVE_FRAC]
        derived from win_prob/loss_prob -- a crushing win leaves most of the
        board standing, a narrow win leaves little. Previously this was a
        flat ``win_prob * 5.0`` / ``loss_prob * 5.0`` that never exceeded 5
        regardless of tavern tier or board size, which meant real-game damage
        escalation (roughly 2-5 early, 15-25 late) never happened and every
        training game ran out the clock at the 40-round cap instead of ending
        by elimination -- see CONTEXT.md 2026-09-01.
        """
        def _power(board: List[dict]) -> float:
            total = 0.0
            for m in board:
                atk, hp = minion_stats(m)
                atk += m.get("perm_atk_bonus", 0) + m.get("game_atk_bonus", 0)
                hp  += m.get("perm_hp_bonus", 0)  + m.get("game_hp_bonus", 0)
                total += (atk + hp) * (2.0 if m.get("golden") else 1.0)
            return total

        def _tier_sum(board: List[dict]) -> int:
            return sum(int(m.get("tier", 1)) for m in board)

        pp = _power(player_board)
        op = _power(opponent_board)
        win_prob = max(0.05, min(0.95, pp / (pp + op + 1e-9)))

        # Heuristic has no tie signal — assume ties are rare (5%)
        tie_prob  = 0.05 * (1.0 - abs(win_prob - 0.5) * 2)  # peaks at 0.05 when evenly matched
        loss_prob = max(0.0, 1.0 - win_prob - tie_prob)

        def _survive_frac(p: float) -> float:
            """Map a clamped win/loss probability to a surviving-board fraction.

            p == 0.5 (evenly matched) -> MIN_SURVIVE_FRAC (narrow win, most of
            the board traded away). p at its clamp bound (0.05 or 0.95) ->
            MAX_SURVIVE_FRAC (crushing win, board mostly intact).
            """
            dominance = max(0.0, min(1.0, (p - 0.5) / DOMINANCE_RANGE))
            return MIN_SURVIVE_FRAC + (MAX_SURVIVE_FRAC - MIN_SURVIVE_FRAC) * dominance

        # expected_damage_dealt: damage the OPPONENT takes when the PLAYER
        # wins, so it is attributed to the PLAYER's own tavern_tier and the
        # PLAYER's own board tier-sum (the winning side's stats), scaled by
        # how dominant a win win_prob implies.
        player_survive_frac = _survive_frac(win_prob)
        expected_damage_dealt = player_tier + _tier_sum(player_board) * player_survive_frac

        # expected_damage_taken: damage the PLAYER takes when the OPPONENT
        # wins, so it is attributed to the OPPONENT's own tavern_tier and the
        # OPPONENT's own board tier-sum, scaled by how dominant a win
        # loss_prob implies for the opponent.
        opp_survive_frac = _survive_frac(loss_prob)
        expected_damage_taken = opp_tier + _tier_sum(opponent_board) * opp_survive_frac

        return SimResult(
            win_prob=win_prob,
            tie_prob=tie_prob,
            loss_prob=loss_prob,
            expected_damage_dealt=expected_damage_dealt,
            expected_damage_taken=expected_damage_taken,
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

        win_prob  = float(data["win_prob"])
        tie_prob  = float(data.get("tie_prob",  0.0))
        loss_prob = float(data.get("loss_prob", max(0.0, 1.0 - win_prob - tie_prob)))
        return SimResult(
            win_prob=win_prob,
            tie_prob=tie_prob,
            loss_prob=loss_prob,
            expected_damage_dealt=float(data.get("expected_damage_dealt", 0.0)),
            expected_damage_taken=float(data.get("expected_damage_taken", 0.0)),
            trials=int(data.get("trials", self.n_trials)),
        )
