"""
BattlegroundsGame — 8-player self-play game loop.

Simulates a full Hearthstone Battlegrounds game using:
  - TavernPool for card draws
  - Matchmaker for pairing
  - SymbolicBoardComputer for board analysis
  - FirestoneClient for combat simulation
  - PPO agents (or random / scripted agents) as players
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from env.player_state import MinionState, OpponentSnapshot, PlayerState
from env.tavern_pool import TavernPool
from env.matchmaker import Matchmaker
from symbolic.board_computer import SymbolicBoardComputer, _board_power
from symbolic.firestone_client import FirestoneClient
from symbolic.combat_sim import BGCombatSim
from symbolic.effect_handler import EffectHandler
from symbolic.hero_handler import HeroPowerHandler
from env.trinket_handler import TrinketHandler
from agent.card_encoder import CardEncoder
from agent.hero_encoder import HERO_DEF_MAP, NULL_HERO_ID

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reward constants (CLAUDE.md)
# ---------------------------------------------------------------------------
#
# --- Unified potential-based reward shaping (Ng, Harada & Russell 1999) ----
#
# A single potential Φ(s) (stored in PlayerState.phi) replaces the old, split
# phi_board/phi_tier scheme. The old scheme reset both potentials at the start
# of every round, which is a one-sided ratchet: the agent was PAID for
# building board/tier strength up within a turn, but was never CHARGED when
# combat (or simply falling behind the tavern-tier curve as round_num
# advances) reduced it, because the baseline was silently re-established at
# the next round's start. This both produced farmable, degenerate incentives
# (a 312-update run showed sell:place climbing 0.46->0.84 and level_rate
# collapsing 0.12->0.036 while reward improved and placement stayed flat --
# see CONTEXT.md 2026-08-31) and broke the policy-invariance guarantee this
# shaping style exists to provide, since Ng et al.'s proof requires Φ to be a
# genuine function of state evaluated consistently across the whole episode.
#
# Fix: Φ(s) is initialised once, to Φ(s_0), in reset() (see PlayerState.phi's
# docstring) and NEVER reset again. BattlegroundsGame._apply_potential_shaping
# is called at EVERY point a reward is finalised for a player -- every
# step_shopping action type, not only PLACE/SELL/LEVEL_UP, and once per round
# right after combat resolves in run_game -- so the shaping term telescopes
# exactly across the episode: with SHAPE_GAMMA matching PPOConfig.gamma
# exactly, the PPO-discounted sum of shaped rewards collapses to
# SHAPE_ALPHA * (SHAPE_GAMMA**T * Φ(s_T) - Φ(s_0)), independent of the path
# taken. Consequently ANY cyclic action sequence (buy/sell churn, level-then-
# idle, freeze/unfreeze, ...) that returns Φ to the same value nets ~0 shaped
# reward (up to the tiny (SHAPE_GAMMA - 1) discount drag on intermediate
# states -- expected and harmless, see the scratchpad telescoping test), while
# a genuinely improving board/tier trajectory that is KEPT nets positive. This
# is the structural guarantee against farming shaping instead of playing.
SHAPE_ALPHA = 0.20   # shaping scale; matches the old BOARD_SHAPE_ALPHA
                      # magnitude -- board-strength was always the dominant of
                      # the two old potentials, and Φ is now normalised to
                      # [0, 1] (see _potential), so this bounds the maximum
                      # possible telescoped shaping magnitude for a whole
                      # episode to +-SHAPE_ALPHA.
SHAPE_GAMMA = 0.997   # MUST match PPOConfig.gamma (agent/ppo.py) -- this is
                      # not a free tuning knob. Using anything else (e.g. the
                      # old undiscounted 1.0) breaks the exact telescoping
                      # identity above, which is what makes the invariance
                      # guarantee hold with PPO's own GAE discounting.
BOARD_SHAPE_TRIALS = 30    # sim trials per _board_win_prob call (~0.5 ms each)

# Deterministic, noise-free board-strength potential -- see BattlegroundsGame.
# shape_stats_weight, which train.py now fixes at 1.0 (see BOARD_SHAPE_STATS_WEIGHT):
# this fully replaced the MC win-probability estimate as the board-shape potential
# 2026-08-31, after a full training run showed reward/placement plateauing and then
# regressing right around when the old anneal-to-0 schedule finished fading this
# potential out (~250k steps) in favour of the noisy win-probability estimate, with
# a rising sell:place ratio and shrinking board size in that same window -- consistent
# with the policy learning to farm noise in the 30-trial MC estimate rather than
# actually improving the board. See CONTEXT.md for the full analysis.
#
# BOARD_SHAPE_STATS_SATURATION is the total effective attack+health+keyword+synergy
# value at which this potential reads 0.5 (rough mid-game board, untuned).
BOARD_SHAPE_STATS_SATURATION = 30.0

# Per-instance bonus for a "punches above its raw stats" keyword -- Divine Shield,
# Taunt, Reborn, and Windfury all add real combat value a plain atk+hp sum misses.
# Untuned initial guess, same as BOARD_SHAPE_STATS_SATURATION.
BOARD_STATS_KEYWORD_BONUS = 3.0

# Flat bonus when the board has >=4 minions of one tribe -- mirrors CLAUDE.md's
# "synergistic" threshold (Symbolic Layer Rule 4). Binary, not per-card, since going
# from 4->5 of a tribe is a much smaller jump than crossing the threshold at all.
BOARD_STATS_SYNERGY_BONUS = 5.0

# Weights combining the board-strength and tavern-tier-pace components into
# the single unified potential Φ(s) (see BattlegroundsGame._potential). Both
# components are already normalised to [0, 1] (see _board_potential /
# _tier_potential), so weights that sum to 1.0 keep Φ(s) in [0, 1] too. The
# 2:1 split mirrors the old BOARD_SHAPE_ALPHA (0.20) : TIER_SHAPE_ALPHA (0.10)
# ratio -- board quality was always weighted about twice as heavily as
# leveling pace.
BOARD_POTENTIAL_WEIGHT = 0.67
TIER_POTENTIAL_WEIGHT  = 1.0 - BOARD_POTENTIAL_WEIGHT  # = 0.33; exact complement
                                                         # guarantees weights sum to 1.0

# --- Dense per-round / per-action reward terms -----------------------------
#
# These terms fire every round or every action, so across a typical ~10-15
# round game they accumulate far more often than the once-per-game
# FINAL_PLACEMENT_REWARD (below), which is the actual objective (spans -4..
# +4). Measured on 20 games of mixed GreedyPlayAgent/HeuristicAgent scripted
# baselines (see CONTEXT.md 2026-08-31 for the full decomposition): with the
# flat per-round survival bonus removed (see compute_round_reward), the
# remaining dense terms summed to ~-6.49/player-game against a
# FINAL_PLACEMENT_REWARD mean of only ~-0.375/player-game (~17x) -- dense
# terms were large enough to swamp even a 1st-place finish (+4) into a
# net-negative total reward, which is exactly the "optimizing the shaping
# instead of the game" failure mode this rebalance fixes.
#
# DENSE_REWARD_SCALE rescales every dense coefficient below by the same
# shared factor (rather than retuning each independently) so the *relative*
# weighting between dense terms — already reasoned about individually when
# each was introduced — is preserved. 0.30 brings the post-removal dense
# total to about -1.9/player-game: comfortably below FINAL_PLACEMENT_REWARD's
# +-4 span (so a strong finish can no longer be drowned out) while still
# leaving a meaningful per-round gradient, roughly the size of one placement-
# rank step (e.g. 2nd->3rd is a swing of 1.0).
DENSE_REWARD_SCALE = 0.30

WIN_REWARD          =  0.5  * DENSE_REWARD_SCALE   # was 0.5;  compute_round_reward win term
LOSS_PENALTY         = -0.3  * DENSE_REWARD_SCALE   # was -0.3; compute_round_reward loss term
DAMAGE_TAKEN_COEF    =  0.05 * DENSE_REWARD_SCALE   # was 0.05; compute_round_reward
DAMAGE_DEALT_COEF    =  0.05 * DENSE_REWARD_SCALE   # was 0.05; compute_round_reward
RANK_DELTA_COEF      =  0.15 * DENSE_REWARD_SCALE   # was 0.15; compute_round_reward
HAND_PENALTY_COEF    =  0.08 * DENSE_REWARD_SCALE   # was 0.08; _end_of_turn_reward
GOLD_PENALTY_COEF    =  0.05 * DENSE_REWARD_SCALE   # was 0.05; _end_of_turn_reward
EMPTY_BOARD_PENALTY  =  0.30 * DENSE_REWARD_SCALE   # was 0.30 flat; step_shopping SELL
REROLL_PENALTY_BASE  =  0.05 * DENSE_REWARD_SCALE   # was 0.05; step_shopping REROLL
REROLL_PENALTY_STEP  =  0.05 * DENSE_REWARD_SCALE   # was 0.05/reroll past 2; step_shopping REROLL

FINAL_PLACEMENT_REWARD: Dict[int, float] = {
    1: +4.0,
    2: +2.0,
    3: +1.0,
    4:  0.0,
    5: -1.0,
    6: -2.0,
    7: -3.0,
    8: -4.0,
}

# Shop sizes per tavern tier
SHOP_SIZE_FOR_TIER = {1: 3, 2: 4, 3: 4, 4: 5, 5: 5, 6: 6, 7: 7}

# Number of minions offered in the shop at each tier
def shop_size(tier: int) -> int:
    return SHOP_SIZE_FOR_TIER.get(tier, 3)


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

def compute_round_reward(
    damage_taken: int,
    damage_dealt: float,
    prev_rank: int,
    cur_rank: int,
    result: str,           # "win" | "loss" | "tie"
    max_health: int = 40,
) -> float:
    """Dense reward shaping for one shopping+combat round.

    Components (coefficients scaled by DENSE_REWARD_SCALE -- see the module
    constants block for why: unscaled, these dense per-round terms summed to
    roughly 17x FINAL_PLACEMENT_REWARD's magnitude, drowning out the actual
    objective)
    ----------
    Combat outcome : +WIN_REWARD win / LOSS_PENALTY loss
    Damage taken   : -DAMAGE_TAKEN_COEF * (damage / max_health)  — penalise health loss
    Damage dealt   : +DAMAGE_DEALT_COEF * (damage / max_health)  — reward hurting opponents
    Rank delta     : (prev_rank - cur_rank) * RANK_DELTA_COEF  — positive when rank
                     improves; fires both on combat health changes AND opponent
                     eliminations

    Note: gold efficiency (-GOLD_PENALTY_COEF * unspent_gold) is applied in
    step_shopping at END_TURN, not here, since it fires mid-round before combat.

    The flat +0.1 per-round survival bonus that used to live here was removed
    2026-08-31: it was unconditional passive income (~+1.2/game, fired merely
    for being alive) that diluted the placement signal without rewarding any
    actual decision -- see CONTEXT.md for the STEP 0 measurement that found it.
    """
    r  = WIN_REWARD          if result == "win"  else 0.0
    r += LOSS_PENALTY         if result == "loss" else 0.0
    r += -DAMAGE_TAKEN_COEF * (damage_taken / max_health)
    r +=  DAMAGE_DEALT_COEF * (damage_dealt  / max_health)
    r += (prev_rank - cur_rank) * RANK_DELTA_COEF
    return r


# ---------------------------------------------------------------------------
# Game result
# ---------------------------------------------------------------------------

@dataclass
class GameResult:
    """Result of a single completed BG game."""

    placements:    Dict[int, int]    # player_id → placement (1=winner, 8=last)
    final_rewards: Dict[int, float]  # player_id → total accumulated reward
    round_history: List[dict]        # per-round summary dicts
    n_rounds:      int


# ---------------------------------------------------------------------------
# Observation building helpers
# ---------------------------------------------------------------------------

def _minion_to_dict(m) -> dict:
    """Convert MinionState or dict to plain dict."""
    if isinstance(m, dict):
        return m
    return m.__dict__ if hasattr(m, "__dict__") else {}


_TRIBE_LIST = [
    "BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECH",
    "MURLOC", "NAGA", "PIRATE", "QUILBOAR", "UNDEAD",
]


def _board_dominant_tribe(board) -> Tuple[Optional[str], int]:
    """Return (dominant_tribe, count) for a list of MinionState/dicts.

    Uses the card's `tribes` field when available; falls back to `tribe`.
    Returns (None, 0) when the board is empty or has no tribal minions.
    """
    from collections import Counter
    counts: Counter = Counter()
    for m in board:
        d = _minion_to_dict(m)
        tribes = d.get("tribes") or ([d["tribe"]] if d.get("tribe") else [])
        for t in tribes:
            t_up = t.upper()
            if t_up in _TRIBE_LIST:
                counts[t_up] += 1
    if not counts:
        return None, 0
    top, cnt = counts.most_common(1)[0]
    return top, cnt


def _pad_list(lst: list, length: int, fill=None) -> list:
    """Pad or truncate list to exactly *length* elements."""
    return list(lst[:length]) + [fill] * max(0, length - len(lst))


def _encode_zone(
    minions: list,
    encoder: CardEncoder,
    max_slots: int,
    *,
    board_size: int = 0,
    dominant_tribe_count: int = 0,
    total_aura_dependency: float = 0.0,
    round_num: int = 1,
    tavern_tier: int = 1,
) -> np.ndarray:
    """Encode a zone (board/shop/hand) to [max_slots, 44] float32."""
    dicts = [_minion_to_dict(m) for m in minions if m is not None]
    return encoder.encode_board(
        dicts,
        board_size=board_size,
        dominant_tribe_count=dominant_tribe_count,
        total_aura_dependency=total_aura_dependency,
        round_num=round_num,
        tavern_tier=tavern_tier,
        max_slots=max_slots,
    )


# ---------------------------------------------------------------------------
# Smart play positioning
# ---------------------------------------------------------------------------

def _smart_position(minion: MinionState, board: list) -> int:
    """Return the board insertion index for a minion being played.

    Priority order: Taunt → Divine Shield → Windfury → Normal (append).
    Taunt and Divine Shield minions go to the front (index 0) so they absorb
    hits early.  Windfury minions go to the back to attack twice safely.
    All other minions append to the end.
    """
    if minion.taunt:
        return 0
    if minion.divine_shield:
        return 0
    if minion.windfury:
        return len(board)
    return len(board)


# ---------------------------------------------------------------------------
# Main game class
# ---------------------------------------------------------------------------

class BattlegroundsGame:
    """Runs a full 8-player Hearthstone Battlegrounds self-play game.

    Parameters
    ----------
    card_defs:
        Mapping card_id → card definition dict from bg_card_definitions.json.
    agents:
        List of agent objects with a ``get_action(obs)`` method, or None
        for a random agent.  Must be length n_players or None.
    board_computer:
        SymbolicBoardComputer instance.
    firestone_client:
        FirestoneClient instance (mock or real).
    matchmaker:
        Matchmaker instance.
    tavern_pool:
        TavernPool instance.
    n_players:
        Number of players (default 8).
    max_rounds:
        Hard cap on rounds before the game is forced to end (default 40).
    seed:
        Optional RNG seed for reproducibility.
    shape_stats_weight:
        Weight in [0, 1] given to the deterministic board-stats potential when
        blended into board-shape reward, vs. the noisy MC win-probability
        estimate (weight 1 - shape_stats_weight). 0 = pure win-probability
        (default, unchanged behaviour). Callers doing training-progress
        annealing (see train.py) pass a decaying value per game.
    """

    def __init__(
        self,
        card_defs: Dict[str, dict],
        agents: Optional[List[Any]],
        board_computer: SymbolicBoardComputer,
        firestone_client: FirestoneClient,
        matchmaker: Matchmaker,
        tavern_pool: TavernPool,
        n_players: int = 8,
        max_rounds: int = 40,
        seed: Optional[int] = None,
        batched: bool = True,
        shape_stats_weight: float = 0.0,
    ) -> None:
        self.card_defs       = card_defs
        self.agents          = agents or [None] * n_players
        self.board_computer  = board_computer
        self.firestone       = firestone_client
        self.matchmaker      = matchmaker
        self.tavern_pool     = tavern_pool
        self.n_players       = n_players
        self.max_rounds      = max_rounds
        self.shape_stats_weight = shape_stats_weight
        self.batched         = batched
        self._rng            = random.Random(seed)
        self.encoder         = CardEncoder(card_defs)
        self.effect_handler  = EffectHandler(card_defs, tavern_pool=self.tavern_pool)
        self.hero_handler    = HeroPowerHandler(card_defs, HERO_DEF_MAP)
        self.trinket_handler = TrinketHandler(card_defs, rng=self._rng if hasattr(self, "_rng") else None)
        self._shape_sim      = BGCombatSim(n_trials=BOARD_SHAPE_TRIALS)

        # Populated by reset()
        self.players: List[PlayerState] = []
        self.round_num: int = 0
        self._accumulated_rewards: Dict[int, float] = {}
        self._placement_counter: int = 0  # counts up from 8 as players die

    # ------------------------------------------------------------------
    # Gold / cost helpers
    # ------------------------------------------------------------------

    def _gold_for_round(self, round_num: int) -> int:
        """Gold available = min(2 + round_num, 10)."""
        return min(2 + round_num, 10)

    def _expected_tier_for_round(self, round_num: int) -> int:
        """Rough 'on-curve' tavern tier for a competent player at this round.

        Used only as the denominator for tier-shape potential (_tier_potential) --
        not a hard target or a claim about optimal play. Untuned; revisit alongside
        TIER_POTENTIAL_WEIGHT if leveling ends up over/under-incentivized in practice.
        """
        if round_num <= 1:
            return 1
        elif round_num <= 3:
            return 2
        elif round_num <= 5:
            return 3
        elif round_num <= 7:
            return 4
        elif round_num <= 9:
            return 5
        else:
            return 6

    def _end_of_turn_reward(self, ps) -> float:
        """Shared reward shaping applied at the end of every shopping phase.

        Empty-board penalty  : -EMPTY_BOARD_PENALTY if the board is empty — breaks
                               level-then-end-turn degenerate policy.
        Hand penalty         : -HAND_PENALTY_COEF per card left in hand — cards in
                               hand don't fight; discourages buying without placing.
        Gold efficiency      : -GOLD_PENALTY_COEF * unspent_gold (scaled down over
                               rounds).

        Coefficients scaled by DENSE_REWARD_SCALE -- see the module constants
        block.
        """
        r = 0.0
        board_size = len(ps.board)
        hand_size  = len(ps.hand)
        # Empty-board penalty fires at the SELL action that empties the board (not here)
        # so that credit assignment is immediate rather than deferred.
        # Hand penalty: bought cards that aren't placed don't help in combat
        r -= HAND_PENALTY_COEF * hand_size
        # Unspent gold penalty (fades to 20% by round 16+)
        gold_scale = max(0.2, 1.0 - (ps.round_num - 1) / 15.0)
        r -= GOLD_PENALTY_COEF * ps.gold * gold_scale
        return r

    def _level_cost_for_tier(self, current_tier: int) -> int:
        """Base upgrade cost: tier1→5, tier2→7, tier3→8, tier4→9, tier5→10."""
        costs = {1: 5, 2: 7, 3: 8, 4: 9, 5: 10, 6: 0}
        return costs.get(current_tier, 0)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> List[dict]:
        """Reset the game and return initial observations for each player."""
        self.tavern_pool.reset()
        self.matchmaker.history.clear()
        self.round_num = 1
        self._placement_counter = self.n_players  # placements count down from n_players (last place)
        self._accumulated_rewards = {i: 0.0 for i in range(self.n_players)}

        # Assign heroes: sample without replacement from active pool
        hero_ids = list(HERO_DEF_MAP.keys())
        active_heroes = [
            hid for hid in hero_ids
            if HERO_DEF_MAP[hid].get("phase", 99) <= 2  # phases 0-2 only
        ]
        chosen_heroes = self._rng.sample(
            active_heroes, min(self.n_players, len(active_heroes))
        )
        # Pad with null hero if not enough distinct heroes
        while len(chosen_heroes) < self.n_players:
            chosen_heroes.append(NULL_HERO_ID)

        self.players = []
        for pid in range(self.n_players):
            hero_card_id = chosen_heroes[pid]
            hdef = HERO_DEF_MAP.get(hero_card_id, HERO_DEF_MAP[NULL_HERO_ID])
            ps = PlayerState(
                player_id=pid,
                hero_card_id=hero_card_id,
                health=40,
                armor=hdef.get("armor", 0),
                max_health=40,
                gold=self._gold_for_round(1),
                max_gold=10,
                tavern_tier=1,
                level_cost=self._level_cost_for_tier(1),
                frozen=False,
                round_num=1,
                alive=True,
                hero_power_cost=hdef.get("power_cost", 0),
                hero_power_charges=hdef.get("total_charges", -1),
                hero_power_counter=0,
                hero_power_x=4,
                buy_cost=3,
                reroll_cost=1,
            )
            # Draw initial shop
            ps.shop = self._draw_shop(ps)
            # Φ(s_0) -- the one and only reset point for ps.phi. See PlayerState.phi
            # and the module constants block for why it must never be reset again.
            ps.phi = self._potential(ps)
            self.players.append(ps)

        return [self._get_observation(pid) for pid in range(self.n_players)]

    # ------------------------------------------------------------------
    # Shop drawing
    # ------------------------------------------------------------------

    def _draw_shop(self, ps: PlayerState) -> List[MinionState]:
        """Draw fresh shop cards for a player, respecting frozen cards."""
        n = shop_size(ps.tavern_tier)
        if ps.frozen:
            # Keep existing shop; only fill empty slots
            existing = list(ps.shop)
            n_needed = max(0, n - len(existing))
            new_cards = self.tavern_pool.draw(ps.tavern_tier, n_needed)
            return existing + [self._dict_to_minion(c) for c in new_cards]
        else:
            # Return old shop cards to the pool, draw fresh
            if ps.shop:
                self.tavern_pool.return_cards(
                    [_minion_to_dict(m) for m in ps.shop]
                )
            drawn = self.tavern_pool.draw(ps.tavern_tier, n)
            return [self._dict_to_minion(c) for c in drawn]

    def _dict_to_minion(self, d: dict) -> MinionState:
        """Convert a TavernPool card dict to a MinionState."""
        card_id = d.get("card_id", d.get("id", ""))
        card_def = self.card_defs.get(card_id, {})
        mechanics = [m.upper() for m in card_def.get("mechanics", [])]
        keywords = card_def.get("keywords", {})
        is_magnetic = (
            "MAGNETIC" in mechanics
            or bool(card_def.get("has_magnetic", False))
            or bool(keywords.get("magnetic", False))
        )
        # Detect spells: explicit flag/type, or card def present but has no stats
        is_spell = (
            bool(card_def.get("is_spell", False))
            or card_def.get("type", "").upper() == "SPELL"
            or (
                card_def
                and "base_atk" not in card_def
                and "base_hp" not in card_def
                and d.get("attack", -1) < 0
                and d.get("health", -1) < 0
            )
        )
        return MinionState(
            card_id=card_id,
            name=d.get("name", ""),
            attack=d.get("attack", 0),
            health=d.get("health", 0),
            max_health=d.get("health", 0),
            tier=d.get("tier", 1),
            magnetic=is_magnetic,
            is_spell=is_spell,
            activate_cost=int(card_def.get("activate_cost") or 0),
        )

    # ------------------------------------------------------------------
    # Board-strength potential Φ(s) for reward shaping
    # ------------------------------------------------------------------

    def _board_win_prob(self, ps) -> float:
        """Estimate win probability for ps's current board via fast Monte Carlo.

        Uses the announced next opponent's last known board as the reference
        opponent.  Falls back to an empty board (win_prob ≈ 1.0) when no
        opponent snapshot is available (early rounds).
        """
        if not ps.board:
            return 0.0
        player_board = [_minion_to_dict(m) for m in ps.board]
        opp_snap = (ps.opponent_snapshots.get(ps.next_opponent_id)
                    if ps.next_opponent_id is not None else None)
        # No snapshot yet (round 1) — return neutral 0.5 so shaping is meaningful
        # from the very first placement rather than saturating at 1.0 vs empty board.
        if not opp_snap or not opp_snap.board:
            return 0.5
        opp_board = [_minion_to_dict(m) for m in opp_snap.board]
        try:
            result = self._shape_sim.simulate(
                player_board, opp_board,
                player_tier=ps.tavern_tier,
                opp_tier=opp_snap.tavern_tier,
            )
            return result.win_prob
        except Exception:
            return 0.5

    def _board_stats_potential(self, ps) -> float:
        """Deterministic, noise-free board-strength potential: total effective
        stats (attack+health, via the same _board_power helper the symbolic
        layer uses elsewhere) plus keyword and tribal-synergy bonuses, saturating
        toward 1.0. Bounded to [0, 1), monotonic in board quality, zero noise.

        This is quality-weighted, not count-weighted -- a board of seven 1/1s
        scores barely above a board of two well-statted minions, so hoarding
        weak minions to fill slots doesn't pay off the way it would under a
        raw board-size proxy. Selling a weak minion to make room for a stronger
        one causes a momentary dip (the sale drops power immediately) but the
        replacement's placement raises the score net-positive, which is what
        makes "sell weak, buy strong" a learnable win rather than a pure cost.
        """
        if not ps.board:
            return 0.0
        board_dicts = [_minion_to_dict(m) for m in ps.board]
        power = _board_power(board_dicts)

        keyword_bonus = 0.0
        for m in board_dicts:
            if m.get("divine_shield"):
                keyword_bonus += BOARD_STATS_KEYWORD_BONUS
            if m.get("taunt"):
                keyword_bonus += BOARD_STATS_KEYWORD_BONUS
            if m.get("reborn"):
                keyword_bonus += BOARD_STATS_KEYWORD_BONUS
            if m.get("windfury"):
                keyword_bonus += BOARD_STATS_KEYWORD_BONUS

        from symbolic.effect_handler import _minion_tribes as _get_tribes
        tribe_counts: Dict[str, int] = {}
        for m in ps.board:
            for t in _get_tribes(m, self.card_defs):
                tribe_counts[t] = tribe_counts.get(t, 0) + 1
        synergy_bonus = BOARD_STATS_SYNERGY_BONUS if tribe_counts and max(tribe_counts.values()) >= 4 else 0.0

        value = power + keyword_bonus + synergy_bonus
        return value / (value + BOARD_SHAPE_STATS_SATURATION)

    def _board_potential(self, ps) -> float:
        """Board-strength component of Φ(s): the deterministic stats potential,
        optionally blended with the noisy MC win-probability estimate.

        shape_stats_weight is fixed at BOARD_SHAPE_STATS_WEIGHT (see train.py) --
        no longer annealed toward 0. At weight 1.0 (the current default) this
        skips _board_win_prob entirely: no point paying for a 30-trial combat
        sim whose result gets multiplied by zero, and it's one less noisy call
        in the hottest path in self-play (every action -- see
        _apply_potential_shaping). Bounded to [0, 1] either way, since both
        _board_stats_potential and _board_win_prob (a probability) are.
        """
        w = self.shape_stats_weight
        if w >= 1.0:
            return self._board_stats_potential(ps)
        if w <= 0.0:
            return self._board_win_prob(ps)
        return (1.0 - w) * self._board_win_prob(ps) + w * self._board_stats_potential(ps)

    def _tier_potential(self, ps) -> float:
        """Tier-pace component of Φ(s): tavern tier as a fraction of the round's
        on-curve tier, capped at 1.0 -- reaching or exceeding the curve fully
        saturates this component, so there's no reward for leveling further
        than useful for the round, only for closing a real gap.
        """
        expected = self._expected_tier_for_round(ps.round_num)
        return min(1.0, ps.tavern_tier / expected)

    def _potential(self, ps) -> float:
        """Unified potential Φ(s) for true potential-based reward shaping.

        Φ(s) = BOARD_POTENTIAL_WEIGHT * _board_potential(s)
             + TIER_POTENTIAL_WEIGHT  * _tier_potential(s)

        Both components are already normalised to [0, 1] and the weights sum
        to 1.0, so Φ(s) ∈ [0, 1] for every reachable state. This single Φ
        replaces the old separate board/tier potentials (see PlayerState.phi
        and the module constants block for why the split-and-reset design was
        broken) -- see _apply_potential_shaping for how it's paid out.
        """
        return (BOARD_POTENTIAL_WEIGHT * self._board_potential(ps)
                + TIER_POTENTIAL_WEIGHT * self._tier_potential(ps))

    def _apply_potential_shaping(self, ps) -> float:
        """Pay potential-based shaped reward for one transition and advance ps.phi.

        r_shaped = SHAPE_ALPHA * (SHAPE_GAMMA * Φ(s') - Φ(s))

        Must be called at EVERY point a reward is finalised for *ps* -- every
        step_shopping action type (BUY, SELL, PLACE, REROLL, FREEZE, LEVEL_UP,
        HERO_POWER, END_TURN, ACTIVATE), plus once per round right after
        combat resolves in run_game -- not only on board-changing actions.
        Telescoping is only exact when Φ is evaluated at every transition;
        skipping any of them would reopen a smaller version of the exact bug
        this replaced (see module constants block / CONTEXT.md 2026-08-31).

        Because ps.phi is never reset mid-episode, and SHAPE_GAMMA matches
        PPOConfig.gamma exactly, the PPO-discounted sum of every shaped reward
        paid out over a whole game collapses (telescopes) to exactly
            SHAPE_ALPHA * (SHAPE_GAMMA**T * Φ(s_T) - Φ(s_0))
        regardless of the path taken to get there -- any cyclic action
        sequence (buy/sell churn, level-then-idle, freeze/unfreeze, ...) that
        returns Φ to the same value nets ~0 shaped reward. This is the
        structural guarantee against farming shaping instead of playing;
        guaranteed not to change the optimal policy (Ng et al. 1999).
        """
        phi_after = self._potential(ps)
        shaped = SHAPE_ALPHA * (SHAPE_GAMMA * phi_after - ps.phi)
        ps.phi = phi_after
        return shaped

    # ------------------------------------------------------------------
    # Shopping phase
    # ------------------------------------------------------------------

    def step_shopping(
        self,
        player_id: int,
        type_action: int,
        ptr_action: int,
    ) -> Tuple[dict, float, bool]:
        """Execute one buy-phase action for a player.

        Parameters
        ----------
        player_id:
            Index into self.players.
        type_action:
            Action type index (0-8), matching ACTION_TYPE_NAMES in policy.py:
            0=buy, 1=sell, 2=place, 3=reroll, 4=freeze, 5=level_up,
            6=hero_power, 7=end_turn, 8=activate.
        ptr_action:
            Card pointer index (0-23) for buy/sell/place/activate; -1 otherwise.
            Layout: shop[0-6] | board[7-13] | hand[14-23].
            activate reuses the board zone (7-13) — it targets the activating
            minion itself, same slot as sell.

        Returns
        -------
        (next_obs, reward, done_with_shopping)
        """
        from agent.policy import PTR_SHOP_OFF, PTR_BOARD_OFF, PTR_HAND_OFF

        ps = self.players[player_id]
        reward = 0.0
        done = False

        # ── Trinket offer in progress: BUY(0/1/2) picks, END_TURN declines ─────
        if ps.trinket_offer_pending:
            if type_action == 0:  # BUY → pick trinket by shop slot index
                choice_idx = ptr_action - PTR_SHOP_OFF
                self.trinket_handler.select(ps, choice_idx)
            else:  # any other action (including END_TURN) declines the offer
                self.trinket_handler.decline(ps)
                if type_action == 7:
                    reward += self._end_of_turn_reward(ps)
                    self.hero_handler.on_end_turn(ps)
                    done = True
            # Potential shaping fires here too -- trinkets can buff the board
            # immediately on selection, and every reward-emitting point must
            # evaluate Φ for the telescoping identity to hold exactly.
            reward += self._apply_potential_shaping(ps)
            return self._get_observation(player_id), reward, done

        # ── Discover in progress: only BUY(0/1/2) is valid ───────────────────
        # The observation encodes discover options in shop slots [0-2].
        if ps.discover_pending:
            choice_idx = ptr_action - PTR_SHOP_OFF
            if type_action == 0 and 0 <= choice_idx < len(ps.discover_pending):
                chosen  = ps.discover_pending[choice_idx]
                rejects = [m for i, m in enumerate(ps.discover_pending)
                           if i != choice_idx]
                # Return unchosen cards to pool as dicts
                if self.tavern_pool is not None:
                    self.tavern_pool.return_cards(
                        [_minion_to_dict(m) for m in rejects]
                    )
                ps.discover_pending = []
                if len(ps.hand) < 10:
                    ps.hand.append(chosen)
            # All other actions are ignored while discover is pending. Still
            # evaluate shaping (see comment above) -- discover doesn't touch
            # board/tier so this is normally a no-op modulo the tiny
            # (SHAPE_GAMMA - 1) discount drag.
            shaped = self._apply_potential_shaping(ps)
            return self._get_observation(player_id), shaped, False

        if type_action == 0:
            # buy: ptr_action is shop slot index (ptr 0-6 → slot 0-6)
            i = ptr_action - PTR_SHOP_OFF
            eff_cost = 0 if ps.first_buy_free else max(0, ps.buy_cost - ps.buy_discount)
            if 0 <= i < len(ps.shop) and ps.shop[i] is not None and ps.gold >= eff_cost:
                minion = ps.shop.pop(i)
                ps.hand.append(minion)
                ps.gold = max(0, ps.gold - eff_cost)
                if ps.first_buy_free:
                    ps.first_buy_free = False  # consumed
                else:
                    ps.buy_discount = 0  # consume one-shot discount
                self.hero_handler.on_buy(ps, minion)
                self.effect_handler.on_buy(ps, minion)
                # Living Prison (Activate): the next minion bought this turn gives
                # its stats to the activating minion, one-shot.
                living_prison_src = getattr(ps, "_living_prison_source", None)
                if living_prison_src is not None:
                    ps._living_prison_source = None  # type: ignore[attr-defined]
                    if living_prison_src in ps.board:
                        living_prison_src.perm_atk_bonus += minion.attack
                        living_prison_src.perm_hp_bonus  += minion.health
                        living_prison_src.max_health     += minion.health
                from env.triple_system import check_and_process_triple
                check_and_process_triple(ps, self.tavern_pool)

        elif type_action == 1:
            # sell: ptr_action is board slot index (ptr 7-13 → slot 0-6)
            i = ptr_action - PTR_BOARD_OFF
            if 0 <= i < len(ps.board) and ps.board[i] is not None:
                minion = ps.board.pop(i)
                ps.gold = min(ps.max_gold, ps.gold + 1)
                self.effect_handler.on_sell(ps, minion)
                self.hero_handler.on_sell(ps, minion)
                # Fungalmancer Flurgl: inject Murloc into shop
                if getattr(ps, "_flurgl_murloc_due", False):
                    ps._flurgl_murloc_due = False  # type: ignore[attr-defined]
                    murlocs = self.tavern_pool.draw(ps.tavern_tier, 1)
                    # (TavernPool.draw doesn't filter by tribe, so this is approximate)
                    for card in murlocs:
                        ps.shop.append(self._dict_to_minion(card))
                # Tad: add a random Murloc to hand
                if getattr(ps, "_tad_due", False):
                    ps._tad_due = False  # type: ignore[attr-defined]
                    if len(ps.hand) < 10:
                        cards = self.tavern_pool.draw(ps.tavern_tier, 1)
                        for card in cards:
                            ps.hand.append(self._dict_to_minion(card))
                if not ps.board:                        # emptied the board — charge penalty here for clean credit assignment
                    reward -= EMPTY_BOARD_PENALTY

        elif type_action == 2:
            # place: ptr_action is hand slot index (ptr 14-23 → slot 0-9)
            h = ptr_action - PTR_HAND_OFF
            # Spells don't occupy a board slot; minions require board space
            if 0 <= h < len(ps.hand) and ps.hand[h] is not None:
                minion = ps.hand[h]
                board_full = len(ps.board) >= 7
                if minion.is_spell or not board_full:
                    ps.hand.pop(h)
                    if minion.is_spell:
                        # Cast the spell and discard — no board slot consumed
                        self._cast_spell(ps, minion)
                    else:
                        # Check Magnetic: merge with rightmost friendly Mech if present
                        mech_targets = [
                            m for m in ps.board
                            if "MECH" in (_minion_to_dict(m).get("tribes") or [])
                            or _minion_to_dict(m).get("tribe", "").upper() == "MECH"
                        ]
                        if minion.magnetic and mech_targets:
                            target = mech_targets[-1]
                            # Drone Duplicator (Activate): doubles the next Magnetization onto it.
                            dbl = 2 if getattr(target, "_magnetize_double_pending", False) else 1
                            if dbl == 2:
                                target._magnetize_double_pending = False  # type: ignore[attr-defined]
                            target.attack += minion.attack * dbl
                            target.health += minion.health * dbl
                            target.max_health += minion.max_health * dbl
                            if minion.divine_shield:
                                target.divine_shield = True
                            if minion.taunt:
                                target.taunt = True
                            if minion.venomous:
                                target.venomous = True
                            if minion.windfury:
                                target.windfury = True
                            if minion.reborn:
                                target.reborn = True
                            # Magnetic minion merged — not added to board
                        else:
                            # Normal placement with smart positioning
                            pos = _smart_position(minion, ps.board)
                            ps.board.insert(pos, minion)
                            # Apply accumulated "this game" tribe buffs
                            self._apply_game_buffs(ps, minion)
                        self._update_multiplier_flags(ps)
                        self.effect_handler.on_play(ps, minion)
                        self.hero_handler.on_play(ps, minion)
                        # P2-F: Mechagnome Interpreter — +2/+1 to played minion if it's a MECH
                        minion_tribes = _minion_to_dict(minion).get("tribes") or []
                        is_mech = (
                            "MECH" in [t.upper() for t in minion_tribes]
                            or _minion_to_dict(minion).get("tribe", "").upper() == "MECH"
                        )
                        if is_mech:
                            for aura_m in ps.board:
                                if "mechagnomeinterpreter" in aura_m.name.lower().replace(" ", "") and aura_m is not minion:
                                    mult = 2 if aura_m.golden else 1
                                    minion.perm_atk_bonus += 2 * mult
                                    minion.perm_hp_bonus  += 1 * mult
                                    minion.max_health     += 1 * mult
                        from env.triple_system import check_and_process_triple
                        check_and_process_triple(ps, self.tavern_pool)
                    # Potential shaping (see the single call at the end of this
                    # method) now fires for BOTH branches above (spell cast or
                    # minion placement) -- the old code only paid it for the
                    # minion branch, silently skipping shaping on spell casts even
                    # though spells (Blood Gem, Timecap'n Hooktail's aura, ...)
                    # can change board strength just as much as a placement can.

        elif type_action == 3:
            # reroll — consume a free refresh (Refreshing Anomaly) before spending gold
            _free = getattr(ps, "_free_refreshes", 0)
            if _free > 0 or ps.gold >= ps.reroll_cost:
                if _free > 0:
                    ps._free_refreshes = _free - 1  # type: ignore[attr-defined]
                else:
                    ps.gold -= ps.reroll_cost
                    # Escalating penalty: 1-2 rerolls ok, 3+ gets expensive fast
                    _n_rerolls = getattr(ps, "_rerolls_this_turn", 0)
                    reward -= REROLL_PENALTY_BASE + REROLL_PENALTY_STEP * max(0, _n_rerolls - 2)
                    ps._rerolls_this_turn = _n_rerolls + 1  # type: ignore[attr-defined]
                ps.frozen = False
                ps.shop = self._draw_shop(ps)
                self.hero_handler.on_refresh(ps)
                # Ysera: inject a Dragon into the shop
                if getattr(ps, "_ysera_dragon_due", False):
                    ps._ysera_dragon_due = False  # type: ignore[attr-defined]
                    extras = self.tavern_pool.draw(ps.tavern_tier, 1)
                    for card in extras:
                        ps.shop.append(self._dict_to_minion(card))

        elif type_action == 4:
            # freeze — semantically "I'm done shopping; save this shop for next turn".
            # Immediately ends the turn so the agent can't freeze then keep buying.
            # Applies the same end-of-turn effects as END_TURN.
            ps.frozen = True
            reward += self._end_of_turn_reward(ps)
            self.hero_handler.on_end_turn(ps)
            self.trinket_handler.apply_on_round_end(ps)
            done = True

        elif type_action == 5:
            # level_up (Millhouse adds 1 to cost)
            millhouse = getattr(ps, "_millhouse", False)
            effective_level_cost = ps.level_cost + (1 if millhouse else 0)
            if ps.tavern_tier < 6 and ps.gold >= effective_level_cost:
                ps.gold = max(0, ps.gold - effective_level_cost)
                ps.tavern_tier = min(6, ps.tavern_tier + 1)
                ps.level_cost = max(0, self._level_cost_for_tier(ps.tavern_tier) - 1)
                ps.frozen = False
                ps.shop = self._draw_shop(ps)
                self.hero_handler.on_tavern_upgrade(ps)

        elif type_action == 6:
            # hero_power: mark as used unconditionally so passive/unsupported heroes
            # can't be spammed — the mask won't offer it again this turn.
            ps.hero_power_used = True
            hdef = HERO_DEF_MAP.get(ps.hero_card_id, {})
            ptype = hdef.get("power_type", "null")
            cost  = ps.hero_power_cost
            if (
                ptype == "active_noptr"
                and ps.gold >= cost
                and (ps.hero_power_charges == -1 or ps.hero_power_charges > 0)
            ):
                ps.gold -= cost
                if ps.hero_power_charges > 0:
                    ps.hero_power_charges -= 1
                self.hero_handler.activate_no_pointer(ps, self.tavern_pool)

        elif type_action == 7:
            reward += self._end_of_turn_reward(ps)
            self.hero_handler.on_end_turn(ps)
            self.trinket_handler.apply_on_round_end(ps)
            done = True

        elif type_action == 8:
            # activate: ptr_action is board slot index (ptr 7-13 → slot 0-6),
            # targeting the activating minion itself (same zone as sell).
            i = ptr_action - PTR_BOARD_OFF
            if 0 <= i < len(ps.board) and ps.board[i] is not None:
                minion = ps.board[i]
                cost = minion.activate_cost
                if cost > 0 and not minion.activated_this_turn and ps.gold >= cost:
                    ps.gold -= cost
                    minion.activated_this_turn = True
                    self.effect_handler.on_activate(ps, minion)

        # Potential shaping fires for every dispatched action type above (BUY,
        # SELL, PLACE, REROLL, FREEZE, LEVEL_UP, HERO_POWER, END_TURN, ACTIVATE)
        # via this single call site, whether or not the action's own
        # preconditions were met -- a no-op/failed action leaves Φ(s) unchanged
        # so this correctly contributes ~0 (modulo the tiny (SHAPE_GAMMA - 1)
        # discount drag), and a single call site is much easier to audit for
        # "every transition is covered" than scattering it through every
        # branch (the previous design's bug, board-shape reward that only
        # fired on the minion sub-branch of PLACE, came from exactly that
        # kind of scattering).
        reward += self._apply_potential_shaping(ps)
        return self._get_observation(player_id), reward, done

    def _apply_game_buffs(self, ps: PlayerState, minion: MinionState) -> None:
        """Apply accumulated 'this game' tribe buffs from ps.game_buffs to *minion*.

        Called after a minion is placed on the board so it receives buffs that
        were registered by earlier battlecries (e.g. Nerubian Deathswarmer).
        """
        from symbolic.effect_handler import _minion_tribes as _get_tribes
        minion_tribes = _get_tribes(minion, self.card_defs)
        for tribe_key, (atk, hp) in ps.game_buffs.items():
            if tribe_key == "ALL":
                match = True
            elif ":" in tribe_key:
                _, token_name = tribe_key.split(":", 1)
                match = token_name.lower() in minion.name.lower()
            else:
                match = tribe_key in minion_tribes
            if match:
                minion.game_atk_bonus += atk
                minion.game_hp_bonus  += hp
                minion.max_health     += hp

    def _update_multiplier_flags(self, ps: PlayerState) -> None:
        """Scan the board and set has_brann / has_titus / has_drakkari flags."""
        board_ids = {_minion_to_dict(m).get("card_id", "") for m in ps.board}
        ps.has_brann    = any("brann"   in cid.lower() or "TB_BaconUps_800" in cid
                              for cid in board_ids)
        ps.has_titus    = any("titus"   in cid.lower() or "TB_BaconUps_116" in cid
                              for cid in board_ids)
        ps.has_drakkari = any("drakkari" in cid.lower() or "TB_BaconUps_090" in cid
                              for cid in board_ids)

    def _trinket_id_to_minion_dict(self, card_id: str) -> dict:
        """Build a minimal card dict for a trinket card_id (used in shop zone encoding)."""
        cdef = self.card_defs.get(card_id, {})
        return {
            "card_id": card_id,
            "name": cdef.get("name", card_id),
            "attack": 0,
            "health": 0,
            "tier": 0,
        }

    def _cast_spell(self, ps: PlayerState, minion: MinionState) -> None:
        """Apply a spell card's effect and discard it.  Falls back to no-op for unknown spells."""
        name = minion.name.lower()
        if "blood gem" in name:
            # Give a random friendly minion +1/+1 (plus Blood Gem bonuses)
            if ps.board:
                target = self._rng.choice(ps.board)
                atk_bonus = 1 + ps.blood_gem_atk_bonus
                hp_bonus  = 1 + ps.blood_gem_hp_bonus
                target.attack     += atk_bonus
                target.health     += hp_bonus
                target.max_health += hp_bonus
        elif "blood gem barrage" in name:
            # AoE version: +1/+1 (+bonuses) to ALL friendly board minions
            atk_bonus = 1 + ps.blood_gem_atk_bonus
            hp_bonus  = 1 + ps.blood_gem_hp_bonus
            for m in ps.board:
                m.attack     += atk_bonus
                m.health     += hp_bonus
                m.max_health += hp_bonus
        elif "tavern spell" in name or "coin" in name:
            # Generic tavern spells / coin: refund 1 gold
            ps.gold = min(ps.max_gold, ps.gold + 1)
        # else: no-op for unrecognized spells

        # P2-F: Post-spell aura triggers
        for aura_m in ps.board:
            aura_key = aura_m.name.lower().replace(" ", "")
            if "timecapnhooktail" in aura_key:
                # +1 ATK to all friendlies whenever a spell is cast
                mult = 2 if aura_m.golden else 1
                for m in ps.board:
                    m.perm_atk_bonus += 1 * mult
            elif "plankwalker" in aura_key:
                # +2/+1 to 3 random friendlies per spell cast
                mult = 2 if aura_m.golden else 1
                others = [m for m in ps.board if m is not aura_m]
                if others:
                    for _ in range(3 * mult):
                        target = self._rng.choice(others)
                        target.perm_atk_bonus += 2
                        target.perm_hp_bonus  += 1
                        target.max_health     += 1

    # ------------------------------------------------------------------
    # Combat phase
    # ------------------------------------------------------------------

    def step_combat(
        self,
        player_id: int,
        opponent_id: int,
    ) -> dict:
        """Simulate combat between two players via FirestoneClient.

        Returns a combat result dict and updates player health.
        opponent_id == -1 means a ghost matchup; the opponent's board is empty
        (player always wins, takes 0 damage).
        """
        ps = self.players[player_id]
        player_board = [_minion_to_dict(m) for m in ps.board]

        if opponent_id == -1 or opponent_id >= len(self.players):
            # Ghost matchup: automatic win, no damage
            result = {
                "result": "win",
                "damage_taken": 0,
                "damage_dealt": 0.0,
                "player_id": player_id,
                "opponent_id": opponent_id,
            }
            ps.last_result = "win"
            ps.last_damage_taken = 0
            ps.last_damage_dealt = 0
            return result

        opp = self.players[opponent_id]

        # Fire start-of-combat trinket effects (e.g. stat buffs) before sim snapshot
        self.trinket_handler.apply_on_combat_start(ps)

        opp_board = [_minion_to_dict(m) for m in opp.board]

        sim = self.firestone.simulate(
            player_board, opp_board,
            player_tier=ps.tavern_tier,
            opp_tier=opp.tavern_tier,
        )

        # Determine concrete outcome by sampling from the full probability distribution
        roll = self._rng.random()
        if roll < sim.win_prob:
            outcome = "win"
        elif roll < sim.win_prob + sim.tie_prob:
            outcome = "tie"
        else:
            outcome = "loss"

        # Damage calculation (simplified: tier + board size when player loses)
        if outcome == "loss":
            damage_taken = int(round(sim.expected_damage_taken))
            if damage_taken == 0:
                damage_taken = max(1, ps.tavern_tier + len(opp.board))
            damage_taken = max(0, damage_taken)
        else:
            damage_taken = 0

        damage_dealt = float(sim.expected_damage_dealt) if outcome == "win" else 0.0

        # Update health
        effective_hp = ps.health + ps.armor
        effective_hp = max(0, effective_hp - damage_taken)
        if effective_hp > ps.health:
            ps.armor = effective_hp - ps.health
        else:
            ps.armor = 0
            ps.health = effective_hp

        ps.last_result       = outcome
        ps.last_damage_taken = damage_taken
        ps.last_damage_dealt = int(round(damage_dealt))

        # Fire post-combat hooks: persistent DR effects (e.g. Anubarak)
        # Pass all board card_ids — the handler decides which ones apply.
        dead_card_ids = [m.card_id for m in ps.board]
        self.effect_handler.on_after_combat(ps, dead_card_ids)
        self.trinket_handler.apply_on_combat_end(ps, outcome)

        # P3-A: Rafaam post-combat steal — copy random minion from opponent's board
        if getattr(ps, "_rafaam_active", False):
            ps._rafaam_active = False  # type: ignore[attr-defined]
            if outcome == "win" and opp.board and len(ps.hand) < 10:
                import copy as _copy
                stolen = _copy.copy(self._rng.choice(opp.board))
                stolen.perm_atk_bonus = 0
                stolen.perm_hp_bonus  = 0
                stolen.game_atk_bonus = 0
                stolen.game_hp_bonus  = 0
                ps.hand.append(stolen)

        # P3-A: Tess post-combat draw — add a random card from the pool to next shop
        if getattr(ps, "_tess_active", False):
            ps._tess_active = False  # type: ignore[attr-defined]
            if self.tavern_pool is not None:
                drawn = self.tavern_pool.draw(ps.tavern_tier, 1)
                for card in drawn:
                    ps.shop.append(self._dict_to_minion(card))

        # Update per-opponent snapshot with everything we now know about them
        dom_tribe, dom_count = _board_dominant_tribe(opp.board)
        prev_snap = ps.opponent_snapshots.get(opponent_id)
        prev_health = prev_snap.health if prev_snap is not None else opp.health
        ps.opponent_snapshots[opponent_id] = OpponentSnapshot(
            board=list(opp.board),
            tavern_tier=opp.tavern_tier,
            health=opp.health,
            prev_health=prev_health,
            armor=opp.armor,
            board_size=len(opp.board),
            dominant_tribe=dom_tribe,
            dominant_tribe_count=dom_count,
            is_synergistic=dom_count >= 4,
            last_seen_round=self.round_num,
        )

        # Unfreeze at end of combat (unless player froze shop)
        if not ps.frozen:
            pass  # shop was already not frozen

        return {
            "result":       outcome,
            "damage_taken": damage_taken,
            "damage_dealt": damage_dealt,
            "win_prob":     sim.win_prob,
            "player_id":    player_id,
            "opponent_id":  opponent_id,
        }

    # ------------------------------------------------------------------
    # Elimination handling
    # ------------------------------------------------------------------

    def _eliminate_players(self, round_num: int) -> List[int]:
        """Kill players at 0 HP and assign placements in reverse kill order.

        Returns list of newly eliminated player_ids.
        """
        newly_dead = []
        for ps in self.players:
            if ps.alive and ps.health <= 0:
                ps.alive = False
                ps.placement = self._placement_counter
                self._placement_counter -= 1
                newly_dead.append(ps.player_id)
                logger.debug(
                    "Player %d eliminated at round %d (placement %d)",
                    ps.player_id, round_num, ps.placement,
                )
        return newly_dead

    # ------------------------------------------------------------------
    # Main game loop
    # ------------------------------------------------------------------

    def run_game(self, agents: Optional[List[Any]] = None) -> GameResult:
        """Run a complete Battlegrounds game to completion.

        Parameters
        ----------
        agents:
            Override the agents list for this game.  None to use
            self.agents.  Each agent should implement ``get_action(obs)``
            returning a ``(type_idx, ptr_idx)`` tuple.

        Returns
        -------
        GameResult with placements, rewards, and per-round history.
        """
        active_agents = agents if agents is not None else self.agents
        initial_obs = self.reset()
        round_history: List[dict] = []
        cumulative_rewards: Dict[int, float] = {i: 0.0 for i in range(self.n_players)}

        for round_num in range(1, self.max_rounds + 1):
            self.round_num = round_num
            alive_players = [p for p in self.players if p.alive]
            if len(alive_players) <= 1:
                break

            round_summary: dict = {"round": round_num, "combats": [], "eliminations": []}

            # ---- Announce pairings BEFORE shopping so each player knows
            #      who they will face this round (mirrors real BG UI).
            pairings = self.matchmaker.pair_players(self.players, round_num)
            for pid_a, pid_b in pairings:
                if pid_a < len(self.players):
                    self.players[pid_a].next_opponent_id = pid_b if pid_b != -1 else None
                if pid_b != -1 and pid_b < len(self.players):
                    self.players[pid_b].next_opponent_id = pid_a

            # ---- Shopping phase ----------------------------------------
            # end_turn_buffers: pid → tuple buffered for post-combat flush
            # (sequential) 4-tuple: (obs, type, ptr, step_reward)
            # (batched)    8-tuple: (obs, type, ptr, step_reward, log_p, val,
            #                        type_mask_np, ptr_mask_np)
            end_turn_buffers: dict = {}

            # Phase 1 — setup all alive players (fast, no inference)
            initial_obs: dict = {}
            round_agents: dict = {}
            for ps in alive_players:
                ps.round_num = round_num
                ps.gold      = self._gold_for_round(round_num)
                ps.level_cost = max(0, ps.level_cost - 1)
                ps.hero_power_used = False
                ps._rerolls_this_turn = 0  # type: ignore[attr-defined]
                for m in ps.board:
                    m.activated_this_turn = False
                # NOTE: ps.phi is intentionally NOT reset here. The old code
                # re-baselined phi_board/phi_tier every round, discarding any
                # potential drop since the last evaluation (e.g. falling behind
                # the tavern-tier curve as round_num just advanced above) --
                # exactly the one-sided ratchet this whole rework removes. The
                # drift since last round's post-combat evaluation is picked up
                # automatically by the next _apply_potential_shaping call (the
                # first shopping action this round, or the forced END_TURN if
                # none is taken) against the still-live ps.phi baseline. See
                # the module constants block and CONTEXT.md (2026-08-31).
                ps.shop = self._draw_shop(ps)
                ps.frozen = False
                self.hero_handler.on_start_of_round(ps)
                self.hero_handler.on_refresh(ps)
                self.trinket_handler.maybe_offer(ps, round_num)
                self.trinket_handler.apply_on_round_start(ps)
                if getattr(ps, "_ysera_dragon_due", False):
                    ps._ysera_dragon_due = False  # type: ignore[attr-defined]
                    extras = self.tavern_pool.draw(ps.tavern_tier, 1)
                    for card in extras:
                        ps.shop.append(self._dict_to_minion(card))
                initial_obs[ps.player_id]  = self._get_observation(ps.player_id)
                round_agents[ps.player_id] = (
                    active_agents[ps.player_id]
                    if ps.player_id < len(active_agents) else None
                )

            # Phase 2 — action loop (batched or sequential)
            if self.batched and self._agents_support_batching(alive_players, round_agents):
                self._run_shopping_phase_batched(
                    alive_players, round_agents, initial_obs,
                    end_turn_buffers, cumulative_rewards,
                )
            else:
                for ps in alive_players:
                    obs   = initial_obs[ps.player_id]
                    agent = round_agents[ps.player_id]
                    max_actions = 30
                    for _ in range(max_actions):
                        prev_obs = obs
                        type_action, ptr_action = self._get_agent_action(agent, obs, ps)
                        obs, step_reward, done = self.step_shopping(
                            ps.player_id, type_action, ptr_action
                        )
                        cumulative_rewards[ps.player_id] += step_reward
                        if done:
                            end_turn_buffers[ps.player_id] = (
                                prev_obs, type_action, ptr_action, step_reward
                            )
                            break
                        if hasattr(agent, "record_transition"):
                            agent.record_transition(
                                prev_obs, type_action, ptr_action,
                                reward=step_reward, done=False,
                            )

            # ---- Combat phase (uses same pairings already announced) ----
            # Snapshot ranks BEFORE combat so the delta includes any kills.
            pre_ranks = {ps.player_id: ps.get_rank(self.players)
                         for ps in self.players if ps.alive}

            combat_results: List[Tuple[int, int, dict, dict]] = []

            for (pid_a, pid_b) in pairings:
                if not self.players[pid_a].alive:
                    continue
                # Ghost matchup
                if pid_b == -1:
                    result = self.step_combat(pid_a, -1)
                    round_summary["combats"].append(result)
                    combat_results.append((pid_a, -1, result, {}))
                    continue

                if not self.players[pid_b].alive:
                    continue

                result_a = self.step_combat(pid_a, pid_b)
                result_b = self.step_combat(pid_b, pid_a)
                round_summary["combats"].append(result_a)
                combat_results.append((pid_a, pid_b, result_a, result_b))

            # ---- Elimination check (before rewards so rank delta includes kills)
            new_dead = self._eliminate_players(round_num)
            round_summary["eliminations"] = new_dead

            # ---- Round rewards + transition flush -------------------------
            for (pid_a, pid_b, result_a, result_b) in combat_results:
                pairs = [(pid_a, result_a)]
                if pid_b != -1:
                    pairs.append((pid_b, result_b))
                for pid, result_info in pairs:
                    ps = self.players[pid]
                    cur_rank = ps.get_rank(self.players)
                    r = compute_round_reward(
                        damage_taken=result_info["damage_taken"],
                        damage_dealt=result_info["damage_dealt"],
                        prev_rank=pre_ranks.get(pid, cur_rank),
                        cur_rank=cur_rank,
                        result=result_info["result"],
                        max_health=ps.max_health,
                    )
                    # Potential shaping, evaluated right after combat -- the
                    # charge the old per-round reset silently absorbed. In this
                    # simulator combat doesn't itself mutate ps.board (win/loss
                    # is resolved as an aggregate win-prob/damage estimate, not
                    # by tracking which individual minions died), so this call
                    # is usually a near-noop today; it exists so (a) any future
                    # change that does let combat weaken the board is charged
                    # automatically with no further wiring, and (b) it closes
                    # the telescoping sum for this round for every player who
                    # fought, including ones eliminated this round (their Φ(s_T)
                    # is fixed here, at the moment their episode ends).
                    r += self._apply_potential_shaping(ps)
                    # Fire placement reward immediately on elimination so the
                    # agent doesn't have to wait until game end for this signal.
                    if pid in new_dead:
                        r += FINAL_PLACEMENT_REWARD.get(ps.placement, -4.0)
                    cumulative_rewards[pid] += r

                    # Flush the buffered end-turn transition with the combined
                    # reward (gold penalty + round reward).  done=True when the
                    # player was just eliminated; False when they survive.
                    buf = end_turn_buffers.pop(pid, None)
                    agent = active_agents[pid] if pid < len(active_agents) else None
                    if buf is not None:
                        if len(buf) == 8:
                            # Batched path: pre-computed log_prob and value stored
                            et_obs, et_type, et_ptr, et_step_reward, \
                                et_log_p, et_val, et_t_mask, et_p_mask = buf
                            if hasattr(agent, "record_transition_precomputed"):
                                agent.record_transition_precomputed(
                                    et_obs, et_type, et_ptr,
                                    reward=et_step_reward + r,
                                    done=not ps.alive,
                                    log_prob=et_log_p, value=et_val,
                                    type_mask=et_t_mask, ptr_mask=et_p_mask,
                                )
                            elif hasattr(agent, "record_transition"):
                                agent.record_transition(
                                    et_obs, et_type, et_ptr,
                                    reward=et_step_reward + r,
                                    done=not ps.alive,
                                )
                        elif hasattr(agent, "record_transition"):
                            # Sequential path: 4-tuple
                            et_obs, et_type, et_ptr, et_step_reward = buf
                            agent.record_transition(
                                et_obs, et_type, et_ptr,
                                reward=et_step_reward + r,
                                done=not ps.alive,
                            )
            round_history.append(round_summary)

            alive_after = [p for p in self.players if p.alive]
            if len(alive_after) <= 1:
                break

        # Assign remaining placements to survivors
        survivors = sorted(
            [p for p in self.players if p.alive],
            key=lambda p: p.total_health,
            reverse=True,
        )
        place = 1
        for ps in survivors:
            ps.placement = place
            place += 1

        # Final placement rewards + terminal transitions
        # Eliminated players already received their placement reward at the
        # moment of elimination; only survivors get it here.
        placements: Dict[int, int] = {}
        final_rewards: Dict[int, float] = {}
        for ps in self.players:
            placement = ps.placement if ps.placement is not None else self.n_players
            placements[ps.player_id] = placement
            final_r = FINAL_PLACEMENT_REWARD.get(placement, -4.0) if ps.alive else 0.0
            final_rewards[ps.player_id] = cumulative_rewards[ps.player_id] + final_r

            # Terminal transition: delivers the placement reward as a done=True
            # step so the PPO value target bootstraps to 0 at game end.
            # Uses the last observation seen by this player and end_turn as the
            # action (a no-op carrier for the reward signal).
            agent = active_agents[ps.player_id] if ps.player_id < len(active_agents) else None
            if hasattr(agent, "record_transition"):
                last_obs = self._get_observation(ps.player_id)
                agent.record_transition(
                    last_obs, 7, -1,   # type=end_turn, no pointer
                    reward=final_r,
                    done=True,
                )

        return GameResult(
            placements=placements,
            final_rewards=final_rewards,
            round_history=round_history,
            n_rounds=self.round_num,
        )

    # ------------------------------------------------------------------
    # Action selection helper
    # ------------------------------------------------------------------

    def _get_agent_action(
        self,
        agent: Any,
        obs: dict,
        ps: PlayerState,
    ) -> Tuple[int, int]:
        """Get a (type_action, ptr_action) from the agent or fall back to random.

        Returns
        -------
        (type_idx, ptr_idx) where ptr_idx is -1 for non-pointer types.
        """
        from agent.policy import build_type_mask, build_pointer_mask, TYPES_WITH_POINTER

        type_mask = build_type_mask(ps)
        valid_types = type_mask.nonzero(as_tuple=True)[0].tolist()
        if not valid_types:
            return 7, -1  # end_turn as last resort

        def _random_action() -> Tuple[int, int]:
            # Weighted random: bias towards end_turn to prevent infinite loops
            weights = [3.0 if t == 7 else 1.0 for t in valid_types]
            total_w = sum(weights)
            r = self._rng.random() * total_w
            cumulative = 0.0
            chosen_type = valid_types[-1]
            for t, w in zip(valid_types, weights):
                cumulative += w
                if r < cumulative:
                    chosen_type = t
                    break
            if chosen_type in TYPES_WITH_POINTER:
                ptr_mask = build_pointer_mask(ps, chosen_type)
                valid_ptrs = ptr_mask.nonzero(as_tuple=True)[0].tolist()
                ptr = self._rng.choice(valid_ptrs) if valid_ptrs else -1
            else:
                ptr = -1
            return chosen_type, ptr

        if agent is None:
            return _random_action()

        try:
            result = agent.get_action(obs)
            if isinstance(result, (list, tuple)) and len(result) == 2:
                type_action, ptr_action = int(result[0]), int(result[1])
                if type_mask[type_action]:
                    return type_action, ptr_action
        except Exception as exc:
            logger.debug("Agent get_action failed: %s", exc)

        return _random_action()

    # ------------------------------------------------------------------
    # Batched shopping helpers
    # ------------------------------------------------------------------

    def _agents_support_batching(self, alive_players, round_agents) -> bool:
        """Return True if all agents expose get_action_batch via their policy
        and none have opted out via supports_batching = False."""
        for ps in alive_players:
            agent = round_agents.get(ps.player_id)
            if agent is None:
                return False
            if not getattr(agent, "supports_batching", True):
                return False
            policy = getattr(agent, "policy", None)
            if policy is None or not hasattr(policy, "get_action_batch"):
                return False
        return True

    def _run_shopping_phase_batched(
        self,
        alive_players,
        round_agents,
        initial_obs: dict,
        end_turn_buffers: dict,
        cumulative_rewards: dict,
    ) -> None:
        """Run the buy-phase for all alive players with batched inference.

        At each step, collects observations for all still-active players,
        runs a single forward pass for the whole batch, then applies each
        player's action and removes players that issued END_TURN.
        """
        import torch as _torch
        import numpy as _np
        from agent.policy import build_type_mask_batch, build_pointer_mask_batch, build_pointer_mask

        first_agent = round_agents[alive_players[0].player_id]
        dev = next(first_agent.policy.parameters()).device

        current_obs = dict(initial_obs)
        active = list(alive_players)
        max_actions = 30

        for _ in range(max_actions):
            if not active:
                break

            obs_list      = [current_obs[ps.player_id] for ps in active]
            player_states = [o["player_state"] for o in obs_list]

            # Group active players by policy object so each distinct policy
            # gets one batched forward pass.  All-same-policy (common case)
            # → one pass, identical to the old behaviour.  Current + historical
            # snapshot → two passes, one per group.
            _groups: dict = {}  # id(policy) → (policy_obj, [local_indices])
            for _gi, _ps in enumerate(active):
                _pol = round_agents[_ps.player_id].policy
                _key = id(_pol)
                if _key not in _groups:
                    _groups[_key] = (_pol, [])
                _groups[_key][1].append(_gi)

            _n = len(active)
            _type_buf = [None] * _n
            _ptr_buf  = [None] * _n
            _lp_buf   = [None] * _n
            _val_buf  = [None] * _n

            for _pol, _g_idxs in _groups.values():
                _g_obs    = [obs_list[i] for i in _g_idxs]
                _g_states = [player_states[i] for i in _g_idxs]
                _board_g  = _torch.tensor(
                    _np.stack([o["board_tokens"]   for o in _g_obs]), dtype=_torch.float32, device=dev)
                _shop_g   = _torch.tensor(
                    _np.stack([o["shop_tokens"]    for o in _g_obs]), dtype=_torch.float32, device=dev)
                _hand_g   = _torch.tensor(
                    _np.stack([o["hand_tokens"]    for o in _g_obs]), dtype=_torch.float32, device=dev)
                _scalar_g = _torch.tensor(
                    _np.stack([o["scalar_context"] for o in _g_obs]), dtype=_torch.float32, device=dev)
                _opp_g    = _torch.tensor(
                    _np.stack([o.get("opp_tokens", _np.zeros((7, 44), dtype=_np.float32))
                               for o in _g_obs]), dtype=_torch.float32, device=dev)
                _t_mask_g   = build_type_mask_batch(_g_states).to(dev)
                _occ_mask_g = _torch.stack(
                    [build_pointer_mask(_s, -1) for _s in _g_states]).to(dev)
                _ta, _pa, _lp, _vl = _pol.get_action_batch(
                    _board_g, _shop_g, _hand_g, _scalar_g,
                    type_mask=_t_mask_g, pointer_mask=_occ_mask_g, opp_tokens=_opp_g,
                )
                for _j, _i in enumerate(_g_idxs):
                    _type_buf[_i] = _ta[_j]
                    _ptr_buf[_i]  = _pa[_j]
                    _lp_buf[_i]   = _lp[_j]
                    _val_buf[_i]  = _vl[_j]

            type_acts = _torch.stack(_type_buf)
            ptr_acts  = _torch.stack(_ptr_buf)
            log_probs = _torch.stack(_lp_buf)
            values    = _torch.stack(_val_buf)
            t_mask    = build_type_mask_batch(player_states).to(dev)
            ptr_masks = build_pointer_mask_batch(player_states, type_acts)

            next_active = []
            for i, ps in enumerate(active):
                agent     = round_agents[ps.player_id]
                prev_obs  = obs_list[i]
                t_a       = int(type_acts[i].item())
                p_a       = int(ptr_acts[i].item())
                log_p     = float(log_probs[i].item())
                val       = float(values[i].item())
                t_mask_np = t_mask[i].cpu().numpy()
                p_mask_np = ptr_masks[i].cpu().numpy()

                next_obs, step_reward, is_done = self.step_shopping(ps.player_id, t_a, p_a)
                cumulative_rewards[ps.player_id] += step_reward

                if is_done:
                    end_turn_buffers[ps.player_id] = (
                        prev_obs, t_a, p_a, step_reward,
                        log_p, val, t_mask_np, p_mask_np,
                    )
                else:
                    if hasattr(agent, "record_transition_precomputed"):
                        agent.record_transition_precomputed(
                            prev_obs, t_a, p_a, step_reward, False,
                            log_p, val, t_mask_np, p_mask_np,
                        )
                    elif hasattr(agent, "record_transition"):
                        agent.record_transition(
                            prev_obs, t_a, p_a, reward=step_reward, done=False,
                        )
                    current_obs[ps.player_id] = next_obs
                    next_active.append(ps)

            active = next_active

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _get_observation(self, player_id: int) -> dict:
        """Build an observation dict for the policy network.

        Returns raw numpy arrays ready for tensor conversion, plus a
        scalar_context vector from SymbolicBoardComputer.
        """
        ps = self.players[player_id]

        board_dicts = [_minion_to_dict(m) for m in ps.board]
        features = self.board_computer.compute(
            board_dicts,
            gold=ps.gold,
            round_num=ps.round_num,
            tavern_tier=ps.tavern_tier,
        )

        dominant_tribe_count = (
            features.tribe_counts.get(features.dominant_tribe, 0)
            if features.dominant_tribe else 0
        )

        ctx = dict(
            board_size=features.board_size,
            dominant_tribe_count=dominant_tribe_count,
            total_aura_dependency=features.total_aura_dependency,
            round_num=ps.round_num,
            tavern_tier=ps.tavern_tier,
        )

        board_tokens = _encode_zone(ps.board, self.encoder, 7,  **ctx)
        # During trinket offer / discover, replace shop zone with the choice options
        if ps.trinket_offer_pending:
            offered = self.trinket_handler.get_pending_offer(ps.player_id)
            shop_source = [self._trinket_id_to_minion_dict(cid) for cid in offered]
        elif ps.discover_pending:
            shop_source = ps.discover_pending
        else:
            shop_source = ps.shop
        shop_tokens  = _encode_zone(shop_source, self.encoder, 7,  **ctx)
        hand_tokens  = _encode_zone(ps.hand,  self.encoder, 10, **ctx)

        # Look up the announced next opponent's snapshot (None on round 1)
        opp_snap: Optional[OpponentSnapshot] = None
        if ps.next_opponent_id is not None:
            opp_snap = ps.opponent_snapshots.get(ps.next_opponent_id)

        opp_board  = opp_snap.board if opp_snap is not None else []
        opp_tokens = _encode_zone(opp_board, self.encoder, 7, **ctx)

        # Own board scalar (24 dims)
        # + all-opponent scalar (7 × 8 = 56 dims, sorted by player_id; own slot zeroed)
        #     each 8-dim block: tier/7, health/40, armor/10, board_size/7,
        #                       dominant_tribe_count/7, is_synergistic,
        #                       rounds_since_seen/10, health_delta/40
        # + lobby scalar (6 dims: num_alive, mean_tier, mean_health,
        #                         num_synergistic, health_rank, tier_rank)
        # = 86 dims total
        own_scalar = features.to_scalar_vector()  # [24]

        # All-opponent block: one 8-dim slot per player_id 0..n_players-1
        all_opp_scalar = np.zeros(self.n_players * 8, dtype=np.float32)
        for opp_pid in range(self.n_players):
            if opp_pid == player_id:
                continue  # own slot stays zero
            snap = ps.opponent_snapshots.get(opp_pid)
            if snap is None:
                continue
            base = opp_pid * 8
            all_opp_scalar[base:base + 8] = [
                snap.tavern_tier / 7.0,
                snap.health / 40.0,
                snap.armor / 10.0,
                snap.board_size / 7.0,
                snap.dominant_tribe_count / 7.0,
                float(snap.is_synergistic),
                (ps.round_num - snap.last_seen_round) / 10.0,
                (snap.health - snap.prev_health) / 40.0,
            ]

        # Lobby-wide summary over all known opponent snapshots
        all_players = self.players
        alive_players = [p for p in all_players if p.alive]
        n_alive = len(alive_players)

        all_snaps = list(ps.opponent_snapshots.values())
        if all_snaps:
            mean_opp_tier   = sum(s.tavern_tier for s in all_snaps) / len(all_snaps)
            mean_opp_health = sum(s.health for s in all_snaps) / len(all_snaps)
            num_synergistic = sum(1 for s in all_snaps if s.is_synergistic)
        else:
            mean_opp_tier   = ps.tavern_tier
            mean_opp_health = 40.0
            num_synergistic = 0

        # Rank among alive players by total health and tavern tier (1 = best)
        alive_sorted_health = sorted(alive_players,
                                     key=lambda p: p.total_health, reverse=True)
        alive_sorted_tier   = sorted(alive_players,
                                     key=lambda p: p.tavern_tier, reverse=True)
        health_rank = next((i + 1 for i, p in enumerate(alive_sorted_health)
                            if p.player_id == player_id), n_alive)
        tier_rank   = next((i + 1 for i, p in enumerate(alive_sorted_tier)
                            if p.player_id == player_id), n_alive)

        lobby_scalar = np.array([
            n_alive / 8.0,
            mean_opp_tier / 7.0,
            mean_opp_health / 40.0,
            num_synergistic / 7.0,
            health_rank / 8.0,
            tier_rank / 8.0,
        ], dtype=np.float32)

        # Economy features the policy needs but can't infer from card tokens (6 dims)
        economy_scalar = np.array([
            ps.gold / 10.0,                               # current gold (0-10)
            float(ps.frozen),                              # froze this turn
            ps.level_cost / 10.0,                         # gold needed to level
            float(ps.hero_power_used),                    # hero power already spent
            len(ps.equipped_trinkets) / 2.0,              # 0 / 0.5 / 1.0 trinkets equipped
            float(ps.trinket_offer_pending),              # trinket pick screen active
        ], dtype=np.float32)

        scalar_ctx = np.concatenate([own_scalar, all_opp_scalar, lobby_scalar, economy_scalar])  # [100]

        return {
            "board_tokens":   board_tokens,   # [7, 44]
            "shop_tokens":    shop_tokens,    # [7, 44]
            "hand_tokens":    hand_tokens,    # [10, 44]
            "opp_tokens":     opp_tokens,     # [7, 44]  next opponent's last board
            "scalar_context": scalar_ctx,     # [98]
            "player_id":      player_id,
            "player_state":   ps,
        }
