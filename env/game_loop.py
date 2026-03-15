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
from symbolic.board_computer import SymbolicBoardComputer
from symbolic.firestone_client import FirestoneClient
from agent.card_encoder import CardEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reward constants (CLAUDE.md)
# ---------------------------------------------------------------------------

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

    Components
    ----------
    Combat outcome : +0.5 win / -0.3 loss
    Damage taken   : -0.05 * (damage / max_health)  — penalise health loss
    Damage dealt   : +0.05 * (damage / max_health)  — reward hurting opponents
    Rank delta     : (prev_rank - cur_rank) * 0.15  — positive when rank improves;
                     fires both on combat health changes AND opponent eliminations
    Survival bonus : +0.1 flat for being alive this round
    """
    r  =  0.5  if result == "win"  else 0.0
    r += -0.3  if result == "loss" else 0.0
    r += -0.05 * (damage_taken / max_health)
    r +=  0.05 * (damage_dealt  / max_health)
    r += (prev_rank - cur_rank) * 0.15
    r +=  0.1   # survival bonus
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
    ) -> None:
        self.card_defs       = card_defs
        self.agents          = agents or [None] * n_players
        self.board_computer  = board_computer
        self.firestone       = firestone_client
        self.matchmaker      = matchmaker
        self.tavern_pool     = tavern_pool
        self.n_players       = n_players
        self.max_rounds      = max_rounds
        self._rng            = random.Random(seed)
        self.encoder         = CardEncoder(card_defs)

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
        self._placement_counter = self.n_players + 1  # placements count down
        self._accumulated_rewards = {i: 0.0 for i in range(self.n_players)}

        self.players = []
        for pid in range(self.n_players):
            ps = PlayerState(
                player_id=pid,
                health=40,
                armor=0,
                max_health=40,
                gold=self._gold_for_round(1),
                max_gold=10,
                tavern_tier=1,
                level_cost=self._level_cost_for_tier(1),
                frozen=False,
                round_num=1,
                alive=True,
            )
            # Draw initial shop
            ps.shop = self._draw_shop(ps)
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
        return MinionState(
            card_id=d.get("card_id", d.get("id", "")),
            name=d.get("name", ""),
            attack=d.get("attack", 0),
            health=d.get("health", 0),
            max_health=d.get("health", 0),
            tier=d.get("tier", 1),
        )

    # ------------------------------------------------------------------
    # Shopping phase
    # ------------------------------------------------------------------

    def step_shopping(
        self,
        player_id: int,
        action: int,
    ) -> Tuple[dict, float, bool]:
        """Execute one buy-phase action for a player.

        Parameters
        ----------
        player_id:
            Index into self.players.
        action:
            Integer action index (see ACTION_NAMES in policy.py).

        Returns
        -------
        (next_obs, reward, done_with_shopping)
        """
        ps = self.players[player_id]
        reward = 0.0
        done = False

        if action <= 6:
            # buy_i: buy shop slot i
            i = action
            if i < len(ps.shop) and ps.shop[i] is not None and ps.gold >= 3:
                minion = ps.shop.pop(i)
                ps.hand.append(minion)
                ps.gold = max(0, ps.gold - 3)

        elif action <= 13:
            # sell_i: sell board slot i
            i = action - 7
            if i < len(ps.board) and ps.board[i] is not None:
                ps.board.pop(i)
                ps.gold = min(ps.max_gold, ps.gold + 1)

        elif 14 <= action <= 83:
            # play_h{h}_p{p}: play hand[h], insert at board position p
            offset = action - 14
            h = offset // 7
            p = offset % 7
            if (h < len(ps.hand) and ps.hand[h] is not None
                    and len(ps.board) < 7 and p <= len(ps.board)):
                minion = ps.hand.pop(h)
                ps.board.insert(p, minion)
                self._update_multiplier_flags(ps)

        elif action == 84:
            # level_up
            if ps.tavern_tier < 6 and ps.gold >= ps.level_cost:
                ps.gold = max(0, ps.gold - ps.level_cost)
                ps.tavern_tier = min(6, ps.tavern_tier + 1)
                ps.level_cost = max(0, self._level_cost_for_tier(ps.tavern_tier) - 1)
                ps.frozen = False
                ps.shop = self._draw_shop(ps)

        elif action == 85:
            # freeze
            ps.frozen = True

        elif action == 86:
            # refresh
            if ps.gold >= 1:
                ps.gold -= 1
                ps.frozen = False
                ps.shop = self._draw_shop(ps)

        elif action == 87:
            # hero_power: no-op placeholder
            pass

        elif action == 88:
            # end_turn
            done = True

        elif 89 <= action <= 94:
            # swap_ij: swap adjacent board positions i ↔ i+1
            i = action - 89
            if i + 1 < len(ps.board) and ps.board[i] is not None and ps.board[i + 1] is not None:
                ps.board[i], ps.board[i + 1] = ps.board[i + 1], ps.board[i]

        return self._get_observation(player_id), reward, done

    def _update_multiplier_flags(self, ps: PlayerState) -> None:
        """Scan the board and set has_brann / has_titus / has_drakkari flags."""
        board_ids = {_minion_to_dict(m).get("card_id", "") for m in ps.board}
        ps.has_brann    = any("brann"   in cid.lower() or "TB_BaconUps_800" in cid
                              for cid in board_ids)
        ps.has_titus    = any("titus"   in cid.lower() or "TB_BaconUps_116" in cid
                              for cid in board_ids)
        ps.has_drakkari = any("drakkari" in cid.lower() or "TB_BaconUps_090" in cid
                              for cid in board_ids)

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
        opp_board = [_minion_to_dict(m) for m in opp.board]

        sim = self.firestone.simulate(
            player_board, opp_board,
            player_tier=ps.tavern_tier,
            opp_tier=opp.tavern_tier,
        )

        # Determine concrete outcome by sampling from win probability
        roll = self._rng.random()
        if roll < sim.win_prob:
            outcome = "win"
        elif roll < sim.win_prob + (1.0 - sim.win_prob) / 2.0:
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
            returning an integer action index.

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
            for ps in alive_players:
                ps.round_num = round_num
                ps.gold      = self._gold_for_round(round_num)
                # Decrease level_cost by 1 each turn (BG mechanic), floor at 0
                ps.level_cost = max(0, ps.level_cost - 1)
                # Redraw shop (respects frozen flag)
                ps.shop = self._draw_shop(ps)
                ps.frozen = False  # reset freeze flag after draw

                obs = self._get_observation(ps.player_id)
                agent = active_agents[ps.player_id] if ps.player_id < len(active_agents) else None

                # Shopping action loop
                max_actions = 30  # safety cap to prevent infinite loops
                for _ in range(max_actions):
                    action = self._get_agent_action(agent, obs, ps)
                    obs, step_reward, done = self.step_shopping(ps.player_id, action)
                    cumulative_rewards[ps.player_id] += step_reward
                    if done:
                        break

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

            # ---- Round rewards (computed post-elimination) ----------------
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
                    cumulative_rewards[pid] += r
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

        # Final placement rewards
        placements: Dict[int, int] = {}
        final_rewards: Dict[int, float] = {}
        for ps in self.players:
            placement = ps.placement if ps.placement is not None else self.n_players
            placements[ps.player_id] = placement
            final_r = FINAL_PLACEMENT_REWARD.get(placement, -4.0)
            final_rewards[ps.player_id] = cumulative_rewards[ps.player_id] + final_r

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
    ) -> int:
        """Get an action from the agent or fall back to a simple random policy."""
        from agent.policy import build_action_mask

        mask = build_action_mask(ps)
        valid = mask.nonzero(as_tuple=True)[0].tolist()
        if not valid:
            return 19  # end_turn as last resort

        if agent is None:
            # Random agent: uniform over valid actions, with END_TURN bias
            # to avoid infinite loops
            weights = []
            for v in valid:
                weights.append(3.0 if v == 19 else 1.0)
            total_w = sum(weights)
            r = self._rng.random() * total_w
            cumulative = 0.0
            for v, w in zip(valid, weights):
                cumulative += w
                if r < cumulative:
                    return v
            return 19

        try:
            action = agent.get_action(obs)
            if isinstance(action, (list, tuple)):
                action = action[0]
            action = int(action)
            if mask[action]:
                return action
        except Exception as exc:
            logger.debug("Agent get_action failed: %s", exc)

        # Fall back to random valid action
        return self._rng.choice(valid)

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
        shop_tokens  = _encode_zone(ps.shop,  self.encoder, 7,  **ctx)
        hand_tokens  = _encode_zone(ps.hand,  self.encoder, 10, **ctx)

        # Look up the announced next opponent's snapshot (None on round 1)
        opp_snap: Optional[OpponentSnapshot] = None
        if ps.next_opponent_id is not None:
            opp_snap = ps.opponent_snapshots.get(ps.next_opponent_id)

        opp_board  = opp_snap.board if opp_snap is not None else []
        opp_tokens = _encode_zone(opp_board, self.encoder, 7, **ctx)

        # Own board scalar (24 dims)
        # + opponent scalar (8 dims: tier, health, armor, board_size,
        #                    dominant_tribe_count, is_synergistic,
        #                    rounds_since_seen, health_delta)
        # + lobby scalar (6 dims: num_alive, mean_tier, mean_health,
        #                         num_synergistic, health_rank, tier_rank)
        # = 38 dims total
        own_scalar = features.to_scalar_vector()  # [24]

        if opp_snap is not None:
            rounds_since_seen = (ps.round_num - opp_snap.last_seen_round) / 10.0
            health_delta = (opp_snap.health - opp_snap.prev_health) / 40.0
            opp_scalar = np.array([
                opp_snap.tavern_tier / 7.0,           # 24
                opp_snap.health / 40.0,               # 25
                opp_snap.armor / 10.0,                # 26
                opp_snap.board_size / 7.0,            # 27
                opp_snap.dominant_tribe_count / 7.0,  # 28
                float(opp_snap.is_synergistic),       # 29
                rounds_since_seen,                    # 30
                health_delta,                         # 31
            ], dtype=np.float32)
        else:
            opp_scalar = np.zeros(8, dtype=np.float32)

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
            n_alive / 8.0,                        # 32
            mean_opp_tier / 7.0,                  # 33
            mean_opp_health / 40.0,               # 34
            num_synergistic / 7.0,                # 35
            health_rank / 8.0,                    # 36
            tier_rank / 8.0,                      # 37
        ], dtype=np.float32)

        scalar_ctx = np.concatenate([own_scalar, opp_scalar, lobby_scalar])  # [38]

        return {
            "board_tokens":   board_tokens,   # [7, 44]
            "shop_tokens":    shop_tokens,    # [7, 44]
            "hand_tokens":    hand_tokens,    # [2, 44]
            "opp_tokens":     opp_tokens,     # [7, 44]  next opponent's last board
            "scalar_context": scalar_ctx,     # [38]
            "player_id":      player_id,
            "player_state":   ps,
        }
