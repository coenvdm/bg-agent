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

from env.player_state import MinionState, OpponentSnapshot, PlayerState, minion_stats
from env.tavern_pool import TavernPool
from env.matchmaker import Matchmaker
from agent.policy import REORDER_BUDGET_PER_TURN
from symbolic.board_computer import SymbolicBoardComputer, _board_power
from symbolic.firestone_client import FirestoneClient
from symbolic.combat_sim import BGCombatSim
from symbolic.effect_handler import EffectHandler
from symbolic.hero_handler import HeroPowerHandler
from env.trinket_handler import TrinketHandler
from agent.card_encoder import CardEncoder
from agent.hero_encoder import HERO_DEF_MAP, NULL_HERO_ID

logger = logging.getLogger(__name__)


def expected_tier_for_round(round_num: int) -> int:
    """Rough 'on-curve' tavern tier for a competent player at this round.

    Module-level so both BattlegroundsGame._tier_potential (the tier-shape
    reward denominator) and the scripted opponents in train.py
    (HeuristicAgent / GreedyPlayAgent leveling logic) share exactly ONE
    definition -- see CLAUDE.md, "Never hardcode card interactions" spirit
    extended to this curve: duplicating it in train.py would let the two
    drift apart silently. Untuned; revisit alongside TIER_POTENTIAL_WEIGHT
    if leveling ends up over/under-incentivized in practice.
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
# SHAPE_ALPHA raised 0.20 -> 1.5 on 2026-09-01 after the first run under the
# telescoping potential went 500 updates (~13,000 games) with eval placement
# flat at chance (4.46 / 4.41 / 4.87 / 4.32 / 4.50 vs greedy) while sell:place
# sat at 0.875 and level_rate at 2%.
#
# The 0.20 was inherited from the OLD BOARD_SHAPE_ALPHA -- but that constant
# was calibrated for a potential that RESET EVERY ROUND, i.e. it paid out once
# per round (~12x per game). Making the potential telescope correctly across
# the whole episode without rescaling alpha therefore shrank the aggregate
# shaping signal by roughly the number of rounds: total telescoped magnitude
# was bounded to +-0.20 against a FINAL_PLACEMENT_REWARD spanning +-4.0, i.e.
# ~5% of the objective. That is far too weak to densify credit assignment,
# which is the entire purpose of the term -- and with churn merely
# reward-NEUTRAL under telescoping (nothing pays for it, but nothing charges
# for it either), there was no gradient discouraging buy/sell cycling.
#
# Sizing: Φ moves ~0.07 per minion placed/sold in practice (measured), so
# alpha=1.5 makes a single good placement worth ~0.10 immediately -- comparable
# to the per-round combat reward (+0.15 win / -0.09 loss) and therefore
# actually able to guide behaviour, while total episode shaping stays ~0.6
# (~15% of the placement range) so it densifies rather than swamps the true
# objective.
#
# Raising alpha is SAFE with respect to degeneracy, and that is the point of
# having made this a true potential function: the invariance result (Ng et al.
# 1999) holds for ANY alpha, so the optimal policy is unchanged no matter how
# large this is. Alpha trades credit-assignment density against gradient
# variance only -- it cannot reintroduce a farmable term, because every cyclic
# sequence still telescopes to ~0 regardless of scale.
SHAPE_ALPHA = 1.5
SHAPE_GAMMA = 0.997   # MUST match PPOConfig.gamma (agent/ppo.py) -- this is
                      # not a free tuning knob. Using anything else (e.g. the
                      # old undiscounted 1.0) breaks the exact telescoping
                      # identity above, which is what makes the invariance
                      # guarantee hold with PPO's own GAE discounting.
BOARD_SHAPE_TRIALS = 30    # sim trials per _board_win_prob call (~0.5 ms each)

# Trials for the REAL combat resolution used by step_combat (not the board-
# shape potential above -- see BattlegroundsGame._combat_sim). Separate
# constant and separate BGCombatSim instance from BOARD_SHAPE_TRIALS /
# _shape_sim on purpose: _shape_sim.simulate() is called on every single
# shopping action (see _board_win_prob), so sharing one instance's RNG stream
# between shaping and real combat resolution would make shaping calls
# perturb the sequence of outcomes real combat draws from, and vice versa.
# 8 trials (vs. 200 default) keeps the per-combat cost low since this runs
# once per player per round rather than once per action.
COMBAT_SIM_TRIALS = 8

# Explicit reward cost charged per successful REORDER.
#
# WHY THIS EXISTS. REORDER costs no gold, so without a charge it is a FREE
# action, and a free action in a discounted MDP is a stalling device: each one
# advances a gamma=0.997 step, so a policy that expects a NEGATIVE return can
# improve its objective purely by delaying the bad news. The gain per free step
# is (1 - gamma) * |V| = 0.003 * |V|, which for a losing agent (V approaching
# the -4.0 FINAL_PLACEMENT_REWARD floor) is up to ~0.012/step. The only
# counterweight already present is the potential-shaping discount drag,
# SHAPE_ALPHA * Phi * (SHAPE_GAMMA - 1) ~= -0.0023/step at Phi=0.5 -- about 5x
# too small.
#
# This is not hypothetical: it was MEASURED happening. Over the first 51
# updates of the 2026-09-02 run, REORDER went 2.4% -> 24.9% of all actions,
# reaching 4.82 of the 6-per-turn budget (80% of the cap) while END_TURN fell
# 12.0% -> 5.0% -- i.e. turns were getting longer, which is the signature of
# stalling rather than positioning. REORDER_BUDGET_PER_TURN bounds the damage
# but does not remove the incentive, so the agent simply walks to the bound.
# This is the same failure mode ACTIVATE showed before its mask was fixed
# (CONTEXT.md 2026-09-01); the lesson is that a budget alone is not enough.
#
# SIZING. 0.03 is ~2.5x the worst-case per-step stalling gain, so the gradient
# against redundant reordering is unambiguous. It is also small against what a
# genuinely useful reposition is worth: board ORDER swings
# (win_prob - loss_prob) by ~0.55 on non-blowout matchups, roughly a full
# placement step, i.e. ~1.0 of FINAL_PLACEMENT_REWARD -- so a real
# repositioning pays for itself ~30x over and is not suppressed. Note this
# does break strict potential-shaping policy-invariance (it is a genuine
# action cost, not a potential difference); that is intentional and is the
# point -- the free action has to stop being free.
REORDER_COST = 0.03

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
# value at which this potential reads 0.5.
#
# RETUNED 2026-09-02 (30.0 -> 60.0) after a live-run trace showed the exact
# degenerate policy this potential exists to prevent: one Suspicious
# Prisonguard pumping a single minion from 3/3 to 36/36 over 18 rounds, board
# held at 4/7 minions, all 10 gold banked every round from round 8, agent
# taking ACTIVATE+FREEZE only and never buying/leveling again -- still won
# the lobby. The 30.0 guess was documented as "rough mid-game board, untuned"
# and was never checked against what boards actually reach under the real
# combat engine.
#
# MEASUREMENT (checkpoint_backups/live_bg_agent_ppo.pt, update 378 -- the
# checkpoint that produced the trace above -- 30 games, real seat mix: 2
# PPOAgent + 2 StaticAgent sharing that policy + 2 HeuristicAgent + 2
# GreedyPlayAgent, 3,474 per-player-round board samples). Raw value = this
# function's numerator (power + keyword_bonus + synergy_bonus), i.e. BEFORE
# the /(value+SATURATION) division, so it's independent of the constant being
# tuned. For the trained-policy seats specifically (the only population this
# potential's gradient actually trains against -- Static/Heuristic/Greedy
# collect no PPO transitions):
#
#   percentile   p10   p25   p50   p75   p90   p95   p99   max
#   raw value      9    26    71   121   162   187   249   308
#
# and by round (median): r8=57, r12=103, r16=131, r20=164, r24=177 -- boards
# keep growing well past round 8, they don't plateau. The degenerate trace's
# own round-18 board (Flighty Scout 36/36 + 3 others) computes to a raw value
# of 137, squarely in the p75-p90 band of ordinary play, not an outlier.
#
# With the OLD constant (30.0), potential(p25=26)=0.46, potential(p50=71)=
# 0.70, potential(p90=162)=0.84 -- already more than half-saturated by the
# 25th percentile and essentially flat (dPhi/dvalue ~ 0.001) by the median.
# That is the mechanism behind "every additional purchase pays ~0 shaped
# reward" from round ~8 onward.
#
# NEW value (60.0) was chosen so potential(value)=0.5 lands near the p50 of
# the trained-policy population (71) while also matching the ORIGINAL intent
# of the comment ("rough mid-game board") once checked against a real
# mid-game board: round 8 (median game length is 21 rounds, so round 8 is
# genuinely early-mid) has a median raw value of 57, almost exactly 60. This
# stretches the useful (non-flat) gradient across roughly the 25th-90th
# percentile of real play: potential(p25=26)=0.30, potential(p50=71)=0.54,
# potential(p90=162)=0.73, potential(p99=249)=0.81 -- a meaningfully wide,
# still-climbing range instead of being pinned near the ceiling by round 8.
#
# Functional shape (value/(value+SATURATION)) was kept rather than replaced:
# it is already a monotonic, noise-free map onto [0, 1) with a single
# legible knob, and the measured problem was that the knob was set wrong, not
# that the curve's shape was structurally unfit -- a differently-shaped curve
# correctly calibrated would land in the same place. Re-review this constant
# again if a future run's board-value distribution (re-run the measurement
# above) drifts materially from the numbers here -- e.g. after a card-pool
# refresh or a combat-sim change that shifts typical board power.
BOARD_SHAPE_STATS_SATURATION = 60.0

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
# --- Damage coefficients ----------------------------------------------------
#
# These two look symmetric but behave in opposite ways, and were retuned
# 2026-09-03 in opposite directions after measuring their variance
# contribution at 0.0% each (see CONTEXT.md).
#
# DAMAGE_TAKEN_COEF telescopes.  Summed over a game,
#   sum_rounds COEF * dmg/max_health  ==  COEF * (max_health - final_health)/max_health
# so the LIFETIME cost of taking all 40 damage and dying is exactly one
# COEF.  At the old 0.05 raw (0.015 scaled) that lifetime maximum was 0.015
# against a FINAL_PLACEMENT_REWARD span of 8 -- 0.2% of the objective.  The
# term wasn't weak, it was inert: the agent was ~indifferent between losing a
# fight at 5 damage and losing it at 20, even though in Battlegrounds that is
# the difference between dying on round 12 and surviving to round 18.
# At 0.6 raw (0.18 scaled) that same gap is worth 0.0675 -- 75% of
# |LOSS_PENALTY| -- so margin now matters about as much as the binary
# outcome, while the dense/placement ratio moves only 0.22 -> 0.25 (measured,
# 40 games; CLAUDE.md's "meaningful but not dominating" band is 0.6-0.8).
#
# DAMAGE_DEALT_COEF does NOT telescope -- it is unbounded positive income
# (8 wins x 15 damage is 3x the coefficient, every game, forever), i.e. the
# same one-sided ratchet the split board/tier shaping was killed for on
# 2026-08-31.  It also triple-counts: WIN_REWARD already pays for winning,
# RANK_DELTA_COEF already pays for opponents dying, and board_potential
# already pays for the board strength that produced the damage -- at every
# action rather than once per round.  Set to 0.  This also makes ghost
# combats need no special case: we don't care by how much you beat anyone,
# so there is nothing to zero out separately for a dead opponent.
DAMAGE_TAKEN_COEF    =  0.6  * DENSE_REWARD_SCALE   # was 0.05; compute_round_reward
DAMAGE_DEALT_COEF    =  0.0  * DENSE_REWARD_SCALE   # was 0.05; removed -- see above
RANK_DELTA_COEF      =  0.15 * DENSE_REWARD_SCALE   # was 0.15; compute_round_reward
HAND_PENALTY_COEF    =  0.08 * DENSE_REWARD_SCALE   # was 0.08; _end_of_turn_reward
GOLD_PENALTY_COEF    =  0.05 * DENSE_REWARD_SCALE   # was 0.05; _end_of_turn_reward
EMPTY_BOARD_PENALTY  =  0.30 * DENSE_REWARD_SCALE   # was 0.30 flat; step_shopping SELL

# --- Reroll penalty ----------------------------------------------------------
#
# Retuned 2026-09-03 alongside the flat GOLD_PENALTY_SCALE change above, after
# that change made the two penalties fight each other. Once gold no longer
# fades away late-game, reroll is the ONLY gold sink available with a full
# board and no affordable buys -- but the escalating per-turn penalty priced
# even the first reroll as a net loss relative to just holding the gold:
#   holding 1 gold to end of turn costs GOLD_PENALTY_COEF * GOLD_PENALTY_SCALE
#     = 0.015 * 0.5 = 0.0075
#   the OLD first reroll cost REROLL_PENALTY_BASE alone = 0.015 -- already 2x
#     the gold-penalty relief it buys, before the per-reroll escalation even
#     starts. Measured live (40 games, current checkpoint): 93% of round>=13
#     end-of-turns left >=5 gold banked (mean 7.47), with buy/reroll/freeze
#     all LEGAL and only 3.2/30 actions used that turn -- the policy was
#     correctly solving the reward as written, which is why it never converged
#     out of it during training rather than being an artifact worth waiting
#     out.
#
# Rerolling should cost LESS than floating the same gold, full stop, so
# spending it (even fruitlessly) always beats sitting on it once nothing else
# is buyable. REROLL_PENALTY_BASE is set below the per-gold holding cost with
# margin (0.003 vs 0.0075 -- 40%, comfortably favouring reroll at every count
# from 1 to the max 10 reachable in a turn) and REROLL_PENALTY_STEP is 0 --
# unlike REORDER, reroll is not a free repeatable action to guard against:
# every use spends real gold hard-capped at ps.max_gold=10 regardless of
# source (every gold grant, hero powers and battlecries included, routes
# through min(ps.max_gold, ...) -- see player_state.py / hero_handler.py /
# effect_handler.py), so it is already self-limiting to at most 10 uses per
# turn with no escalation needed, even if gold regenerates mid-turn.
#
# This bound is exact, not just typical, because a plain reroll changes ONLY
# ps.shop -- never ps.board or ps.tavern_tier -- so board_potential and
# tier_potential, and therefore Phi(s), are LITERALLY unchanged by it. That
# makes the module's existing "potential-shaping discount drag" (the same
# SHAPE_ALPHA * Phi * (SHAPE_GAMMA - 1) term the REORDER_COST comment above
# derives for REORDER, and the module-constants block calls "the tiny
# (SHAPE_GAMMA - 1) discount drag on intermediate states") an EXACT per-reroll
# tax, stacking with REROLL_PENALTY_BASE:
#     cost(Phi) = REROLL_PENALTY_BASE + SHAPE_ALPHA * (1 - SHAPE_GAMMA) * Phi
#               = 0.003 + 0.0045 * Phi
# Since Phi(s) in [0, 1] for every reachable state (CLAUDE.md's Reward
# Shaping (3): both potential components are already in [0, 1] and their
# weights sum to 1), cost(Phi) <= 0.003 + 0.0045 = 0.0075 for EVERY reachable
# state, with equality only in the limit of a maximally saturated board and
# tier -- so a reroll that improves nothing is weakly cheaper than holding
# the same gold at every reachable state, not merely on average. Whenever a
# reroll instead surfaces something worth buying, BUY's own positive
# potential-shaping term (Phi(s') > Phi(s)) is the incentive that follows,
# and it is not bounded the way this tax is -- so reroll can never be
# reward-optimal AS A SUBSTITUTE for buying real value, only as a substitute
# for sitting on gold when nothing is worth buying, which is the case this
# retune targets. The one case where floating gold is legitimately correct
# -- freezing a shop that has a minion worth waiting a turn to afford --
# pays only the ordinary flat gold penalty already covered above, exactly
# once, same as any other unspent gold; FREEZE has never carried a separate
# penalty (see step_shopping type_action == 4), so that case needed no
# change here.
REROLL_PENALTY_BASE  =  0.01  * DENSE_REWARD_SCALE   # was 0.05 raw (0.015 scaled); see above
REROLL_PENALTY_STEP  =  0.0   * DENSE_REWARD_SCALE   # was 0.05 raw; escalation removed -- see above

# --- Unspent-gold penalty (flat) --------------------------------------------
#
# _end_of_turn_reward charges -GOLD_PENALTY_COEF * gold * GOLD_PENALTY_SCALE.
# This used to be a round-indexed schedule (_gold_penalty_scale, removed
# 2026-09-03) that faded 1.0 -> 0.2 by round 13 on the theory that
# early/mid-game gold retention is "saving for a level spike." That theory
# is wrong for this game: `ps.gold` is unconditionally OVERWRITTEN by
# `_gold_for_round(round_num)` at the top of every round (see reset() /
# the per-round setup loop) -- there is no carry-over and no interest, so
# gold held back this turn buys literally nothing next turn (next turn's
# gold is fixed by round number alone, matching real Battlegrounds rules,
# unlike TFT/Underlords-style banking). `ps.level_cost` also decays on its
# own every round regardless of spending, so "wait to level" never requires
# holding gold either. There is therefore no round at which leftover gold
# is anything but pure waste -- a flat coefficient is the more CORRECT
# model, not a simplification traded for accuracy.
#
# The round-indexed schedule also had an independent bug the flat model
# sidesteps entirely: its late-game ramp back up to GOLD_SCALE_LATE_CEIL
# was timed to reach full strength by round 23, sized against a ~21-round
# median measured on an older checkpoint. As the policy improved, games got
# SHORTER (this checkpoint: 15-21 rounds, mean 17.2, measured below) -- the
# ramp's punishing teeth increasingly fell in a round range most games
# never reached, so it was chasing a target that moves away from it as
# training progresses (see CONTEXT.md 2026-09-02 "gold ramp mistimed").
# Flat has no round dependence, so this failure mode cannot recur.
#
# GOLD_PENALTY_SCALE = 0.5 was picked (not guessed) by replaying 2,436 real
# end-of-turn events across 24 games on the real seat mix (4 policy seats
# sharing one live checkpoint + 2 HeuristicAgent + 2 GreedyPlayAgent) and
# comparing candidate flat scales against what the OLD schedule actually
# charged on those SAME trajectories:
#   old schedule total cost  = 29.28   (mean 0.0120/event)
#   flat 0.5 total cost      = 29.93   (mean 0.0123/event, 1.02x old)
#   flat 1.0 total cost      = 59.87   (2.04x old)
# 0.5 reproduces the old schedule's aggregate magnitude almost exactly, so
# the already-validated dense/placement balance (see DENSE_REWARD_SCALE
# above and CLAUDE.md's re-measurement mandate) carries over unchanged --
# this fixes the false premise and the round-mistiming bug without also
# gambling on a new, unvalidated penalty magnitude. Measured on the same
# 24 games: mean|dense_plus_shaping| = 0.823 vs mean|placement_reward| =
# 2.125 per player-game (ratio 0.39, comfortably under the ~0.6-0.8 band
# prior sessions treated as "meaningful but not dominating"), and gold
# events skew low already (median leftover = 0 gold, p90 = 7) -- this
# checkpoint mostly isn't hoarding, so the fix mainly matters for the tail
# and for future checkpoints, not as a wholesale reward-magnitude increase.
# At GOLD_PENALTY_SCALE=0.5, a full 10-gold purse costs -0.075/turn -- half
# a WIN_REWARD (0.15), i.e. meaningful every turn without being punitive on
# a turn where 1-2 gold is structurally unspendable (shop/board/hand full).
GOLD_PENALTY_SCALE = 0.5

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
    outcome_dist: Optional[dict] = None,
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

    outcome_dist
    ------------
    When supplied (keys: p_win, p_loss, exp_damage_taken, exp_damage_dealt --
    the last two already UNCONDITIONAL, i.e. probability-weighted), the
    outcome and damage terms are paid at their *expectation* over the combat
    distribution rather than at the single sampled realisation.  This is
    Rao-Blackwellisation: E[r | boards] is a conditional expectation given
    everything the agent's actions control, so it is unbiased
    (E[rbar] == E[r]) and by the orthogonal decomposition
    Var(sampled) == Var(expected) + Var(noise) it can never increase
    variance.  It is also free: BGCombatSim already runs COMBAT_SIM_TRIALS
    full combats and aggregates them, and step_combat was collapsing that
    aggregate back to one coin flip purely to label the reward.

    Measured effect (40 games, 2026-09-03): removes 22.6% of per-combat
    reward variance but only 2.3% of TOTAL return variance -- because 76.5%
    of combats are already near-decisive (win_prob <= 0.125 or >= 0.875),
    where the sample IS the expectation, and because placement variance
    (sd 2.50) dwarfs combat-reward variance (sd 0.63) by design. Taken
    because it is free and provably non-negative, not because it moves
    training.

    NOTE: only the REWARD is Rao-Blackwellised.  The dynamics stay sampled --
    health still moves by the realised damage.  Expectation-ing the dynamics
    too would make a 55%-win board never lose, delete risk management from a
    game whose objective (FINAL_PLACEMENT_REWARD) is a step function over
    placement, and tune the policy for a game it does not play.
    """
    if outcome_dist is not None:
        p_win  = outcome_dist["p_win"]
        p_loss = outcome_dist["p_loss"]
        r  = WIN_REWARD  * p_win
        r += LOSS_PENALTY * p_loss
        r += -DAMAGE_TAKEN_COEF * (outcome_dist["exp_damage_taken"] / max_health)
        r +=  DAMAGE_DEALT_COEF * (outcome_dist["exp_damage_dealt"] / max_health)
        r += (prev_rank - cur_rank) * RANK_DELTA_COEF
        return r

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
    use_real_combat:
        When True (default), step_combat resolves combats via the full
        turn-by-turn BGCombatSim (self._combat_sim) instead of
        firestone_client.simulate(). Set False to route real combat back
        through firestone_client (whatever backend it is configured with --
        e.g. the mock_mode heuristic) for A/B comparison against the old
        behaviour. Does not affect board-shape potential, which always uses
        self._shape_sim / _board_stats_potential regardless of this flag.
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
        use_real_combat: bool = True,
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
        self.use_real_combat = use_real_combat
        self.batched         = batched
        self._rng            = random.Random(seed)
        self.encoder         = CardEncoder(card_defs)
        self.effect_handler  = EffectHandler(card_defs, tavern_pool=self.tavern_pool)
        self.hero_handler    = HeroPowerHandler(card_defs, HERO_DEF_MAP)
        self.trinket_handler = TrinketHandler(card_defs, rng=self._rng if hasattr(self, "_rng") else None)
        # Both sims get their OWN RNG stream, but both are DERIVED from the
        # game seed rather than left unseeded. Two separate requirements:
        #
        #   - Separate streams, because _shape_sim.simulate() is called on
        #     every shopping action while _combat_sim.simulate() is called
        #     once per player per round; sharing one instance would let the
        #     number of shaping calls (i.e. the policy's action count) shift
        #     the outcomes real combat draws.
        #   - Derived from `seed`, because leaving them unseeded takes RNG
        #     from OS entropy and makes a seeded game NON-reproducible. That
        #     matters concretely: evaluate_policy() pins a fixed per-game seed
        #     precisely so eval points are comparable across updates, and
        #     before real combat was wired in this held (the old mock backend
        #     returned deterministic probabilities and only self._rng sampled
        #     the outcome). Unseeded sims would have silently reintroduced
        #     eval noise that no one could attribute.
        #
        # The offsets just decorrelate the two streams from each other and
        # from self._rng; any distinct constants would do.
        self._shape_sim      = BGCombatSim(
            n_trials=BOARD_SHAPE_TRIALS,
            seed=None if seed is None else seed + 0x51A9E,
        )
        self._combat_sim     = BGCombatSim(
            n_trials=COMBAT_SIM_TRIALS,
            seed=None if seed is None else seed + 0xC0B7A,
        )

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
        not a hard target or a claim about optimal play. Delegates to the
        module-level expected_tier_for_round() so there is exactly ONE
        definition of the curve (also used by train.py's scripted agents).
        """
        return expected_tier_for_round(round_num)

    def _end_of_turn_reward(self, ps) -> float:
        """Shared reward shaping applied at the end of every shopping phase.

        Empty-board penalty  : -EMPTY_BOARD_PENALTY if the board is empty — breaks
                               level-then-end-turn degenerate policy.
        Hand penalty         : -HAND_PENALTY_COEF per card left in hand — cards in
                               hand don't fight; discourages buying without placing.
        Gold efficiency      : -GOLD_PENALTY_COEF * unspent_gold * GOLD_PENALTY_SCALE --
                               flat (round-independent): gold never carries over between
                               rounds (see the GOLD_PENALTY_SCALE block comment for why
                               that makes leftover gold pure waste at every round, and
                               for the 2026-09-03 measurement behind the 0.5 value).

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
        # Unspent gold penalty -- see GOLD_PENALTY_SCALE block comment.
        r -= GOLD_PENALTY_COEF * ps.gold * GOLD_PENALTY_SCALE
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
        # `d` is normally a raw TavernPool draw -- keys "base_atk"/"base_hp"
        # (see bg_card_pipeline.py) -- but this is also called with dicts that
        # already look like a MinionState (keys "attack"/"health", e.g. the
        # partial dict triple_system.check_and_process_triple returns to the
        # pool). minion_stats() tolerates both shapes so this stays the ONE
        # place a card-def/pool dict becomes a MinionState; see its docstring
        # in env/player_state.py for the full story (this used to silently
        # construct every minion with attack=0, health=0).
        atk, hp = minion_stats(d)
        return MinionState(
            card_id=card_id,
            name=d.get("name", ""),
            attack=atk,
            health=hp,
            max_health=hp,
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
                    # Flat per-reroll cost, deliberately cheaper than holding
                    # the same gold to end of turn -- see REROLL_PENALTY_BASE
                    # block comment. _n_rerolls is kept (rather than a flat
                    # `-= REROLL_PENALTY_BASE`) only so REROLL_PENALTY_STEP
                    # still works if it's ever retuned off 0.
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

        elif type_action == 9:
            # reorder: ptr_action is a board slot (ptr 7-13 -> slot 0-6); the
            # minion there is moved to the FRONT of the board. Position is a
            # first-class Battlegrounds decision -- measured over non-blowout
            # matchups in symbolic/combat_sim.py, reordering the same minions
            # swings (win_prob - loss_prob) by a mean of ~0.55 -- and until
            # this action existed the policy had no way to express it, since
            # minions are otherwise appended in play order.
            #
            # Move-to-front rather than an explicit (from, to) pair because it
            # needs only ONE pointer and still reaches every permutation in at
            # most n-1 moves, so no second pointer head is required.
            i = ptr_action - PTR_BOARD_OFF
            if 1 <= i < len(ps.board) and ps.board[i] is not None and ps.reorders_left > 0:
                ps.board.insert(0, ps.board.pop(i))
                ps.reorders_left -= 1
                # Charged only on a reorder that actually happened. An invalid
                # one cannot be sampled (build_type_mask/build_pointer_mask
                # exclude slot 0, an exhausted budget, and boards under 2
                # minions), so there is no way to dodge the cost by aiming at
                # a no-op. See REORDER_COST for the sizing argument.
                reward -= REORDER_COST

        # Potential shaping fires for every dispatched action type above (BUY,
        # SELL, PLACE, REROLL, FREEZE, LEVEL_UP, HERO_POWER, END_TURN, ACTIVATE,
        # REORDER)
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

        GHOSTS.  ``opponent_id`` may name a *dead* player: that is a ghost
        matchup, and it is fought for real against that player's final board.
        Dead players keep their board (nothing here ever assigns or clears
        ``.board``, and the shopping phase iterates alive players only), so
        ``opp.board`` is exactly the board they died with -- which is what a
        Battlegrounds ghost is.  Losing to a ghost costs full damage and can
        eliminate you; winning deals damage to nobody, because there is no
        one left to damage.  That last part needs no special case now that
        DAMAGE_DEALT_COEF is 0.

        Until 2026-09-03 this method short-circuited every ghost to an
        automatic win with zero damage, handing out a free WIN_REWARD.  That
        was 7.5% of all pairings (measured: 3.92 per lobby-game, concentrated
        at rounds 9-16 against a ~16.6-round mean game), i.e. the agent was
        told its board did not matter in exactly the window where board
        strength decides placement.  ``Matchmaker.get_ghost`` had been written
        for this and was never called from anywhere.

        ``opponent_id == -1`` now means only "no ghost source exists yet"
        (no player has died -- reachable only with an odd ``n_players``), and
        keeps the old free-win behaviour as a degenerate fallback.
        """
        ps = self.players[player_id]
        player_board = [_minion_to_dict(m) for m in ps.board]

        if opponent_id == -1 or opponent_id >= len(self.players):
            # No ghost source available at all: nothing to fight.
            result = {
                "result": "win",
                "damage_taken": 0,
                "damage_dealt": 0.0,
                "player_id": player_id,
                "opponent_id": opponent_id,
                "is_ghost": True,
                "outcome_dist": {"p_win": 1.0, "p_loss": 0.0,
                                 "exp_damage_taken": 0.0, "exp_damage_dealt": 0.0},
            }
            ps.last_result = "win"
            ps.last_damage_taken = 0
            ps.last_damage_dealt = 0
            return result

        opp = self.players[opponent_id]
        is_ghost = not opp.alive

        # Fire start-of-combat trinket effects (e.g. stat buffs) before sim snapshot
        self.trinket_handler.apply_on_combat_start(ps)

        opp_board = [_minion_to_dict(m) for m in opp.board]

        # Real combat resolution: full turn-by-turn BGCombatSim (taunt, divine
        # shield, venomous, reborn, windfury, deathrattles, positioning,
        # tribes, multiplier cards -- see symbolic/combat_sim.py) via its own
        # instance/RNG stream (self._combat_sim), independent of the board-
        # shape potential's self._shape_sim. use_real_combat=False falls back
        # to the old firestone_client path (whatever backend it is configured
        # with) for A/B comparison. Same SimResult shape either way, so
        # nothing downstream (the outcome-sampling code right below) changes.
        if self.use_real_combat:
            sim = self._combat_sim.simulate(
                player_board, opp_board,
                player_tier=ps.tavern_tier,
                opp_tier=opp.tavern_tier,
            )
        else:
            sim = self.firestone.simulate(
                player_board, opp_board,
                player_tier=ps.tavern_tier,
                opp_tier=opp.tavern_tier,
            )

        # Determine concrete outcome by sampling from the full probability
        # distribution.  The DYNAMICS stay sampled -- health must move by a
        # realised amount, and expectation-ing it would delete risk from a
        # game whose objective is a step function over placement.  Only the
        # REWARD is paid at expectation (outcome_dist below).
        roll = self._rng.random()
        if roll < sim.win_prob:
            outcome = "win"
        elif roll < sim.win_prob + sim.tie_prob:
            outcome = "tie"
        else:
            outcome = "loss"

        # Rao-Blackwellised reward terms.  BGCombatSim already ran
        # COMBAT_SIM_TRIALS full combats to produce these; collapsing them to
        # the single roll above purely to label the reward threw that work
        # away.  NOTE sim.expected_damage_{dealt,taken} are CONDITIONAL on
        # winning/losing (combat_sim.py divides by max(wins,1)/max(losses,1)),
        # so they must be probability-weighted here to become unconditional.
        p_win  = sim.win_prob
        p_loss = max(0.0, 1.0 - sim.win_prob - sim.tie_prob)
        outcome_dist = {
            "p_win":  p_win,
            "p_loss": p_loss,
            "exp_damage_taken": p_loss * float(sim.expected_damage_taken),
            "exp_damage_dealt": 0.0 if is_ghost else p_win * float(sim.expected_damage_dealt),
        }

        # Damage calculation (simplified: tier + board size when player loses)
        if outcome == "loss":
            damage_taken = int(round(sim.expected_damage_taken))
            if damage_taken == 0:
                damage_taken = max(1, ps.tavern_tier + len(opp.board))
            damage_taken = max(0, damage_taken)
        else:
            damage_taken = 0

        # A ghost win damages nobody -- there is no player left to damage.
        damage_dealt = (0.0 if is_ghost
                        else float(sim.expected_damage_dealt) if outcome == "win"
                        else 0.0)

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
            "is_ghost":     is_ghost,
            "outcome_dist": outcome_dist,
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
                # Announce back only to a LIVE opponent.  For a ghost matchup
                # pid_b is a real but dead player, and pid_a must still get
                # the announcement: it is what makes the ghost's board visible
                # through the existing opponent_snapshots machinery during
                # shopping.  Without it next_opponent_id stayed None, opp_tokens
                # encoded an EMPTY board and _board_shape_potential fell back to
                # 0.5 -- so simulating ghost fights while leaving this alone
                # would add real lethal risk while keeping the agent blind to
                # it, which is strictly worse than the old free win.
                if pid_b != -1 and pid_b < len(self.players) and self.players[pid_b].alive:
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
                ps.reorders_left = REORDER_BUDGET_PER_TURN
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

                # Ghost matchup: pid_b is either -1 (nobody has died yet) or a
                # real DEAD player whose final board we fight for real.  Either
                # way only pid_a fights and only pid_a is paid -- hence the
                # -1 sentinel kept in the combat_results tuple below, which is
                # what the reward loop keys off to skip the dead side.
                if pid_b == -1 or not self.players[pid_b].alive:
                    result = self.step_combat(pid_a, pid_b)
                    round_summary["combats"].append(result)
                    combat_results.append((pid_a, -1, result, {}))
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
                        outcome_dist=result_info.get("outcome_dist"),
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
        from agent.policy import build_type_mask_batch, build_pointer_mask

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
            _pmask_buf = [None] * _n

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
                # Bug fix (2026-09-01, see CONTEXT.md): the pointer mask used to
                # SAMPLE must be the same, type-specific mask that gets STORED
                # in the transition -- previously this passed a type-agnostic
                # full-occupancy mask (build_pointer_mask(_s, -1)) here, then
                # separately recomputed the correct type-specific mask below
                # (via build_pointer_mask_batch) only for storage. That let the
                # policy sample pointers get_action_batch's own zone-restriction
                # would never allow through the real per-type rule (e.g.
                # ACTIVATE requires cost>0/affordable/not-yet-activated, not
                # just "board slot occupied"), producing ~92% no-op ACTIVATE
                # actions, AND made evaluate_actions() at PPO-update time score
                # the stored action under a different, narrower distribution
                # than the one it was actually sampled from -- biasing the
                # importance ratio on every pointer-type action even when the
                # policy hadn't changed. ptr_mask_fn closes over this group's
                # states so get_action_batch can build the exact type-specific
                # mask itself, AFTER sampling the type, and hand back the mask
                # it actually used -- see get_action_batch's docstring.
                _ta, _pa, _lp, _vl, _pm = _pol.get_action_batch(
                    _board_g, _shop_g, _hand_g, _scalar_g,
                    type_mask=_t_mask_g, opp_tokens=_opp_g,
                    ptr_mask_fn=lambda _i, _t, _states=_g_states: build_pointer_mask(_states[_i], _t),
                )
                for _j, _i in enumerate(_g_idxs):
                    _type_buf[_i]  = _ta[_j]
                    _ptr_buf[_i]   = _pa[_j]
                    _lp_buf[_i]    = _lp[_j]
                    _val_buf[_i]   = _vl[_j]
                    _pmask_buf[_i] = _pm[_j]

            type_acts = _torch.stack(_type_buf)
            ptr_acts  = _torch.stack(_ptr_buf)
            log_probs = _torch.stack(_lp_buf)
            values    = _torch.stack(_val_buf)
            t_mask    = build_type_mask_batch(player_states).to(dev)
            # Stored mask == sampled mask by construction: these are exactly
            # the masks get_action_batch returned above, not a recomputation.
            # (Do NOT reintroduce a separate build_pointer_mask_batch(...) call
            # here -- that is precisely the bug described above.)
            ptr_masks = _torch.stack(_pmask_buf)

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
