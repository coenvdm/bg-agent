# Hearthstone Battlegrounds — RL Agent

You are helping build a Reinforcement Learning agent that plays Hearthstone
Battlegrounds competitively. This file defines the architecture, symbolic
feature layer, and card definitions the agent uses.

---

## ⚠️ MANDATORY END-OF-SESSION CHECKLIST

**These steps are REQUIRED at the end of every session in which any file is changed.
Do not consider a session complete until all four steps are done.**

### Step 1 — Append to CONTEXT.md

Use a Bash heredoc to append directly — do NOT read the file first (it can be large):

```bash
# If CONTEXT.md doesn't exist yet, create it with the header first:
# echo "# bg_agent — Development Context Log" > CONTEXT.md

cat >> CONTEXT.md << 'EOF'

---
### [DATE] — [SHORT SESSION TITLE]
**Files changed:** `list/of/files.py`
**What was done:** 2–4 sentences summarizing the changes made and why.
**Current state:** One sentence on where things stand now.
**Open questions / next steps:** Bullet list of unresolved issues or planned next actions.
---
EOF
```

- Use `>>` (append), never `>` (overwrite).
- Never use the Read or Edit tools on CONTEXT.md — always append via Bash `>>`.

### Step 2 — Stage source files

```bash
git add <specific changed files>
```

Do **not** stage: `*.pt` model weights, `data/` logs, `__pycache__/`, or `.env` files.

### Step 3 — Commit

```bash
git commit -m "Session YYYY-MM-DD — <short title matching CONTEXT.md entry>"
```

### Step 4 — Push

```bash
git push origin master
```

If no remote exists yet:

```bash
git remote add origin <your-github-repo-url>
git push -u origin master
```

---

## Architecture Overview

The agent uses a **neurosymbolic** design: a hand-coded symbolic layer computes
all deterministic mechanical quantities (auras, deathrattle types, multipliers,
combat win probabilities), and passes them as structured features to a neural
network that learns only the **strategic** decisions.

```
Raw game state
      │
      ├──→ Symbolic layer (bg_card_pipeline.py)  →  board features
      │         deterministic, no gradients            (auras, DRs, triggers)
      │
      └──→ Firestone combat sim (subprocess)     →  win_prob, expected_damage
                Monte Carlo over 200 trials
                        │
                        ▼
               Neural network (PPO)
               learns: when to level, pivot tribes,
                       freeze, scout, manage economy,
                       and every real card CHOICE
```

**Policy network shape lives in exactly one place**: `agent/policy.py`'s
`POLICY_ARCH` / `make_policy()`. Never construct `BGPolicyNetwork(...)` with
inline kwargs — seven call sites used to do that, and a single disagreeing
number does not raise: `PPOTrainer.load_checkpoint` catches the mismatch, logs
a warning, returns False, and training silently restarts from zero.

Current architecture (2026-09-04): `d_model=256`, `nhead=8`, **`num_layers=6`**
(was 4), 32 tokens, ~5M params. Pointer scoring is a **pointer network**: each
of six scorer ROLES (buy / sell / place / activate / choose_target /
choose_option — see `_TYPE_ROLE`) is a fixed linear probe on the token *plus* a
scaled dot-product against a per-role query emitted from the global
`[CLS ‖ scalar]` state. The query is **zero-initialised**, so it starts as an
exact no-op and the scorers reduce to the fixed probes they replaced. It exists
because a fixed probe asks the *same* question of a token regardless of context:
"is this Beast worth buying?" has a different answer at 3 gold on round 4 than at
10 gold on round 12 with a full Murloc board.

`forward()` returns pointer logits as a single **`[B, N_ACTION_TYPES, 24]`**
stack indexed by action type, not as parallel tensors each call site picks
between with its own `if type == 8` branch. That shape makes the
sample/evaluate-mismatch bug class unrepresentable — it has bitten this project
twice (see `CONTEXT.md` 2026-08-31/09-01 and the ACTIVATE scorer fix), because
three call sites had to agree and a disagreement corrupts PPO's importance ratio
while every loss curve still looks healthy.

---

## Symbolic Layer Rules

When computing features, always follow these conventions:

1. **Multipliers first**: check for Brann (battlecries trigger twice), Titus
   Rivendare (deathrattles trigger an extra time), and Drakkari Enchanter
   (end-of-turn effects trigger twice) before computing any effect counts.

2. **Aura dependency**: for every aura source on the board, compute
   `aura_dependency_score = (power_with_aura - power_without) / power_with`.
   High scores mean a fragile board if the aura source dies early.

3. **Effect duration matters**: tag every effect as `PERMANENT` (persists
   between rounds), `THIS_COMBAT` (resets next round), or `THIS_GAME`
   (lasts the full game). Permanent buffs are worth ~3× a combat-only buff
   of equivalent stats.

4. **Tribe counts drive synergy scores**: compute tribal density for all
   10 tribes (Murloc, Beast, Mech, Demon, Dragon, Pirate, Elemental,
   Quilboar, Naga, Undead). A board is "synergistic" when ≥4 minions share
   a tribe.

5. **Never hardcode card interactions in the neural network**. If a new card
   appears that fits an existing effect category, add it to the card
   definitions JSON and the symbolic layer handles it automatically.

---

## Action Space

Each buy-phase turn is a sequence of atomic actions until END_TURN:

The implemented space is 12 action types, each optionally carrying ONE pointer
into a 24-slot zone layout (shop 0-6, board 7-13, hand 14-23). Source of truth
is `agent/policy.py` (`N_ACTION_TYPES`, `ACTION_TYPE_NAMES`, `TYPES_WITH_POINTER`).

```python
0 BUY(shop_idx)        # pointer: shop slot
1 SELL(board_idx)      # pointer: board slot
2 PLACE(hand_idx)      # pointer: hand slot -- APPENDS to the board; the
                       #   position is chosen afterwards with REORDER
3 REROLL               # no pointer
4 FREEZE               # no pointer
5 LEVEL_UP             # no pointer
6 HERO_POWER           # no pointer -- masked ON only for heroes whose
                       #   power_type is "active_noptr" (17 of 29 are
                       #   passive/null and could not act; fixed 2026-09-05)
7 END_TURN             # no pointer
8 ACTIVATE(board_idx)  # pointer: board slot -- the minion's own Activate (N)
9 REORDER(board_idx)   # pointer: board slot 1-6 -- moves that minion to the
                       #   FRONT of the board
10 CHOOSE_TARGET(board_idx)   # pointer: board slot -- resolves a pending
                       #   "choose a minion" effect (see Choice Mechanics)
11 CHOOSE_OPTION(opt_idx)     # pointer: shop slot 0..n-1 -- resolves a pending
                       #   "Choose One" battlecry branch
```

Types 10 and 11 are legal **only** while `ps.choice_pending` is set, and then
they are the *only* legal type -- see Choice Mechanics below.

`REORDER` is move-to-front rather than an explicit `(from, to)` pair: that
needs only one pointer yet still reaches **every** permutation of n minions in
at most n-1 moves (verified by BFS over all 5,040 orderings of a 7-minion
board). `REORDER_BUDGET_PER_TURN = 6` is exactly that worst case for a full
board, so the budget never blocks a reachable arrangement.

Board slot 0 is never a valid REORDER target, and the budget is finite,
because a *costless, repeatable* action is a discounting exploit: each one
advances a `gamma=0.997` step, so spamming them discounts the pending
END_TURN penalties. ACTIVATE demonstrated exactly this failure mode before its
mask was fixed (see `CONTEXT.md` 2026-09-01) -- do not add another no-cost
action without a budget or a real state change.

Always mask invalid actions (no gold to buy, board full, etc.).

---

## Reward Shaping

**Source of truth is `env/game_loop.py` — this section is a summary, not a
spec. Update this section (not the other way around) whenever the constants
below change.**

Four components combine into the total reward. **Note:** an earlier version
of this section (pre-2026-09) described a *split* `phi_board`/`phi_tier`
scheme with separate `_apply_board_shape`/`_apply_tier_shape` methods that
reset each potential every round. That scheme was replaced 2026-08-31 by the
unified telescoping potential described in (3) below — the split/reset
design was a one-sided ratchet (paid for building board/tier strength within
a turn, never charged for losing it) and broke the policy-invariance
guarantee this shaping relies on. If you find references to
`BOARD_SHAPE_ALPHA`/`TIER_SHAPE_ALPHA`/`ps.phi_board`/`ps.phi_tier`
anywhere, they are stale — `env/game_loop.py` is the only source of truth.

1. **`compute_round_reward`** — dense per-combat-round signal (coefficients
   shown already multiplied by `DENSE_REWARD_SCALE = 0.30`; the raw values in
   the source are 0.5/-0.3/0.6/0.0/0.15 respectively — see the module
   constants block for why everything in this component and (2) is scaled
   down 0.30x, and quote the *scaled* number when reasoning about magnitude).
   Since 2026-09-03 the outcome and damage terms are paid at their
   **expectation** over the combat distribution (`outcome_dist`), not at the
   single sampled roll:
   ```python
   r  =  0.15   * p_win                          # WIN_REWARD
   r += -0.09   * p_loss                         # LOSS_PENALTY
   r += -0.18   * (E[damage_taken] / max_health) # DAMAGE_TAKEN_COEF
   r +=  0.0    * (E[damage_dealt]  / max_health) # DAMAGE_DEALT_COEF (removed)
   r += (prev_rank - cur_rank) * 0.045           # RANK_DELTA_COEF
   ```
   This is Rao-Blackwellisation and it is free: `BGCombatSim` already runs
   `COMBAT_SIM_TRIALS = 8` full combats and aggregates them, and `step_combat`
   was collapsing that aggregate back to one coin flip purely to label the
   reward. Unbiased and provably non-increasing in variance, but **small**:
   measured 22.6% of per-combat reward variance and only 2.3% of *total*
   return variance, because 76.5% of combats are already near-decisive
   (`win_prob <= 0.125` or `>= 0.875`) and placement variance (sd 2.50) dwarfs
   combat-reward variance (sd 0.63) by design. Only the **reward** is
   expectation-ed — the dynamics stay sampled, deliberately: expectation-ing
   health too would make a 55%-win board never lose and delete risk management
   from a game whose objective is a step function over placement.
   The two damage coefficients were retuned in **opposite** directions on
   2026-09-03 after both measured at 0.0% of reward variance:
   - `DAMAGE_TAKEN_COEF` 0.05 → **0.6 raw** (`0.18` scaled). It *telescopes*:
     summed over a game the damage penalty equals `COEF * (max_health -
     final_health)/max_health`, so the **lifetime** cost of taking all 40
     damage and dying was exactly `0.015` against a `FINAL_PLACEMENT_REWARD`
     span of 8. Not weak — inert: the agent was indifferent between losing a
     fight at 5 damage and at 20, which in Battlegrounds is the difference
     between dying on round 12 and surviving to round 18. At 0.18 that gap is
     worth `0.0675`, ~75% of `|LOSS_PENALTY|`.
   - `DAMAGE_DEALT_COEF` 0.05 → **0**. It does *not* telescope — 8 wins × 15
     damage is unbounded positive income, the same one-sided ratchet the split
     board/tier shaping was killed for on 2026-08-31 — and it triple-counts
     `WIN_REWARD`, `RANK_DELTA_COEF` and `board_potential`.
   There is no flat per-round survival bonus — one existed historically
   (`+0.1` unconditionally) but was removed 2026-08-31: it was passive income
   that fired merely for being alive and diluted the placement signal
   without rewarding any actual decision (see `CONTEXT.md`).
2. **`_end_of_turn_reward`** — fired on END_TURN, and on the forced turn-end
   when a player exhausts the 30-action budget without ending its turn (that
   forced path exists so burning the budget cannot be used to *dodge* the
   penalties below). **No longer fired on FREEZE**: as of 2026-09-04 FREEZE is
   a plain shop toggle that does not end the turn, matching Hearthstone — see
   Game Dynamics below. `-HAND_PENALTY_COEF`
   (`0.024`) per card left in hand, and `-GOLD_PENALTY_COEF * gold *
   GOLD_PENALTY_SCALE` for unspent gold, `GOLD_PENALTY_COEF = 0.015`.
   `GOLD_PENALTY_SCALE` is **flat** (`0.5`), replacing a round-indexed
   V-shaped schedule (2026-09-02 → 2026-09-03). The V-shape was built on the
   premise that early/mid-game gold retention is sometimes correct ("saving
   for a level spike"), but that premise is false for this game: `ps.gold`
   is unconditionally overwritten by `_gold_for_round(round_num)` every
   round (no carry-over, no interest — matching real Battlegrounds, unlike
   TFT/Underlords-style banking) and `ps.level_cost` decays on its own each
   round regardless of spending, so leftover gold buys nothing next turn at
   *any* round — there's no round at which holding it is anything but pure
   waste. The V-shape also had an independent mistiming bug: its late-game
   ramp was tuned to reach full strength by round 23 against a ~21-round
   median, but as the policy improved games got *shorter* (15-21 rounds on
   the checkpoint that motivated this fix), so the ramp's punishing teeth
   increasingly fell outside the round range most games actually reached
   (see `CONTEXT.md` 2026-09-02 "gold ramp mistimed"). A flat coefficient
   has no round dependence, so this can't recur.
   `GOLD_PENALTY_SCALE = 0.5` was sized by replaying 2,436 real end-of-turn
   (round, gold) events from 24 games on the real seat mix against the OLD
   schedule: flat `0.5` reproduces the old schedule's aggregate cost almost
   exactly (29.93 vs 29.28 total, 1.02x) on those same trajectories, so the
   already-validated dense/placement balance carries over rather than being
   re-gambled (measured on the same games: mean|dense_plus_shaping| = 0.823
   vs mean|placement_reward| = 2.125 per player-game, ratio 0.39, well under
   the ~0.6-0.8 band prior sessions treated as "meaningful but not
   dominating" — see `CONTEXT.md` 2026-09-03). At `GOLD_PENALTY_SCALE = 0.5`,
   a full 10-gold purse costs `-0.075`/turn — half of `WIN_REWARD = 0.15`,
   every turn, not just late.

   `step_shopping`'s REROLL branch charges its own separate,
   **non-escalating** flat cost, `-REROLL_PENALTY_BASE` = `-0.003` per use
   (`REROLL_PENALTY_STEP = 0`). Retuned 2026-09-03 in the same session as the
   flat gold penalty above, because the two fought each other: with a full
   board and no affordable buy, reroll is the *only* gold sink, but the old
   escalating cost (`REROLL_PENALTY_BASE = 0.015`, `+0.015`/reroll past 2)
   priced even the **first** reroll as a net loss against just holding the
   gold — holding 1 gold to end of turn only costs `GOLD_PENALTY_COEF *
   GOLD_PENALTY_SCALE = 0.0075`, half the old reroll cost. Measured live (40
   games, the checkpoint that motivated this fix): 93% of round-13+
   end-of-turns banked ≥5 gold (mean 7.47) with buy/reroll/freeze all legal
   and only 3.2/30 actions used that turn — the policy was solving the
   reward exactly as written, not failing to converge. Reroll must cost
   *less* than floating the same gold at every count for spending it to ever
   be worth it once nothing else is buyable: `0.003` is 40% of `0.0075`, and
   with no escalation that margin holds at every reroll count up to the
   max of 10 reachable in a turn (gold is real and non-regenerating —
   `min(2+round,10)`/turn, no carry-over — so unlike `REORDER` this needed no
   budget to stay safe from a discounting exploit). Freezing a shop to wait a
   turn for a minion you can't yet afford is unaffected and intentionally
   so: `FREEZE` has never carried a penalty beyond the ordinary flat gold
   charge above, paid once, same as any other unspent gold.
3. **Unified potential-based shaping** (`_apply_potential_shaping`) — a
   single potential Φ(s) ∈ [0, 1] (Ng, Harada & Russell 1999), paid out at
   **every** shopping action (BUY/SELL/PLACE/REROLL/FREEZE/LEVEL_UP/
   HERO_POWER/END_TURN/ACTIVATE/REORDER/CHOOSE_TARGET/CHOOSE_OPTION) plus
   once per round right after combat resolves:
   ```python
   r_shaped = SHAPE_ALPHA * (SHAPE_GAMMA * Φ(s') − Φ(s))   # SHAPE_ALPHA=1.5, SHAPE_GAMMA=0.997
   Φ(s) = 0.67 * board_potential(s) + 0.33 * tier_potential(s)   # BOARD_/TIER_POTENTIAL_WEIGHT
   ```
   `ps.phi` is initialised once in `reset()` and **never reset mid-episode**
   — this is what makes the sum telescope exactly to `SHAPE_ALPHA *
   (SHAPE_GAMMA**T * Φ(s_T) − Φ(s_0))` regardless of path, so any cyclic
   action sequence (buy/sell churn, freeze/unfreeze, ...) nets ~0 shaped
   reward. This replaced the old per-round-reset split scheme in 2026-08-31;
   `SHAPE_ALPHA` was itself raised `0.20 → 1.5` on 2026-09-01 once telescoping
   made the old value pay out ~5% of what it used to (see `CONTEXT.md`).
   - `board_potential(s)`: deterministic, noise-free — `value / (value +
     field)`, where `field` is the mean board value of the **other alive
     players**, floored at `BOARD_SHAPE_FIELD_FLOOR = 60.0`
     (`_field_denominator`), and
     `value = BOARD_STATS_MINION_SCALE * Σ_minion (atk+hp)**BOARD_STATS_MINION_EXPONENT`
     (per-minion effective stats via `symbolic.board_computer._board_power`)
     plus a flat `+5` per Divine Shield/Taunt/Reborn/Windfury instance
     (`BOARD_STATS_KEYWORD_BONUS`) and a flat `+9` if any tribe reaches the
     CLAUDE.md "synergistic" threshold of 4+ (`BOARD_STATS_SYNERGY_BONUS`).
     Minions in **hand** count too, at `BOARD_STATS_HAND_WEIGHT = 0.5`
     (spells excluded).

     **The denominator is the FIELD, not a constant** (2026-09-05). It was a
     fixed `BOARD_SHAPE_STATS_SATURATION = 60`, standing in for "a typical
     opponent" — but real opponent boards grow from ~8 at round 1 to ~190 by
     round 16, so Φ saturated exactly when the board is full and upgrading is
     the only lever left. Measured against `BGCombatSim` as ground truth, the
     shaped reward paid per 1.0 of **real win probability** gained was `0.103`
     at round 6, `0.061` at round 14, `0.026` at round 18 — a late upgrade
     worth **+0.445 win probability was paid +0.0114**, barely more than the
     `0.009` saved by not rerolling three times, while costing 2 net gold and
     3 actions instead of 1. The live policy responded exactly as priced: from
     round ~15 its action mix was REROLL + END_TURN and *nothing else*, with
     SELL at 2.0% of actions and 43% of all gold burned on rerolls. Dividing
     by the field makes Φ a ratio-to-field — a Bradley-Terry-shaped
     win-probability proxy, which is what placement actually depends on.
     Measured after: calibration spread `4.02x → 2.32x`, late payoff roughly
     doubled, **early payoff unchanged** (`+0.0763 → +0.0768`) so the width
     fix's early-game incentive survives — by construction, since the floor
     equals the old constant and the field mean only passes 60 around round 6,
     making this a strict *late-game* correction.
     Three details that matter: the field is **snapshotted once per round**
     (`_refresh_field_values`), so Φ stays a deterministic function of state —
     a denominator drifting as other seats shop concurrently is not, and
     within a turn Φ then moves only through the agent's own board; **self is
     excluded** from the mean, since including it lets the agent's own
     improvement inflate its own denominator and damps the true gradient ~12.5%
     at n=8; and `reset()` **clears** the snapshot, because a reused
     `BattlegroundsGame` would otherwise carry game N's endgame lobby into
     game N+1's initial Φ. It is **self-calibrating** — it reads the live
     lobby rather than a fitted per-round curve, so it cannot go stale as the
     policy improves, which is the exact failure mode that killed the
     round-indexed gold ramp (2026-09-02).

     `BOARD_STATS_HAND_WEIGHT` exists because BUY used to pay **exactly
     `+0.0000`**: the minion moves shop → hand, the board is unchanged, so 3
     gold bought a shaped signal of "nothing happened" and the whole payoff was
     deferred to a separate PLACE action, making BUY read as pure cost in a
     per-action advantage. Note this does **not** change the total value of
     buy-and-place and cannot — Φ telescopes, so the endpoints fix the sum
     (measured `+0.0188` either way). It is purely credit-assignment smoothing,
     and it matters because "sell first, hope to buy better" is a bootstrapping
     trap: if the policy rarely upgrades then `V(post-sell)` is low, so
     `A(sell)` is negative, so it never learns to sell.

     The per-minion **exponent** (`0.7`, with `SCALE = 3.0`) was added
     2026-09-05 and is the load-bearing part. Before it, `value` was a flat
     sum of attack+health over the whole board, which made Φ **blind to board
     width**: measured on the live u1806 checkpoint, one 40/40 scored
     `Φ=0.571` and seven 6/6s scored `Φ=0.562` — indistinguishable, when in
     Battlegrounds the wide board wins that fight overwhelmingly (seven bodies
     and seven attacks against one, and since the 2026-09-04 first-attacker
     fix it also strikes first). The old code defended the flat sum as
     "quality-weighted, not count-weighted", guarding against hoarding 1/1s to
     fill slots — but it left the **opposite** degenerate board completely
     unguarded, and that is the one the policy actually found. Measured over 6
     games on that checkpoint: **135 of 150 choices (90%) were Suspicious
     Prisonguard's "+3/+3 to another minion"** (22.5/game), **24% of all gold
     went to ACTIVATE and only 13.6% to buying minions**, and board size sat
     at **4.42/7, never filling**. Per gold the pump paid `0.0129` ΔΦ against
     BUY's `0.0098` — the agent was right to pump; Φ was wrong. Combat win
     rate fell 0.558 → 0.486 and gauntlet Elo peaked at update 800 (`+299`)
     and decayed to `+144` by update 1800.
     Concavity is applied **per minion, then summed** — never over the board
     total, which would be width-blind in the same way. It is not a penalty
     bolted on: a 50/50 genuinely does not beat a 25/25 twice as reliably,
     since it still makes one attack and still dies to one Venomous or Zapp.
     Ordering it now produces on equal-total-stat boards: 1×40/40 `0.518` <
     2×20/20 `0.569` < 4×10/10 `0.620` < 7×6/6 `0.652`; BUY-and-place now pays
     **3.17×** a `+3/+3` pump (was 1.62×). The 1/1-hoarding case stays handled
     — seven 1/1s score `0.362`, below the single 40/40.
     `EXPONENT`/`SCALE` were fitted on **624 real end-of-turn boards** so that
     Φ's *median is unchanged* (flat sum: value p50 34.0 → Φ 0.362; new: value
     p50 59.8 → Φ 0.499 at `SAT=60`): the point is to change the **shape** of
     the potential, not its magnitude, so the dense-vs-placement balance stays
     where it was validated. `BOARD_SHAPE_STATS_SATURATION` therefore stays at
     `60.0`, and the keyword/synergy bonuses were rescaled `3→5` and `5→9`
     purely to hold their existing *share* of `value` under the 1.76× median
     shift — neither is a retune.

     *Historical (the constant this replaced).*
     `BOARD_SHAPE_STATS_SATURATION` was raised **30.0 → 60.0** on 2026-09-02:
     a 30-game measurement on the real seat mix (2 PPOAgent + 2 StaticAgent +
     2 HeuristicAgent + 2 GreedyPlayAgent) found the trained-policy
     population's board `value` already at p50=71/p90=162 by the time boards
     matured, and the degenerate live-run board that triggered this fix
     (one Suspicious Prisonguard-pumped minion, 4/7 board, round 18) computed
     to `value=137` — squarely mid-distribution, not an outlier. At the old
     `30.0`, that distribution was already >50% saturated by its 25th
     percentile, so essentially every purchase past round ~8 paid ≈0 shaped
     reward. `60.0` lands potential=0.5 near both the trained-policy median
     (71) and the round-8 median (57, genuinely "mid-game" given the ~21-round
     median game length) — see the long comment above
     `BOARD_SHAPE_STATS_SATURATION` in `env/game_loop.py` for the full
     percentile table. **Note that this 2026-09-02 change also amplified the
     width-blindness above** — it was diagnosed *from* a Suspicious-Prisonguard
     -pumped board, and desaturating Φ made each pump pay more, not less. The
     percentile numbers quoted here are on the OLD flat-sum `value` scale and
     do not transfer to the new one; the refitted figures are in the exponent
     paragraph above. An empty-board penalty (`-EMPTY_BOARD_PENALTY` =
     `-0.09`) fires separately, on the SELL action that empties the board.
   - `tier_potential(s) = min(1.0, tavern_tier / _expected_tier_for_round(round_num))`
     — reaching/exceeding the round's on-curve tier fully saturates this
     component, so there's no reward for leveling further than useful.
   - Both components are already in [0, 1] and the weights sum to 1.0, so
     Φ(s) ∈ [0, 1] for every reachable state **regardless of
     `BOARD_SHAPE_STATS_SATURATION`** — retuning that constant changes how
     quickly Φ climbs, never the [0,1] bound the telescoping proof depends
     on, and therefore cannot by itself blow up the total per-episode
     shaping magnitude (bounded ≈ ±`SHAPE_ALPHA * BOARD_POTENTIAL_WEIGHT`
     ≈ ±1.0 either way).
   - `REORDER_COST = 0.03` is charged as a separate, un-shaped action cost on
     every REORDER that actually applies (not potential-based — it
     deliberately breaks strict policy-invariance to kill a
     `gamma`-discounting stalling exploit; see `env/game_loop.py`).
4. **`FINAL_PLACEMENT_REWARD`** — fires immediately at the moment a player
   is eliminated (not at game end); survivors receive it at game end instead.
   ```python
   FINAL_PLACEMENT_REWARD = {1:+4.0, 2:+2.0, 3:+1.0, 4:0.0,
                              5:-1.0, 6:-2.0, 7:-3.0, 8:-4.0}
   ```

PPO uses `gamma=0.997`, `gae_lambda=0.95` (raised from `gamma=0.99` so the
final placement reward doesn't decay away before reaching early-round
decisions); `SHAPE_GAMMA` in (3) must match this exactly for the telescoping
identity to hold.

Historically, the dense per-round/per-action terms in (1)/(2) summed to
roughly **17x** `FINAL_PLACEMENT_REWARD`'s magnitude before
`DENSE_REWARD_SCALE` was introduced (see `CONTEXT.md`, 2026-08-31) — drowning
out the actual objective. Whenever retuning any dense coefficient (including
the gold-penalty scale above), re-measure the dense-sum-per-player-game
vs. `FINAL_PLACEMENT_REWARD` balance on the real seat mix and confirm it
hasn't drifted back toward dominating; the 2026-09-02 gold-schedule retune
was checked this way (mean dense_sum stayed near 0, well inside the ±4
placement span, on a 30-game/240-player-game sample), and so was the
2026-09-03 flattening (dense/placement ratio 0.39 on a 24-game/192-player-game
sample — see `CONTEXT.md`).
When retuning leveling incentives, watch `level_rate` *and* `board_size`/
placement together (not `level_rate` alone) — a naive leveling incentive can
reproduce the same kind of degenerate policy a flat `board_size` reward
caused historically (agent chases the proxy metric at the expense of what it
actually stands for).

---

## Progress Metrics & Model Selection

**Only the gauntlet Elo is trustworthy.** `evaluate_policy(..., opponent=
'gauntlet')` seats the policy against a spread of its own frozen past selves
and fits Bradley-Terry ratings anchored to the **oldest** reference
(`fit_gauntlet_elo` in `train.py`), so successive values share a scale and
"higher is better" holds across a whole run. Nothing in the shaping can
inflate it.

The other numbers are not substitutes:

- `greedy`/`heuristic` eval **saturate and then stop discriminating**. On the
  2026-09-04 run they pinned at top1 ≈ 0.92 / top4 ≈ 0.94 from update ~350
  onward — 80% of the run reading flat while the policy regressed 155 Elo.
- **Gauntlet *placement* is not comparable across time** even though the Elo
  is: the reference set grows stronger as new refs are frozen, so placement
  drifts toward 4.5 on its own. Quote the Elo.
- `game_rewards` / `avg10` is **not a progress metric at all.** It has sd ≈
  2.5 (dominated by placement), so a 10-game mean has sd ≈ 0.8 and its running
  max is a luck record. `bg_agent_ppo_best.pt` used to be selected on it and
  froze at **update 309** with `best_avg10 = 3.452` — never beaten across the
  next 1,500 updates, so the run's genuinely best weights (update ~800) were
  never saved anywhere. Since 2026-09-05 `best` is selected on gauntlet Elo
  (`best_elo`, persisted in history so a resumed run doesn't re-save on the
  first eval); `best_avg10` is still tracked and printed, but selects nothing.

When judging a run, read gauntlet Elo first, then `update_board_avg`,
`update_cwin_avg` and the per-round action mix together — a policy that is
reward-hacking shows up there (narrow board, gold flowing somewhere other
than BUY) long before any placement number moves.

---

## Choice Mechanics

Every card effect where the real Hearthstone client gives the player a genuine
decision is resolved by the **agent**, not by RNG. Before 2026-09-04 none of
this existed: "Choose One" had no mechanic at all (six of the seven cards did
nothing when played), and every "choose a minion" effect called `rng.choice`.

`PlayerState.choice_pending` (an `env/player_state.py:PendingChoice`) pauses the
shopping phase until it is resolved. Two kinds:

- **`kind="target"`** — "Choose a friendly Demon", "Give another minion +3/+3",
  "Set another minion's stats to 50/50". `targets` holds legal **board indices**
  and the agent points at one with `CHOOSE_TARGET`. Needs no extra encoding: the
  board zone of the observation already describes every candidate.
- **`kind="option"`** — a Choose One battlecry, whose branches are whole
  *effects*. Each branch is rendered as a **pseudo-card** in the shop zone by
  `game_loop._choice_option_tokens`, so the existing 44-dim card encoder
  describes it: "+4 Attack and Windfury" becomes a token with `attack=4` and the
  windfury bit set. Resolved with `CHOOSE_OPTION`.

Rules that matter:

- **A choice with one legal target is applied immediately, never raised.**
  Pausing to offer a single option would burn one of the turn's 30 actions on a
  non-decision.
- **Choices queue** (`ps.choice_queue`). Brann doubles a Choose One battlecry
  into *two independent* choices, and a branch can itself raise a target choice
  (Sprightly Scarab: pick "Beast +1/+1 and Reborn", then pick which Beast).
- **A choice can never be dropped.** If the 30-action budget runs out mid-
  decision, `_force_resolve_choices` applies the resolver's own default. Letting
  it lapse would make "waste the action budget" a way to cancel an effect.
- Branch/target effects are dispatched by tag through
  `EffectHandler._apply_choice_effect`, so a Choose One branch and a plain
  targeted effect share the same small set of composable effects.
- **Still random, correctly:** "Get a random X" and Discover are random in real
  Hearthstone too. Mind Muck chooses *which friendly Demon* consumes, but *which
  Tavern minion* is consumed stays random.

Triple rewards are a real Discover as well (`env/triple_system.py`): three
Tier+1 cards into `ps.discover_pending`, agent picks. It used to take
`candidates[0]` unconditionally.

---

## Game Dynamics — Hearthstone Fidelity

Fixed 2026-09-04 after an audit against the real game. Each was a silent
divergence, not a known approximation:

| Rule | Was | Now |
|---|---|---|
| Tavern upgrade | Rerolled the shop (free refresh every level) and cleared `frozen` | Shop and freeze are untouched; new-tier minions appear on the next refresh |
| Upgrade cost | New tier started at `base - 1`, then round-start decay took another 1 | Starts at full `base`; only the round-start decay reduces it |
| FREEZE | Ended the turn immediately | A shop toggle; play continues (once per turn via the mask) |
| Combat first attacker | 50/50 coin flip | Side with **more minions** attacks first; coin flip only on an exact tie |
| Start-of-combat order | Rolled independently of attack order | Same precedence, decided once from pre-combat minion counts |
| Loss-damage fallback | `ps.tavern_tier + len(opp.board)` — the *loser's* tier and a minion *count* | `opp.tavern_tier + sum(opponent's surviving minion tiers)`, matching `CombatSide.win_damage` |
| `round_history` combats | Recorded only one side of each pairing | Records both |
| Triple-reward minion | Built without card_defs, so it arrived with no Taunt/Divine Shield/Activate | Keywords and `activate_cost` looked up, same as a shop purchase |

The first-attacker rule is the largest behavioural change: board **width** is
valuable in Battlegrounds partly *because* it buys the first attack, and a coin
flip gave a 4-minion board the same expected tempo as a 7-minion one — so
nothing in the environment ever taught the policy to go wide.

The upgrade fixes cut the other way: levelling is now strictly more expensive
(one gold more from the turn after every upgrade) and no longer comes with a
free shop refresh, so `level_rate` is expected to *fall* relative to pre-2026-09-04
runs. That is the intended correction, not a regression.

---

## Ghost Matchups

When an odd number of players is alive, one player is paired against a
**ghost**: the board of the **most recently eliminated** player, frozen at the
moment they died. Dead players keep their board (nothing in `env/game_loop.py`
ever assigns or clears `.board`, and the shopping phase iterates alive players
only), so `ps.board` on a dead player *is* their board at death.

Rules, as of 2026-09-03:

- The fight is **simulated for real** through `BGCombatSim`, like any other
  combat. Losing to a ghost costs full damage and can eliminate you.
- Winning deals damage to nobody — there is no player left to damage. This
  needs no special case now that `DAMAGE_DEALT_COEF` is 0.
- `Matchmaker.pair_players` returns `(recipient, dead_player_id)` with a
  **real** id. The `-1` sentinel now means only "nobody has died yet"
  (reachable only with an odd `n_players`) and keeps a degenerate free-win
  fallback. A ghost matchup is identified by the opponent not being `alive`.
- `next_opponent_id` **must** point at the ghost, so its board is visible
  during shopping through the normal `opponent_snapshots` machinery. Before
  this it was set to `None`, which encoded an *empty* opponent board and made
  `_board_shape_potential` fall back to 0.5. Simulating ghost fights while
  leaving that alone would add lethal risk while keeping the agent blind to
  it — strictly worse than the old free win.
- The recipient is chosen avoiding whoever got the ghost last round.

Until 2026-09-03 every ghost was short-circuited to an automatic win with zero
damage — a free `WIN_REWARD` on 7.5% of all pairings (3.92 per lobby-game,
concentrated at rounds 9–16). `Matchmaker.get_ghost` had been written for this
and was never called from anywhere.

**Do not over-estimate this fix.** Measured over 40 games, ghosts resolve to
93% win / 4% loss / 3% tie with mean `win_prob` 0.919 — dead players' boards
are weak *by selection*, since they died of them. The old auto-win was a
decent approximation of a ~92%-win fight. This is a correctness fix (a weak
board *can* now be punished, and the 4% losses cost ~13.7 damage), not a large
behavioural lever.

---

## Current Active Card Pool

Full card listings (275 minions, Tiers 1–7) are in **[CARDS.md](CARDS.md)**.
Read that file when working on `bg_card_pipeline.py`, the symbolic layer specs
(`DEATHRATTLE_SPECS`, `AURA_SPECS`, `TRIGGER_SPECS`), or any card-specific logic.

The Key Multiplier and Key Aura tables below are the subset needed most often
and are kept here for quick reference.

---

## Key Multiplier Cards (Highest Priority in Symbolic Layer)

These cards change how many times effects trigger. Always detect them first:

| Card | Effect | Detection text |
|---|---|---|
| Brann Bronzebeard (T5) | Battlecries trigger twice | "Battlecries trigger twice" |
| Titus Rivendare (T5) | Deathrattles trigger an extra time | "Deathrattles trigger an extra time" |
| Drakkari Enchanter (T5) | End-of-turn effects trigger twice | "end of turn effects trigger twice" |
| Balinda Stonehearth (T6) | Spells targeting friendly minions cast twice | "spells that target friendly minions cast twice" |

## Key Aura Cards (Affect Effective Stats Computation)

| Card | Aura | Target |
|---|---|---|
| Roaring Recruiter (T3) | +3/+1 per Dragon attack | attacking Dragon |
| Timecap'n Hooktail (T3) | +1 Attack per Tavern spell cast | all friendly minions |
| Cage Gnawer (T4) | +2/+1 per Beast attack | all friendly Beasts |
| Charging Czarina (T5) | +4 Attack per Tavern spell cast | friendly Divine Shield minions |
| Gunpowder Courier (T4) | +2 Attack per 5 Gold spent | friendly Pirates |
| Torrential Ruiner (T6) | +2/+3 per spell cast on a friendly Naga | all friendly minions |

The full auto-detected aura/multiplier sets live in `MULTIPLIER_CARDS` /
`AURA_CARDS` in `bg_card_pipeline.py` — re-review both after every card pool
refresh, since removed cards must be dropped from these sets manually.

## Mechanics Glossary

- **Rally**: triggers when this minion attacks during combat
- **Spellcraft**: a spell this minion can teach; buying it gives you that spell
- **Avenge (N)**: triggers after N friendly minions die in combat
- **Activate (N)**: a repeatable, once-per-turn minion ability costing N Gold to trigger — wired into the Action Space as type 8
- **Blood Gem**: Quilboar mechanic, +1/+1 buff item generated by Quilboar cards
- **Magnetic / Volumizer**: Mech mechanic — Magnetize attaches stats to a Mech
- **Pass**: Duos mechanic — passing a card to your teammate
- **Reborn**: resurrect with 1 HP after dying once in combat
- **Bonus Keyword**: one of Divine Shield, Venomous, Windfury, Taunt, Reborn — some cards grant random ones

---

## File Structure (to implement)

```
bg_agent/
├── CLAUDE.md                    ← this file
├── bg_card_pipeline.py          ← fetches + builds card definitions from HearthstoneJSON
├── bg_card_definitions.json     ← generated card DB (re-run after each patch)
├── symbolic/
│   ├── board_computer.py        ← SymbolicBoardComputer (auras, multipliers, DR types)
│   ├── shop_analyzer.py         ← per-card buy-phase value estimates
│   └── firestone_client.py      ← subprocess wrapper around Firestone sim
├── env/
│   ├── game_loop.py             ← BattlegroundsGame (8-player loop)
│   ├── player_state.py          ← PlayerState dataclass
│   ├── tavern_pool.py           ← shared card pool with draw/return
│   └── matchmaker.py            ← pairing logic with ghost support
├── agent/
│   ├── policy.py                ← Transformer-based policy + value network
│   ├── card_encoder.py          ← structured card → 44-dim feature vector
│   └── ppo.py                   ← PPO training loop with action masking
└── train.py                     ← entry point: self-play + population trainer
```

---

## Updating This File After a Patch

Run the pipeline to regenerate card definitions:

```bash
python bg_card_pipeline.py --output bg_card_definitions.json --stats
```

Then update `CARDS.md` by re-running the HearthstoneJSON scraper and filtering
on `isBattlegroundsPoolMinion: true`. Cards without this flag are retired and
should not appear in the symbolic layer.

## Context Logging & Git Commits

See the **MANDATORY END-OF-SESSION CHECKLIST** at the top of this file.