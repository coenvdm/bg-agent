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
                       freeze, scout, manage economy
```

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

The implemented space is 10 action types, each optionally carrying ONE pointer
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
6 HERO_POWER           # no pointer
7 END_TURN             # no pointer
8 ACTIVATE(board_idx)  # pointer: board slot -- the minion's own Activate (N)
9 REORDER(board_idx)   # pointer: board slot 1-6 -- moves that minion to the
                       #   FRONT of the board
```

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
   the source are 0.5/-0.3/0.05/0.05/0.15 respectively — see the module
   constants block for why everything in this component and (2) is scaled
   down 0.30x, and quote the *scaled* number when reasoning about magnitude):
   ```python
   r  =  0.15   if result == "win"  else 0.0     # WIN_REWARD
   r += -0.09   if result == "loss" else 0.0     # LOSS_PENALTY
   r += -0.015 * (damage_taken / max_health)     # DAMAGE_TAKEN_COEF
   r +=  0.015 * (damage_dealt  / max_health)    # DAMAGE_DEALT_COEF
   r += (prev_rank - cur_rank) * 0.045           # RANK_DELTA_COEF
   ```
   There is no flat per-round survival bonus — one existed historically
   (`+0.1` unconditionally) but was removed 2026-08-31: it was passive income
   that fired merely for being alive and diluted the placement signal
   without rewarding any actual decision (see `CONTEXT.md`).
2. **`_end_of_turn_reward`** — fired on END_TURN/FREEZE: `-HAND_PENALTY_COEF`
   (`0.024`) per card left in hand, and `-GOLD_PENALTY_COEF * gold *
   gold_scale` for unspent gold, `GOLD_PENALTY_COEF = 0.015`.
   `gold_scale` is **V-shaped**, retuned 2026-09-02: it fades linearly 1.0 →
   `GOLD_SCALE_FLOOR = 0.2` by `GOLD_SCALE_FADE_ROUND = 13` (early/mid-game
   gold retention is sometimes correct — saving to level), then *ramps back
   up* at `GOLD_SCALE_LATE_RAMP = 0.13`/round to `GOLD_SCALE_LATE_CEIL = 1.5`
   by round 23. The old schedule was flat at `0.2` from round 13 to the end
   of the game however long it ran, which priced a full 10-gold purse at
   `-0.03`/turn against `WIN_REWARD = 0.15` — and a live-run trace showed a
   trained policy duly banking all 10 gold every round from round 8 onward
   for the rest of an 18-round game while it stopped buying/leveling
   entirely (see `CONTEXT.md` 2026-09-02). Late-game idle gold now costs up
   to `-0.225`/turn (round 23+, gold=10) — more than a full `WIN_REWARD`.
   Rounds 1-12 are numerically unchanged from the old schedule (see
   `_gold_penalty_scale` in `env/game_loop.py`).
3. **Unified potential-based shaping** (`_apply_potential_shaping`) — a
   single potential Φ(s) ∈ [0, 1] (Ng, Harada & Russell 1999), paid out at
   **every** shopping action (BUY/SELL/PLACE/REROLL/FREEZE/LEVEL_UP/
   HERO_POWER/END_TURN/ACTIVATE/REORDER) plus once per round right after
   combat resolves:
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
     BOARD_SHAPE_STATS_SATURATION)`, where `value` = total effective
     attack+health (`symbolic.board_computer._board_power`) plus a flat
     `+3` per Divine Shield/Taunt/Reborn/Windfury instance
     (`BOARD_STATS_KEYWORD_BONUS`) and a flat `+5` if any tribe reaches the
     CLAUDE.md "synergistic" threshold of 4+ (`BOARD_STATS_SYNERGY_BONUS`).
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
     percentile table. An empty-board penalty (`-EMPTY_BOARD_PENALTY` =
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
the gold-penalty schedule above), re-measure the dense-sum-per-player-game
vs. `FINAL_PLACEMENT_REWARD` balance on the real seat mix and confirm it
hasn't drifted back toward dominating; the 2026-09-02 gold-schedule retune
was checked this way (mean dense_sum stayed near 0, well inside the ±4
placement span, on a 30-game/240-player-game sample — see `CONTEXT.md`).
When retuning leveling incentives, watch `level_rate` *and* `board_size`/
placement together (not `level_rate` alone) — a naive leveling incentive can
reproduce the same kind of degenerate policy a flat `board_size` reward
caused historically (agent chases the proxy metric at the expense of what it
actually stands for).

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