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

Five components combine into the total reward:

1. **`compute_round_reward`** — dense per-combat-round signal:
   ```python
   r  =  0.5  if result == "win"  else 0.0
   r += -0.3  if result == "loss" else 0.0
   r += -0.05 * (damage_taken / max_health)
   r +=  0.05 * (damage_dealt  / max_health)
   r += (prev_rank - cur_rank) * 0.15
   r +=  0.1   # flat survival bonus for being alive this round
   ```
2. **`_end_of_turn_reward`** — fired on END_TURN: `-0.08` per card left in
   hand, and `-0.05 * gold * gold_scale` for unspent gold, where `gold_scale`
   fades from 1.0 down to a floor of 0.2 by round 16+.
3. **Potential-based board-strength shaping** (`_apply_board_shape`) — on
   every PLACE/SELL, runs a 30-trial Monte Carlo combat sim to estimate
   win-probability Φ(s), and pays `α · (γ · Φ(s') − Φ(s))` with
   `BOARD_SHAPE_ALPHA = 0.20`, `BOARD_SHAPE_GAMMA = 1.0`, applied identically
   (unclipped) to both PLACE and SELL — an earlier version clipped PLACE to
   `max(0, shaped)` while leaving SELL unclipped, which broke the
   policy-invariance guarantee this shaping relies on (Ng et al. 1999) and
   biased the policy toward selling; fixed 2026-08-30, see `CONTEXT.md`.
   `ps.phi_board` resets at the start of every round so shaping never leaks
   value across turns. An empty board penalty (`-0.30`) fires on the SELL
   action that empties the board.
4. **Potential-based leveling shaping** (`_apply_tier_shape`) — on a
   successful LEVEL_UP, pays `α_tier · (γ_tier · Φ_tier(s') − Φ_tier(s))` with
   `TIER_SHAPE_ALPHA = 0.10`, `TIER_SHAPE_GAMMA = 1.0`, where
   `Φ_tier(s) = min(1.0, tavern_tier / _expected_tier_for_round(round_num))`.
   Added 2026-08-31 because LEVEL_UP previously had no reward term at all,
   so PLACE (immediate board-shape payout for the same gold) strictly
   dominated it — see `CONTEXT.md`. The `min(1.0, …)` cap means leveling past
   the round's rough curve pays nothing extra, and `ps.phi_tier` resets at
   round start like `ps.phi_board`, so it only ever rewards closing a real
   gap, never repeat-farming or racing ahead of what's useful.
5. **`FINAL_PLACEMENT_REWARD`** — fires immediately at the moment a player
   is eliminated (not at game end); survivors receive it at game end instead.
   ```python
   FINAL_PLACEMENT_REWARD = {1:+4.0, 2:+2.0, 3:+1.0, 4:0.0,
                              5:-1.0, 6:-2.0, 7:-3.0, 8:-4.0}
   ```

PPO uses `gamma=0.997`, `gae_lambda=0.95` (raised from `gamma=0.99` so the
final placement reward doesn't decay away before reaching early-round
decisions).

The board-shaping constants (`BOARD_SHAPE_ALPHA/GAMMA`, the empty-board
penalty's firing point) went through several exploit-driven fixes on
2026-04-19 — see `CONTEXT.md` for the history before retuning them again.
When retuning `TIER_SHAPE_ALPHA`, watch `level_rate` *and* `board_size`/
placement together (not `level_rate` alone) — a naive leveling incentive can
reproduce the same kind of degenerate policy the old flat `board_size`
reward caused (agent chases the proxy metric at the expense of what it
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