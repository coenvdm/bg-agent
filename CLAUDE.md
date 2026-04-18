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

```python
class Action(Enum):
    BUY(shop_idx)           # 0-6
    SELL(board_idx)         # 0-6
    PLAY(hand_idx, pos)     # play from hand to board position
    REORDER(from_idx, to)   # reposition on board
    LEVEL_UP                # spend gold to upgrade tavern tier
    FREEZE                  # freeze shop for next turn
    REFRESH                 # spend 1 gold to reroll shop
    HERO_POWER              # use hero power (hero-specific)
    END_TURN
```

Always mask invalid actions (no gold to buy, board full, etc.).

---

## Reward Shaping

```python
def round_reward(player, result):
    r  = +0.5  if result.win   else 0.0
    r += -0.3  if result.loss  else 0.0
    r += -0.05 * damage_taken          # normalized by max health
    r += +0.05 * damage_dealt
    r += (4 - current_rank) * 0.1     # rank in lobby (1=best)
    return r

FINAL_PLACEMENT_REWARD = {1:+4.0, 2:+2.0, 3:+1.0, 4:0.0,
                           5:-1.0, 6:-2.0, 7:-3.0, 8:-4.0}
```

---

## Current Active Card Pool

Full card listings (264 minions, Tiers 1–7) are in **[CARDS.md](CARDS.md)**.
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
| Young Murk-Eye (T6) | Triggers adjacent battlecry each turn | "trigger the Battlecry of an adjacent" |

## Key Aura Cards (Affect Effective Stats Computation)

| Card | Aura | Target |
|---|---|---|
| Twilight Watcher (T5) | +1/+3 per Dragon attack | all friendly Dragons |
| Roaring Recruiter (T3) | +3/+1 per Dragon attack | attacking Dragon |
| Shore Marauder (T6) | +1/+1 passive | Pirates and Elementals |
| Lord of the Ruins (T6) | +2/+1 after Demon deals damage | all other friendlies |
| Hardy Orca (T3) | +1/+1 when this takes damage | all other friendlies |
| Iridescent Skyblazer (T5) | +3/+1 when Beast takes damage | another friendly Beast |

## Mechanics Glossary

- **Rally**: triggers when this minion attacks during combat
- **Spellcraft**: a spell this minion can teach; buying it gives you that spell
- **Avenge (N)**: triggers after N friendly minions die in combat
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