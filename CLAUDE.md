# Hearthstone Battlegrounds — RL Agent

You are helping build a Reinforcement Learning agent that plays Hearthstone
Battlegrounds competitively. This file defines the architecture, symbolic
feature layer, and card definitions the agent uses.

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

Source: HearthstoneJSON API, filtered to `isBattlegroundsPoolMinion: true`.
Scraped: 2026-03-14. **264 minions** across 7 tiers.

Format: `Name (ATK/HP) [Tribe]: Card text`

> **Important for symbolic layer**: cards below are the ground truth for
> building `DEATHRATTLE_SPECS`, `AURA_SPECS`, and `TRIGGER_SPECS` dicts.
> For each card, identify:
> - trigger type (battlecry / deathrattle / end_of_turn / start_of_combat /
>   on_sell / on_buy / avenge / passive / rally / spellcraft)
> - effect target (self / adjacent / tribe / all_friendly / random_enemy)
> - effect duration (permanent / this_combat / this_game / instant)
> - whether it scales with board state (`for each`, `for every`)

---

### Tier 1 (22 minions)

- Annoy-o-Tron (1/2) [MECHANICAL]: Taunt. Divine Shield.
- Aureate Laureate (1/1) [PIRATE]: Divine Shield. Battlecry: Make this minion Golden.
- Bubble Gunner (2/3) [MURLOC]: Battlecry: Gain a random Bonus Keyword.
- Buzzing Vermin (1/1) [BEAST]: Taunt. Deathrattle: Summon a 2/2 Beetle.
- Cord Puller (1/1) [MECHANICAL]: Divine Shield. Deathrattle: Summon a 1/1 Microbot.
- Crackling Cyclone (2/1) [ELEMENTAL]: Divine Shield. Windfury.
- Dune Dweller (3/2) [ELEMENTAL]: Battlecry: Give Elementals in the Tavern +1/+1 this game.
- Flighty Scout (3/3) [MURLOC]: Start of Combat: If this minion is in your hand, summon a copy of it.
- Harmless Bonehead (1/1) [UNDEAD]: Deathrattle: Summon two 1/1 Skeletons.
- Minted Corsair (1/3) [PIRATE]: When you sell this, get a Tavern Coin.
- Misfit Dragonling (2/1) [DRAGON]: Start of Combat: Gain stats equal to your Tier.
- Ominous Seer (2/1) [DEMON/NAGA]: Battlecry: The next Tavern spell you buy costs (1) less.
- Passenger (2/2): The first time your team Passes each turn, gain +1/+2.
- Picky Eater (1/1) [DEMON]: Battlecry: Consume a random minion in the Tavern to gain its stats.
- Razorfen Geomancer (2/1) [QUILBOAR]: Battlecry: Get 2 Blood Gems.
- Risen Rider (2/1) [UNDEAD]: Taunt. Reborn.
- Rot Hide Gnoll (1/4) [UNDEAD]: Has +1 Attack for each friendly minion that died this combat.
- Surf n' Surf (1/1) [NAGA/BEAST]: Spellcraft: Give a minion "Deathrattle: Summon a 3/2 Crab" until next turn.
- Swampstriker (1/5) [MURLOC]: Windfury. After you summon a Murloc, gain +1 Attack.
- Tusked Camper (2/3) [QUILBOAR]: Rally: This plays a Blood Gem on itself.
- Twilight Hatchling (1/1) [DRAGON]: Deathrattle: Summon a 3/3 Whelp that attacks immediately.
- Wrath Weaver (1/4) [DEMON]: After you play a Demon, deal 1 damage to your hero and gain +2/+1.

---

### Tier 2 (38 minions)

- Blue Volumizer (1/3) [MECHANICAL]: Magnetic. The first time this is played or Magnetized, your Volumizers have +3 Health this game (wherever they are).
- Briarback Bookie (3/3) [QUILBOAR]: At the end of your turn, get a Blood Gem.
- Defiant Shipwright (2/5) [PIRATE]: Whenever this gains Attack from other sources, gain +1 Health.
- Embalming Expert (3/2) [UNDEAD]: After the Tavern is Refreshed, give its right-most minion +2 Attack and Reborn.
- Eternal Knight (4/1) [UNDEAD]: Has +4/+1 for each friendly Eternal Knight that died this game (wherever this is).
- Expert Aviator (3/4) [MURLOC]: Rally: Give the left-most minion in your hand +1/+1 and summon it for this combat only.
- Fire Baller (4/3) [ELEMENTAL]: When you sell this, give your minions +1 Attack. Improve your future Ballers.
- Forest Rover (1/1) [BEAST]: Battlecry: Your Beetles have +2/+1 this game. Deathrattle: Summon a 2/2 Beetle.
- Freedealing Gambler (3/3) [PIRATE]: This minion sells for 3 Gold.
- Friendly Saloonkeeper (3/4): Battlecry: Your teammate gets a Tavern Coin.
- Gathering Stormer (5/1) [ELEMENTAL]: When you sell this, your teammate gains 1 Gold. (Improves each turn!)
- Generous Geomancer (1/1) [QUILBOAR]: Deathrattle: You and your teammate each get a Blood Gem.
- Green Volumizer (3/3) [MECHANICAL]: Magnetic. The first time this is played or Magnetized, your Volumizers have +1/+1 this game (wherever they are).
- Humming Bird (1/4) [BEAST]: Start of Combat: For the rest of this combat, your Beasts have +1 Attack.
- Intrepid Botanist (3/4): Choose One — Your Tavern spells give an extra +1 Attack this game; or +1 Health.
- Irate Rooster (3/4) [BEAST]: Start of Combat: Deal 1 damage to adjacent minions and give them +4 Attack.
- Lava Lurker (2/5) [NAGA]: The first Spellcraft spell played from hand on this each turn is permanent. (1 left!)
- Mechagnome Interpreter (2/3) [MECHANICAL]: Whenever you play or Magnetize a Mech, give it +2/+1.
- Mind Muck (3/2) [DEMON]: Battlecry: Choose a friendly Demon. It consumes a minion in the Tavern to gain its stats.
- Moon-Bacon Jazzer (1/3) [QUILBOAR]: Battlecry: Your Blood Gems give an extra +1 Health this game.
- Nerubian Deathswarmer (1/4) [UNDEAD]: Battlecry: Your Undead have +1 Attack this game (wherever they are).
- Oozeling Gladiator (2/2): Battlecry: Get two Slimy Shields that give +1/+1 and Taunt.
- Patient Scout (1/1): When you sell this, Discover a Tier 1 minion. (Improves each turn!)
- Prophet of the Boar (2/3): Taunt. After you play a Quilboar, get a Blood Gem.
- Red Volumizer (3/1) [MECHANICAL]: Magnetic. The first time this is played or Magnetized, your Volumizers have +3 Attack this game (wherever they are).
- Reef Riffer (3/2) [NAGA]: Spellcraft: Give a minion stats equal to your Tier until next turn.
- Saltscale Honcho (5/2) [MURLOC]: After you summon a Murloc, give a friendly Murloc other than it +2 Health.
- Sellemental (3/3) [ELEMENTAL]: When you sell this, get a 3/3 Elemental.
- Shell Collector (4/3) [NAGA]: Battlecry: Get a Tavern Coin.
- Sleepy Supporter (3/4) [DRAGON]: Rally: Give another random friendly Dragon +2/+3.
- Snow Baller (3/4) [ELEMENTAL]: When you sell this, give your minions +1 Health. Improve your future Ballers.
- Soul Rewinder (4/1) [DEMON]: After your hero takes damage, rewind it and give this +1 Health.
- Surfing Sylvar (1/2) [PIRATE]: At the end of your turn, give adjacent minions +1 Attack. Repeat for each friendly Golden minion.
- Tad (2/2) [MURLOC]: When you sell this, get a random Murloc.
- Tarecgosa (3/3) [DRAGON]: This permanently keeps Bonus Keywords and stats gained in combat.
- Wanderer Cho (4/3): One Pass each turn is free. (1 left!)
- Whelp Watcher (1/4) [DRAGON]: Rally: Summon a 3/3 Whelp to attack the target first.
- Worgen Executive (2/5): Rally: After the Tavern is Refreshed this game, give its right-most minion +1/+1.

---

### Tier 3 (47 minions)

- Amber Guardian (3/2) [DRAGON]: Taunt. Start of Combat: Give another friendly Dragon +2/+2 and Divine Shield.
- Annoy-o-Module (2/4) [MECHANICAL]: Magnetic. Divine Shield. Taunt.
- Anub'arak, Nerubian King (3/2) [UNDEAD]: Deathrattle: Your Undead have +1 Attack this game (wherever they are).
- Aranasi Alchemist (1/2) [DEMON]: Taunt. Reborn. Deathrattle: Give minions in the Tavern +1 Health this game.
- Auto Accelerator (3/3) [MECHANICAL]: Battlecry: Get a random Magnetic Volumizer.
- Bassgill (5/2) [MURLOC]: Deathrattle: Summon the highest-Health Murloc from your hand for this combat only.
- Bird Buddy (2/4): Avenge (1): Give your Beasts +1/+1.
- Blue Whelp (1/5) [DRAGON]: Rally: Your Tavern spells give an extra +1 Health this game.
- Bottom Feeder (3/4) [MURLOC]: At the end of your turn, you and your teammate each get a random Tier 1 card.
- Bountiful Bedrock (3/4) [ELEMENTAL]: At the end of every 2 turns, get a random Elemental. (2 turns left!)
- Briarback Drummer (5/2) [QUILBOAR]: Battlecry: Get a Blood Gem Barrage.
- Briny Bootlegger (4/2) [PIRATE]: Deathrattle: Get a Tavern Coin.
- Budding Greenthumb (1/5): Avenge (3): Give adjacent minions +2/+2 permanently.
- Cadaver Caretaker (3/3) [UNDEAD]: Deathrattle: Summon three 1/1 Skeletons.
- Coldlight Diver (1/1) [MURLOC]: Battlecry and Deathrattle: Get a random Tier 1 Tavern spell.
- Deadly Spore (1/1): Venomous.
- Deep-Sea Angler (2/3) [NAGA]: Spellcraft: Give a minion +2/+6 and Taunt until next turn.
- Doting Dracthyr (4/3) [DRAGON]: At the end of your turn, give your teammate's minions +1 Attack.
- Felemental (3/3) [ELEMENTAL/DEMON]: Battlecry: Give minions in the Tavern +2/+1 this game.
- Felhorn (2/4) [DEMON/BEAST]: Battlecry: Give your other Demons and Beasts +1/+2 and deal 1 damage to them, twice.
- Gemsplitter (2/1) [QUILBOAR]: Divine Shield. After a friendly minion loses Divine Shield, get a Blood Gem.
- Goldgrubber (3/2) [PIRATE]: At the end of your turn, gain +3/+2 for each friendly Golden minion.
- Handless Forsaken (2/1) [UNDEAD]: Deathrattle: Summon a 2/1 Hand with Reborn.
- Hardy Orca (1/6) [BEAST]: Taunt. Whenever this takes damage, give your other minions +1/+1.
- Jelly Belly (2/3) [UNDEAD]: After a friendly minion is Reborn, gain +2/+3 permanently.
- Jumping Jack (3/4) [ALL]: After the first time this is sold, Pass it.
- Orc-estra Conductor (4/4): Battlecry: Give a minion +2/+2 (Improved by each Orc-estra your team has played this game).
- Peggy Sturdybone (2/1) [PIRATE]: Whenever a card is added to your hand, give another friendly Pirate +2/+1.
- Plunder Pal (2/2) [PIRATE]: At the start of your turn, you and your teammate each gain 1 Gold.
- Prickly Piper (5/1) [QUILBOAR]: Deathrattle: Your Blood Gems give an extra +1 Attack this game.
- Profound Thinker (2/3) [NAGA]: Rally: Get a random Spellcraft spell. This casts a copy of it (targets this if possible).
- Puddle Prancer (4/4) [MURLOC]: After this is Passed, gain +4/+4.
- Rampager (8/8) [BEAST]: Rally: Deal 1 damage to your other minions.
- Relentless Deflector (5/4) [MECHANICAL]: Has Taunt while this has Divine Shield. Avenge (3): Gain Divine Shield.
- Roadboar (3/4) [QUILBOAR]: Rally: Get 2 Blood Gems.
- Roaring Recruiter (2/8) [DRAGON]: Whenever another friendly Dragon attacks, give it +3/+1.
- Scourfin (3/3) [MURLOC]: Deathrattle: Give a random minion in your hand +5/+5.
- Shoalfin Mystic (4/4) [MURLOC]: When you sell this, your Tavern spells give an extra +1/+1 this game.
- Sprightly Scarab (3/1) [BEAST]: Choose One — Give a Beast +1/+1 and Reborn; or +4 Attack and Windfury.
- Technical Element (5/6) [ELEMENTAL/MECHANICAL]: Magnetic. Can Magnetize to both Mechs and Elementals.
- The Glad-iator (3/3) [NAGA]: Divine Shield. Whenever you cast a spell, gain +1 Attack.
- Timecap'n Hooktail (1/4) [DRAGON/PIRATE]: Whenever you cast a Tavern spell, give your minions +1 Attack.
- Underhanded Dealer (3/3) [PIRATE]: After you gain Gold, gain +1/+2.
- Waveling (6/1) [ELEMENTAL]: Deathrattle: After the Tavern is Refreshed this game, give its right-most minion +3/+3.
- Wheeled Crewmate (6/3) [PIRATE]: Deathrattle: Reduce the Cost of upgrading your team's Taverns by (1).
- Wildfire Elemental (6/3) [ELEMENTAL]: After this attacks and kills a minion, deal excess damage to an adjacent minion.
- Zesty Shaker (5/6) [NAGA]: Once per turn, when a Spellcraft spell is played on this, get a new copy of it.

---

### Tier 4 (56 minions)

- Accord-o-Tron (5/5) [MECHANICAL]: Magnetic. At the start of your turn, gain 1 Gold.
- Apprentice of Sefin (4/4) [MURLOC]: At the end of your turn, gain +2/+2 and a random Bonus Keyword.
- Blade Collector (3/2) [PIRATE]: Also damages the minions next to whomever this attacks.
- Bonker (2/7) [QUILBOAR]: Rally: This plays 2 Blood Gems on all your other minions.
- Bream Counter (4/4) [MURLOC]: While this is in your hand, after you play a Murloc, gain +4/+4.
- Conveyor Construct (5/2) [MECHANICAL]: Deathrattle: Get a random Magnetic Volumizer.
- Daggerspine Thrasher (3/5) [NAGA]: Whenever you cast a spell, gain Divine Shield, Windfury, or Venomous until next turn.
- Deep Blue Crooner (2/3) [NAGA]: Spellcraft: Give a minion +2/+3 until next turn. Improve your future Deep Blues.
- Devout Hellcaller (2/2) [DEMON]: After another friendly Demon deals damage, gain +1/+2 permanently.
- En-Djinn Blazer (4/4) [ELEMENTAL]: Battlecry: After the Tavern is Refreshed this game, give its right-most minion +7/+7.
- Fearless Foodie (2/4) [QUILBOAR]: Choose One — Your Blood Gems give an extra +1/+1 this game; or Get 4 Blood Gems.
- Feisty Freshwater (6/4) [ELEMENTAL]: Deathrattle: You and your teammate each gain two free Refreshes.
- Flaming Enforcer (4/5) [ELEMENTAL/DEMON]: At the end of your turn, consume the highest-Health minion in the Tavern to gain its stats.
- Friendly Geist (6/3) [UNDEAD]: Deathrattle: Your Tavern spells give an extra +1 Attack this game.
- Geomagus Roogug (4/6) [QUILBOAR]: Divine Shield. Whenever a Blood Gem is played on this, this plays a Blood Gem on a different friendly minion.
- Gormling Gourmet (4/3) [MURLOC]: Taunt. Battlecry and Deathrattle: Get a Seafood Stew.
- Grave Narrator (2/7) [UNDEAD]: Avenge (3): Your teammate gets a random minion of their most common type.
- Grease Bot (2/4) [MECHANICAL]: Divine Shield. After a friendly minion loses Divine Shield, give it +2/+2 permanently.
- Gunpowder Courier (2/6) [PIRATE]: Whenever you spend 6 Gold, give your Pirates +2 Attack. (6 Gold left!)
- Heroic Underdog (1/10): Stealth. Rally: Gain the target's Attack.
- Humon'gozz (5/5): Divine Shield. Your Tavern spells give an extra +1/+2.
- Hunting Tiger Shark (3/5) [BEAST]: Battlecry: Discover a Beast.
- Ichoron the Protector (3/1) [ELEMENTAL]: Divine Shield. Whenever you play an Elemental, give it Divine Shield until next turn.
- Imposing Percussionist (4/4) [DEMON]: Battlecry: Discover a Demon. Deal damage to your hero equal to its Tier.
- Industrious Deckhand (3/5) [PIRATE]: At the start of your turn, get 2 Tavern Coins.
- Lovesick Balladist (3/2) [PIRATE]: Battlecry: Give a Pirate +1 Health. (Improved by each Gold you spent this turn!)
- Malchezaar, Prince of Dance (5/4) [DEMON]: Two Refreshes each turn cost Health instead of Gold. (2 left!)
- Mantid King (3/3): After your team Passes, randomly gain Venomous, Taunt, or Divine Shield until next turn.
- Marquee Ticker (1/5) [MECHANICAL]: At the end of your turn, get a random Tavern spell.
- Mirror Monster (4/4) [ALL]: When you buy or Discover this, get an extra copy and Pass it.
- Monstrous Macaw (5/4) [BEAST]: Rally: Trigger your left-most Deathrattle (except this minion's).
- Persistent Poet (2/3) [DRAGON]: Divine Shield. Adjacent Dragons permanently keep Bonus Keywords and stats gained in combat.
- Plankwalker (6/4) [UNDEAD/PIRATE]: Whenever you cast a Tavern spell, give three random friendly minions +2/+1.
- Private Chef (5/4) [NAGA]: Spellcraft: Choose a minion. Get a different random minion of its type, then Pass it.
- Prized Promo-Drake (1/1) [DRAGON]: Start of Combat: Give your Dragons +4/+4.
- Prosthetic Hand (3/1) [UNDEAD/MECHANICAL]: Magnetic. Reborn. Can Magnetize to Mechs or Undead.
- Razorfen Flapper (5/3) [QUILBOAR]: Deathrattle: Get a Blood Gem Barrage.
- Refreshing Anomaly (4/5) [ELEMENTAL]: Battlecry: Gain 2 free Refreshes.
- Runed Progenitor (2/8) [BEAST]: Avenge (3): Your Beetles have +2/+2 this game. Deathrattle: Summon a 2/2 Beetle.
- Rylak Metalhead (5/3) [BEAST]: Taunt. Deathrattle: Trigger the Battlecry of an adjacent minion.
- San'layn Scribe (4/4) [UNDEAD]: Has +4/+4 for each of your team's San'layn Scribes that died this game (wherever this is).
- Shifty Snake (6/1) [BEAST]: Deathrattle: Your teammate gets a random Deathrattle minion.
- Silent Enforcer (6/2) [DEMON]: Taunt. Deathrattle: Deal 2 damage to all minions (except friendly Demons).
- Sin'dorei Straight Shot (3/4): Divine Shield. Windfury. Rally: Remove Reborn and Taunt from the target.
- Soulsplitter (4/2) [UNDEAD]: Reborn. Start of Combat: Give a friendly Undead Reborn.
- Spirit Drake (1/8) [UNDEAD/DRAGON]: Avenge (3): Get a random Tavern spell.
- Stellar Freebooter (8/4) [PIRATE]: Taunt. Deathrattle: Give a friendly minion Health equal to this minion's Attack.
- Tavern Tempest (2/2) [ELEMENTAL]: Battlecry: Get a random Elemental.
- Tortollan Blue Shell (3/6): If you lost your last combat, this minion sells for 5 Gold.
- Trench Fighter (6/6) [QUILBOAR]: At the end of your turn, get a Gem Confiscation.
- Trigore the Lasher (9/3) [BEAST]: Whenever another friendly Beast takes damage, gain +2 Health permanently.
- Tunnel Blaster (3/7): Taunt. Deathrattle: Deal 3 damage to all minions.
- Wannabe Gargoyle (9/1) [DRAGON]: Reborn. This is Reborn with full Attack.
- Waverider (2/8) [NAGA]: Spellcraft: Give a minion +2/+2. If it's a Naga, also give it Windfury until next turn.
- Weary Mage (5/1) [NAGA]: Spellcraft: Give a minion +2/+2. If it's a Naga, also give it Reborn until next turn.
- Witchwing Nestmatron (3/5): Avenge (3): Get a random Battlecry minion.

---

### Tier 5 (55 minions)

- Air Revenant (3/6) [ELEMENTAL]: After you spend 7 Gold, get Easterly Winds. (7 left!)
- Ashen Corruptor (6/6) [DEMON]: After your hero takes damage, rewind it and give minions in the Tavern +1/+1 this turn.
- Azsharan Veteran (4/5) [NAGA]: Spellcraft: Give your minions +2/+1 for each different Bonus Keyword in your warband.
- Bile Spitter (1/10) [MURLOC]: Venomous. Rally: Give another friendly Murloc Venomous.
- Brann Bronzebeard (2/4): Your Battlecries trigger twice.
- Burgeoning Whelp (5/5) [DRAGON]: Battlecry and Deathrattle: Your Whelps have +3/+3 this game.
- Cannon Corsair (3/7) [PIRATE]: After you gain Gold, give your Pirates +1/+1.
- Carapace Raiser (6/3) [UNDEAD]: Deathrattle: Get a Haunted Carapace.
- Catacomb Crasher (6/10) [UNDEAD]: Whenever you would summon a minion that doesn't fit in your warband, give your minions +2/+1 permanently.
- Champion of the Primus (2/10) [UNDEAD]: Avenge (2): Your Undead have +1 Attack this game (wherever they are).
- Charging Czarina (4/2) [MECHANICAL]: Divine Shield. Whenever you cast a Tavern spell, give your minions with Divine Shield +4 Attack.
- Costume Enthusiast (4/5) [MURLOC]: Divine Shield. Start of Combat: Gain the Attack of the highest-Attack minion in your hand.
- Drakkari Enchanter (1/5): Your end of turn effects trigger twice.
- Eternal Tycoon (2/9) [UNDEAD]: Avenge (5): Summon an Eternal Knight. It attacks immediately.
- Felboar (2/6) [DEMON/QUILBOAR]: After you cast 3 spells, consume a minion in the Tavern to gain its stats. (3 left!)
- Firescale Hoarder (5/5) [NAGA/DRAGON]: Battlecry and Deathrattle: Get a Shiny Ring.
- Furious Driver (3/3) [DEMON]: Battlecry: Your other Demons each consume a minion in the Tavern to gain its stats.
- Gentle Djinni (4/5) [ELEMENTAL]: Taunt. Deathrattle: Get a random Elemental.
- Glowscale (4/6) [NAGA]: Taunt. Spellcraft: Give a minion Divine Shield until next turn.
- Hackerfin (5/3) [MURLOC]: Battlecry: Give your other minions +1/+2. (Improved by each different Bonus Keyword in your warband!)
- Insatiable Ur'zul (4/6) [DEMON]: Taunt. After you play a Demon, consume a random minion in the Tavern to gain its stats.
- Iridescent Skyblazer (3/8) [BEAST]: Whenever a friendly Beast takes damage, give a friendly Beast other than it +3/+1 permanently.
- Junk Jouster (6/5) [MECHANICAL]: Whenever a minion is Magnetized to this, give your minions +3/+2.
- Leeroy the Reckless (6/2): Deathrattle: Destroy the minion that killed this.
- Magnanimoose (5/2) [BEAST]: Deathrattle: Summon a copy of a minion from your teammate's warband. Set its Health to 1 (except Magnanimoose).
- Man'ari Messenger (9/6) [DEMON]: Battlecry: Minions in your team's Taverns have +1/+1 this game.
- Metal Dispenser (2/8) [MECHANICAL]: Divine Shield. Avenge (3): Magnetize a random Volumizer to this. Get a copy of it.
- Mrglin' Burglar (8/6) [MURLOC]: After you play a Murloc, give a friendly minion and a minion in your hand +4/+4.
- Nalaa the Redeemer (5/7): Whenever you cast a Tavern spell, give a friendly minion of each type +3/+2.
- Niuzao (7/6) [BEAST]: Rally: Deal damage equal to this minion's Attack to a random enemy minion other than the target.
- Nomi, Kitchen Nightmare (4/4): After you play an Elemental, give Elementals in the Tavern +2/+2 this game.
- Photobomber (6/6) [MECHANICAL]: Deathrattle: Deal 2 damage to the highest-Health enemy minion. (Improved by Tavern spells you've cast this game!)
- Primalfin Lookout (3/2) [MURLOC]: Battlecry: If you control another Murloc, Discover a Murloc.
- Razorfen Vineweaver (5/5) [QUILBOAR]: Rally: This plays 3 permanent Blood Gems on itself.
- Rodeo Performer (3/4): Battlecry: Discover a Tavern spell.
- Selfless Sightseer (6/2) [DRAGON]: Battlecry: Increase your team's maximum Gold by (1).
- Shadowdancer (5/4) [DEMON]: Taunt. Deathrattle: Get a Staff of Enrichment.
- Showy Cyclist (4/3) [NAGA]: Deathrattle: Give your Naga +2/+2. (Improved by every 4 spells you've cast this game!)
- Silithid Burrower (5/4) [BEAST]: Deathrattle: Give your Beasts +1/+1. Avenge (1): Improve this by +1/+1 permanently.
- Spiked Savior (8/2) [BEAST]: Taunt. Reborn. Deathrattle: Give your minions +1 Health and deal 1 damage to them.
- Storm Splitter (5/5) [NAGA]: Once per turn, after you Pass a Tavern spell, get a new copy of it.
- Stuntdrake (14/5) [DRAGON]: Avenge (3): Give this minion's Attack to two different friendly minions.
- Support System (4/5) [MECHANICAL]: At the end of your turn, give a minion in your teammate's warband Divine Shield.
- Thieving Rascal (5/6) [PIRATE]: At the start of your turn, gain 1 Gold. Repeat for each friendly Golden minion.
- Three Lil' Quilboar (3/3) [QUILBOAR]: Deathrattle: This plays 3 Blood Gems on all your Quilboar.
- Titus Rivendare (1/7): Your Deathrattles trigger an extra time.
- Tranquil Meditative (3/8) [NAGA]: Spellcraft: Your Tavern spells give an extra +1/+1 this game.
- Turquoise Skitterer (4/4) [BEAST]: Deathrattle: Your Beetles have +4/+4 this game. Summon a 2/2 Beetle.
- Twilight Broodmother (7/4) [DRAGON]: Deathrattle: Summon 2 Twilight Hatchlings. Give them Taunt.
- Twilight Watcher (3/7) [DRAGON]: Whenever a friendly Dragon attacks, give your Dragons +1/+3.
- Unforgiving Treant (3/12): Taunt. Whenever this takes damage, give your minions +2 Attack permanently.
- Unleashed Mana Surge (5/4) [ELEMENTAL]: After you play an Elemental, give your Elementals +2/+2.
- Visionary Shipman (5/5) [PIRATE]: After you gain Gold 5 times, get a random Tavern spell. (5 left!)
- Well Wisher (6/6): Spellcraft: Pass a different non-Golden minion.
- Wintergrasp Ghoul (5/3) [UNDEAD]: Deathrattle: Get a Tomb Turning.

---

### Tier 6 (33 minions)

- Acid Rainfall (8/8) [ELEMENTAL]: After you Refresh 5 times, gain the stats of the right-most minion in the Tavern. (5 left!)
- Apexis Guardian (7/5) [MECHANICAL]: Deathrattle and Rally: Magnetize a random Magnetic Volumizer to another friendly Mech.
- Archaedas (10/10): Battlecry: Get a random Tier 5 minion.
- Arid Atrocity (6/6) [ALL]: Deathrattle: Summon a 6/6 Golem. (Improved by each friendly minion type that died this combat!)
- Avalanche Caller (6/5) [ELEMENTAL]: At the end of your turn, get a Mounting Avalanche.
- Bluesy Siren (8/8) [NAGA]: Whenever a friendly Naga attacks, this casts Deep Blues for +2/+3 on it. (3 times per combat.)
- Charlga (3/3) [QUILBOAR]: At the end of your turn, this plays 2 Blood Gems on all your other minions.
- Dark Dazzler (4/7) [DEMON]: After your teammate sells a minion, gain its stats. (Once per turn.)
- Deathly Striker (8/8) [UNDEAD]: Avenge (4): Get a random Undead. Deathrattle: Summon it from your hand for this combat only.
- Dramaloc (10/2) [MURLOC]: Deathrattle: Give 2 other friendly Murlocs the Attack of the highest-Attack minion in your hand.
- Eternal Summoner (8/1) [UNDEAD]: Reborn. Deathrattle: Summon 1 Eternal Knight.
- Famished Felbat (9/5) [DEMON]: At the end of your turn, your Demons each consume a minion in the Tavern to gain its stats.
- Fauna Whisperer (4/9) [NAGA]: At the end of your turn, cast Natural Blessing on adjacent minions.
- Felfire Conjurer (7/6) [DEMON/DRAGON]: At the end of your turn, your Tavern spells give an extra +1/+1 this game.
- Fire-forged Evoker (8/5) [DRAGON]: Start of Combat: Give your Dragons +2/+1. After you cast a Tavern spell, improve this.
- Fleet Admiral Tethys (5/6) [PIRATE]: After you spend 10 Gold, get a random Pirate. (10 Gold left!)
- Forsaken Weaver (3/8) [UNDEAD]: After you cast a Tavern spell, your Undead have +2 Attack this game (wherever they are).
- Ignition Specialist (8/8) [DRAGON]: At the end of your turn, get 2 random Tavern spells.
- Lord of the Ruins (5/6) [DEMON]: After a friendly Demon deals damage, give friendly minions other than it +2/+1.
- Loyal Mobster (6/5) [QUILBOAR]: At the end of your turn, this plays a Blood Gem on all your teammate's minions.
- Magicfin Mycologist (4/8) [MURLOC]: Once per turn, after you buy a Tavern spell, get a 1/1 Murloc and teach it that spell. (1 left!)
- Needling Crone (5/4) [QUILBOAR]: Your Blood Gems give twice their stats during combat.
- Nightmare Par-tea Guest (6/6) [ALL]: Battlecry and Deathrattle: Get a Misplaced Tea Set.
- Paint Smudger (5/5) [QUILBOAR]: Whenever you cast a Tavern spell, this plays a Blood Gem on 3 friendly minions.
- Rabid Panther (4/8) [BEAST]: After you play a Beast, give your Beasts +3/+3 and deal 1 damage to them.
- Sanguine Refiner (3/10) [QUILBOAR]: Rally: Your Blood Gems give an extra +1/+1 this game.
- Shore Marauder (8/10) [ELEMENTAL/PIRATE]: Your Pirates and Elementals give an extra +1/+1.
- Silky Shimmermoth (3/8) [BEAST]: Whenever this takes damage, your Beetles have +2/+2 this game. Deathrattle: Summon a 2/2 Beetle.
- Sundered Matriarch (7/4) [NAGA]: Whenever you cast a spell, give your minions +3 Health.
- Transport Reactor (1/1) [MECHANICAL]: Magnetic. Has +1/+1 for each time your team has Passed this game (wherever this is).
- Utility Drone (4/6) [MECHANICAL]: At the end of your turn, give your minions +4/+4 for each Magnetization they have.
- Whirling Lass-o-Matic (6/3) [MECHANICAL]: Divine Shield. Windfury. Rally: Get a random Tavern spell.
- Young Murk-Eye (9/6) [MURLOC]: At the end of your turn, trigger the Battlecry of an adjacent minion.

---

### Tier 7 (13 minions)

- Captain Sanders (9/9) [PIRATE]: Battlecry: Make a friendly minion from Tier 6 or below Golden.
- Champion of Sargeras (10/10) [DEMON]: Battlecry and Deathrattle: Minions in the Tavern have +5/+5 this game.
- Futurefin (7/13) [MURLOC]: At the end of your turn, give this minion's stats to the left-most minion in your hand.
- Highkeeper Ra (6/6): Battlecry, Deathrattle and Rally: Get a random Tier 6 minion.
- Obsidian Ravager (7/7) [DRAGON]: Rally: Deal damage equal to this minion's Attack to the target and an adjacent minion.
- Polarizing Beatboxer (5/10) [MECHANICAL]: Whenever you Magnetize another minion, it also Magnetizes to this.
- Sandy (1/1): Start of Combat: Transform into a copy of your teammate's highest-Health minion.
- Sanguine Champion (18/3) [QUILBOAR]: Battlecry and Deathrattle: Your Blood Gems give an extra +1/+1 this game.
- Sea Witch Zar'jira (4/5) [NAGA]: Spellcraft: Choose a different minion in the Tavern to get a copy of.
- Stalwart Kodo (16/32) [BEAST]: After you summon a minion in combat, give it this minion's maximum stats. (3 times per combat.)
- Stitched Salvager (16/4) [UNDEAD]: Start of Combat: Destroy the minion to the left. Deathrattle: Summon an exact copy of it. (Except Stitched Salvager.)
- Stone Age Slab (5/10) [ELEMENTAL]: After you buy a minion, give it +10/+10 and double its stats. (Once per turn.)
- The Last One Standing (12/12) [ALL]: Rally: Give a friendly minion of each type +12/+12 permanently.

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

Then update `CLAUDE.md` card tables by re-running the HearthstoneJSON scraper
and filtering on `isBattlegroundsPoolMinion: true`. Cards without this flag
are retired and should not appear in the symbolic layer.

## Context Logging

After every session, append a new entry to `CONTEXT.md` in the project root.
Each entry should follow this format:

---
### [DATE] — [SHORT SESSION TITLE]
**Files changed:** `list/of/files.py`
**What was done:** 2–4 sentences summarizing the changes made and why.
**Current state:** One sentence on where things stand now.
**Open questions / next steps:** Bullet list of unresolved issues or planned next actions.
---

Do not overwrite previous entries. Always append. If `CONTEXT.md` does not exist, create it with a header:
# bg_agent — Development Context Log