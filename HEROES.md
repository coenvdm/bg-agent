# Hearthstone Battlegrounds — Hero Powers Reference

> Last updated: 2026-03-31 (Patches 34.4.2 + 34.6.x current).
> Sources: [hearthstone.wiki.gg](https://hearthstone.wiki.gg/wiki/Battlegrounds/Hero), [blizzardwatch.com](https://blizzardwatch.com/2025/02/21/hearthstone-battlegrounds-kerrigan-raynor-artanis-anomalies/), [fantasywarden.com](https://fantasywarden.com/games/hearthstone-battlegrounds-heroes)
>
> **Known recent changes:**
> - **Added (Patch 31.6):** Artanis, Jim Raynor, Kerrigan (StarCraft heroes)
> - **Added (Patch 34.6):** Morchie (Warped Conflux), Loh redesigned (Heroic Inspiration)
> - **Temporarily removed (Patch 34.4.2):** Loh, Living Legend (old version) — since redesigned and re-added
> - **Farseer Nobundo** added in Season 9

Armor values range 5–20. Lower armor = stronger hero (more balanced by weaker starting health cushion).

---

## Implementation Tiers for the RL Agent

| Priority | Complexity | Heroes |
|---|---|---|
| **Phase 1** | Passive / simple gold effect | Hoggar, Trade Prince Gallywix, Forest Warden Omu, Ysera, Millificent Manastorm, Greybough, Pyramad, Enhance-o-Mechano, Ozumat, Queen Wagtoggle |
| **Phase 2** | Active, no-pointer (random target) | Infinite Toki, George the Fallen, Arch-Villain Rafaam, Galakrond, Death Speaker Blackthorn, The Great Akazamzarak, Illidan Stormrage, Al'Akir the Windlord |
| **Phase 3** | Active, requires pointer (targeted) | Vol'jin, Shudderwock, Xyrella, Mutanus the Devourer, Maiev Shadowsong, Overlord Saurfang |
| **Phase 4** | Complex / multi-stage / combat-phase | A. F. Kay, Thorim, Reno Jackson, Teron Gorefiend, Sir Finley, Professor Putricide, Ragnaros, N'Zoth |

---

## Full Hero Roster

### Active Hero Powers — No Pointer (random or self-targeting)

| Hero | Hero Power | Cost | Effect | Armor |
|---|---|---|---|---|
| **Infinite Toki** | Temporal Tavern | 1 | Refresh the Tavern with 2 minions from a Tier higher than your current tier included. | 10 |
| **Death Speaker Blackthorn** | Bloodbound | 1 | Get 2 Blood Gems. (Triggers twice per turn.) | 10 |
| **Doctor Holli'dae** | Blessing of the Nine Frogs | 1 | Get a random Tavern Spell. | 10 |
| **Snake Eyes** | Lucky Roll | 1 | Roll a 6-sided die. Gain that much Gold this turn. | 10 |
| **Arch-Villain Rafaam** | I'll Take That! | 1 | Next combat, if an enemy minion dies, add a copy of it to your hand. | 10 |
| **Tess Greymane** | Bob's Burgles | 1 | Refresh the Tavern with copies of the last opponent's minions. | 10 |
| **The Great Akazamzarak** | Prestidigitation | 1 | Discover a Secret and put it on the battlefield. | 10 |
| **Yogg-Saron, Hope's End** | Puzzle Box | 1 | Steal a random minion from the Tavern. Give it +1/+1. | 10 |
| **Scabbs Cutterbutter** | I Spy | 2 | Discover a plain copy of one of your opponent's warband minions. | 10 |
| **George the Fallen** | Boon of Light | 2 | Give a friendly minion Divine Shield. | 10 |
| **Zephrys the Great** | Three Wishes | 3 | Choose a minion in your hand or play. If you have two copies, discover a third. (3 uses per game.) | 10 |
| **E.T.C., Band Manager** | Sign a New Artist | 3 | Discover a Buddy (from Tier 2+). | 10 |
| **Captain Eudora** | Buried Treasure | 5 | Dig! After 5 uses, unearth a Golden minion from the pool. | 10 |
| **Professor Putricide** | Build-An-Undead | 4 | Craft a custom Undead minion. (3 creations per game.) | 10 |
| **Heistbaron Togwaggle** | The Perfect Crime | 10 | Steal all cards in the Tavern. Each turn without buying, next use costs (1) less. | 10 |

### Active Hero Powers — Pointer (targeted)

| Hero | Hero Power | Cost | Effect | Armor | Pointer zone |
|---|---|---|---|---|---|
| **Vol'jin** | Spirit Swap | 0 | Choose two minions. Swap their Attack and Health. | 5 | board × 2 |
| **Shudderwock** | Snicker Snack | 0 | Trigger a friendly minion's Battlecry. | 10 | board |
| **Galakrond** | Galakrond's Greed | 1 | Choose a minion in the Tavern. Discover a minion of a higher Tier to replace it. | 10 | shop |
| **Mutanus the Devourer** | Devour | 1 | Remove a friendly minion. Spit its stats onto another. Gain 1 Gold. | 10 | board × 2 |
| **Xyrella** | Set the Light Free | 2 | Choose a minion. Set its stats to 2/2 and add it to your hand. | 10 | shop |
| **Overlord Saurfang** | For The Horde! | 2 | Give a Tavern minion +X/+X (X increases by 1 each turn). | 10 | shop |
| **Maiev Shadowsong** | Imprison | 2 | Choose a friendly minion. It goes Dormant for 2 rounds, returning with +2/+2. | 10 | board |
| **King Mukla** | Bananarama | 2 | Give all opponents a Banana (+1/+1 spell). You get 2. | — | — |

### Passive Hero Powers

| Hero | Hero Power | Effect | Armor |
|---|---|---|---|
| **Trade Prince Gallywix** | Smart Savings | After selling a minion, gain 1 extra Gold on your next turn. | 10 |
| **Forest Warden Omu** | Everbloom | After you upgrade your Tavern, gain 2 extra Gold this turn. | 10 |
| **Ysera** | Dream Portal | Whenever you refresh, add a Dragon to your Tavern (at your current Tier). | 10 |
| **Millificent Manastorm** | Tinker | Whenever you summon a Mech in the Tavern, give it +2 Attack. | 10 |
| **Millhouse Manastorm** | Manastorm | All Minions and Refreshes cost 2 Gold. All Tavern upgrades cost 1 more. | 5 |
| **Greybough** | Sprout It Out! | Start of Combat: give all friendly minions summoned during combat +1/+2 and Taunt. | 10 |
| **Queen Wagtoggle** | Wax Warband | Start of Combat: give one friendly minion of each minion type bonus stats equal to its Tier. | 10 |
| **Ozumat** | Tentacular | At Start of Combat, summon a 2/2 Tentacle with Taunt. Each friendly minion you sell this game gives it +1/+1. | 10 |
| **Enhance-o-Mechano** | Enchancification | After you Refresh, give a random friendly minion Taunt, Windfury, Reborn, or Divine Shield. | 10 |
| **Alexstrasza** | Queen of Dragons | After you upgrade your Tavern to Tier 5, Discover two Dragons. | 10 |
| **Aranna Starseeker** | Demon Hunter Training | After 14 friendly minion attacks, your first minion buy each turn is free. | 10 |
| **Ini Stormcoil** | Tinker | After 10 friendly minions die, get a random Mech. (Resets.) | 10 |
| **Fungalmancer Flurgl** | Murloc Evolution | After you sell a Murloc, add a Murloc to your Tavern. | 10 |
| **Inge the Iron Hymn** | Armor of Ages | Gain Armor equal to the number of friendly minions that died in combat. | 10 |
| **Drek'Thar** | ??? | Passive — details vary by patch. | 10 |

### Economy / Gold Heroes

| Hero | Hero Power | Cost | Effect | Armor |
|---|---|---|---|---|
| **Cap'n Hoggarr** | Somethin' Shiny | Passive | Gain 1 extra Gold each turn. | 15 |
| **Pyramad** | Might of Pyramid | 0 | Give a random friendly minion +4 Health. (Increases by +1 each turn it's unused.) | 15 |
| **Rakanishu** | Lantern of Power | 0 | Give your minions +X/+X where X = number of minions with Divine Shield. | 15 |

### Hero Powers with Progression / Quests

| Hero | Hero Power | Effect | Armor |
|---|---|---|---|
| **A. F. Kay** | Procrastinate | Skip first two buy phases. Begin turn 3 with a Tier 3 minion and two Tier 2 minions. | 5 |
| **Sir Finley Mrrgglton** | Adventure! | Discover a new Hero Power. (Passive version of discovered power applies.) | 10 |
| **Master Nguyen** | Power of the Storm | Each turn, choose between two new Hero Powers. | 10 |
| **Reno Jackson** | Gonna Be Rich! | Make a friendly Golden minion (once per game). | 10 |
| **Patches the Pirate** | Pirate Parrrrty! | Get a Pirate. Next use costs (1) less. | 10 |
| **Silas Darkmoon** | Come One, Come All! | Get 3 Darkmoon Faire Tickets. Spend them all to Discover a minion of your current Tier. | 10 |
| **Thorim, Stormlord** | Choose Your Champion | After spending 65 total Gold, Discover a Tier 7 minion. | 10 |
| **Kurtrus Ashfallen** | Glaive Ricochet | After buying 3 minions, get a copy of one of them. | 10 |
| **Galewing** | Dungar's Gryphon | Choose a Flightpath. Complete its milestones for rewards. | 10 |
| **Queen Azshara** | Azshara's Ambition | Passive until your warband totals 25 Attack, then: Discover a Naga minion up to your Tier for 1 Gold. | 10 |
| **Sire Denathrius** | Whodunit? | At game start, choose one of two Quests. Complete it for a powerful reward. | 10 |
| **Cookie the Cook** | Stir the Pot | Throw a minion into your cooking pot. Once you have 3, Discover from their types. | 10 |
| **Tickatus** | Prize Wall | Every 4 turns, Discover a Darkmoon Prize. | 10 |
| **Dinotamer Brann** | Survival of the Fittest | After buying 15 Beasts, get a random Legendary Beast. | 10 |
| **Elise Starseeker** | Lead Explorer | After you level to Tier 4, get Map to the Golden Monkey (5 Gold: replaces all Tavern minions with Goldens once). | 10 |
| **Jandice Barov** | Swap the Minions | Swap a minion in hand with a random Tavern minion (of the same Tier). | 0 |
| **Captain Hooktusk** | Raid the Docks | Attack the Tavern 4 times. Each hit reduces the cost of the minion you hit by (1). | 0 |

### Combat-Phase Passive Heroes

| Hero | Hero Power | Effect | Armor |
|---|---|---|---|
| **The Lich King** | Reborn Rites | Give a random friendly minion Reborn at the start of your next combat. | 10 |
| **Ragnaros the Firelord** | Sulfuras | At end of combat, deal 1 damage to all enemies. After attacking 20 times total, deal 3 instead. | 10 |
| **N'Zoth** | Tentacles for Arms | At end of combat, summon a 0/3 Tentacle with Taunt for each Deathrattle minion that died (up to 3). | 10 |
| **Teron Gorefiend** | Rapid Reanimation | Destroy a friendly minion at the start of combat. At end of combat, re-summon exact copy with all buffs. | 10 |
| **Illidan Stormrage** | Wingman | Your leftmost and rightmost minions gain +2/+1 and immediately attack at Start of Combat. | 10 |
| **Al'Akir the Windlord** | Swatting Insects | Give your leftmost minion Windfury, Divine Shield, and Taunt. | 0 |
| **The Curator** | Mixed Company | Start of Combat: give +1/+1 to your minions for each distinct minion type you have. | 10 |
| **Lich Baz'hial** | Graveyard Shift | Steal a Tavern minion. Take damage equal to its Tavern Tier. | 0 |
| **Deathwing** | ALL WILL BURN | All minions have +3 Attack. | 10 |
| **Sindragosa** | Stay Frosty | At end of combat, give all Frozen minions in Bob's Tavern +2/+1. | 10 |
| **Nozdormu** | Timeform | Your hero takes no damage from ties. | 20 |
| **C'Thun** | Saturday C'Thuns! | At end of combat, C'Thun attacks a random enemy minion. C'Thun grows as your minions take damage. | 10 |
| **Y'Shaarj** | Embrace the Pain | Start of Combat: give your most powerful minion a permanent +2/+2. | 10 |
| **Sylvanas Windrunner** | Banshee's Wail | Start of Combat: steal a random enemy minion until end of combat. | 10 |
| **Malygos** | Arcane Alteration | At start of combat, give your minions +1/+1 and a random keyword. | 10 |

### Special / Unusual Mechanics

| Hero | Hero Power | Cost | Effect | Armor |
|---|---|---|---|---|
| **Patchwerk** | All Patched Up | Passive | Start with 60 Health and no armor. No Hero Power. | 0 |
| **Sneed** | Sneed's Replicator | Passive | After buying a Golden minion, get a copy of it. | 10 |
| **Guff Runetotem** | Natural Balance | 0 | Give your minions of the least-represented type +2/+2. | 10 |
| **Varden Dawngrasp** | Frost Nova | 2 | Freeze all Tavern minions. Gain +2/+2 on each. | 10 |
| **Bru'kan** | Earth Invocation | 0 | Give a random friendly minion +4/+4. (Rotates through: Earth / Water / Fire / Air each giving different effects.) | 15 |
| **Cariel Roame** | Righteous Cause | Passive | Whenever you buy a minion with Divine Shield, give all friendly minions +1/+1. | 10 |
| **Tamsin Roame** | From the Shadows | 0 | Give a Deathrattle minion +2/+2 and summon a Haunted Spirit when it dies. | 10 |
| **The Jailer** | Fel Chains | Passive | Start with 5 Armor. All enemies' non-combat effects are disabled. | 5 |
| **Lord Barov** | Whelp Training | 0 | Give a friendly minion +1/+1 for each Armor you have. Lose 2 Armor. | 10 |
| **Skycap'n Kragg** | Kragg's Contract | 1 | Get a random Tavern Spell. It costs (0). | 10 |
| **Vanndar Stormpike** | Battle Readiness | Passive | Start with 2 extra minions in your warband. They start with 1/1. | 10 |
| **Rock Master Voone** | Dragon Taming | 0 | Give all friendly Dragons +1/+1. | 10 |
| **Maiev Shadowsong** | Imprison | 2 | Choose a Tavern minion. It goes Dormant for 2 turns and gains +2/+2. | 10 |
| **Rokara** | Rampage | 0 | Give your weakest minion +3/+3. | 10 |
| **Ambassador Faeilin** | Naga Synergy | 0 | Give a random friendly Naga +2/+2. | 10 |
| **Tae'thelan Bloodwatcher** | On the House | 0 | The next minion you buy costs (1) less this turn. (Resets each turn.) | 10 |
| **Onyxia** | Brood Mother | Passive | At end of combat, summon 1/1 Whelps equal to damage dealt. | 10 |
| **Kael'thas Sunstrider** | Sunstrider's Favor | Passive | Every third minion you buy costs (0). | 10 |
| **Dancin' Deryl** | Dance Lessons | 1 | Give two random minions in your hand +1/+1. | 10 |
| **Murloc Holmes** | Detective for Hire | 0 | Guess a minion from your opponent's warband (shown two options). Get a Clue if correct. | 10 |
| **Edwin VanCleef** | Preparation | Passive | After playing a card from your hand, give your next buy a discount of (1). | 10 |
| **Tavish Stormpike** | Trueshot Aura | 0 | Give your leftmost minion +2 Attack. (Rotates left each turn.) | 10 |
| **Lady Vashj** | Coilfang Crest | 0 | Give a random friendly Naga +2/+2. | 10 |
| **Mr. Bigglesworth** | Kitteh's Scheme | Passive | After each player dies, your minions gain a random buff. | 10 |
| **Chenvaala** | Avalanche | Passive | After you play 3 Elementals, reduce your Tavern upgrade cost by (3). | 10 |
| **Liadrin** | Blood Knight | Passive | After you play a minion with Divine Shield, give your minions +1/+1. | 10 |

---

## Implementation Notes for RL Agent

### Pointer-type action changes needed (Phase 3)
`TYPES_WITH_POINTER` currently = `{0 BUY, 1 SELL, 2 PLAY}`. Hero powers that target a specific board or shop slot need `6 HERO_POWER` added. The pointer mask for hero power should be zone-specific:
- **board-targeting heroes**: use `PTR_BOARD_OFF … PTR_BOARD_OFF+7`
- **shop-targeting heroes**: use `PTR_SHOP_OFF … PTR_SHOP_OFF+7`
- **two-target heroes** (Vol'jin, Mutanus): need two sequential pointer actions — add action type `9 HERO_POWER_TARGET2`

### Hero state fields to add to `PlayerState`
```python
hero_card_id: str = ""           # already exists
hero_power_used: bool = False    # reset each buy phase
hero_power_charges: int = 1      # for multi-use heroes (e.g. Eudora)
hero_power_counter: int = 0      # for progression (e.g. Eudora digs, Thorim gold spent)
hero_power_cost: int = 0         # base cost (set from hero definition at game start)
armor: int = 0                   # already tracked, feeds Lord Barov
```

### Phase 1 heroes — no new infrastructure needed
All of these can be wired into `step_shopping` with a simple dispatch, or as passive hooks in the existing game loop events:

| Hero | Hook location |
|---|---|
| Gallywix | `on_sell` callback — set `extra_gold_next_turn += 1` |
| Hoggar | `_gold_for_round(player)` — `+= 1` unconditionally |
| Omu | `on_tavern_upgrade` event — `ps.gold += 2` |
| Ysera | `do_refresh` — draw an extra Dragon if available |
| Millificent | `_dict_to_minion` — `+= 2 atk` for Mechs from Tavern |
| Millhouse | `_gold_for_round` — reroll/upgrade costs modified |
| Chenvaala | `on_play` — count Elementals, apply discount |
| Kael'thas | `on_buy` — track count mod 3, apply 0-cost |
| Cariel Roame | `on_buy` — if divine_shield, buff all friendlies |
| Edwin | `on_play_from_hand` — discount next buy by 1 |

---

## Recently Added Heroes (Patches 31.6 – 34.6)

### StarCraft Heroes (added Patch 31.6, Feb 2025)

| Hero | Hero Power | Cost | Effect | Complexity |
|---|---|---|---|---|
| **Artanis** | Warp Gate | Passive | At game start, choose between two powerful Protoss minions. They unlock after you buy 16 cards. All are Tier 5 Mechs; some (e.g. Colossus) upgrade during the game, others (e.g. Mothership) trigger Avenge effects. | Medium — needs a sub-pool of Protoss token minions and a buy-counter. |
| **Jim Raynor** | Lift Off | Passive | Start with a 2/2 Battlecruiser on the board. Spell upgrades appear in the Tavern each turn; buying them buffs the Battlecruiser (+atk/+hp, keywords). Upgrade quality scales with Tavern Tier. | Hard — unique "upgrade spell" cards must appear in shop; Battlecruiser state tracked separately. |
| **Kerrigan, Queen of Blades** | Spawn More Overlords | Passive | Start with a 2/2 Larva that evolves each turn into a different Zerg minion. Has a parallel leveling track (starts at 7 Gold, -1 per turn idle) to unlock Tier 2 and Tier 3 Zerg forms. | Hard — dual gold-economy system and Zerg transformation state machine. |

### Other New Heroes

| Hero | Hero Power | Cost | Effect | Added |
|---|---|---|---|---|
| **Farseer Nobundo** | The Galaxy's Lens | 3→2→1→0 | Get a copy of the last Tavern Spell you played. Cost decreases by 1 each turn (resets when used). | Season 9 (Dec 2024) |
| **Morchie** | Warped Conflux | Passive | On Turn 5, visit the Minor Timewarp. (Presumably: step back in time to revisit the Tavern from a previous turn.) | Patch 34.6 (Mar 2025) |
| **Loh, Living Legend** *(redesigned)* | Heroic Inspiration | Passive | After 15 friendly minions attack, get a Triple Reward. Buddy: Stoneshell Guardian (5/5) — the first Triple Reward you play each turn Discovers from Golden minions. | Patch 34.6 (redesigned) |

---

## Heroes Confirmed NOT in Current Pool (as of March 2026)

- **Loh, Living Legend** (original version) — temporarily removed Patch 34.4.2, then redesigned and re-added in 34.6
- Any hero not listed above may be on rotation; Blizzard rotates ~5–10 heroes per season

---

## Quick-Reference: Hero Power Cost Summary

| Cost | Heroes |
|---|---|
| **0 (free)** | Vol'jin, Al'Akir, Shudderwock, Jandice Barov, Captain Hooktusk, Pyramad, Rakanishu, Guff, Bru'kan, Tamsin, Lord Barov, Rock Master Voone, Rokara, Ambassador Faeilin, Tae'thelan, Murloc Holmes |
| **1** | Infinite Toki, Death Speaker Blackthorn, Doctor Holli'dae, Snake Eyes, Arch-Villain Rafaam, Tess Greymane, The Great Akazamzarak, Yogg-Saron, Galakrond, Mutanus, Skycap'n Kragg |
| **2** | Scabbs, George the Fallen, Xyrella, Overlord Saurfang, Maiev Shadowsong, King Mukla, Varden Dawngrasp, Dancin' Deryl |
| **3** | Zephrys, E.T.C., Farseer Nobundo (first use) |
| **4** | Professor Putricide |
| **5** | Captain Eudora |
| **10** | Heistbaron Togwaggle (decreases each turn) |
| **Passive** | Gallywix, Hoggar, Omu, Ysera, Millificent, Millhouse, Greybough, Queen Wagtoggle, Ozumat, Enhance-o-Mechano, Alexstrasza, Aranna, Ini Stormcoil, Fungalmancer Flurgl, Inge, Cariel, Edwin, Kael'thas, Sneed, Patchwerk, Nozdormu, Deathwing, Lich King, Ragnaros, N'Zoth, Illidan, The Curator, Sindragosa, C'Thun, Y'Shaarj, Sylvanas, Malygos, Onyxia, Artanis, Jim Raynor, Kerrigan, Loh, Mr. Bigglesworth, Chenvaala, Liadrin, Millhouse, Dinotamer Brann |
