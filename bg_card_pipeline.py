"""
bg_card_pipeline.py — Hearthstone Battlegrounds card pipeline.

Parses the embedded CLAUDE.md card list and optionally fetches fresh data
from HearthstoneJSON API to produce bg_card_definitions.json.

Usage:
    python bg_card_pipeline.py [--output bg_card_definitions.json] [--fetch] [--stats]
"""

import argparse
import json
import re
import sys
import urllib.request
import urllib.error
from datetime import date

# ---------------------------------------------------------------------------
# Embedded card data: (name, atk, hp, tribes_list, text)
# ---------------------------------------------------------------------------

TIER_CARDS = {
    1: [
        ("Annoy-o-Tron",        1,  2, ["MECH"],         "Taunt. Divine Shield."),
        ("Aureate Laureate",     1,  1, ["PIRATE"],       "Divine Shield. Battlecry: Make this minion Golden."),
        ("Bubble Gunner",        2,  3, ["MURLOC"],       "Battlecry: Gain a random Bonus Keyword."),
        ("Buzzing Vermin",       1,  1, ["BEAST"],        "Taunt. Deathrattle: Summon a 2/2 Beetle."),
        ("Cord Puller",          1,  1, ["MECH"],         "Divine Shield. Deathrattle: Summon a 1/1 Microbot."),
        ("Crackling Cyclone",    2,  1, ["ELEMENTAL"],    "Divine Shield. Windfury."),
        ("Dune Dweller",         3,  2, ["ELEMENTAL"],    "Battlecry: Give Elementals in the Tavern +1/+1 this game."),
        ("Flighty Scout",        3,  3, ["MURLOC"],       "Start of Combat: If this minion is in your hand, summon a copy of it."),
        ("Harmless Bonehead",    1,  1, ["UNDEAD"],       "Deathrattle: Summon two 1/1 Skeletons."),
        ("Minted Corsair",       1,  3, ["PIRATE"],       "When you sell this, get a Tavern Coin."),
        ("Misfit Dragonling",    2,  1, ["DRAGON"],       "Start of Combat: Gain stats equal to your Tier."),
        ("Ominous Seer",         2,  1, ["DEMON", "NAGA"],"Battlecry: The next Tavern spell you buy costs (1) less."),
        ("Passenger",            2,  2, [],               "The first time your team Passes each turn, gain +1/+2."),
        ("Picky Eater",          1,  1, ["DEMON"],        "Battlecry: Consume a random minion in the Tavern to gain its stats."),
        ("Razorfen Geomancer",   2,  1, ["QUILBOAR"],     "Battlecry: Get 2 Blood Gems."),
        ("Risen Rider",          2,  1, ["UNDEAD"],       "Taunt. Reborn."),
        ("Rot Hide Gnoll",       1,  4, ["UNDEAD"],       "Has +1 Attack for each friendly minion that died this combat."),
        ("Surf n Surf",          1,  1, ["NAGA", "BEAST"],"Spellcraft: Give a minion \"Deathrattle: Summon a 3/2 Crab\" until next turn."),
        ("Swampstriker",         1,  5, ["MURLOC"],       "Windfury. After you summon a Murloc, gain +1 Attack."),
        ("Tusked Camper",        2,  3, ["QUILBOAR"],     "Rally: This plays a Blood Gem on itself."),
        ("Twilight Hatchling",   1,  1, ["DRAGON"],       "Deathrattle: Summon a 3/3 Whelp that attacks immediately."),
        ("Wrath Weaver",         1,  4, ["DEMON"],        "After you play a Demon, deal 1 damage to your hero and gain +2/+1."),
    ],
    2: [
        ("Blue Volumizer",       1,  3, ["MECH"],         "Magnetic. The first time this is played or Magnetized, your Volumizers have +3 Health this game."),
        ("Briarback Bookie",     3,  3, ["QUILBOAR"],     "At the end of your turn, get a Blood Gem."),
        ("Defiant Shipwright",   2,  5, ["PIRATE"],       "Whenever this gains Attack from other sources, gain +1 Health."),
        ("Embalming Expert",     3,  2, ["UNDEAD"],       "After the Tavern is Refreshed, give its right-most minion +2 Attack and Reborn."),
        ("Eternal Knight",       4,  1, ["UNDEAD"],       "Has +4/+1 for each friendly Eternal Knight that died this game."),
        ("Expert Aviator",       3,  4, ["MURLOC"],       "Rally: Give the left-most minion in your hand +1/+1 and summon it for this combat only."),
        ("Fire Baller",          4,  3, ["ELEMENTAL"],    "When you sell this, give your minions +1 Attack. Improve your future Ballers."),
        ("Forest Rover",         1,  1, ["BEAST"],        "Battlecry: Your Beetles have +2/+1 this game. Deathrattle: Summon a 2/2 Beetle."),
        ("Freedealing Gambler",  3,  3, ["PIRATE"],       "This minion sells for 3 Gold."),
        ("Friendly Saloonkeeper",3,  4, [],               "Battlecry: Your teammate gets a Tavern Coin."),
        ("Gathering Stormer",    5,  1, ["ELEMENTAL"],    "When you sell this, your teammate gains 1 Gold. (Improves each turn!)"),
        ("Generous Geomancer",   1,  1, ["QUILBOAR"],     "Deathrattle: You and your teammate each get a Blood Gem."),
        ("Green Volumizer",      3,  3, ["MECH"],         "Magnetic. The first time this is played or Magnetized, your Volumizers have +1/+1 this game."),
        ("Humming Bird",         1,  4, ["BEAST"],        "Start of Combat: For the rest of this combat, your Beasts have +1 Attack."),
        ("Intrepid Botanist",    3,  4, [],               "Choose One - Your Tavern spells give an extra +1 Attack this game; or +1 Health."),
        ("Irate Rooster",        3,  4, ["BEAST"],        "Start of Combat: Deal 1 damage to adjacent minions and give them +4 Attack."),
        ("Lava Lurker",          2,  5, ["NAGA"],         "The first Spellcraft spell played from hand on this each turn is permanent. (1 left!)"),
        ("Mechagnome Interpreter",2, 3, ["MECH"],         "Whenever you play or Magnetize a Mech, give it +2/+1."),
        ("Mind Muck",            3,  2, ["DEMON"],        "Battlecry: Choose a friendly Demon. It consumes a minion in the Tavern to gain its stats."),
        ("Moon-Bacon Jazzer",    1,  3, ["QUILBOAR"],     "Battlecry: Your Blood Gems give an extra +1 Health this game."),
        ("Nerubian Deathswarmer",1,  4, ["UNDEAD"],       "Battlecry: Your Undead have +1 Attack this game (wherever they are)."),
        ("Oozeling Gladiator",   2,  2, [],               "Battlecry: Get two Slimy Shields that give +1/+1 and Taunt."),
        ("Patient Scout",        1,  1, [],               "When you sell this, Discover a Tier 1 minion. (Improves each turn!)"),
        ("Prophet of the Boar",  2,  3, [],               "Taunt. After you play a Quilboar, get a Blood Gem."),
        ("Red Volumizer",        3,  1, ["MECH"],         "Magnetic. The first time this is played or Magnetized, your Volumizers have +3 Attack this game."),
        ("Reef Riffer",          3,  2, ["NAGA"],         "Spellcraft: Give a minion stats equal to your Tier until next turn."),
        ("Saltscale Honcho",     5,  2, ["MURLOC"],       "After you summon a Murloc, give a friendly Murloc other than it +2 Health."),
        ("Sellemental",          3,  3, ["ELEMENTAL"],    "When you sell this, get a 3/3 Elemental."),
        ("Shell Collector",      4,  3, ["NAGA"],         "Battlecry: Get a Tavern Coin."),
        ("Sleepy Supporter",     3,  4, ["DRAGON"],       "Rally: Give another random friendly Dragon +2/+3."),
        ("Snow Baller",          3,  4, ["ELEMENTAL"],    "When you sell this, give your minions +1 Health. Improve your future Ballers."),
        ("Soul Rewinder",        4,  1, ["DEMON"],        "After your hero takes damage, rewind it and give this +1 Health."),
        ("Surfing Sylvar",       1,  2, ["PIRATE"],       "At the end of your turn, give adjacent minions +1 Attack. Repeat for each friendly Golden minion."),
        ("Tad",                  2,  2, ["MURLOC"],       "When you sell this, get a random Murloc."),
        ("Tarecgosa",            3,  3, ["DRAGON"],       "This permanently keeps Bonus Keywords and stats gained in combat."),
        ("Wanderer Cho",         4,  3, [],               "One Pass each turn is free. (1 left!)"),
        ("Whelp Watcher",        1,  4, ["DRAGON"],       "Rally: Summon a 3/3 Whelp to attack the target first."),
        ("Worgen Executive",     2,  5, [],               "Rally: After the Tavern is Refreshed this game, give its right-most minion +1/+1."),
    ],
    3: [
        ("Amber Guardian",       3,  2, ["DRAGON"],       "Taunt. Start of Combat: Give another friendly Dragon +2/+2 and Divine Shield."),
        ("Annoy-o-Module",       2,  4, ["MECH"],         "Magnetic. Divine Shield. Taunt."),
        ("Anubarak Nerubian King",3,  2, ["UNDEAD"],      "Deathrattle: Your Undead have +1 Attack this game (wherever they are)."),
        ("Aranasi Alchemist",    1,  2, ["DEMON"],        "Taunt. Reborn. Deathrattle: Give minions in the Tavern +1 Health this game."),
        ("Auto Accelerator",     3,  3, ["MECH"],         "Battlecry: Get a random Magnetic Volumizer."),
        ("Bassgill",             5,  2, ["MURLOC"],       "Deathrattle: Summon the highest-Health Murloc from your hand for this combat only."),
        ("Bird Buddy",           2,  4, [],               "Avenge (1): Give your Beasts +1/+1."),
        ("Blue Whelp",           1,  5, ["DRAGON"],       "Rally: Your Tavern spells give an extra +1 Health this game."),
        ("Bottom Feeder",        3,  4, ["MURLOC"],       "At the end of your turn, you and your teammate each get a random Tier 1 card."),
        ("Bountiful Bedrock",    3,  4, ["ELEMENTAL"],    "At the end of every 2 turns, get a random Elemental. (2 turns left!)"),
        ("Briarback Drummer",    5,  2, ["QUILBOAR"],     "Battlecry: Get a Blood Gem Barrage."),
        ("Briny Bootlegger",     4,  2, ["PIRATE"],       "Deathrattle: Get a Tavern Coin."),
        ("Budding Greenthumb",   1,  5, [],               "Avenge (3): Give adjacent minions +2/+2 permanently."),
        ("Cadaver Caretaker",    3,  3, ["UNDEAD"],       "Deathrattle: Summon three 1/1 Skeletons."),
        ("Coldlight Diver",      1,  1, ["MURLOC"],       "Battlecry and Deathrattle: Get a random Tier 1 Tavern spell."),
        ("Deadly Spore",         1,  1, [],               "Venomous."),
        ("Deep-Sea Angler",      2,  3, ["NAGA"],         "Spellcraft: Give a minion +2/+6 and Taunt until next turn."),
        ("Doting Dracthyr",      4,  3, ["DRAGON"],       "At the end of your turn, give your teammate's minions +1 Attack."),
        ("Felemental",           3,  3, ["ELEMENTAL", "DEMON"], "Battlecry: Give minions in the Tavern +2/+1 this game."),
        ("Felhorn",              2,  4, ["DEMON", "BEAST"],"Battlecry: Give your other Demons and Beasts +1/+2 and deal 1 damage to them, twice."),
        ("Gemsplitter",          2,  1, ["QUILBOAR"],     "Divine Shield. After a friendly minion loses Divine Shield, get a Blood Gem."),
        ("Goldgrubber",          3,  2, ["PIRATE"],       "At the end of your turn, gain +3/+2 for each friendly Golden minion."),
        ("Handless Forsaken",    2,  1, ["UNDEAD"],       "Deathrattle: Summon a 2/1 Hand with Reborn."),
        ("Hardy Orca",           1,  6, ["BEAST"],        "Taunt. Whenever this takes damage, give your other minions +1/+1."),
        ("Jelly Belly",          2,  3, ["UNDEAD"],       "After a friendly minion is Reborn, gain +2/+3 permanently."),
        ("Jumping Jack",         3,  4, ["ALL"],          "After the first time this is sold, Pass it."),
        ("Orc-estra Conductor",  4,  4, [],               "Battlecry: Give a minion +2/+2 (Improved by each Orc-estra your team has played this game)."),
        ("Peggy Sturdybone",     2,  1, ["PIRATE"],       "Whenever a card is added to your hand, give another friendly Pirate +2/+1."),
        ("Plunder Pal",          2,  2, ["PIRATE"],       "At the start of your turn, you and your teammate each gain 1 Gold."),
        ("Prickly Piper",        5,  1, ["QUILBOAR"],     "Deathrattle: Your Blood Gems give an extra +1 Attack this game."),
        ("Profound Thinker",     2,  3, ["NAGA"],         "Rally: Get a random Spellcraft spell. This casts a copy of it (targets this if possible)."),
        ("Puddle Prancer",       4,  4, ["MURLOC"],       "After this is Passed, gain +4/+4."),
        ("Rampager",             8,  8, ["BEAST"],        "Rally: Deal 1 damage to your other minions."),
        ("Relentless Deflector", 5,  4, ["MECH"],         "Has Taunt while this has Divine Shield. Avenge (3): Gain Divine Shield."),
        ("Roadboar",             3,  4, ["QUILBOAR"],     "Rally: Get 2 Blood Gems."),
        ("Roaring Recruiter",    2,  8, ["DRAGON"],       "Whenever another friendly Dragon attacks, give it +3/+1."),
        ("Scourfin",             3,  3, ["MURLOC"],       "Deathrattle: Give a random minion in your hand +5/+5."),
        ("Shoalfin Mystic",      4,  4, ["MURLOC"],       "When you sell this, your Tavern spells give an extra +1/+1 this game."),
        ("Sprightly Scarab",     3,  1, ["BEAST"],        "Choose One - Give a Beast +1/+1 and Reborn; or +4 Attack and Windfury."),
        ("Technical Element",    5,  6, ["ELEMENTAL", "MECH"], "Magnetic. Can Magnetize to both Mechs and Elementals."),
        ("The Glad-iator",       3,  3, ["NAGA"],         "Divine Shield. Whenever you cast a spell, gain +1 Attack."),
        ("Timecapn Hooktail",    1,  4, ["DRAGON", "PIRATE"], "Whenever you cast a Tavern spell, give your minions +1 Attack."),
        ("Underhanded Dealer",   3,  3, ["PIRATE"],       "After you gain Gold, gain +1/+2."),
        ("Waveling",             6,  1, ["ELEMENTAL"],    "Deathrattle: After the Tavern is Refreshed this game, give its right-most minion +3/+3."),
        ("Wheeled Crewmate",     6,  3, ["PIRATE"],       "Deathrattle: Reduce the Cost of upgrading your team's Taverns by (1)."),
        ("Wildfire Elemental",   6,  3, ["ELEMENTAL"],    "After this attacks and kills a minion, deal excess damage to an adjacent minion."),
        ("Zesty Shaker",         5,  6, ["NAGA"],         "Once per turn, when a Spellcraft spell is played on this, get a new copy of it."),
    ],
    4: [
        ("Accord-o-Tron",        5,  5, ["MECH"],         "Magnetic. At the start of your turn, gain 1 Gold."),
        ("Apprentice of Sefin",  4,  4, ["MURLOC"],       "At the end of your turn, gain +2/+2 and a random Bonus Keyword."),
        ("Blade Collector",      3,  2, ["PIRATE"],       "Also damages the minions next to whomever this attacks."),
        ("Bonker",               2,  7, ["QUILBOAR"],     "Rally: This plays 2 Blood Gems on all your other minions."),
        ("Bream Counter",        4,  4, ["MURLOC"],       "While this is in your hand, after you play a Murloc, gain +4/+4."),
        ("Conveyor Construct",   5,  2, ["MECH"],         "Deathrattle: Get a random Magnetic Volumizer."),
        ("Daggerspine Thrasher", 3,  5, ["NAGA"],         "Whenever you cast a spell, gain Divine Shield, Windfury, or Venomous until next turn."),
        ("Deep Blue Crooner",    2,  3, ["NAGA"],         "Spellcraft: Give a minion +2/+3 until next turn. Improve your future Deep Blues."),
        ("Devout Hellcaller",    2,  2, ["DEMON"],        "After another friendly Demon deals damage, gain +1/+2 permanently."),
        ("En-Djinn Blazer",      4,  4, ["ELEMENTAL"],    "Battlecry: After the Tavern is Refreshed this game, give its right-most minion +7/+7."),
        ("Fearless Foodie",      2,  4, ["QUILBOAR"],     "Choose One - Your Blood Gems give an extra +1/+1 this game; or Get 4 Blood Gems."),
        ("Feisty Freshwater",    6,  4, ["ELEMENTAL"],    "Deathrattle: You and your teammate each gain two free Refreshes."),
        ("Flaming Enforcer",     4,  5, ["ELEMENTAL", "DEMON"], "At the end of your turn, consume the highest-Health minion in the Tavern to gain its stats."),
        ("Friendly Geist",       6,  3, ["UNDEAD"],       "Deathrattle: Your Tavern spells give an extra +1 Attack this game."),
        ("Geomagus Roogug",      4,  6, ["QUILBOAR"],     "Divine Shield. Whenever a Blood Gem is played on this, this plays a Blood Gem on a different friendly minion."),
        ("Gormling Gourmet",     4,  3, ["MURLOC"],       "Taunt. Battlecry and Deathrattle: Get a Seafood Stew."),
        ("Grave Narrator",       2,  7, ["UNDEAD"],       "Avenge (3): Your teammate gets a random minion of their most common type."),
        ("Grease Bot",           2,  4, ["MECH"],         "Divine Shield. After a friendly minion loses Divine Shield, give it +2/+2 permanently."),
        ("Gunpowder Courier",    2,  6, ["PIRATE"],       "Whenever you spend 6 Gold, give your Pirates +2 Attack. (6 Gold left!)"),
        ("Heroic Underdog",      1, 10, [],               "Stealth. Rally: Gain the target's Attack."),
        ("Humongozz",            5,  5, [],               "Divine Shield. Your Tavern spells give an extra +1/+2."),
        ("Hunting Tiger Shark",  3,  5, ["BEAST"],        "Battlecry: Discover a Beast."),
        ("Ichoron the Protector",3,  1, ["ELEMENTAL"],    "Divine Shield. Whenever you play an Elemental, give it Divine Shield until next turn."),
        ("Imposing Percussionist",4, 4, ["DEMON"],        "Battlecry: Discover a Demon. Deal damage to your hero equal to its Tier."),
        ("Industrious Deckhand", 3,  5, ["PIRATE"],       "At the start of your turn, get 2 Tavern Coins."),
        ("Lovesick Balladist",   3,  2, ["PIRATE"],       "Battlecry: Give a Pirate +1 Health. (Improved by each Gold you spent this turn!)"),
        ("Malchezaar Prince of Dance", 5, 4, ["DEMON"],   "Two Refreshes each turn cost Health instead of Gold. (2 left!)"),
        ("Mantid King",          3,  3, [],               "After your team Passes, randomly gain Venomous, Taunt, or Divine Shield until next turn."),
        ("Marquee Ticker",       1,  5, ["MECH"],         "At the end of your turn, get a random Tavern spell."),
        ("Mirror Monster",       4,  4, ["ALL"],          "When you buy or Discover this, get an extra copy and Pass it."),
        ("Monstrous Macaw",      5,  4, ["BEAST"],        "Rally: Trigger your left-most Deathrattle (except this minion's)."),
        ("Persistent Poet",      2,  3, ["DRAGON"],       "Divine Shield. Adjacent Dragons permanently keep Bonus Keywords and stats gained in combat."),
        ("Plankwalker",          6,  4, ["UNDEAD", "PIRATE"], "Whenever you cast a Tavern spell, give three random friendly minions +2/+1."),
        ("Private Chef",         5,  4, ["NAGA"],         "Spellcraft: Choose a minion. Get a different random minion of its type, then Pass it."),
        ("Prized Promo-Drake",   1,  1, ["DRAGON"],       "Start of Combat: Give your Dragons +4/+4."),
        ("Prosthetic Hand",      3,  1, ["UNDEAD", "MECH"],"Magnetic. Reborn. Can Magnetize to Mechs or Undead."),
        ("Razorfen Flapper",     5,  3, ["QUILBOAR"],     "Deathrattle: Get a Blood Gem Barrage."),
        ("Refreshing Anomaly",   4,  5, ["ELEMENTAL"],    "Battlecry: Gain 2 free Refreshes."),
        ("Runed Progenitor",     2,  8, ["BEAST"],        "Avenge (3): Your Beetles have +2/+2 this game. Deathrattle: Summon a 2/2 Beetle."),
        ("Rylak Metalhead",      5,  3, ["BEAST"],        "Taunt. Deathrattle: Trigger the Battlecry of an adjacent minion."),
        ("Sanlayn Scribe",       4,  4, ["UNDEAD"],       "Has +4/+4 for each of your team's San'layn Scribes that died this game (wherever this is)."),
        ("Shifty Snake",         6,  1, ["BEAST"],        "Deathrattle: Your teammate gets a random Deathrattle minion."),
        ("Silent Enforcer",      6,  2, ["DEMON"],        "Taunt. Deathrattle: Deal 2 damage to all minions (except friendly Demons)."),
        ("Sindorei Straight Shot",3,  4, [],              "Divine Shield. Windfury. Rally: Remove Reborn and Taunt from the target."),
        ("Soulsplitter",         4,  2, ["UNDEAD"],       "Reborn. Start of Combat: Give a friendly Undead Reborn."),
        ("Spirit Drake",         1,  8, ["UNDEAD", "DRAGON"], "Avenge (3): Get a random Tavern spell."),
        ("Stellar Freebooter",   8,  4, ["PIRATE"],       "Taunt. Deathrattle: Give a friendly minion Health equal to this minion's Attack."),
        ("Tavern Tempest",       2,  2, ["ELEMENTAL"],    "Battlecry: Get a random Elemental."),
        ("Tortollan Blue Shell", 3,  6, [],               "If you lost your last combat, this minion sells for 5 Gold."),
        ("Trench Fighter",       6,  6, ["QUILBOAR"],     "At the end of your turn, get a Gem Confiscation."),
        ("Trigore the Lasher",   9,  3, ["BEAST"],        "Whenever another friendly Beast takes damage, gain +2 Health permanently."),
        ("Tunnel Blaster",       3,  7, [],               "Taunt. Deathrattle: Deal 3 damage to all minions."),
        ("Wannabe Gargoyle",     9,  1, ["DRAGON"],       "Reborn. This is Reborn with full Attack."),
        ("Waverider",            2,  8, ["NAGA"],         "Spellcraft: Give a minion +2/+2. If it's a Naga, also give it Windfury until next turn."),
        ("Weary Mage",           5,  1, ["NAGA"],         "Spellcraft: Give a minion +2/+2. If it's a Naga, also give it Reborn until next turn."),
        ("Witchwing Nestmatron", 3,  5, [],               "Avenge (3): Get a random Battlecry minion."),
    ],
    5: [
        ("Air Revenant",         3,  6, ["ELEMENTAL"],    "After you spend 7 Gold, get Easterly Winds. (7 left!)"),
        ("Ashen Corruptor",      6,  6, ["DEMON"],        "After your hero takes damage, rewind it and give minions in the Tavern +1/+1 this turn."),
        ("Azsharan Veteran",     4,  5, ["NAGA"],         "Spellcraft: Give your minions +2/+1 for each different Bonus Keyword in your warband."),
        ("Bile Spitter",         1, 10, ["MURLOC"],       "Venomous. Rally: Give another friendly Murloc Venomous."),
        ("Brann Bronzebeard",    2,  4, [],               "Your Battlecries trigger twice."),
        ("Burgeoning Whelp",     5,  5, ["DRAGON"],       "Battlecry and Deathrattle: Your Whelps have +3/+3 this game."),
        ("Cannon Corsair",       3,  7, ["PIRATE"],       "After you gain Gold, give your Pirates +1/+1."),
        ("Carapace Raiser",      6,  3, ["UNDEAD"],       "Deathrattle: Get a Haunted Carapace."),
        ("Catacomb Crasher",     6, 10, ["UNDEAD"],       "Whenever you would summon a minion that doesn't fit in your warband, give your minions +2/+1 permanently."),
        ("Champion of the Primus",2,10, ["UNDEAD"],       "Avenge (2): Your Undead have +1 Attack this game (wherever they are)."),
        ("Charging Czarina",     4,  2, ["MECH"],         "Divine Shield. Whenever you cast a Tavern spell, give your minions with Divine Shield +4 Attack."),
        ("Costume Enthusiast",   4,  5, ["MURLOC"],       "Divine Shield. Start of Combat: Gain the Attack of the highest-Attack minion in your hand."),
        ("Drakkari Enchanter",   1,  5, [],               "Your end of turn effects trigger twice."),
        ("Eternal Tycoon",       2,  9, ["UNDEAD"],       "Avenge (5): Summon an Eternal Knight. It attacks immediately."),
        ("Felboar",              2,  6, ["DEMON", "QUILBOAR"], "After you cast 3 spells, consume a minion in the Tavern to gain its stats. (3 left!)"),
        ("Firescale Hoarder",    5,  5, ["NAGA", "DRAGON"], "Battlecry and Deathrattle: Get a Shiny Ring."),
        ("Furious Driver",       3,  3, ["DEMON"],        "Battlecry: Your other Demons each consume a minion in the Tavern to gain its stats."),
        ("Gentle Djinni",        4,  5, ["ELEMENTAL"],    "Taunt. Deathrattle: Get a random Elemental."),
        ("Glowscale",            4,  6, ["NAGA"],         "Taunt. Spellcraft: Give a minion Divine Shield until next turn."),
        ("Hackerfin",            5,  3, ["MURLOC"],       "Battlecry: Give your other minions +1/+2. (Improved by each different Bonus Keyword in your warband!)"),
        ("Insatiable Urzul",     4,  6, ["DEMON"],        "Taunt. After you play a Demon, consume a random minion in the Tavern to gain its stats."),
        ("Iridescent Skyblazer", 3,  8, ["BEAST"],        "Whenever a friendly Beast takes damage, give a friendly Beast other than it +3/+1 permanently."),
        ("Junk Jouster",         6,  5, ["MECH"],         "Whenever a minion is Magnetized to this, give your minions +3/+2."),
        ("Leeroy the Reckless",  6,  2, [],               "Deathrattle: Destroy the minion that killed this."),
        ("Magnanimoose",         5,  2, ["BEAST"],        "Deathrattle: Summon a copy of a minion from your teammate's warband. Set its Health to 1 (except Magnanimoose)."),
        ("Manari Messenger",     9,  6, ["DEMON"],        "Battlecry: Minions in your team's Taverns have +1/+1 this game."),
        ("Metal Dispenser",      2,  8, ["MECH"],         "Divine Shield. Avenge (3): Magnetize a random Volumizer to this. Get a copy of it."),
        ("Mrglin Burglar",       8,  6, ["MURLOC"],       "After you play a Murloc, give a friendly minion and a minion in your hand +4/+4."),
        ("Nalaa the Redeemer",   5,  7, [],               "Whenever you cast a Tavern spell, give a friendly minion of each type +3/+2."),
        ("Niuzao",               7,  6, ["BEAST"],        "Rally: Deal damage equal to this minion's Attack to a random enemy minion other than the target."),
        ("Nomi Kitchen Nightmare",4, 4, [],               "After you play an Elemental, give Elementals in the Tavern +2/+2 this game."),
        ("Photobomber",          6,  6, ["MECH"],         "Deathrattle: Deal 2 damage to the highest-Health enemy minion. (Improved by Tavern spells you've cast this game!)"),
        ("Primalfin Lookout",    3,  2, ["MURLOC"],       "Battlecry: If you control another Murloc, Discover a Murloc."),
        ("Razorfen Vineweaver",  5,  5, ["QUILBOAR"],     "Rally: This plays 3 permanent Blood Gems on itself."),
        ("Rodeo Performer",      3,  4, [],               "Battlecry: Discover a Tavern spell."),
        ("Selfless Sightseer",   6,  2, ["DRAGON"],       "Battlecry: Increase your team's maximum Gold by (1)."),
        ("Shadowdancer",         5,  4, ["DEMON"],        "Taunt. Deathrattle: Get a Staff of Enrichment."),
        ("Showy Cyclist",        4,  3, ["NAGA"],         "Deathrattle: Give your Naga +2/+2. (Improved by every 4 spells you've cast this game!)"),
        ("Silithid Burrower",    5,  4, ["BEAST"],        "Deathrattle: Give your Beasts +1/+1. Avenge (1): Improve this by +1/+1 permanently."),
        ("Spiked Savior",        8,  2, ["BEAST"],        "Taunt. Reborn. Deathrattle: Give your minions +1 Health and deal 1 damage to them."),
        ("Storm Splitter",       5,  5, ["NAGA"],         "Once per turn, after you Pass a Tavern spell, get a new copy of it."),
        ("Stuntdrake",          14,  5, ["DRAGON"],       "Avenge (3): Give this minion's Attack to two different friendly minions."),
        ("Support System",       4,  5, ["MECH"],         "At the end of your turn, give a minion in your teammate's warband Divine Shield."),
        ("Thieving Rascal",      5,  6, ["PIRATE"],       "At the start of your turn, gain 1 Gold. Repeat for each friendly Golden minion."),
        ("Three Lil Quilboar",   3,  3, ["QUILBOAR"],     "Deathrattle: This plays 3 Blood Gems on all your Quilboar."),
        ("Titus Rivendare",      1,  7, [],               "Your Deathrattles trigger an extra time."),
        ("Tranquil Meditative",  3,  8, ["NAGA"],         "Spellcraft: Your Tavern spells give an extra +1/+1 this game."),
        ("Turquoise Skitterer",  4,  4, ["BEAST"],        "Deathrattle: Your Beetles have +4/+4 this game. Summon a 2/2 Beetle."),
        ("Twilight Broodmother", 7,  4, ["DRAGON"],       "Deathrattle: Summon 2 Twilight Hatchlings. Give them Taunt."),
        ("Twilight Watcher",     3,  7, ["DRAGON"],       "Whenever a friendly Dragon attacks, give your Dragons +1/+3."),
        ("Unforgiving Treant",   3, 12, [],               "Taunt. Whenever this takes damage, give your minions +2 Attack permanently."),
        ("Unleashed Mana Surge", 5,  4, ["ELEMENTAL"],    "After you play an Elemental, give your Elementals +2/+2."),
        ("Visionary Shipman",    5,  5, ["PIRATE"],       "After you gain Gold 5 times, get a random Tavern spell. (5 left!)"),
        ("Well Wisher",          6,  6, [],               "Spellcraft: Pass a different non-Golden minion."),
        ("Wintergrasp Ghoul",    5,  3, ["UNDEAD"],       "Deathrattle: Get a Tomb Turning."),
    ],
    6: [
        ("Acid Rainfall",        8,  8, ["ELEMENTAL"],    "After you Refresh 5 times, gain the stats of the right-most minion in the Tavern. (5 left!)"),
        ("Apexis Guardian",      7,  5, ["MECH"],         "Deathrattle and Rally: Magnetize a random Magnetic Volumizer to another friendly Mech."),
        ("Archaedas",           10, 10, [],               "Battlecry: Get a random Tier 5 minion."),
        ("Arid Atrocity",        6,  6, ["ALL"],          "Deathrattle: Summon a 6/6 Golem. (Improved by each friendly minion type that died this combat!)"),
        ("Avalanche Caller",     6,  5, ["ELEMENTAL"],    "At the end of your turn, get a Mounting Avalanche."),
        ("Bluesy Siren",         8,  8, ["NAGA"],         "Whenever a friendly Naga attacks, this casts Deep Blues for +2/+3 on it. (3 times per combat.)"),
        ("Charlga",              3,  3, ["QUILBOAR"],     "At the end of your turn, this plays 2 Blood Gems on all your other minions."),
        ("Dark Dazzler",         4,  7, ["DEMON"],        "After your teammate sells a minion, gain its stats. (Once per turn.)"),
        ("Deathly Striker",      8,  8, ["UNDEAD"],       "Avenge (4): Get a random Undead. Deathrattle: Summon it from your hand for this combat only."),
        ("Dramaloc",            10,  2, ["MURLOC"],       "Deathrattle: Give 2 other friendly Murlocs the Attack of the highest-Attack minion in your hand."),
        ("Eternal Summoner",     8,  1, ["UNDEAD"],       "Reborn. Deathrattle: Summon 1 Eternal Knight."),
        ("Famished Felbat",      9,  5, ["DEMON"],        "At the end of your turn, your Demons each consume a minion in the Tavern to gain its stats."),
        ("Fauna Whisperer",      4,  9, ["NAGA"],         "At the end of your turn, cast Natural Blessing on adjacent minions."),
        ("Felfire Conjurer",     7,  6, ["DEMON", "DRAGON"], "At the end of your turn, your Tavern spells give an extra +1/+1 this game."),
        ("Fire-forged Evoker",   8,  5, ["DRAGON"],       "Start of Combat: Give your Dragons +2/+1. After you cast a Tavern spell, improve this."),
        ("Fleet Admiral Tethys", 5,  6, ["PIRATE"],       "After you spend 10 Gold, get a random Pirate. (10 Gold left!)"),
        ("Forsaken Weaver",      3,  8, ["UNDEAD"],       "After you cast a Tavern spell, your Undead have +2 Attack this game (wherever they are)."),
        ("Ignition Specialist",  8,  8, ["DRAGON"],       "At the end of your turn, get 2 random Tavern spells."),
        ("Lord of the Ruins",    5,  6, ["DEMON"],        "After a friendly Demon deals damage, give friendly minions other than it +2/+1."),
        ("Loyal Mobster",        6,  5, ["QUILBOAR"],     "At the end of your turn, this plays a Blood Gem on all your teammate's minions."),
        ("Magicfin Mycologist",  4,  8, ["MURLOC"],       "Once per turn, after you buy a Tavern spell, get a 1/1 Murloc and teach it that spell. (1 left!)"),
        ("Needling Crone",       5,  4, ["QUILBOAR"],     "Your Blood Gems give twice their stats during combat."),
        ("Nightmare Par-tea Guest",6, 6, ["ALL"],         "Battlecry and Deathrattle: Get a Misplaced Tea Set."),
        ("Paint Smudger",        5,  5, ["QUILBOAR"],     "Whenever you cast a Tavern spell, this plays a Blood Gem on 3 friendly minions."),
        ("Rabid Panther",        4,  8, ["BEAST"],        "After you play a Beast, give your Beasts +3/+3 and deal 1 damage to them."),
        ("Sanguine Refiner",     3, 10, ["QUILBOAR"],     "Rally: Your Blood Gems give an extra +1/+1 this game."),
        ("Shore Marauder",       8, 10, ["ELEMENTAL", "PIRATE"], "Your Pirates and Elementals give an extra +1/+1."),
        ("Silky Shimmermoth",    3,  8, ["BEAST"],        "Whenever this takes damage, your Beetles have +2/+2 this game. Deathrattle: Summon a 2/2 Beetle."),
        ("Sundered Matriarch",   7,  4, ["NAGA"],         "Whenever you cast a spell, give your minions +3 Health."),
        ("Transport Reactor",    1,  1, ["MECH"],         "Magnetic. Has +1/+1 for each time your team has Passed this game (wherever this is)."),
        ("Utility Drone",        4,  6, ["MECH"],         "At the end of your turn, give your minions +4/+4 for each Magnetization they have."),
        ("Whirling Lass-o-Matic",6,  3, ["MECH"],         "Divine Shield. Windfury. Rally: Get a random Tavern spell."),
        ("Young Murk-Eye",       9,  6, ["MURLOC"],       "At the end of your turn, trigger the Battlecry of an adjacent minion."),
    ],
    7: [
        ("Captain Sanders",      9,  9, ["PIRATE"],       "Battlecry: Make a friendly minion from Tier 6 or below Golden."),
        ("Champion of Sargeras",10, 10, ["DEMON"],        "Battlecry and Deathrattle: Minions in the Tavern have +5/+5 this game."),
        ("Futurefin",            7, 13, ["MURLOC"],       "At the end of your turn, give this minion's stats to the left-most minion in your hand."),
        ("Highkeeper Ra",        6,  6, [],               "Battlecry, Deathrattle and Rally: Get a random Tier 6 minion."),
        ("Obsidian Ravager",     7,  7, ["DRAGON"],       "Rally: Deal damage equal to this minion's Attack to the target and an adjacent minion."),
        ("Polarizing Beatboxer", 5, 10, ["MECH"],         "Whenever you Magnetize another minion, it also Magnetizes to this."),
        ("Sandy",                1,  1, [],               "Start of Combat: Transform into a copy of your teammate's highest-Health minion."),
        ("Sanguine Champion",   18,  3, ["QUILBOAR"],     "Battlecry and Deathrattle: Your Blood Gems give an extra +1/+1 this game."),
        ("Sea Witch Zarjira",    4,  5, ["NAGA"],         "Spellcraft: Choose a different minion in the Tavern to get a copy of."),
        ("Stalwart Kodo",       16, 32, ["BEAST"],        "After you summon a minion in combat, give it this minion's maximum stats. (3 times per combat.)"),
        ("Stitched Salvager",   16,  4, ["UNDEAD"],       "Start of Combat: Destroy the minion to the left. Deathrattle: Summon an exact copy of it. (Except Stitched Salvager.)"),
        ("Stone Age Slab",       5, 10, ["ELEMENTAL"],    "After you buy a minion, give it +10/+10 and double its stats. (Once per turn.)"),
        ("The Last One Standing",12, 12, ["ALL"],         "Rally: Give a friendly minion of each type +12/+12 permanently."),
    ],
}

# ---------------------------------------------------------------------------
# Multiplier and aura card sets (for fast lookup)
# ---------------------------------------------------------------------------

MULTIPLIER_CARDS = {
    "brann_bronzebeard",
    "titus_rivendare",
    "drakkari_enchanter",
    "young_murk-eye",
}

# Cards with passive auras boosting other friendlies
AURA_CARDS = {
    "shore_marauder",
    "twilight_watcher",
    "lord_of_the_ruins",
    "hardy_orca",
    "roaring_recruiter",
    "iridescent_skyblazer",
    "mechagnome_interpreter",
    "timecapn_hooktail",
    "junk_jouster",
    "geomagus_roogug",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_card_id(name: str) -> str:
    """Lowercase name, spaces→underscores, remove apostrophes and commas."""
    cid = name.lower()
    cid = cid.replace("'", "").replace(",", "").replace(" ", "_")
    return cid


def detect_keywords(text: str) -> dict:
    t = text.lower()
    return {
        "taunt":        "taunt" in t,
        "divine_shield": "divine shield" in t,
        "reborn":       "reborn" in t,
        "windfury":     "windfury" in t,
        "venomous":     "venomous" in t,
        "magnetic":     "magnetic" in t,
        "stealth":      "stealth" in t,
    }


def detect_trigger_type(text: str) -> str:
    """Determine the primary trigger type from card text."""
    t = text.lower()
    # Order matters: check most specific first
    if "battlecry" in t and "deathrattle" in t:
        return "battlecry"
    if "battlecry" in t:
        return "battlecry"
    if "deathrattle" in t:
        return "deathrattle"
    if "start of combat" in t:
        return "start_of_combat"
    if "at the end of your turn" in t or "end of your turn" in t:
        return "end_of_turn"
    if "when you sell this" in t or "when you sell" in t:
        return "on_sell"
    if "when you buy" in t:
        return "on_buy"
    avenge_match = re.search(r"avenge\s*\(\d+\)", t)
    if avenge_match:
        return "avenge"
    if "rally:" in t:
        return "rally"
    if "spellcraft:" in t:
        return "spellcraft"
    return "passive"


def detect_effect_target(text: str) -> str:
    """Determine the primary effect target."""
    t = text.lower()
    if "adjacent" in t:
        return "adjacent"
    # tribe-specific targeting
    tribe_patterns = [
        r"your\s+(murloc|beast|mech|demon|dragon|pirate|elemental|quilboar|naga|undead|beetles|whelps|pirates|elementals|murlocs|beasts|dragons|demons|undead|nagas)s?\b"
    ]
    for pat in tribe_patterns:
        if re.search(pat, t):
            return "tribe"
    if "your other minions" in t or "your other" in t:
        return "all_friendly"
    if "your minions" in t or "your teammate's minions" in t or "all your" in t:
        return "all_friendly"
    if "random enemy" in t or "enemy minion" in t or "enemy" in t:
        return "random_enemy"
    if "minions in the tavern" in t or "right-most minion in the tavern" in t or "tavern" in t:
        return "tavern"
    if "friendly minion" in t or "another friendly" in t:
        return "all_friendly"
    if "a minion" in t or "give a minion" in t:
        return "single_target"
    return "self"


def detect_effect_duration(text: str) -> str:
    """Determine effect duration."""
    t = text.lower()
    if "this game" in t:
        return "this_game"
    if "permanently" in t:
        return "permanent"
    if "until next turn" in t or "for this combat" in t or "this combat" in t:
        return "this_combat"
    return "instant"


def detect_scales_with_board(text: str) -> bool:
    t = text.lower()
    return bool(
        re.search(r"for each\b", t) or
        re.search(r"for every\b", t) or
        "equal to your tier" in t
    )


def detect_avenge_count(text: str):
    m = re.search(r"avenge\s*\((\d+)\)", text.lower())
    if m:
        return int(m.group(1))
    return None


def detect_is_aura(text: str, card_id: str) -> bool:
    """
    True if card is in the explicit aura set, or has 'whenever' plus
    'give' or 'gain' targeting other friendlies.
    """
    if card_id in AURA_CARDS:
        return True
    t = text.lower()
    if "whenever" in t and ("give" in t or "gain" in t):
        # Must target others, not just self
        if (
            "your minions" in t or
            "your other" in t or
            "friendly" in t or
            re.search(r"your\s+\w+s?\b", t)
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_card_entry(name: str, tier: int, atk: int, hp: int,
                     tribes: list, text: str) -> dict:
    card_id = make_card_id(name)
    keywords = detect_keywords(text)
    trigger_type = detect_trigger_type(text)
    effect_target = detect_effect_target(text)
    effect_duration = detect_effect_duration(text)
    scales_with_board = detect_scales_with_board(text)
    avenge_count = detect_avenge_count(text)
    is_multiplier = card_id in MULTIPLIER_CARDS
    is_aura = detect_is_aura(text, card_id)
    has_magnetic = "magnetic" in text.lower()

    return {
        "name": name,
        "tier": tier,
        "base_atk": atk,
        "base_hp": hp,
        "tribes": tribes,
        "keywords": keywords,
        "trigger_type": trigger_type,
        "effect_target": effect_target,
        "effect_duration": effect_duration,
        "scales_with_board": scales_with_board,
        "avenge_count": avenge_count,
        "is_multiplier": is_multiplier,
        "is_aura": is_aura,
        "has_magnetic": has_magnetic,
        "raw_text": text,
    }


def build_from_embedded() -> dict:
    cards = {}
    for tier, entries in TIER_CARDS.items():
        for (name, atk, hp, tribes, text) in entries:
            card_id = make_card_id(name)
            cards[card_id] = build_card_entry(name, tier, atk, hp, tribes, text)
    return cards


# ---------------------------------------------------------------------------
# HearthstoneJSON fetch (optional)
# ---------------------------------------------------------------------------

HEARTHSTONE_JSON_URL = "https://api.hearthstonejson.com/v1/latest/enUS/cards.json"

TRIBE_MAP = {
    "MURLOC":    "MURLOC",
    "BEAST":     "BEAST",
    "MECH":      "MECH",
    "DEMON":     "DEMON",
    "DRAGON":    "DRAGON",
    "PIRATE":    "PIRATE",
    "ELEMENTAL": "ELEMENTAL",
    "QUILBOAR":  "QUILBOAR",
    "NAGA":      "NAGA",
    "UNDEAD":    "UNDEAD",
    "ALL":       "ALL",
}


def fetch_hearthstone_json(timeout: int = 30) -> list:
    """
    Fetch cards.json from HearthstoneJSON API.
    Returns list of raw card dicts, or [] on failure.
    """
    try:
        print(f"Fetching {HEARTHSTONE_JSON_URL} ...", file=sys.stderr)
        req = urllib.request.Request(
            HEARTHSTONE_JSON_URL,
            headers={"User-Agent": "bg-card-pipeline/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        all_cards = json.loads(raw)
        bg_cards = [c for c in all_cards if c.get("isBattlegroundsPoolMinion") is True]
        print(f"  Fetched {len(all_cards)} total cards, {len(bg_cards)} BG pool minions.", file=sys.stderr)
        return bg_cards
    except Exception as exc:
        print(f"  WARNING: fetch failed ({exc}). Proceeding with embedded data only.", file=sys.stderr)
        return []


def merge_api_data(cards: dict, api_cards: list) -> dict:
    """
    Attempt to merge fresh API data into our card dict.
    We match by normalised name. API data wins for base_atk / base_hp
    when available. New cards from the API that aren't in our embedded
    data are added with best-effort parsed fields.
    """
    api_by_name = {}
    for ac in api_cards:
        raw_name = ac.get("name", "")
        nid = make_card_id(raw_name)
        api_by_name[nid] = ac

    merged = 0
    added = 0

    for cid, entry in cards.items():
        if cid in api_by_name:
            ac = api_by_name[cid]
            if "attack" in ac:
                entry["base_atk"] = ac["attack"]
            if "health" in ac:
                entry["base_hp"] = ac["health"]
            merged += 1

    # Add any API cards not already in embedded set
    for nid, ac in api_by_name.items():
        if nid not in cards:
            name = ac.get("name", nid)
            atk = ac.get("attack", 0)
            hp = ac.get("health", 0)
            tier = ac.get("techLevel", 1)
            text = ac.get("text", "")
            # Strip HTML tags from text
            text = re.sub(r"<[^>]+>", "", text)

            raw_races = ac.get("races", ac.get("race", []))
            if isinstance(raw_races, str):
                raw_races = [raw_races]
            tribes = [TRIBE_MAP.get(r.upper(), r.upper()) for r in raw_races]

            cards[nid] = build_card_entry(name, tier, atk, hp, tribes, text)
            added += 1

    print(f"  Merged {merged} existing cards, added {added} new cards from API.", file=sys.stderr)
    return cards


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(cards: dict) -> None:
    total = len(cards)
    print(f"\n=== bg_card_definitions stats ===")
    print(f"Total cards: {total}")

    # By tier
    tier_counts = {}
    for c in cards.values():
        t = c["tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1
    print("\nBy tier:")
    for t in sorted(tier_counts):
        print(f"  Tier {t}: {tier_counts[t]}")

    # By tribe
    tribe_counts = {}
    for c in cards.values():
        for tribe in c["tribes"]:
            tribe_counts[tribe] = tribe_counts.get(tribe, 0) + 1
        if not c["tribes"]:
            tribe_counts["NEUTRAL"] = tribe_counts.get("NEUTRAL", 0) + 1
    print("\nBy tribe:")
    for tribe, count in sorted(tribe_counts.items()):
        print(f"  {tribe}: {count}")

    # By trigger type
    trigger_counts = {}
    for c in cards.values():
        tt = c["trigger_type"]
        trigger_counts[tt] = trigger_counts.get(tt, 0) + 1
    print("\nBy trigger type:")
    for tt, count in sorted(trigger_counts.items(), key=lambda x: -x[1]):
        print(f"  {tt}: {count}")

    # Multipliers
    mults = [c["name"] for c in cards.values() if c["is_multiplier"]]
    print(f"\nMultiplier cards ({len(mults)}): {', '.join(mults)}")

    # Auras
    auras = [c["name"] for c in cards.values() if c["is_aura"]]
    print(f"\nAura cards ({len(auras)}): {', '.join(auras)}")

    # Magnetic
    mags = [c["name"] for c in cards.values() if c["has_magnetic"]]
    print(f"\nMagnetic cards ({len(mags)}): {', '.join(mags)}")

    # Avenge
    avenges = [(c["name"], c["avenge_count"]) for c in cards.values() if c["avenge_count"] is not None]
    print(f"\nAvenge cards ({len(avenges)}):")
    for name, n in sorted(avenges, key=lambda x: x[1]):
        print(f"  {name} (Avenge {n})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build bg_card_definitions.json from embedded CLAUDE.md card list."
    )
    parser.add_argument(
        "--output", "-o",
        default="bg_card_definitions.json",
        help="Output JSON file path (default: bg_card_definitions.json)",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Try to fetch fresh data from HearthstoneJSON API and merge.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print summary statistics after building.",
    )
    args = parser.parse_args()

    # Build from embedded data
    print("Building card definitions from embedded data...", file=sys.stderr)
    cards = build_from_embedded()
    print(f"  Embedded cards loaded: {len(cards)}", file=sys.stderr)

    # Optionally fetch and merge API data
    if args.fetch:
        api_cards = fetch_hearthstone_json()
        if api_cards:
            cards = merge_api_data(cards, api_cards)

    # Assemble output
    output = {
        "version": str(date.today()),
        "total": len(cards),
        "cards": cards,
    }

    # Write JSON
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {len(cards)} cards to {args.output}", file=sys.stderr)

    if args.stats:
        print_stats(cards)


if __name__ == "__main__":
    main()
