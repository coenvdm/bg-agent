"""
bg_card_pipeline.py — Hearthstone Battlegrounds card pipeline.

Parses the embedded card list and optionally fetches fresh data
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
# Last updated: 2026-04-17 from HearthstoneJSON API (270 minions, 7 tiers)
# ---------------------------------------------------------------------------

TIER_CARDS = {
    1: [
        ("Annoy-o-Tron",          1,  2, ["MECH"],                   "Taunt. Divine Shield."),
        ("Aureate Laureate",      1,  1, ["PIRATE"],                 "Divine Shield. Battlecry: Make this minion Golden."),
        ("Cord Puller",           1,  1, ["MECH"],                   "Divine Shield. Deathrattle: Summon a 1/1 Microbot."),
        ("Crackling Cyclone",     2,  1, ["ELEMENTAL"],              "Divine Shield. Windfury."),
        ("Dune Dweller",          3,  2, ["ELEMENTAL"],              "Battlecry: Give Elementals in the Tavern +1/+1 this game."),
        ("Flighty Scout",         3,  3, ["MURLOC"],                 "Start of Combat: If this minion is in your hand, summon a copy of it."),
        ("Gluttonous Trogg",      2,  3, [],                         "Once you buy 4 cards, gain +4/+4."),
        ("Harmless Bonehead",     1,  1, ["UNDEAD"],                 "Deathrattle: Summon two 1/1 Skeletons."),
        ("Manasaber",             4,  1, ["BEAST"],                  "Deathrattle: Summon two 0/1 Cublings with Taunt."),
        ("Ominous Seer",          2,  1, ["DEMON", "NAGA"],          "Battlecry: The next Tavern spell you buy costs (1) less."),
        ("Passenger",             2,  2, [],                         "The first time your team Passes each turn, gain +1/+2."),
        ("Picky Eater",           1,  1, ["DEMON"],                  "Battlecry: Consume a random minion in the Tavern to gain its stats."),
        ("Razorfen Geomancer",    2,  1, ["QUILBOAR"],               "Battlecry: Get 2 Blood Gems."),
        ("Risen Rider",           2,  1, ["UNDEAD"],                 "Taunt. Reborn."),
        ("River Skipper",         1,  1, ["MURLOC"],                 "When you sell this, get a random Tier 1 minion."),
        ("Rot Hide Gnoll",        1,  4, ["UNDEAD"],                 "Has +1 Attack for each friendly minion that died this combat."),
        ("Scarlet Survivor",      3,  3, ["DRAGON"],                 "Once this reaches 6 Attack, gain Divine Shield."),
        ("Southsea Busker",       3,  1, ["PIRATE"],                 "Battlecry: Gain 1 Gold next turn."),
        ("Sun-Bacon Relaxer",     2,  3, ["QUILBOAR"],               "When you sell this, get 2 Blood Gems."),
        ("Surf n' Surf",          1,  1, ["NAGA", "BEAST"],          "Spellcraft: Give a minion \"Deathrattle: Summon a 3/2 Crab\" until next turn."),
        ("Twilight Hatchling",    1,  1, ["DRAGON"],                 "Deathrattle: Summon a 3/3 Whelp that attacks immediately."),
        ("Wrath Weaver",          1,  4, ["DEMON"],                  "After you play a Demon, deal 1 damage to your hero and gain +2/+1."),
    ],
    2: [
        ("Alert Alarmist",        1,  1, ["MECH"],                   "Taunt. Deathrattle: The next Tavern spell you buy costs (1) less."),
        ("Ancestral Automaton",   3,  4, ["MECH"],                   "Has +3/+2 for each other Ancestral Automaton you've summoned this game (wherever this is)."),
        ("Blazing Skyfin",        2,  4, ["MURLOC", "DRAGON"],       "After you trigger a Battlecry, gain +1/+1."),
        ("Bristleback Bully",     3,  2, ["QUILBOAR"],               "Taunt. Deathrattle: Get a Blood Gem that also gives a Quilboar Taunt."),
        ("Defiant Shipwright",    2,  5, ["PIRATE"],                 "Whenever this gains Attack from other sources, gain +1 Health."),
        ("Eternal Knight",        4,  1, ["UNDEAD"],                 "Has +4/+1 for each friendly Eternal Knight that died this game (wherever this is)."),
        ("Expert Aviator",        3,  4, ["MURLOC"],                 "Rally: Summon the highest-Attack minion from your hand for this combat only."),
        ("Fire Baller",           4,  3, ["ELEMENTAL"],              "When you sell this, give your minions +1 Attack. Improve your future Ballers."),
        ("Freedealing Gambler",   3,  3, ["PIRATE"],                 "This minion sells for 3 Gold."),
        ("Friendly Saloonkeeper", 3,  4, [],                         "Battlecry: Your teammate gets a Tavern Coin."),
        ("Gathering Stormer",     5,  1, ["ELEMENTAL"],              "When you sell this, your teammate gains 1 Gold. (Improves each turn!)"),
        ("Generous Geomancer",    1,  1, ["QUILBOAR"],               "Deathrattle: You and your teammate each get a Blood Gem."),
        ("Glowgullet Warlord",    2,  2, ["QUILBOAR"],               "Deathrattle: Summon two 1/1 Quilboar with Taunt. This plays a Blood Gem on them."),
        ("Humming Bird",          1,  4, ["BEAST"],                  "Start of Combat: For the rest of this combat, your Beasts have +1 Attack."),
        ("Intrepid Botanist",     3,  4, [],                         "Choose One - Your Tavern spells give an extra +1 Attack this game; or +1 Health."),
        ("Laboratory Assistant",  3,  4, ["DEMON"],                  "Battlecry: Add a Fodder to your next 3 Refreshes."),
        ("Lava Lurker",           2,  5, ["NAGA"],                   "The first Spellcraft spell played from hand on this each turn is permanent."),
        ("Metallic Hunter",       2,  1, ["MECH"],                   "Deathrattle: Get a Pointy Arrow."),
        ("Nerubian Deathswarmer", 1,  4, ["UNDEAD"],                 "Battlecry: Your Undead have +1 Attack this game (wherever they are)."),
        ("Old Soul",              3,  4, ["UNDEAD"],                 "After 15 friendly minions die while this is in your hand, make this Golden."),
        ("Oozeling Gladiator",    2,  2, [],                         "Battlecry: Get two Slimy Shields that give +1/+1 and Taunt."),
        ("Patient Scout",         1,  1, [],                         "When you sell this, Discover a Tier 1 minion. (Improves each turn!)"),
        ("Prophet of the Boar",   2,  3, [],                         "Taunt. After you play a Quilboar, get a Blood Gem."),
        ("Reef Riffer",           3,  2, ["NAGA"],                   "Spellcraft: Give a minion stats equal to your Tier until next turn."),
        ("Scarlet Skull",         2,  1, ["UNDEAD"],                 "Reborn. Deathrattle: Give a friendly Undead +1/+2."),
        ("Sellemental",           3,  3, ["ELEMENTAL"],              "When you sell this, get a 3/3 Elemental."),
        ("Sewer Rat",             3,  2, ["BEAST"],                  "Deathrattle: Summon a 2/3 Turtle with Taunt."),
        ("Shell Collector",       4,  3, ["NAGA"],                   "Battlecry: Get a Tavern Coin."),
        ("Sleepy Supporter",      4,  3, ["DRAGON"],                 "Rally: Give the minion to the right of this +2/+2."),
        ("Snow Baller",           3,  4, ["ELEMENTAL"],              "When you sell this, give your minions +1 Health. Improve your future Ballers."),
        ("Soul Rewinder",         4,  1, ["DEMON"],                  "After your hero takes damage, rewind it and give this +1 Health."),
        ("Surfing Sylvar",        1,  2, ["PIRATE"],                 "At the end of your turn, give adjacent minions +1 Attack. Repeat for each friendly Golden minion."),
        ("Tad",                   2,  2, ["MURLOC"],                 "When you sell this, get a random Murloc."),
        ("Tarecgosa",             4,  4, ["DRAGON"],                 "This permanently keeps Bonus Keywords and stats gained in combat."),
        ("Tide Raiser",           2,  1, ["NAGA"],                   "Taunt. Deathrattle: Cast Shifting Tide on an adjacent minion."),
        ("Very Hungry Winterfinner", 2, 5, ["MURLOC"],               "Taunt. Whenever this takes damage, give a minion in your hand +2/+1."),
        ("Wanderer Cho",          4,  3, [],                         "One Pass each turn is free."),
    ],
    3: [
        ("Accord-o-Tron",         3,  3, ["MECH"],                   "Magnetic. At the start of your turn, gain 1 Gold."),
        ("Amber Guardian",        3,  2, ["DRAGON"],                 "Taunt. Start of Combat: Give another friendly Dragon +2/+2 and Divine Shield."),
        ("Annoy-o-Module",        2,  4, ["MECH"],                   "Magnetic. Divine Shield. Taunt."),
        ("Auto Assembler",        2,  2, ["MECH"],                   "Magnetic. Deathrattle: Summon an Ancestral Automaton."),
        ("Black Chromadrake",     2,  6, ["DRAGON"],                 "Battlecry: Your Tavern spells give an extra +1 Health this game."),
        ("Blue Chromadrake",      4,  4, ["DRAGON"],                 "Battlecry: Get a random 2-Cost Tavern spell."),
        ("Bottom Feeder",         3,  4, ["MURLOC"],                 "At the end of your turn, you and your teammate each get a random Tier 1 card."),
        ("Bristlemane Scrapsmith",4,  4, ["QUILBOAR"],               "After a friendly minion with Taunt dies, get a Blood Gem."),
        ("Bronze Chromadrake",    5,  3, ["DRAGON"],                 "Battlecry: Give your other Dragons +5 Attack."),
        ("Cadaver Caretaker",     3,  3, ["UNDEAD"],                 "Deathrattle: Summon three 1/1 Skeletons."),
        ("Deadly Spore",          1,  1, [],                         "Venomous."),
        ("Deep Blue Crooner",     2,  4, ["NAGA"],                   "Spellcraft: Give a minion +1/+2 until next turn. Improve your future Deep Blues."),
        ("Deep-Sea Angler",       2,  3, ["NAGA"],                   "Spellcraft: Give a minion +2/+6 and Taunt until next turn."),
        ("Deflect-o-Bot",         3,  2, ["MECH"],                   "Divine Shield. Whenever you summon a Mech during combat, gain +2 Attack and Divine Shield."),
        ("Disguised Graverobber", 4,  4, [],                         "Battlecry: Destroy a friendly Undead to get a plain copy of it."),
        ("Doting Dracthyr",       4,  3, ["DRAGON"],                 "At the end of your turn, give your teammate's minions +1 Attack."),
        ("Dustbone Devastator",   2,  6, ["UNDEAD"],                 "Rally: Your Undead have +1 Attack this game (wherever they are)."),
        ("Felemental",            3,  3, ["ELEMENTAL", "DEMON"],     "Battlecry: Give minions in the Tavern +2/+1 this game."),
        ("Floating Watcher",      4,  4, ["DEMON"],                  "Whenever your hero takes damage on your turn, gain +2/+2."),
        ("Green Chromadrake",     3,  5, ["DRAGON"],                 "Battlecry: Give your other Dragons +5 Health."),
        ("Gunpowder Courier",     2,  6, ["PIRATE"],                 "Whenever you spend 5 Gold, give your Pirates +1 Attack."),
        ("Handless Forsaken",     2,  1, ["UNDEAD"],                 "Deathrattle: Summon a 2/1 Hand with Reborn."),
        ("Hardy Orca",            1,  6, ["BEAST"],                  "Taunt. Whenever this takes damage, give your other minions +1/+1."),
        ("Jumping Jack",          3,  4, ["ALL"],                    "After the first time this is sold, Pass it."),
        ("King Bagurgle",         2,  3, ["MURLOC"],                 "Battlecry: Give all other Murlocs in your hand and board +2/+3."),
        ("Leeching Felhound",     3,  3, ["DEMON"],                  "This costs Health instead of Gold to buy."),
        ("Lost City Looter",      1,  1, ["PIRATE"],                 "Taunt. At the end of your turn, get a random Bounty."),
        ("Moon-Bacon Jazzer",     2,  5, ["QUILBOAR"],               "Battlecry: Your Blood Gems give an extra +1 Health this game."),
        ("Mummifier",             5,  2, ["UNDEAD"],                 "Deathrattle: Give a different friendly Undead Reborn."),
        ("Orc-estra Conductor",   4,  4, [],                         "Battlecry: Give a minion +2/+2 (Improved by each Orc-estra your team has played this game)."),
        ("Peggy Sturdybone",      2,  1, ["PIRATE"],                 "Whenever a card is added to your hand, give another friendly Pirate +2/+1."),
        ("Plunder Pal",           2,  2, ["PIRATE"],                 "At the start of your turn, you and your teammate each gain 1 Gold."),
        ("Prickly Piper",         5,  1, ["QUILBOAR"],               "Deathrattle: Your Blood Gems give an extra +1 Attack this game."),
        ("Puddle Prancer",        4,  4, ["MURLOC"],                 "After this is Passed, gain +4/+4."),
        ("Pufferquil",            2,  6, ["QUILBOAR", "NAGA"],       "Whenever a spell is cast on this, gain Venomous until next turn."),
        ("Red Chromadrake",       6,  2, ["DRAGON"],                 "Battlecry: Your Tavern spells give an extra +1 Attack this game."),
        ("Roaring Recruiter",     2,  8, ["DRAGON"],                 "Whenever another friendly Dragon attacks, give it +3/+1."),
        ("Scourfin",              4,  3, ["MURLOC"],                 "Deathrattle: Give a random minion in your hand +7/+7."),
        ("Shoalfin Mystic",       4,  4, ["MURLOC"],                 "When you sell this, your Tavern spells give an extra +1/+1 this game."),
        ("Skulking Bristlemane",  5,  2, ["QUILBOAR"],               "Taunt. Deathrattle: This plays a permanent Blood Gem on adjacent minions."),
        ("Sly Raptor",            1,  3, ["BEAST"],                  "Deathrattle: Summon a random Beast. Set its stats to 6/6."),
        ("Sprightly Scarab",      3,  1, ["BEAST"],                  "Choose One - Give a Beast +1/+1 and Reborn; or +4 Attack and Windfury."),
        ("Technical Element",     5,  6, ["ELEMENTAL", "MECH"],      "Magnetic. Can Magnetize to both Mechs and Elementals."),
        ("Timecap'n Hooktail",    1,  4, ["DRAGON", "PIRATE"],       "Whenever you cast a Tavern spell, give your minions +1 Attack."),
        ("Waveling",              5,  1, ["ELEMENTAL"],              "Deathrattle: After the Tavern is Refreshed this game, give a random minion in it +4/+4."),
        ("Wheeled Crewmate",      6,  3, ["PIRATE"],                 "Deathrattle: Reduce the Cost of upgrading your team's Taverns by (1)."),
        ("Wildfire Elemental",    6,  3, ["ELEMENTAL"],              "After this attacks and kills a minion, deal excess damage to an adjacent minion."),
    ],
    4: [
        ("Abyssal Bruiser",       1,  1, ["NAGA"],                   "Divine Shield. Has +1/+1 for each Tavern spell you've cast this game."),
        ("Banana Slamma",         3,  6, ["BEAST"],                  "After you summon a Beast in combat, double its Attack."),
        ("Bigwig Bandit",         4,  6, ["PIRATE"],                 "Rally: Get a random Bounty."),
        ("Blade Collector",       3,  2, ["PIRATE"],                 "Also damages the minions next to whomever this attacks."),
        ("Bream Counter",         4,  4, ["MURLOC"],                 "While this is in your hand, after you play a Murloc, gain +4/+4."),
        ("Deepwater Chieftain",   3,  2, ["MURLOC"],                 "Battlecry and Deathrattle: Get a Deepwater Clan."),
        ("Determined Defender",   5,  5, [],                         "Taunt. Deathrattle: Give adjacent minions +5/+5 and Taunt."),
        ("Diremuck Forager",      5,  6, ["MURLOC"],                 "Start of Combat: When you have space, summon the highest-Attack minion from your hand for this combat only."),
        ("Dual-Wield Corsair",    2,  4, ["PIRATE"],                 "Whenever you spend 5 Gold, give two friendly Pirates +3/+3."),
        ("Egg of the Endtimes",   0,  5, [],                         "After this is in your hand for 2 turns, choose a Tier 6 Dragon to hatch into."),
        ("En-Djinn Blazer",       5,  5, ["ELEMENTAL"],              "Battlecry: After the Tavern is Refreshed this game, give a random minion in it +8/+8."),
        ("Enchanted Sentinel",    3,  5, ["MECH"],                   "Magnetic. Your Tavern spells give an extra +1/+1."),
        ("Eternal Tycoon",        2,  6, ["UNDEAD"],                 "Avenge (5): Summon an Eternal Knight. It attacks immediately."),
        ("Fearless Foodie",       2,  4, ["QUILBOAR"],               "Choose One - Your Blood Gems give an extra +1/+1 this game; or Get 4 Blood Gems."),
        ("Feisty Freshwater",     6,  4, ["ELEMENTAL"],              "Deathrattle: You and your teammate each gain two free Refreshes."),
        ("Flaming Enforcer",      4,  5, ["ELEMENTAL", "DEMON"],     "At the end of your turn, consume the highest-Health minion in the Tavern to gain its stats."),
        ("Friendly Geist",        6,  3, ["UNDEAD"],                 "Deathrattle: Your Tavern spells give an extra +1 Attack this game."),
        ("Geomagus Roogug",       4,  6, ["QUILBOAR"],               "Divine Shield. Whenever a Blood Gem is played on this, this plays a Blood Gem on a different friendly minion."),
        ("Grave Narrator",        2,  7, ["UNDEAD"],                 "Avenge (3): Your teammate gets a random minion of their most common type."),
        ("Heroic Underdog",       1, 10, [],                         "Stealth. Rally: Gain the target's Attack."),
        ("Hired Ritualist",       3,  6, ["QUILBOAR"],               "Once per turn, after a Blood Gem is played on this, gain 2 Gold."),
        ("Humon'gozz",            5,  5, [],                         "Divine Shield. Your Tavern spells give an extra +1/+2."),
        ("Hunting Tiger Shark",   3,  5, ["BEAST"],                  "Battlecry: Discover a Beast."),
        ("Ichoron the Protector", 3,  1, ["ELEMENTAL"],              "Divine Shield. Whenever you play an Elemental, give it Divine Shield until next turn."),
        ("Imposing Percussionist",4,  4, ["DEMON"],                  "Battlecry: Discover a Demon. Deal damage to your hero equal to its Tier."),
        ("Incubation Researcher", 2,  8, ["DRAGON"],                 "Avenge (4): Get a random Chromadrake."),
        ("Leyline Surfacer",      4,  6, ["ELEMENTAL"],              "Battlecry and Deathrattle: Get an Arcane Absorption."),
        ("Lovesick Balladist",    3,  2, ["PIRATE"],                 "Battlecry: Give a Pirate +1 Health. (Improved by each Gold you spent this turn!)"),
        ("Malchezaar, Prince of Dance", 5, 4, ["DEMON"],             "Two Refreshes each turn cost Health instead of Gold."),
        ("Mama Mrrglton",         6,  3, ["MURLOC"],                 "Battlecry: Give your other Murlocs +3 Attack. (Improved by each Mrrglton you played this game!)"),
        ("Mantid King",           3,  3, [],                         "After your team Passes, randomly gain Venomous, Taunt, or Divine Shield until next turn."),
        ("Marquee Ticker",        1,  5, ["MECH"],                   "At the end of your turn, get a random Tavern spell."),
        ("Mirror Monster",        4,  4, ["ALL"],                    "When you buy or Discover this, get an extra copy and Pass it."),
        ("Monstrous Macaw",       5,  4, ["BEAST"],                  "Rally: Trigger your left-most Deathrattle (except this minion's)."),
        ("Papa Mrrglton",         3,  6, ["MURLOC"],                 "Battlecry: Give your other Murlocs +3 Health. (Improved by each Mrrglton you played this game!)"),
        ("Persistent Poet",       2,  3, ["DRAGON"],                 "Divine Shield. Adjacent Dragons permanently keep Bonus Keywords and stats gained in combat."),
        ("Plaguerunner",          4,  2, ["UNDEAD"],                 "Deathrattle: Your Undead have +2 Attack this game, wherever they are. (+4 if this died outside combat!)"),
        ("Private Chef",          5,  4, ["NAGA"],                   "Spellcraft: Choose a minion. Get a different random minion of its type, then Pass it."),
        ("Prized Promo-Drake",    1,  1, ["DRAGON"],                 "Start of Combat: Give your Dragons +4/+4."),
        ("Prosthetic Hand",       3,  1, ["UNDEAD", "MECH"],         "Magnetic. Reborn. Can Magnetize to Mechs or Undead."),
        ("Redtusk Thornraiser",   1,  6, ["QUILBOAR"],               "At the end of your turn, get a Blood Gem that also gives a Quilboar Reborn."),
        ("Refreshing Anomaly",    4,  5, ["ELEMENTAL"],              "Battlecry: Gain 2 free Refreshes."),
        ("Rimescale Priestess",   3,  3, ["NAGA"],                   "Spellcraft: Get a random Tavern spell that gives stats."),
        ("Roving Sailor",         7,  3, ["PIRATE"],                 "Battlecry: Give a friendly minion +2/+2. (Improved by each Tavern spell you cast this turn!)"),
        ("Rylak Metalhead",       5,  3, ["BEAST"],                  "Taunt. Deathrattle: Trigger the Battlecry of an adjacent minion."),
        ("San'layn Scribe",       4,  4, ["UNDEAD"],                 "Has +4/+4 for each of your team's San'layn Scribes that died this game (wherever this is)."),
        ("Seafloor Recruiter",    3,  5, ["NAGA"],                   "Rally: Cast Chef's Choice on the minion to the right."),
        ("Shadowdancer",          4,  2, ["DEMON"],                  "Taunt. Deathrattle: Get a Staff of Enrichment."),
        ("Shifty Snake",          6,  1, ["BEAST"],                  "Deathrattle: Your teammate gets a random Deathrattle minion."),
        ("Sin'dorei Straight Shot",3, 4, [],                         "Divine Shield. Windfury. Rally: Remove Reborn and Taunt from the target."),
        ("Stomping Stegodon",     3,  3, ["BEAST"],                  "Rally: Give your other Beasts +4 Attack and this Rally."),
        ("Tavern Tempest",        2,  2, ["ELEMENTAL"],              "Battlecry: Get a random Elemental."),
        ("Tortollan Blue Shell",  3,  6, [],                         "If you lost your last combat, this minion sells for 5 Gold."),
        ("Trigore the Lasher",    9,  3, ["BEAST"],                  "Whenever another friendly Beast takes damage, gain +2 Health permanently."),
        ("Tunnel Blaster",        3,  7, [],                         "Taunt. Deathrattle: Deal 3 damage to all minions."),
        ("Waverider",             2,  8, ["NAGA"],                   "Spellcraft: Give a minion +2/+2. If it's a Naga, also give it Windfury until next turn."),
        ("Woodland Defiler",      5,  6, ["DEMON"],                  "At the end of your turn, add a Fodder to your next 3 Refreshes."),
        ("Wyvern Outrider",       2,  8, ["BEAST"],                  "Whenever this takes damage, gain a free Refresh. (3 times per turn.)"),
        ("Zesty Shaker",          6,  7, ["NAGA"],                   "Once per turn, when a Spellcraft spell is played on this, get a new copy of it."),
    ],
    5: [
        ("Air Revenant",          3,  6, ["ELEMENTAL"],              "After you spend 7 Gold, cast Easterly Winds."),
        ("Ashen Corruptor",       6,  6, ["DEMON"],                  "After your hero takes damage, rewind it and give minions in the Tavern +1/+1 this turn."),
        ("Bazaar Dealer",         4,  6, ["DEMON"],                  "One Tavern spell each turn costs Health instead of Gold to buy."),
        ("Bile Spitter",          1, 10, ["MURLOC"],                 "Venomous. Rally: Give another friendly Murloc Venomous."),
        ("Brann Bronzebeard",     2,  4, [],                         "Your Battlecries trigger twice."),
        ("Brazen Buccaneer",      4,  4, ["PIRATE"],                 "At the end of your turn, give your left-most Pirate +2/+2. Repeat for each card you played this turn."),
        ("Cataclysmic Harbinger", 6, 10, [],                         "At the end of your turn, get a copy of the last Tavern spell you cast."),
        ("Catacomb Crasher",      6, 10, ["UNDEAD"],                 "Whenever you would summon a minion that doesn't fit in your warband, give your minions +2/+1 permanently."),
        ("Charging Czarina",      6,  2, ["MECH"],                   "Divine Shield. Whenever you cast a Tavern spell, give your minions with Divine Shield +3 Attack."),
        ("Cousin Errgl",          5,  5, ["MURLOC"],                 "At the end of your turn, get a Mama Mrrglton or a Papa Mrrglton."),
        ("Darkcrest Strategist",  4,  5, ["NAGA"],                   "Spellcraft: Get a random Tier 1 Naga. (Improves each turn!)"),
        ("Darkgaze Elder",        5,  6, ["QUILBOAR"],               "Whenever you spend 6 Gold, this plays a Blood Gem on all your Quilboar."),
        ("Divine Sparkbot",       4,  2, ["MECH"],                   "Taunt. Divine Shield. Deathrattle: Get a Sanctify."),
        ("Draconic Warden",       7,  4, ["DRAGON"],                 "Battlecry and Deathrattle: Get a random Chromadrake."),
        ("Drakkari Enchanter",    1,  5, [],                         "Your end of turn effects trigger twice."),
        ("Drustfallen Butcher",   2,  9, ["UNDEAD"],                 "Avenge (4): Get a Butchering."),
        ("Felfire Conjurer",      6,  5, ["DEMON", "DRAGON"],        "At the end of your turn, your Tavern spells give an extra +1/+1 this game."),
        ("Firelands Fugitive",    5,  7, ["ELEMENTAL"],              "Battlecry: Get a Conflagration."),
        ("Gem Smuggler",          4,  5, ["QUILBOAR"],               "Battlecry: This plays 2 Blood Gems on all your other minions."),
        ("Glowscale",             4,  6, ["NAGA"],                   "Taunt. Spellcraft: Give a minion Divine Shield until next turn."),
        ("Hog Watcher",           5,  5, [],                         "Battlecry: Get a Blood Gem that also gives a Quilboar Divine Shield."),
        ("Hot-Air Surveyor",      4,  8, ["QUILBOAR"],               "Blood Gems played from your hand cast an extra time."),
        ("Iridescent Skyblazer",  3,  8, ["BEAST"],                  "Whenever a friendly Beast takes damage, give a friendly Beast other than it +3/+1 permanently."),
        ("Kangor's Apprentice",   3,  6, [],                         "Deathrattle: Summon plain copies of your first 2 Mechs that died this combat."),
        ("Lantern Lava",          5,  5, ["ELEMENTAL"],              "Get a plain copy of the first Elemental you sell each turn (except Lantern Lava)."),
        ("Leeroy the Reckless",   6,  2, [],                         "Deathrattle: Destroy the minion that killed this."),
        ("Living Azerite",        5,  5, ["ELEMENTAL"],              "Whenever you cast a Tavern spell, give Elementals in the Tavern +3/+3 this game."),
        ("Lurking Leviathan",     3,  8, ["BEAST"],                  "Whenever you summon a Beast, give it +2 Attack and improve this permanently."),
        ("Maelstrom Emergent",    2,  7, ["NAGA"],                   "Your Tavern spells cast an extra time in combat."),
        ("Magnanimoose",          5,  2, ["BEAST"],                  "Deathrattle: Summon a copy of a minion from your teammate's warband. Set its Health to 1 (except Magnanimoose)."),
        ("Man'ari Messenger",     9,  6, ["DEMON"],                  "Battlecry: Minions in your team's Taverns have +1/+1 this game."),
        ("Mrglin' Burglar",      10,  8, ["MURLOC"],                 "After you play a Murloc, give a friendly minion and a minion in your hand +5/+4."),
        ("Nalaa the Redeemer",    5,  7, [],                         "Whenever you cast a Tavern spell, give a friendly minion of each type +3/+2."),
        ("Nightmare Par-tea Guest",3, 3, ["ALL"],                    "Battlecry and Deathrattle: Get a Misplaced Tea Set."),
        ("Primalfin Lookout",     3,  2, ["MURLOC"],                 "Battlecry: If you control another Murloc, Discover a Murloc."),
        ("Proud Privateer",       8,  8, ["PIRATE"],                 "Your Bounties cast twice."),
        ("Ring Bearer",           3,  8, ["NAGA", "DRAGON"],         "Whenever 2 friendly minions attack, cast Shiny Ring."),
        ("Rodeo Performer",       3,  4, [],                         "Battlecry: Discover a Tavern spell."),
        ("Scrap Scraper",         6,  5, ["MECH"],                   "Deathrattle: Get a random Magnetic Mech."),
        ("Selfless Sightseer",    6,  2, ["DRAGON"],                 "Battlecry: Increase your team's maximum Gold by (1)."),
        ("Sewer Lord",            4,  6, ["BEAST"],                  "Deathrattle: Summon two Sewer Rats that summon 2/3 Turtles with Taunt."),
        ("Shipwrecked Rascal",    5,  4, ["PIRATE"],                 "Battlecry and Deathrattle: Get a random Bounty."),
        ("Sinrunner Blanchy",     4,  4, ["UNDEAD", "BEAST"],        "Reborn. This is Reborn with full stats and Bonus Keywords."),
        ("Skeletal Strafer",      6,  6, ["UNDEAD"],                 "At the end of your turn, give your minions +1/+1. Avenge (2): Improve this permanently."),
        ("Spiked Savior",         8,  2, ["BEAST"],                  "Taunt. Reborn. Deathrattle: Give your minions +1 Health and deal 1 damage to them."),
        ("Storm Splitter",        5,  5, ["NAGA"],                   "Once per turn, after you Pass a Tavern spell, get a new copy of it."),
        ("Support System",        4,  5, ["MECH"],                   "At the end of your turn, give a minion in your teammate's warband Divine Shield."),
        ("Three Lil' Quilboar",   3,  3, ["QUILBOAR"],               "Deathrattle: This plays 3 Blood Gems on all your Quilboar."),
        ("Tichondrius",           3,  6, ["DEMON"],                  "After your hero takes damage, give your Demons +3/+2."),
        ("Titus Rivendare",       1,  7, [],                         "Your Deathrattles trigger an extra time."),
        ("Tranquil Meditative",   3,  8, ["NAGA"],                   "Spellcraft: Your Tavern spells give an extra +1/+1 this game."),
        ("Twilight Broodmother",  7,  4, ["DRAGON"],                 "Deathrattle: Summon 2 Twilight Hatchlings. Give them Taunt."),
        ("Twisted Wrathguard",    4,  4, ["DEMON"],                  "After you sell a minion, add a Fodder to your next Refresh."),
        ("Void Pup Trainer",      6,  6, ["DEMON"],                  "Battlecry: Give minions in the Tavern from Tier 3 and below +4/+4 this game."),
        ("Well Wisher",           6,  6, [],                         "Spellcraft: Pass a different non-Golden minion."),
        ("Wintergrasp Ghoul",     5,  3, ["UNDEAD"],                 "Deathrattle: Get a Tomb Turning."),
    ],
    6: [
        ("Balinda Stonehearth",   6,  6, [],                         "Your spells that target friendly minions cast twice."),
        ("Batty Terrorguard",     6,  2, ["DEMON"],                  "After you cast a Tavern spell, another friendly Demon consumes a minion in the Tavern to gain its stats."),
        ("Bristlebach",           3, 10, ["QUILBOAR"],               "Avenge (2): This plays 2 Blood Gems on all your Quilboar."),
        ("Choral Mrrrglr",        6,  6, ["MURLOC"],                 "Start of Combat: Gain the stats of all the minions in your hand."),
        ("Consummate Conqueror",  9,  7, ["DEMON"],                  "Whenever a minion is consumed, give minions in the Tavern +1/+1 this turn."),
        ("Dark Dazzler",          4,  7, ["DEMON"],                  "After your teammate sells a minion, gain its stats. (Once per turn.)"),
        ("Dastardly Drust",       5,  3, ["PIRATE"],                 "Whenever you get a Pirate, give your minions +2/+1. Give Golden ones +4/+2 instead."),
        ("Deathly Striker",       8,  8, ["UNDEAD"],                 "Avenge (4): Get a random Undead. Deathrattle: Summon it from your hand for this combat only."),
        ("Earthsong Shaman",      4,  5, ["QUILBOAR"],               "Windfury. At the end of your turn, play a Blood Gem on all your minions. Repeat for each Bonus Keyword this has."),
        ("Elemental of Surprise", 8,  8, ["ELEMENTAL"],              "Divine Shield. This minion can triple with any Elemental."),
        ("Eternal Summoner",      8,  1, ["UNDEAD"],                 "Reborn. Deathrattle: Summon 1 Eternal Knight."),
        ("Falling Sky Golem",     4,  2, ["MECH"],                   "Divine Shield. Has +4/+2 for each Deathrattle you've triggered this game (wherever this is)."),
        ("Famished Felbat",       9,  5, ["DEMON"],                  "At the end of your turn, your Demons each consume a minion in the Tavern to gain its stats."),
        ("Fire-forged Evoker",    8,  5, ["DRAGON"],                 "Start of Combat: Give your Dragons +1/+1. Improves permanently after you cast a Tavern spell."),
        ("Forsaken Weaver",       3,  8, ["UNDEAD"],                 "After you cast a Tavern spell, your Undead have +1 Attack this game (wherever they are)."),
        ("Goldrinn, the Great Wolf", 8, 8, ["BEAST"],                "Deathrattle: For the rest of this combat, your Beasts have +8/+8."),
        ("Groundbreaker",         5,  4, ["NAGA"],                   "After you play a Naga, gain +1/+1. (Improved by every 4 spells you've cast this game!)"),
        ("Ignition Specialist",   8,  8, ["DRAGON"],                 "At the end of your turn, get 2 random Tavern spells."),
        ("Junk Jouster",          8,  7, ["MECH"],                   "After you Magnetize a minion, give your minions +6/+6."),
        ("Kalecgos, Arcane Aspect", 4, 12, ["DRAGON"],               "After you trigger a Battlecry, give your Dragons +2/+2."),
        ("Loyal Mobster",         6,  5, ["QUILBOAR"],               "At the end of your turn, this plays a Blood Gem on all your teammate's minions."),
        ("Magicfin Mycologist",   4,  8, ["MURLOC"],                 "Once per turn, after you buy a Tavern spell, get a 1/1 Murloc and teach it that spell."),
        ("Moonsteel Juggernaut",  6,  6, ["MECH"],                   "At the end of your turn, get a 6/6 Magnetic Satellite and improve this."),
        ("Nightbane, Ignited",   16,  8, ["UNDEAD", "DRAGON"],       "Taunt. Deathrattle: Give 2 different friendly minions this minion's Attack."),
        ("One-Amalgam Tour Group",6,  7, ["ALL"],                    "Whenever you play a card, give friendly minions of its Tier or lower +2/+1."),
        ("P-0UL-TR-0N",           8,  8, ["MECH", "BEAST"],          "Avenge (4): Gain Divine Shield and attack immediately."),
        ("Primitive Painter",     3,  8, ["MURLOC"],                 "After you play a card from Tier 3 or below, give your Murlocs +1/+2."),
        ("Rabid Panther",         4,  8, ["BEAST"],                  "After you play a Beast, give your Beasts +3/+3 and deal 1 damage to them."),
        ("Ruthless Queensguard",  3,  3, ["NAGA"],                   "Battlecry, Deathrattle, and Rally: Cast Queen's Command."),
        ("Ship Jumper",           6,  6, ["PIRATE"],                 "Deathrattle: Summon a 1/1 Sky Pirate and give it this minion's Attack. It attacks immediately."),
        ("Ship Master Eudora",   10,  5, ["PIRATE"],                 "Deathrattle: Give your minions +8/+8. Golden ones keep it permanently."),
        ("Sky Admiral Rogers",    4,  6, ["PIRATE"],                 "After you spend 10 Gold, get a random Bounty."),
        ("Tidemistress Athissa",  6,  7, ["NAGA"],                   "Whenever you cast a spell, give all your Naga +1/+1 permanently."),
        ("Transport Reactor",     1,  1, ["MECH"],                   "Magnetic. Has +1/+1 for each time your team has Passed this game (wherever this is)."),
        ("Ultraviolet Ascendant", 6,  3, ["ELEMENTAL"],              "Start of Combat: Give your other Elementals +3/+2. (Improves after you play an Elemental!)"),
        ("Vinespeaker",           7,  8, ["QUILBOAR"],               "After a friendly Deathrattle minion dies, your Blood Gems give an extra +1 Attack this game."),
    ],
    7: [
        ("Captain Sanders",       9,  9, ["PIRATE"],                 "Battlecry: Make a friendly minion from Tier 6 or below Golden."),
        ("Champion of Sargeras", 10, 10, ["DEMON"],                  "Battlecry and Deathrattle: Minions in the Tavern have +5/+5 this game."),
        ("Futurefin",             7, 13, ["MURLOC"],                 "At the end of your turn, give this minion's stats to the left-most minion in your hand."),
        ("Highkeeper Ra",         6,  6, [],                         "Battlecry, Deathrattle, and Rally: Get a random Tier 6 minion."),
        ("Obsidian Ravager",      7,  7, ["DRAGON"],                 "Rally: Deal damage equal to this minion's Attack to the target and an adjacent minion."),
        ("Polarizing Beatboxer",  5, 10, ["MECH"],                   "Whenever you Magnetize to a different minion, it also Magnetizes to this."),
        ("Sandy",                 1,  1, [],                         "Start of Combat: Transform into a copy of your teammate's highest-Health minion."),
        ("Sanguine Champion",    18,  3, ["QUILBOAR"],               "Battlecry and Deathrattle: Your Blood Gems give an extra +1/+1 this game."),
        ("Sea Witch Zar'jira",    4,  5, ["NAGA"],                   "Spellcraft: Choose a different minion in the Tavern to get a copy of."),
        ("Stalwart Kodo",        16, 32, ["BEAST"],                  "After you summon a minion in combat, give it this minion's maximum stats. (3 times per combat.)"),
        ("Stitched Salvager",    16,  4, ["UNDEAD"],                 "Start of Combat: Destroy the minion to the left. Deathrattle: Summon an exact copy of it. (Except Stitched Salvager.)"),
        ("Stone Age Slab",        5, 10, ["ELEMENTAL"],              "After you buy a minion, give it +10/+10 and double its stats. (Once per turn.)"),
        ("The Last One Standing",12, 12, ["ALL"],                    "Rally: Give a friendly minion of each type +12/+12 permanently."),
    ],
}

# ---------------------------------------------------------------------------
# Multiplier and aura card sets (for fast lookup)
# ---------------------------------------------------------------------------

# Cards that change how many times effects trigger — detect FIRST
MULTIPLIER_CARDS = {
    "brann_bronzebeard",       # battlecries trigger twice
    "titus_rivendare",         # deathrattles trigger extra time
    "drakkari_enchanter",      # end-of-turn effects trigger twice
    "balinda_stonehearth",     # spells targeting friendlies cast twice
    "hot-air_surveyor",        # Blood Gems cast extra time
    "maelstrom_emergent",      # Tavern spells cast extra time in combat
}

# Cards with passive auras or reactive triggers boosting other friendlies
AURA_CARDS = {
    "twilight_watcher",
    "roaring_recruiter",
    "hardy_orca",
    "iridescent_skyblazer",
    "timecapn_hooktail",
    "junk_jouster",
    "geomagus_roogug",
    "tidemistress_athissa",
    "one-amalgam_tour_group",
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
    "MECHANICAL": "MECH",
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


def fetch_hearthstone_json(timeout: int = 60) -> tuple:
    """
    Fetch cards.json from HearthstoneJSON API.
    Returns (bg_minions, bg_trinkets) lists, or ([], []) on failure.
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
        bg_minions = [c for c in all_cards if c.get("isBattlegroundsPoolMinion") is True]
        bg_trinkets = [
            c for c in all_cards
            if c.get("type") == "BATTLEGROUND_TRINKET"
            and "Portrait" not in c.get("name", "")
            and "Sticker" not in c.get("name", "")
            and "battlegroundsNormalDbfId" not in c
        ]
        print(
            f"  Fetched {len(all_cards)} total cards, "
            f"{len(bg_minions)} BG pool minions, "
            f"{len(bg_trinkets)} trinkets.",
            file=sys.stderr,
        )
        return bg_minions, bg_trinkets
    except Exception as exc:
        print(f"  WARNING: fetch failed ({exc}). Proceeding with embedded data only.", file=sys.stderr)
        return [], []


def _clean_api_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = re.sub(r"\[x\]", "", text)
    text = re.sub(r"\|4\s*\([^)]+\)", "", text)
    text = re.sub(r"[\x00-\x1f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def merge_api_data(cards: dict, api_minions: list) -> dict:
    """
    Merge fresh API minion data into our card dict.
    API data wins for base_atk / base_hp. New cards are added.
    """
    api_by_name = {}
    for ac in api_minions:
        nid = make_card_id(ac.get("name", ""))
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

    for nid, ac in api_by_name.items():
        if nid not in cards:
            name = ac.get("name", nid)
            atk = ac.get("attack", 0)
            hp = ac.get("health", 0)
            tier = ac.get("techLevel", 1)
            text = _clean_api_text(ac.get("text", ""))
            raw_races = ac.get("races", ac.get("race", []))
            if isinstance(raw_races, str):
                raw_races = [raw_races]
            tribes = [TRIBE_MAP.get(r.upper(), r.upper()) for r in raw_races]
            cards[nid] = build_card_entry(name, tier, atk, hp, tribes, text)
            added += 1

    print(f"  Merged {merged} existing cards, added {added} new cards from API.", file=sys.stderr)
    return cards


def build_trinket_list(api_trinkets: list) -> list:
    """Build a simplified list of trinket dicts from API data."""
    trinkets = []
    for c in api_trinkets:
        cost = c.get("cost", 0)
        trinkets.append({
            "name": c.get("name", ""),
            "cost": cost,
            "tier": "lesser" if cost <= 3 else "greater",
            "text": _clean_api_text(c.get("text", "")),
        })
    return sorted(trinkets, key=lambda x: (x["tier"], x["name"]))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(cards: dict) -> None:
    total = len(cards)
    print(f"\n=== bg_card_definitions stats ===")
    print(f"Total cards: {total}")

    tier_counts = {}
    for c in cards.values():
        t = c["tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1
    print("\nBy tier:")
    for t in sorted(tier_counts):
        print(f"  Tier {t}: {tier_counts[t]}")

    tribe_counts = {}
    for c in cards.values():
        for tribe in c["tribes"]:
            tribe_counts[tribe] = tribe_counts.get(tribe, 0) + 1
        if not c["tribes"]:
            tribe_counts["NEUTRAL"] = tribe_counts.get("NEUTRAL", 0) + 1
    print("\nBy tribe:")
    for tribe, count in sorted(tribe_counts.items()):
        print(f"  {tribe}: {count}")

    trigger_counts = {}
    for c in cards.values():
        tt = c["trigger_type"]
        trigger_counts[tt] = trigger_counts.get(tt, 0) + 1
    print("\nBy trigger type:")
    for tt, count in sorted(trigger_counts.items(), key=lambda x: -x[1]):
        print(f"  {tt}: {count}")

    mults = [c["name"] for c in cards.values() if c["is_multiplier"]]
    print(f"\nMultiplier cards ({len(mults)}): {', '.join(mults)}")

    auras = [c["name"] for c in cards.values() if c["is_aura"]]
    print(f"\nAura cards ({len(auras)}): {', '.join(auras)}")

    mags = [c["name"] for c in cards.values() if c["has_magnetic"]]
    print(f"\nMagnetic cards ({len(mags)}): {', '.join(mags)}")

    avenges = [(c["name"], c["avenge_count"]) for c in cards.values() if c["avenge_count"] is not None]
    print(f"\nAvenge cards ({len(avenges)}):")
    for name, n in sorted(avenges, key=lambda x: x[1]):
        print(f"  {name} (Avenge {n})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build bg_card_definitions.json from embedded card list."
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

    print("Building card definitions from embedded data...", file=sys.stderr)
    cards = build_from_embedded()
    print(f"  Embedded cards loaded: {len(cards)}", file=sys.stderr)

    trinkets = []
    if args.fetch:
        api_minions, api_trinkets = fetch_hearthstone_json()
        if api_minions:
            cards = merge_api_data(cards, api_minions)
        if api_trinkets:
            trinkets = build_trinket_list(api_trinkets)
            print(f"  Trinkets loaded: {len(trinkets)}", file=sys.stderr)

    output = {
        "version": str(date.today()),
        "total": len(cards),
        "cards": cards,
    }
    if trinkets:
        output["trinkets"] = trinkets
        output["trinket_count"] = len(trinkets)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {len(cards)} cards to {args.output}", file=sys.stderr)

    if args.stats:
        print_stats(cards)


if __name__ == "__main__":
    main()
