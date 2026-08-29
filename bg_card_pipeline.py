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
from typing import Dict

# ---------------------------------------------------------------------------
# Embedded card data: (name, atk, hp, tribes_list, text)
# Last updated: 2026-08-29 from HearthstoneJSON API (275 minions, 7 tiers)
# ---------------------------------------------------------------------------

TIER_CARDS = {
    1: [
        ("Aureate Laureate",           2,  2, ["PIRATE"],                  "Divine Shield. This minion is always Golden, but doesn't give a Triple Reward."),
        ("Buzzing Vermin",             1,  1, ["BEAST"],                   "Taunt. Deathrattle: Summon a 2/2 Beetle."),
        ("Cord Puller",                1,  1, ["MECH"],                    "Divine Shield. Deathrattle: Summon a 1/1 Microbot."),
        ("Crackling Cyclone",          2,  1, ["ELEMENTAL"],               "Divine Shield. Windfury."),
        ("Fleeing Fugitive",           5,  2, ["NAGA"],                    "Whenever you cast a spell on this, gain +1 Health."),
        ("Flighty Scout",              3,  3, ["MURLOC"],                  "Start of Combat: If this minion is in your hand, summon a copy of it."),
        ("Flittering Bat",             1,  4, ["BEAST"],                   "Rally: Summon a 1/1 Beast."),
        ("Glim Guardian",              1,  4, ["DRAGON"],                  "Rally: Gain +2 Attack."),
        ("Harmless Bonehead",          1,  1, ["UNDEAD"],                  "Deathrattle: Summon two 1/1 Skeletons."),
        ("Lullabot",                   2,  2, ["MECH"],                    "Magnetic. At the end of your turn, gain +1 Health."),
        ("Mini-Myrmidon",              1,  4, ["NAGA"],                    "Spellcraft: Give a minion +2 Attack until next turn."),
        ("Molten Rock",                3,  3, ["ELEMENTAL"],               "After you play an Elemental, gain +1 Health."),
        ("Ominous Seer",               2,  1, ["DEMON", "NAGA"],           "Battlecry: The next Tavern spell you buy costs (1) less."),
        ("Passenger",                  2,  2, [],                          "The first time your team Passes each turn, gain +1/+2."),
        ("Razorfen Geomancer",         2,  1, ["QUILBOAR"],                "Battlecry: Get 2 Blood Gems."),
        ("Risen Rider",                2,  1, ["UNDEAD"],                  "Taunt. Reborn."),
        ("River Skipper",              1,  1, ["MURLOC"],                  "When you sell this, get a random Tier 1 minion."),
        ("Rot Hide Gnoll",             1,  4, ["UNDEAD"],                  "Has +1 Attack for each friendly minion that died this combat."),
        ("Scarlet Survivor",           3,  3, ["DRAGON"],                  "Once this reaches 6 Attack, gain Divine Shield."),
        ("Southsea Busker",            3,  1, ["PIRATE"],                  "Battlecry: Gain 1 Gold next turn."),
        ("Suspicious Prisonguard",     3,  3, [],                          "Activate (1): Give another minion +3/+3."),
        ("Tusked Camper",              2,  3, ["QUILBOAR"],                "Rally: This plays a Blood Gem on itself."),
        ("Wrath Weaver",               1,  3, ["DEMON"],                   "After you play a Demon, deal 1 damage to your hero and gain +2/+2."),
    ],
    2: [
        ("Ancestral Automaton",        3,  4, ["MECH"],                    "Has +3/+2 for each other Ancestral Automaton you've summoned this game (wherever this is)."),
        ("Bilgewater Breakout",        3,  2, ["PIRATE"],                  "Battlecry: Get a Lockbox. If you already have one, it opens 1 turn sooner instead."),
        ("Clever Castaway",            2,  3, ["PIRATE"],                  "Activate (2): Discover a Tavern spell."),
        ("Crater Miner",               2,  2, ["QUILBOAR"],                "Choose One - Get 2 Blood Gems; or Get a Gem Day."),
        ("Decoy Conjurer",             3,  4, [],                          "Activate (2): Steal the highest-Attack minion in the Tavern."),
        ("Electric Synthesizer",       3,  4, ["DRAGON"],                  "Battlecry and Start of Combat: Give your other Dragons +1/+1."),
        ("Eternal Knight",             4,  2, ["UNDEAD"],                  "Has +4/+2 for each friendly Eternal Knight that died this game (wherever this is)."),
        ("Expert Aviator",             3,  4, ["MURLOC"],                  "Rally: Summon the highest-Attack minion from your hand for this combat only."),
        ("Fire Baller",                4,  3, ["ELEMENTAL"],               "When you sell this, give your minions +1 Attack. Improve your future Ballers."),
        ("Forest Rover",               1,  1, ["BEAST"],                   "Battlecry: Your Beetles have +2/+1 this game. Deathrattle: Summon a 2/2 Beetle."),
        ("Friendly Saloonkeeper",      3,  4, [],                          "Battlecry: Your teammate gets a Tavern Coin."),
        ("Gathering Stormer",          5,  1, ["ELEMENTAL"],               "When you sell this, your teammate gains 1 Gold. (Improves each turn!)"),
        ("Generous Geomancer",         1,  1, ["QUILBOAR"],                "Deathrattle: You and your teammate each get a Blood Gem."),
        ("Humming Bird",               1,  4, ["BEAST"],                   "Start of Combat: For the rest of this combat, your Beasts have +1 Attack."),
        ("Intrepid Botanist",          3,  4, [],                          "Choose One - Your Tavern spells give an extra +1 Attack this game; or +1 Health."),
        ("Laboratory Assistant",       3,  4, ["DEMON"],                   "Battlecry: Add a Fodder to your next 3 Refreshes."),
        ("Lava Lurker",                2,  5, ["NAGA"],                    "The first Spellcraft spell played from hand on this each turn is permanent. (1 left!)"),
        ("Lurking Lionfish",           3,  4, ["BEAST"],                   "Activate (2): Choose a card in the Tavern. Replace it with a Fishbait for your left-most Beast to attack."),
        ("Mechagnome Interpreter",     3,  1, ["MECH"],                    "Whenever you play or Magnetize a Mech, give it +3/+1."),
        ("Metallic Hunter",            4,  2, ["MECH"],                    "Deathrattle: Get a Pointy Arrow."),
        ("Mind Muck",                  3,  2, ["DEMON"],                   "Battlecry: Choose a friendly Demon. It consumes a minion in the Tavern to gain its stats."),
        ("Nerubian Deathswarmer",      1,  4, ["UNDEAD"],                  "Battlecry: Your Undead have +1 Attack this game (wherever they are)."),
        ("Oozeling Gladiator",         2,  2, [],                          "Battlecry: Get two Slimy Shields that give +1/+1 and Taunt."),
        ("Patient Scout",              1,  1, [],                          "When you sell this, Discover a Tier 1 minion. (Improves each turn!)"),
        ("Prodigious Tusker",          1,  3, ["QUILBOAR"],                "Whenever another friendly minion attacks, this plays a Blood Gem on it."),
        ("Roadboar",                   2,  4, ["QUILBOAR"],                "Rally: Get a Blood Gem."),
        ("Scarlet Skull",              2,  1, ["UNDEAD"],                  "Reborn. Deathrattle: Give a friendly Undead +1/+2."),
        ("Sellemental",                3,  3, ["ELEMENTAL"],               "When you sell this, get a 3/3 Elemental."),
        ("Shell Collector",            4,  3, ["NAGA"],                    "Battlecry: Get a Tavern Coin."),
        ("Snow Baller",                3,  4, ["ELEMENTAL"],               "When you sell this, give your minions +1 Health. Improve your future Ballers."),
        ("Soul Rewinder",              4,  1, ["DEMON"],                   "After your hero takes damage, rewind it and give this +1 Health."),
        ("Surfing Sylvar",             1,  2, ["PIRATE"],                  "At the end of your turn, give adjacent minions +1 Attack. Repeat for each friendly Golden minion."),
        ("Tad",                        2,  2, ["MURLOC"],                  "When you sell this, get a random Murloc."),
        ("Tarecgosa",                  4,  4, ["DRAGON"],                  "This permanently keeps Bonus Keywords and stats gained in combat."),
        ("Thaumaturgist",              1,  2, ["NAGA"],                    "Spellcraft: Give a minion +1/+1 until next turn. (Improved by every 3 spells you've cast this game!)"),
        ("Thousandth Paper Drake",     2,  3, ["DRAGON"],                  "Start of Combat: Give your left-most Dragon +1/+2 and Windfury."),
        ("Very Hungry Winterfinner",   2,  6, ["MURLOC"],                  "Taunt. Whenever this takes damage, give a minion in your hand +2/+1."),
        ("Wanderer Cho",               4,  3, [],                          "One Pass each turn is free. (1 left!)"),
    ],
    3: [
        ("Accord-o-Tron",              3,  3, ["MECH"],                    "Magnetic. At the start of your turn, gain 1 Gold."),
        ("Amber Guardian",             3,  2, ["DRAGON"],                  "Taunt. Start of Combat: Give another friendly Dragon +2/+2 and Divine Shield."),
        ("Annoy-o-Module",             2,  4, ["MECH"],                    "Magnetic. Divine Shield. Taunt."),
        ("Azsharan Cutlassier",        6,  4, ["PIRATE"],                  "Battlecry: Your Tavern spells give an extra +1 Attack this game."),
        ("Blue Whelp",                 1,  5, ["DRAGON"],                  "Rally: Your Tavern spells give an extra +1 Health this game."),
        ("Bottom Feeder",              3,  4, ["MURLOC"],                  "At the end of your turn, you and your teammate each get a random Tier 1 card."),
        ("Breakout Mastermind",        5,  5, ["MURLOC"],                  "Activate (2): Get a random Murloc."),
        ("Briarback Drummer",          5,  3, ["QUILBOAR"],                "Battlecry: Get a Blood Gem Barrage."),
        ("Cadaver Caretaker",          3,  3, ["UNDEAD"],                  "Deathrattle: Summon three 1/1 Skeletons."),
        ("Cagey Conjurer",             5,  3, ["NAGA"],                    "Activate (1): Cast 2 random Tavern spells (targets this if possible)."),
        ("Deadly Spore",               1,  1, [],                          "Venomous."),
        ("Deep-Sea Angler",            2,  3, ["NAGA"],                    "Spellcraft: Give a minion +2/+6 and Taunt. until next turn."),
        ("Deflect-o-Bot",              3,  2, ["MECH"],                    "Divine Shield. Whenever you summon a Mech during combat, gain +2 Attack and Divine Shield."),
        ("Devout Hellcaller",          2,  2, ["DEMON"],                   "After another friendly Demon deals damage, gain +1/+2 permanently."),
        ("Diremuck Forager",           4,  5, ["MURLOC"],                  "Start of Combat: When you have space, summon the highest-Attack Murloc from your hand for this combat only."),
        ("Disguised Graverobber",      4,  4, [],                          "Battlecry: Destroy a friendly Undead to get a plain copy of it."),
        ("Doting Dracthyr",            4,  3, ["DRAGON"],                  "At the end of your turn, give your teammate's minions +1 Attack."),
        ("Dustbone Devastator",        2,  6, ["UNDEAD"],                  "Rally: Your Undead have +1 Attack this game (wherever they are)."),
        ("Fruit Vendor",               3,  6, [],                          "Activate (1): Get 2 Tavern Dish Bananas."),
        ("Gem Rat",                    4,  4, ["QUILBOAR"],                "At the end of your turn, get a Gem Day."),
        ("Handless Forsaken",          2,  1, ["UNDEAD"],                  "Deathrattle: Summon a 2/1 Hand with Reborn."),
        ("Hired Mount",                3,  5, ["DRAGON"],                  "Activate (2): Get a random Chromadrake."),
        ("Jumping Jack",               3,  4, ["ALL"],                     "After the first time this is sold, Pass it."),
        ("Locked-up Mutineer",         6,  3, ["PIRATE"],                  "Deathrattle: Get a Lockbox. If you already have one, it opens 1 turn sooner instead."),
        ("Malchezaar, Prince of Dance",  2,  1, ["DEMON"],                   "Two Refreshes each turn cost Health instead of Gold. (2 left!)"),
        ("Mama Mrrglton",              4,  2, ["MURLOC"],                  "Battlecry: Give your other Murlocs +3 Attack. (Improved by each Mrrglton you played this game!)"),
        ("Meteorite Crasher",          4,  4, ["ELEMENTAL"],               "After you sell an Elemental, gain +4/+4."),
        ("Mummifier",                  5,  2, ["UNDEAD"],                  "Deathrattle: Give a different friendly Undead Reborn."),
        ("Orc-estra Conductor",        4,  4, [],                          "Battlecry: Give a minion +2/+2 (Improved by each Orc-estra your team has played this game)."),
        ("Papa Mrrglton",              2,  4, ["MURLOC"],                  "Battlecry: Give your other Murlocs +3 Health. (Improved by each Mrrglton you played this game!)"),
        ("Plunder Pal",                2,  2, ["PIRATE"],                  "At the start of your turn, you and your teammate each gain 1 Gold."),
        ("Private Investigator",       2,  4, ["PIRATE"],                  "Activate (1): Gain 3 Gold next turn."),
        ("Prosthetic Hand",            3,  1, ["UNDEAD", "MECH"],          "Magnetic, Reborn. Can Magnetize to Mechs or Undead."),
        ("Puddle Prancer",             4,  4, ["MURLOC"],                  "After this is Passed, gain +4/+4."),
        ("Rescue Bot",                 2,  1, ["MECH"],                    "Taunt. Deathrattle: Get a Repair Job."),
        ("Roaring Recruiter",          2,  8, ["DRAGON"],                  "Whenever another friendly Dragon attacks, give it +3/+1."),
        ("Sand Swirler",               3,  2, ["ELEMENTAL"],               "Battlecry: Your Elementals give an extra +2 Attack this game."),
        ("Sly Infiltrator",            4,  5, ["QUILBOAR"],                "Choose One - Gain 2 free Refreshes; or Get 3 Blood Gems."),
        ("Sly Raptor",                 1,  3, ["BEAST"],                   "Deathrattle: Summon a random Beast. Set its stats to 6/6."),
        ("Sprightly Scarab",           3,  1, ["BEAST"],                   "Choose One - Give a Beast +1/+1 and Reborn; or +4 Attack and Windfury."),
        ("Tasty Lobster",              1,  1, ["BEAST"],                   "Deathrattle: Give a random friendly Beast +1/+1. Improve your future Tasty Lobsters."),
        ("Timecap'n Hooktail",         1,  4, ["DRAGON", "PIRATE"],        "Whenever you cast a Tavern spell, give your minions +1 Attack."),
        ("Trapped Clapper",            2,  2, ["DEMON"],                   "Deathrattle: Add a Fodder to your next 3 Refreshes."),
        ("Treasure Parrot",            5,  5, ["BEAST", "PIRATE"],         "Once this deals 35 damage, get a Golden Touch. (35 left!)"),
        ("Trench Fighter",             3,  3, ["QUILBOAR"],                "At the end of your turn, get a Gem Confiscation."),
        ("Waveling",                   5,  1, ["ELEMENTAL"],               "Deathrattle: After the Tavern is Refreshed this game, give a random minion in it +4/+4."),
        ("Waverider",                  2,  6, ["NAGA"],                    "Spellcraft: Give a minion +2/+2. If it's a Naga, also give it Windfury. until next turn."),
        ("Wheeled Crewmate",           6,  3, ["PIRATE"],                  "Deathrattle: Reduce the Cost of upgrading your team's Taverns by (1)."),
        ("Wildfire Elemental",         6,  3, ["ELEMENTAL"],               "After this attacks and kills a minion, deal excess damage to an adjacent enemy."),
        ("Wolf Pup",                   3,  6, ["BEAST"],                   "Rally: Give your other minions +4/+1."),
    ],
    4: [
        ("Abyssal Bruiser",            2,  1, ["NAGA"],                    "Divine Shield. Has +2/+1 for each Tavern spell you've cast this game."),
        ("Air Baller",                 6,  6, ["ELEMENTAL"],               "When you sell this, give your minions +2/+2. Improve your future Ballers."),
        ("Ashen Corruptor",            5,  6, ["DEMON"],                   "After your hero takes damage, rewind it and give minions in the Tavern +1/+1 this turn."),
        ("Auto Assembler",             2,  2, ["MECH"],                    "Magnetic. Deathrattle: Summon an Ancestral Automaton."),
        ("Banana Slamma",              3,  6, ["BEAST"],                   "After you summon a Beast in combat, double its Attack."),
        ("Bigwig Bandit",              4,  6, ["PIRATE"],                  "Rally: Get a random Bounty."),
        ("Blade Collector",            3,  2, ["PIRATE"],                  "Also damages the enemies next to whomever this attacks."),
        ("Bonker",                     2,  7, ["QUILBOAR"],                "Windfury. Rally: This plays a Blood Gem on all your other minions."),
        ("Boom-in-a-Box",              5, 10, [],                          "Taunt. Start of Combat: Deal 3 damage to all other minions."),
        ("Bramble Tunneler",           3,  6, ["QUILBOAR"],                "Rally: Get a random Choose One card."),
        ("Bream Counter",              6,  6, ["MURLOC"],                  "While this is in your hand, after you play a Murloc, gain +6/+6."),
        ("Bronze Timewalker",          2,  9, ["DRAGON"],                  "Rally: Get a random Chromadrake."),
        ("Cage Gnawer",                2,  7, ["BEAST"],                   "Whenever a friendly Beast attacks, give your Beasts +2/+1."),
        ("Captain Cookie",             5,  3, ["MURLOC", "PIRATE"],        "Deathrattle: Get a Chef's Choice."),
        ("Clunker Junker",             3,  4, ["MECH"],                    "Battlecry: Choose a friendly Mech. Discover a Mech to Magnetize to it."),
        ("Dead Bellringer",            3,  6, ["UNDEAD"],                  "Activate (1): Give a different friendly Undead Reborn. Then destroy it to gain +4/+4."),
        ("Deepwater Chieftain",        3,  2, ["MURLOC"],                  "Battlecry and Deathrattle: Get a Deepwater Clan."),
        ("Drone Duplicator",           5,  2, ["MECH"],                    "Divine Shield. Activate (1): The next Magnetization to this minion this turn is doubled."),
        ("En-Djinn Blazer",            5,  5, ["ELEMENTAL"],               "Battlecry: After the Tavern is Refreshed this game, give a random minion in it +10/+10."),
        ("Enchanted Sentinel",         3,  5, ["MECH"],                    "Magnetic. Your Tavern spells give an extra +1/+1."),
        ("Fearless Foodie",            2,  4, ["QUILBOAR"],                "Choose One - Your Blood Gems give an extra +1/+1 this game; or Get 4 Blood Gems."),
        ("Feisty Freshwater",          6,  4, ["ELEMENTAL"],               "Deathrattle: You and your teammate each gain two free Refreshes."),
        ("Flaming Enforcer",           4,  5, ["ELEMENTAL", "DEMON"],      "At the end of your turn, consume the highest-Health minion in the Tavern to gain its stats."),
        ("Friendly Geist",             6,  3, ["UNDEAD"],                  "Deathrattle: Your Tavern spells give an extra +1 Attack this game."),
        ("Gearfin",                    6,  5, ["MECH", "MURLOC"],          "At the end of your turn, get two 1-Cost Tavern spells."),
        ("Glambot",                    4,  4, ["MECH"],                    "Whenever you cast a spell on a Mech, Magnetize a 4/4 Satellite to it."),
        ("Glowing Cinder",             4,  1, ["ELEMENTAL"],               "Deathrattle: Your Elementals give an extra +2 Health this game."),
        ("Grave Narrator",             2,  7, ["UNDEAD"],                  "Avenge (3): Your teammate gets a random minion of their most common type."),
        ("Gunpowder Courier",          2,  6, ["PIRATE"],                  "Whenever you spend 5 Gold, give your Pirates +2 Attack. (5 Gold left!)"),
        ("Headhunter Gryphon",         3,  5, ["BEAST"],                   "Rally: Get a random Beast."),
        ("Heroic Underdog",            1, 10, [],                          "Stealth. Rally: Gain the target's Attack."),
        ("Humon'gozz",                 5,  5, [],                          "Divine Shield. Your Tavern spells give an extra +1/+2."),
        ("Imp-lusionist",              4,  2, ["DEMON"],                   "Deathrattle: Get a Methodical Madness."),
        ("Imposing Percussionist",     4,  4, ["DEMON"],                   "Battlecry: Discover a Demon. Deal damage to your hero equal to its Tier."),
        ("Kelp Keeper",                5,  5, ["MURLOC"],                  "Activate (1): Trigger a friendly minion's Battlecry."),
        ("Living Prison",              4,  5, ["ELEMENTAL"],               "Activate (1): Gain the stats of the next minion you buy this turn."),
        ("Lovesick Balladist",         3,  2, ["PIRATE"],                  "Battlecry: Give a Pirate +1 Health. (Improved by each Gold you spent this turn!)"),
        ("Mantid King",                3,  3, [],                          "After your team Passes, randomly gain Venomous, Taunt, or Divine Shield. until next turn."),
        ("Maritime Extortionist",      7,  7, ["PIRATE"],                  "Has +7/+7 for each Golden minion you've played this game (wherever this is)."),
        ("Maw Caster",                 4,  5, ["UNDEAD"],                  "Battlecry: Destroy a friendly Undead to Discover an Undead."),
        ("Mirror Monster",             4,  4, ["ALL"],                     "When you buy or Discover this, get an extra copy and Pass it."),
        ("Motley Phalanx",             3,  3, ["ALL"],                     "Taunt. Deathrattle: Give a friendly minion of each type +3/+3 permanently."),
        ("Persistent Poet",            2,  3, ["DRAGON"],                  "Divine Shield. Adjacent Dragons permanently keep Bonus Keywords and stats gained in combat."),
        ("Plaguerunner",               4,  2, ["UNDEAD"],                  "Deathrattle: Your Undead have +2 Attack this game, wherever they are. (+4 if triggered outside combat!)"),
        ("Private Chef",               5,  4, ["NAGA"],                    "Spellcraft: Choose a minion. Get a different random minion of its type, then Pass it."),
        ("Razorfen Flapper",           6,  2, ["QUILBOAR"],                "Deathrattle: Get a Blood Gem Barrage."),
        ("Refreshing Anomaly",         4,  5, ["ELEMENTAL"],               "Battlecry: Gain 2 free Refreshes."),
        ("Rimescale Priestess",        3,  3, ["NAGA"],                    "Spellcraft: Get a random Tavern spell that gives stats."),
        ("Runic Arcanist",             2,  4, ["DRAGON"],                  "Start of Combat: Cast Shiny Ring twice."),
        ("San'layn Scribe",            4,  4, ["UNDEAD"],                  "Has +4/+4 for each of your team's San'layn Scribes that died this game (wherever this is)."),
        ("Seafloor Recruiter",         3,  5, ["NAGA"],                    "Rally: Cast Chef's Choice on the minion to the right."),
        ("Shifty Snake",               6,  1, ["BEAST"],                   "Deathrattle: Your teammate gets a random Deathrattle minion."),
        ("Sin'dorei Straight Shot",    3,  4, [],                          "Divine Shield, Windfury. Rally: Remove Reborn and Taunt from the target."),
        ("Sky-hatch Runaway",          4,  7, ["DRAGON"],                  "Activate (1): Trigger a friendly minion's Rally."),
        ("Snare Trapper",              4,  4, ["QUILBOAR"],                "Choose One - Get a random Quilboar; or Increase your maximum Gold by 1."),
        ("Snarky Shark",               4,  5, ["BEAST"],                   "When you sell this, Refresh the Tavern with a Fishbait. Your left-most Beast attacks it."),
        ("Soulkeeping Jailer",         3,  5, ["DEMON"],                   "Activate (2): Your Demons each consume a random minion in the Tavern to gain its stats."),
        ("Tavern Tempest",             2,  2, ["ELEMENTAL"],               "Battlecry: Get a random Elemental."),
        ("Thorned Trailblazer",        4,  5, ["QUILBOAR"],                "One Choose One card each turn has both effects combined. (1 left!)"),
        ("Tortollan Blue Shell",       3,  6, [],                          "If you lost your last combat, this minion sells for 5 Gold."),
        ("Twilight Tidehunter",        4,  6, ["MURLOC"],                  "Whenever you cast a spell on this, give the left-most minion in your hand +8/+8."),
        ("Zesty Shaker",               6,  7, ["NAGA"],                    "Once per turn, when a Spellcraft spell is played on this, get a new copy of it."),
    ],
    5: [
        ("Air Revenant",               3,  6, ["ELEMENTAL"],               "After you spend 7 Gold, cast Easterly Winds. (7 left!)"),
        ("Barrier Banshee",            7,  7, ["UNDEAD"],                  "After a friendly minion is Reborn, gain Divine Shield and +7/+7."),
        ("Bile Spitter",               1, 10, ["MURLOC"],                  "Venomous. Rally: Give another friendly Murloc Venomous."),
        ("Brann Bronzebeard",          2,  4, [],                          "Your Battlecries trigger twice."),
        ("Cataclysmic Harbinger",      6, 10, [],                          "At the end of your turn, get a copy of the last Tavern spell you cast."),
        ("Charging Czarina",           4,  1, ["MECH"],                    "Divine Shield. Whenever you cast a Tavern spell, give your minions with Divine Shield +4 Attack."),
        ("Costume Enthusiast",         4,  5, ["MURLOC"],                  "Divine Shield. Start of Combat: Gain the Attack of the highest-Attack minion in your hand."),
        ("Cousin Errgl",               5,  5, ["MURLOC"],                  "At the end of your turn, get a Mama Mrrglton or a Papa Mrrglton."),
        ("Dancing Barnstormer",        4,  4, ["ELEMENTAL"],               "Battlecry and Deathrattle: Give Elementals in the Tavern +8/+8 this game."),
        ("Darkcrest Strategist",       4,  5, ["NAGA"],                    "Spellcraft: Get a random Tier 1 Naga. (Improves each turn!)"),
        ("Deft Deserter",              8,  8, ["DEMON"],                   "Activate (1): Give all minions in the Tavern +8/+8 and Taunt, Divine Shield, or Windfury."),
        ("Devilish Distractor",        4,  7, ["DEMON"],                   "Whenever you cast a spell on this, give minions in the Tavern +2/+2 this game."),
        ("Draconic Warden",            7,  4, ["DRAGON"],                  "Battlecry and Deathrattle: Get a random Chromadrake."),
        ("Drakkari Enchanter",         1,  5, [],                          "Your end of turn effects trigger twice."),
        ("Drustfallen Butcher",        2,  9, ["UNDEAD"],                  "Avenge (4): Get a Butchering."),
        ("Dual-Wield Corsair",         4,  5, ["PIRATE"],                  "Whenever you spend 5 Gold, give two friendly Pirates +4/+5. (5 Gold left!)"),
        ("Enterprising Escapee",       6,  6, ["PIRATE"],                  "After you spend 5 Gold, get a Lockbox. If you already have one, it opens 1 turns sooner instead. (5 Gold left!)"),
        ("Felboar",                    2,  6, ["DEMON", "QUILBOAR"],       "After you cast 3 spells, consume a minion in the Tavern to gain its stats. (3 left!)"),
        ("Felfire Conjurer",           6,  5, ["DEMON", "DRAGON"],         "At the end of your turn, your Tavern spells give an extra +1/+1 this game."),
        ("Firescale Hoarder",          5,  5, ["NAGA", "DRAGON"],          "Battlecry and Deathrattle: Get a Shiny Ring."),
        ("Glowscale",                  4,  6, ["NAGA"],                    "Taunt. Spellcraft: Give a minion Divine Shield until next turn."),
        ("Hoarding Hyena",             5,  6, ["BEAST"],                   "Rally: Summon a Tasty Lobster."),
        ("Insatiable Ur'zul",          4,  6, ["DEMON"],                   "Taunt. After you play a Demon, consume a random minion in the Tavern to gain its stats."),
        ("Kalecgos, Arcane Aspect",    4, 12, ["DRAGON"],                  "After you trigger a Battlecry, give your Dragons +2/+2."),
        ("Kangor's Apprentice",        3,  6, [],                          "Deathrattle: Summon plain copies of your first 2 Mechs that died this combat."),
        ("Leeroy the Reckless",        6,  2, [],                          "Deathrattle: Destroy the minion that killed this."),
        ("Lurking Leviathan",          3,  8, ["BEAST"],                   "Whenever you summon a Beast, give it +2 Attack and improve this permanently."),
        ("Man'ari Messenger",          9,  6, ["DEMON"],                   "Battlecry: Minions in your team's Taverns have +1/+1 this game."),
        ("Nightmare Par-tea Guest",    3,  3, ["ALL"],                     "Battlecry and Deathrattle: Get a Misplaced Tea Set."),
        ("Nomi, Kitchen Nightmare",    6,  6, [],                          "After you play an Elemental, give Elementals in the Tavern +4/+4 this game."),
        ("Primalfin Lookout",          3,  2, ["MURLOC"],                  "Battlecry: If you control another Murloc, Discover a Murloc."),
        ("Proud Privateer",            8,  8, ["PIRATE"],                  "Your Bounties cast twice."),
        ("Razorfen Vineweaver",        5,  5, ["QUILBOAR"],                "Rally: This plays 3 permanent Blood Gems on itself."),
        ("Rodeo Performer",            3,  4, [],                          "Battlecry: Discover a Tavern spell."),
        ("Sanguine Refiner",           2,  8, ["QUILBOAR"],                "Rally: Your Blood Gems. give an extra +1/+1 this game."),
        ("Scrap Scraper",              6,  5, ["MECH"],                    "Deathrattle: Get a random Magnetic Mech."),
        ("Selfless Sightseer",         6,  2, ["DRAGON"],                  "Battlecry: Increase your team's maximum Gold by (1)."),
        ("Sewer Lord",                 4,  6, ["BEAST"],                   "Deathrattle: Summon two Sewer Rats that summon 2/3 Turtles with Taunt."),
        ("Shamanic Tidecaller",        5,  7, ["MURLOC"],                  "Whenever you cast a spell on a Murloc, give Murlocs in your hand and board +3/+3."),
        ("Shipwrecked Rascal",         5,  4, ["PIRATE"],                  "Battlecry and Deathrattle: Get a random Bounty."),
        ("Showy Cyclist",              4,  2, ["NAGA"],                    "Deathrattle: Give all your Naga +2/+1. (Improved by every 3 spells you've cast this game!)"),
        ("Spark Snapper",              5,  5, ["MECH"],                    "Whenever you play a Mech, Magnetize a 2/2 Satellite to it and improve this."),
        ("Storm Splitter",             5,  5, ["NAGA"],                    "Once per turn, after you Pass a Tavern spell, get a new copy of it."),
        ("Support System",             4,  5, ["MECH"],                    "At the end of your turn, give a minion in your teammate's warband Divine Shield."),
        ("Tichondrius",                3,  3, ["DEMON"],                   "After your hero takes damage, give your Demons +3/+3."),
        ("Titus Rivendare",            1,  7, [],                          "Your Deathrattles trigger an extra time."),
        ("Tranquil Meditative",        3,  8, ["NAGA"],                    "Spellcraft: Your Tavern spells give an extra +1/+1 this game."),
        ("Turquoise Skitterer",        5,  5, ["BEAST"],                   "Deathrattle: Your Beetles have +5/+5 this game. Summon a 2/2 Beetle."),
        ("Vigilant Bristlemane",       3,  5, ["QUILBOAR"],                "Whenever you cast a spell on this, it plays a Blood Gem. on adjacent minions."),
        ("Void Pup Trainer",           7,  7, ["DEMON"],                   "Battlecry: Give minions in the Tavern from Tier 3 and below +3/+3 this game."),
        ("Well Wisher",                6,  6, [],                          "Spellcraft: Pass a different non-Golden minion."),
    ],
    6: [
        ("Balinda Stonehearth",        6,  6, [],                          "Your spells that target friendly minions cast twice."),
        ("Choral Mrrrglr",             6,  6, ["MURLOC"],                  "Start of Combat: Gain the stats of all the minions in your hand."),
        ("Crimson Vindicator",         8,  9, ["DRAGON"],                  "Divine Shield. Rally: Cast Mighty Dragonbreath."),
        ("Dark Dazzler",               4,  7, ["DEMON"],                   "After your teammate sells a minion, gain its stats. (Once per turn.)"),
        ("Deathly Striker",            8,  8, ["UNDEAD"],                  "Avenge (4): Get a random Undead. Deathrattle: Summon it from your hand for this combat only."),
        ("Deathstrider",              10, 11, ["BEAST"],                   "After a friendly Rally. minion attacks, trigger your left-most Deathrattle."),
        ("Elemental of Surprise",      8,  8, ["ELEMENTAL"],               "Divine Shield. This minion can triple with any Elemental."),
        ("Eredar Escapist",            6,  6, ["DEMON"],                   "After your hero takes 3 damage, cast Shiny Ring. (3 left!)"),
        ("Eternal Summoner",           8,  1, ["UNDEAD"],                  "Reborn. Deathrattle: Summon 1 Eternal Knight."),
        ("Falling Sky Golem",          4,  2, ["MECH"],                    "Divine Shield. Has +4/+2 for each Deathrattle you've triggered this game (wherever this is)."),
        ("Fauna Whisperer",            4,  9, ["NAGA"],                    "At the end of your turn, cast Natural Blessing on adjacent minions."),
        ("Fire-forged Evoker",         8,  5, ["DRAGON"],                  "Start of Combat: Give your Dragons +2/+1. Improves permanently after you cast a Tavern spell."),
        ("Forsaken Weaver",            3, 10, ["UNDEAD"],                  "After you cast a Tavern spell, your Undead have +2 Attack this game (wherever they are)."),
        ("Gatekeeper Amalgam",         6,  6, ["ALL"],                     "Whenever you cast a spell on this, it casts Misplaced Tea Set."),
        ("Gentle Djinni",              5,  6, ["ELEMENTAL"],               "Taunt. Battlecry and Deathrattle: Get a random Elemental."),
        ("Goldrinn, the Great Wolf",   8,  8, ["BEAST"],                   "Deathrattle: Your Beasts have +8/+8 until next turn."),
        ("Groundbreaker",              6,  4, ["NAGA"],                    "After you play a Naga, gain +1/+1. (Improved by every 3 spells you've cast this game!)"),
        ("Hooktusk, Master Marauder",  4,  4, ["PIRATE"],                  "After you Discover a card, give your other Pirates +1/+1. (Improved by Golden minions you played this game!)"),
        ("Ignition Specialist",        8,  8, ["DRAGON"],                  "At the end of your turn, get 2 random Tavern spells."),
        ("Loyal Mobster",              6,  5, ["QUILBOAR"],                "At the end of your turn, this plays a Blood Gem. on all your teammate's minions."),
        ("Magicfin Mycologist",        4,  8, ["MURLOC"],                  "Once per turn, after you buy a Tavern spell, get a 1/1 Murloc and teach it that spell. (1 left!)"),
        ("Moat Custodian",             5, 10, ["ELEMENTAL"],               "Rally: Your Elementals give an extra +2/+2 this game."),
        ("Primitive Painter",          3,  8, ["MURLOC"],                  "After you play a card from Tier 3 or below, give your Murlocs +3/+3."),
        ("Ravaging Scorpid",           6,  7, ["BEAST"],                   "After a friendly minion attacks, your Beetles have +5/+5 this game. Deathrattle: Summon a 2/2 Beetle."),
        ("Sanguine Champion",          9,  3, ["QUILBOAR"],                "Battlecry and Deathrattle: Your Blood Gems give an extra +1/+1 this game."),
        ("Silent Deliverer",           7,  7, ["PIRATE"],                  "Battlecry: Get a random Golden minion from Tier 4. It doesn’t give a Triple Reward."),
        ("Sky Admiral Rogers",         4,  5, ["PIRATE"],                  "After you spend 9 Gold, get a random Bounty. (9 Gold left!)"),
        ("Snazzy Phantom",             6,  8, ["UNDEAD"],                  "After a friendly minion is Reborn, give stats equal to its Attack to your right-most Undead."),
        ("Torrential Ruiner",          6,  3, ["NAGA"],                    "Whenever you cast a spell on a Naga, give your minions +2/+3."),
        ("Transport Reactor",          1,  1, ["MECH"],                    "Magnetic. Has +1/+1 for each time your team has Passed this game (wherever this is)."),
        ("Turbo Hogrider",             5,  7, ["QUILBOAR"],                "After you play a Choose One. card, this plays a Blood Gem on all your other Quilboar."),
        ("Twisted Wrathguard",         8,  8, ["DEMON"],                   "After you sell a minion, add a Fodder to your next Refresh."),
        ("Tyrael",                    10, 10, [],                          "Activate (2): Set another minion's stats to 50/50."),
        ("Unbound Tempest",            3, 12, ["ELEMENTAL"],               "After you play 3 Elementals, gain the stats of the highest-Health minion in the Tavern. (3 left!)"),
        ("Unleashed Mana Surge",       6,  5, ["ELEMENTAL"],               "After you play an Elemental, give your Elementals +4/+4."),
        ("Utility Drone",              4,  6, ["MECH"],                    "At the end of your turn, give your minions +4/+4 for each Magnetization. they have."),
        ("Veteran Brigand",            8,  8, ["QUILBOAR"],                "Choose One - This plays 3 Blood Gems on all your minions; or cast Blood Gem Barrage 3 times."),
        ("Warpwing",                  12,  4, ["DRAGON"],                  "Immune while attacking."),
    ],
    7: [
        ("Captain Sanders",            9,  9, ["PIRATE"],                  "Battlecry: Make a friendly minion from Tier 6 or below Golden."),
        ("Champion of Sargeras",       8,  8, ["DEMON"],                   "Battlecry and Deathrattle: Give minions in the Tavern +8/+8 this game."),
        ("Futurefin",                  7, 13, ["MURLOC"],                  "At the end of your turn, give this minion's stats to the left-most minion in your hand."),
        ("Highkeeper Ra",              6,  6, [],                          "Battlecry, Deathrattle, and Rally: Get a random Tier 6 minion."),
        ("Jailbird Juggernaut",        6, 15, ["QUILBOAR"],                "Rally: Summon a Golem with stats equal to this minion's Blood Gems to attack the target first. (0/0)"),
        ("Obsidian Ravager",           7,  7, ["DRAGON"],                  "Rally: Deal damage equal to this minion's Attack to the target and an adjacent minion."),
        ("Polarizing Beatboxer",       5, 10, ["MECH"],                    "Whenever you Magnetize to a different minion, it also Magnetizes to this."),
        ("Sandy",                      1,  1, [],                          "Start of Combat: Transform into a copy of your teammate's highest-Health minion."),
        ("Sea Witch Zar'jira",         4,  5, ["NAGA"],                    "Spellcraft: Choose a different minion in the Tavern to get a copy of."),
        ("Stalwart Kodo",             16, 32, ["BEAST"],                   "After you summon a minion in combat, give it this minion's maximum stats. (3 times per combat.)"),
        ("Stitched Salvager",         16,  4, ["UNDEAD"],                  "Start of Combat: Destroy the minion to the left. Deathrattle: Summon an exact copy of it. (Except Stitched Salvager.)"),
        ("Stone Age Slab",            10, 10, ["ELEMENTAL"],               "After you buy a minion, give it +20/+20 and double its stats. (Once per turn.)"),
        ("The Last One Standing",     15, 15, ["ALL"],                     "Rally: Give a friendly minion of each type +15/+15 permanently."),
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
}

# Cards with passive auras or reactive triggers boosting other friendlies
AURA_CARDS = {
    "roaring_recruiter",
    "timecapn_hooktail",
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
        "activate":     bool(re.search(r"\bactivate\s*\(\d+\)", t)),
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
    if re.search(r"\bactivate\s*\(\d+\)", t):
        return "activate"
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


def detect_activate_cost(text: str):
    """Activate (N): a repeatable, once-per-turn minion ability costing N Gold."""
    m = re.search(r"activate\s*\((\d+)\)", text.lower())
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
    activate_cost = detect_activate_cost(text)
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
        "activate_cost": activate_cost,
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
    text = text or ""
    # HearthstoneJSON breaks back-to-back bolded keywords onto separate lines
    # (e.g. "<b>Magnetic</b>\n<b>Divine Shield</b>") — join those with a period.
    # A bare '\n' elsewhere is just a display-width wrap inside running prose.
    text = re.sub(r"(?<!:)</b>\s*\n\s*", "</b>. ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[x\]", "", text)
    text = re.sub(r"\|4\s*\([^)]+\)", "", text)
    text = re.sub(r"-\s*\n\s*", "-", text)  # rejoin hyphenated words split by line-wrap
    text = text.replace("\n", " ")
    text = re.sub(r"[\x00-\x1f\xa0]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\.\s*\.", ".", text)
    if text and not text.endswith((".", "!", ")")):
        text += "."
    return text


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


def _trinket_tier(c: dict) -> str:
    """Return 'lesser' or 'greater'. spellSchool is authoritative; cost is a fallback
    for older API snapshots that don't set it (some Greater trinkets cost <= 3)."""
    school = (c.get("spellSchool") or "").upper()
    if school == "GREATER_TRINKET":
        return "greater"
    if school == "LESSER_TRINKET":
        return "lesser"
    return "lesser" if c.get("cost", 0) <= 3 else "greater"


# Effect-type detection rules for parse_trinket_effect(), checked in order.
# Each entry: (regex, builder(match, cost) -> effect dict). First match wins.
# Anything unmatched falls through to a labeled-but-inert "complex" bucket so
# TrinketHandler can log it instead of silently doing nothing.
_TRINKET_RULES = []


def _trinket_rule(pattern):
    regex = re.compile(pattern, re.IGNORECASE)
    def _decorator(fn):
        _TRINKET_RULES.append((regex, fn))
        return fn
    return _decorator


@_trinket_rule(r"at the (?:start|end) of (?:each|your|every) turn,?\s*(?:gain|increase your maximum gold by)")
def _rule_gold_per_round(m, text, cost):
    max_m = re.search(r"increase your maximum gold by (\d+)", text, re.I)
    if max_m:
        return {"type": "max_gold_per_round", "amount": int(max_m.group(1))}
    gain_m = re.search(r"gain (\d+) gold", text, re.I)
    if gain_m:
        return {"type": "gold_per_round", "amount": int(gain_m.group(1))}
    return None


@_trinket_rule(r"^gain (\d+) armor\b")
def _rule_armor(m, text, cost):
    return {"type": "armor", "amount": int(m.group(1))}


@_trinket_rule(r"^gain (\d+) gold\b")
def _rule_gold_gain(m, text, cost):
    effect = {"type": "gold_gain", "amount": int(m.group(1))}
    max_m = re.search(r"increase your maximum gold by (\d+)", text, re.I)
    if max_m:
        effect["max_gold_increase"] = int(max_m.group(1))
    return effect


@_trinket_rule(r"reduce the cost of upgrading the tavern by \((\d+)\)")
def _rule_level_cost(m, text, cost):
    amount = int(m.group(1))
    if re.search(r"at the (start|end) of (each|every) turn|repeat this", text, re.I):
        return {"type": "level_cost_reduction_per_round", "amount": amount}
    return {"type": "level_cost_reduction", "amount": amount}


@_trinket_rule(r"^your minions have \+(\d+)(?:/\+(\d+))?\s*(attack|health)?\.?$")
def _rule_stat_buff_all(m, text, cost):
    atk = int(m.group(1))
    hp = int(m.group(2)) if m.group(2) else 0
    if m.group(3) and m.group(3).lower() == "health" and m.group(2) is None:
        atk, hp = 0, atk
    return {"type": "stat_buff_all", "atk": atk, "hp": hp}


@_trinket_rule(r"^your ([a-z]+?)s? have \+(\d+)(?:/\+(\d+))?\s*(attack|health)?\.?$")
def _rule_stat_buff_tribe(m, text, cost):
    tribe = m.group(1).upper()
    if tribe not in {"BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECH", "MURLOC",
                      "NAGA", "PIRATE", "QUILBOAR", "UNDEAD"}:
        return None  # not a recognised tribe word (e.g. "Your Tavern spells...")
    atk = int(m.group(2))
    hp = int(m.group(3)) if m.group(3) else 0
    if m.group(4) and m.group(4).lower() == "health" and m.group(3) is None:
        atk, hp = 0, atk
    return {"type": "stat_buff_tribe", "tribe": tribe, "atk": atk, "hp": hp}


@_trinket_rule(r"your minions from tier (\d+) or (?:lower|below) have \+(\d+)(?:/\+(\d+))?")
def _rule_stat_buff_low_tier(m, text, cost):
    max_tier = int(m.group(1))
    atk = int(m.group(2))
    hp = int(m.group(3)) if m.group(3) else 0
    return {"type": "stat_buff_low_tier", "max_tier": max_tier, "atk": atk, "hp": hp}


@_trinket_rule(r"^at the end of (?:each|your) turn,? give your left-most minion \+(\d+)(?:/\+(\d+))?")
def _rule_eot_leftmost(m, text, cost):
    atk = int(m.group(1))
    hp = int(m.group(2)) if m.group(2) else 0
    return {"type": "end_of_turn_buff_leftmost", "atk": atk, "hp": hp}


@_trinket_rule(r"^at the end of (?:each|your) turn,? give your minions \+(\d+)(?:/\+(\d+))?\.?$")
def _rule_eot_all(m, text, cost):
    atk = int(m.group(1))
    hp = int(m.group(2)) if m.group(2) else 0
    return {"type": "end_of_turn_buff_all", "atk": atk, "hp": hp}


@_trinket_rule(r"^at the end of (?:each|your) turn,? give your ([a-z]+?)s? \+(\d+)(?:/\+(\d+))?")
def _rule_eot_tribe(m, text, cost):
    tribe = m.group(1).upper()
    if tribe not in {"BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECH", "MURLOC",
                      "NAGA", "PIRATE", "QUILBOAR", "UNDEAD"}:
        return None
    atk = int(m.group(2))
    hp = int(m.group(3)) if m.group(3) else 0
    return {"type": "end_of_turn_buff_tribe", "tribe": tribe, "atk": atk, "hp": hp}


@_trinket_rule(r"^start of combat:\s*give your minions \+(\d+)(?:/\+(\d+))?\.?$")
def _rule_soc_all(m, text, cost):
    atk = int(m.group(1))
    hp = int(m.group(2)) if m.group(2) else 0
    return {"type": "start_of_combat_buff_all", "atk": atk, "hp": hp}


@_trinket_rule(r"^start of combat:\s*give your ([a-z]+?)s? \+(\d+)(?:/\+(\d+))?")
def _rule_soc_tribe(m, text, cost):
    tribe = m.group(1).upper()
    if tribe not in {"BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECH", "MURLOC",
                      "NAGA", "PIRATE", "QUILBOAR", "UNDEAD"}:
        return None
    atk = int(m.group(2))
    hp = int(m.group(3)) if m.group(3) else 0
    return {"type": "start_of_combat_buff_tribe", "tribe": tribe, "atk": atk, "hp": hp}


@_trinket_rule(r"avenge\s*\((\d+)\)")
def _rule_avenge(m, text, cost):
    return {"type": "avenge", "count": int(m.group(1))}


@_trinket_rule(r"^spellcraft:")
def _rule_spellcraft(m, text, cost):
    return {"type": "spellcraft"}


@_trinket_rule(r"\bdiscover\b")
def _rule_discover(m, text, cost):
    return {"type": "discover"}


@_trinket_rule(
    r"\b(whenever|after)\b.*\b(attacks?|deals? damage|takes? damage|dies|is reborn|"
    r"loses divine shield|is magnetized|magnetize|summon a minion|consumes?)\b"
)
def _rule_combat_trigger(m, text, cost):
    return {"type": "combat_trigger"}


@_trinket_rule(r"at the (?:start|end) of (?:each|every) turn")
def _rule_round_start_effect(m, text, cost):
    return {"type": "round_start_effect"}


def parse_trinket_effect(text: str, cost: int) -> dict:
    """Classify a trinket's raw text into a structured effect dict.

    Tries each rule in _TRINKET_RULES in order and returns the first match.
    Falls back to {"type": "complex"} — a safe, labeled no-op — when nothing
    matches, so TrinketHandler can log unimplemented cards instead of
    silently doing nothing for an unrecognised type string.
    """
    for regex, builder in _TRINKET_RULES:
        m = regex.search(text)
        if not m:
            continue
        effect = builder(m, text, cost)
        if effect is not None:
            return effect
    return {"type": "complex"}


def build_trinket_list(api_trinkets: list) -> list:
    """Build a list of trinket dicts (with card_id + trinket_effect) from API data."""
    trinkets = []
    seen_ids: Dict[str, int] = {}
    for c in api_trinkets:
        cost = c.get("cost", 0)
        tier = _trinket_tier(c)
        text = _clean_api_text(c.get("text", ""))
        base_id = make_card_id(c.get("name", "")) + ("_g" if tier == "greater" else "_l")
        n = seen_ids.get(base_id, 0)
        seen_ids[base_id] = n + 1
        card_id = base_id if n == 0 else f"{base_id}_{n + 1}"
        trinkets.append({
            "card_id": card_id,
            "name": c.get("name", ""),
            "cost": cost,
            "tier": tier,
            "text": text,
            "trinket_effect": parse_trinket_effect(text, cost),
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
