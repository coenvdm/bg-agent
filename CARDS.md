# BG Card Pool — Current Active Card Definitions

Source: HearthstoneJSON API, filtered to `isBattlegroundsPoolMinion: true` (minions) and `type: BATTLEGROUND_TRINKET` (trinkets).
Scraped: 2026-04-17. **270 minions** across 7 tiers + **177 trinkets** (116 Lesser, 61 Greater).

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

- Annoy-o-Tron (1/2) [MECH]: Taunt. Divine Shield.
- Aureate Laureate (1/1) [PIRATE]: Divine Shield. Battlecry: Make this minion Golden.
- Cord Puller (1/1) [MECH]: Divine Shield. Deathrattle: Summon a 1/1 Microbot.
- Crackling Cyclone (2/1) [ELEMENTAL]: Divine Shield. Windfury.
- Dune Dweller (3/2) [ELEMENTAL]: Battlecry: Give Elementals in the Tavern +1/+1 this game.
- Flighty Scout (3/3) [MURLOC]: Start of Combat: If this minion is in your hand, summon a copy of it.
- Gluttonous Trogg (2/3): Once you buy 4 cards, gain +4/+4.
- Harmless Bonehead (1/1) [UNDEAD]: Deathrattle: Summon two 1/1 Skeletons.
- Manasaber (4/1) [BEAST]: Deathrattle: Summon two 0/1 Cublings with Taunt.
- Ominous Seer (2/1) [DEMON/NAGA]: Battlecry: The next Tavern spell you buy costs (1) less.
- Passenger (2/2): The first time your team Passes each turn, gain +1/+2.
- Picky Eater (1/1) [DEMON]: Battlecry: Consume a random minion in the Tavern to gain its stats.
- Razorfen Geomancer (2/1) [QUILBOAR]: Battlecry: Get 2 Blood Gems.
- Risen Rider (2/1) [UNDEAD]: Taunt. Reborn.
- River Skipper (1/1) [MURLOC]: When you sell this, get a random Tier 1 minion.
- Rot Hide Gnoll (1/4) [UNDEAD]: Has +1 Attack for each friendly minion that died this combat.
- Scarlet Survivor (3/3) [DRAGON]: Once this reaches 6 Attack, gain Divine Shield.
- Southsea Busker (3/1) [PIRATE]: Battlecry: Gain 1 Gold next turn.
- Sun-Bacon Relaxer (2/3) [QUILBOAR]: When you sell this, get 2 Blood Gems.
- Surf n' Surf (1/1) [NAGA/BEAST]: Spellcraft: Give a minion "Deathrattle: Summon a 3/2 Crab" until next turn.
- Twilight Hatchling (1/1) [DRAGON]: Deathrattle: Summon a 3/3 Whelp that attacks immediately.
- Wrath Weaver (1/4) [DEMON]: After you play a Demon, deal 1 damage to your hero and gain +2/+1.

---

### Tier 2 (37 minions)

- Alert Alarmist (1/1) [MECH]: Taunt. Deathrattle: The next Tavern spell you buy costs (1) less.
- Ancestral Automaton (3/4) [MECH]: Has +3/+2 for each other Ancestral Automaton you've summoned this game (wherever this is).
- Blazing Skyfin (2/4) [MURLOC/DRAGON]: After you trigger a Battlecry, gain +1/+1.
- Bristleback Bully (3/2) [QUILBOAR]: Taunt. Deathrattle: Get a Blood Gem that also gives a Quilboar Taunt.
- Defiant Shipwright (2/5) [PIRATE]: Whenever this gains Attack from other sources, gain +1 Health.
- Eternal Knight (4/1) [UNDEAD]: Has +4/+1 for each friendly Eternal Knight that died this game (wherever this is).
- Expert Aviator (3/4) [MURLOC]: Rally: Summon the highest-Attack minion from your hand for this combat only.
- Fire Baller (4/3) [ELEMENTAL]: When you sell this, give your minions +1 Attack. Improve your future Ballers.
- Freedealing Gambler (3/3) [PIRATE]: This minion sells for 3 Gold.
- Friendly Saloonkeeper (3/4): Battlecry: Your teammate gets a Tavern Coin.
- Gathering Stormer (5/1) [ELEMENTAL]: When you sell this, your teammate gains 1 Gold. (Improves each turn!)
- Generous Geomancer (1/1) [QUILBOAR]: Deathrattle: You and your teammate each get a Blood Gem.
- Glowgullet Warlord (2/2) [QUILBOAR]: Deathrattle: Summon two 1/1 Quilboar with Taunt. This plays a Blood Gem on them.
- Humming Bird (1/4) [BEAST]: Start of Combat: For the rest of this combat, your Beasts have +1 Attack.
- Intrepid Botanist (3/4): Choose One — Your Tavern spells give an extra +1 Attack this game; or +1 Health.
- Laboratory Assistant (3/4) [DEMON]: Battlecry: Add a Fodder to your next 3 Refreshes.
- Lava Lurker (2/5) [NAGA]: The first Spellcraft spell played from hand on this each turn is permanent.
- Metallic Hunter (2/1) [MECH]: Deathrattle: Get a Pointy Arrow.
- Nerubian Deathswarmer (1/4) [UNDEAD]: Battlecry: Your Undead have +1 Attack this game (wherever they are).
- Old Soul (3/4) [UNDEAD]: After 15 friendly minions die while this is in your hand, make this Golden.
- Oozeling Gladiator (2/2): Battlecry: Get two Slimy Shields that give +1/+1 and Taunt.
- Patient Scout (1/1): When you sell this, Discover a Tier 1 minion. (Improves each turn!)
- Prophet of the Boar (2/3): Taunt. After you play a Quilboar, get a Blood Gem.
- Reef Riffer (3/2) [NAGA]: Spellcraft: Give a minion stats equal to your Tier until next turn.
- Scarlet Skull (2/1) [UNDEAD]: Reborn. Deathrattle: Give a friendly Undead +1/+2.
- Sellemental (3/3) [ELEMENTAL]: When you sell this, get a 3/3 Elemental.
- Sewer Rat (3/2) [BEAST]: Deathrattle: Summon a 2/3 Turtle with Taunt.
- Shell Collector (4/3) [NAGA]: Battlecry: Get a Tavern Coin.
- Sleepy Supporter (4/3) [DRAGON]: Rally: Give the minion to the right of this +2/+2.
- Snow Baller (3/4) [ELEMENTAL]: When you sell this, give your minions +1 Health. Improve your future Ballers.
- Soul Rewinder (4/1) [DEMON]: After your hero takes damage, rewind it and give this +1 Health.
- Surfing Sylvar (1/2) [PIRATE]: At the end of your turn, give adjacent minions +1 Attack. Repeat for each friendly Golden minion.
- Tad (2/2) [MURLOC]: When you sell this, get a random Murloc.
- Tarecgosa (4/4) [DRAGON]: This permanently keeps Bonus Keywords and stats gained in combat.
- Tide Raiser (2/1) [NAGA]: Taunt. Deathrattle: Cast Shifting Tide on an adjacent minion.
- Very Hungry Winterfinner (2/5) [MURLOC]: Taunt. Whenever this takes damage, give a minion in your hand +2/+1.
- Wanderer Cho (4/3): One Pass each turn is free.

---

### Tier 3 (47 minions)

- Accord-o-Tron (3/3) [MECH]: Magnetic. At the start of your turn, gain 1 Gold.
- Amber Guardian (3/2) [DRAGON]: Taunt. Start of Combat: Give another friendly Dragon +2/+2 and Divine Shield.
- Annoy-o-Module (2/4) [MECH]: Magnetic. Divine Shield. Taunt.
- Auto Assembler (2/2) [MECH]: Magnetic. Deathrattle: Summon an Ancestral Automaton.
- Black Chromadrake (2/6) [DRAGON]: Battlecry: Your Tavern spells give an extra +1 Health this game.
- Blue Chromadrake (4/4) [DRAGON]: Battlecry: Get a random 2-Cost Tavern spell.
- Bottom Feeder (3/4) [MURLOC]: At the end of your turn, you and your teammate each get a random Tier 1 card.
- Bristlemane Scrapsmith (4/4) [QUILBOAR]: After a friendly minion with Taunt dies, get a Blood Gem.
- Bronze Chromadrake (5/3) [DRAGON]: Battlecry: Give your other Dragons +5 Attack.
- Cadaver Caretaker (3/3) [UNDEAD]: Deathrattle: Summon three 1/1 Skeletons.
- Deadly Spore (1/1): Venomous.
- Deep Blue Crooner (2/4) [NAGA]: Spellcraft: Give a minion +1/+2 until next turn. Improve your future Deep Blues.
- Deep-Sea Angler (2/3) [NAGA]: Spellcraft: Give a minion +2/+6 and Taunt until next turn.
- Deflect-o-Bot (3/2) [MECH]: Divine Shield. Whenever you summon a Mech during combat, gain +2 Attack and Divine Shield.
- Disguised Graverobber (4/4): Battlecry: Destroy a friendly Undead to get a plain copy of it.
- Doting Dracthyr (4/3) [DRAGON]: At the end of your turn, give your teammate's minions +1 Attack.
- Dustbone Devastator (2/6) [UNDEAD]: Rally: Your Undead have +1 Attack this game (wherever they are).
- Felemental (3/3) [ELEMENTAL/DEMON]: Battlecry: Give minions in the Tavern +2/+1 this game.
- Floating Watcher (4/4) [DEMON]: Whenever your hero takes damage on your turn, gain +2/+2.
- Green Chromadrake (3/5) [DRAGON]: Battlecry: Give your other Dragons +5 Health.
- Gunpowder Courier (2/6) [PIRATE]: Whenever you spend 5 Gold, give your Pirates +1 Attack.
- Handless Forsaken (2/1) [UNDEAD]: Deathrattle: Summon a 2/1 Hand with Reborn.
- Hardy Orca (1/6) [BEAST]: Taunt. Whenever this takes damage, give your other minions +1/+1.
- Jumping Jack (3/4) [ALL]: After the first time this is sold, Pass it.
- King Bagurgle (2/3) [MURLOC]: Battlecry: Give all other Murlocs in your hand and board +2/+3.
- Leeching Felhound (3/3) [DEMON]: This costs Health instead of Gold to buy.
- Lost City Looter (1/1) [PIRATE]: Taunt. At the end of your turn, get a random Bounty.
- Moon-Bacon Jazzer (2/5) [QUILBOAR]: Battlecry: Your Blood Gems give an extra +1 Health this game.
- Mummifier (5/2) [UNDEAD]: Deathrattle: Give a different friendly Undead Reborn.
- Orc-estra Conductor (4/4): Battlecry: Give a minion +2/+2 (Improved by each Orc-estra your team has played this game).
- Peggy Sturdybone (2/1) [PIRATE]: Whenever a card is added to your hand, give another friendly Pirate +2/+1.
- Plunder Pal (2/2) [PIRATE]: At the start of your turn, you and your teammate each gain 1 Gold.
- Prickly Piper (5/1) [QUILBOAR]: Deathrattle: Your Blood Gems give an extra +1 Attack this game.
- Puddle Prancer (4/4) [MURLOC]: After this is Passed, gain +4/+4.
- Pufferquil (2/6) [QUILBOAR/NAGA]: Whenever a spell is cast on this, gain Venomous until next turn.
- Red Chromadrake (6/2) [DRAGON]: Battlecry: Your Tavern spells give an extra +1 Attack this game.
- Roaring Recruiter (2/8) [DRAGON]: Whenever another friendly Dragon attacks, give it +3/+1.
- Scourfin (4/3) [MURLOC]: Deathrattle: Give a random minion in your hand +7/+7.
- Shoalfin Mystic (4/4) [MURLOC]: When you sell this, your Tavern spells give an extra +1/+1 this game.
- Skulking Bristlemane (5/2) [QUILBOAR]: Taunt. Deathrattle: This plays a permanent Blood Gem on adjacent minions.
- Sly Raptor (1/3) [BEAST]: Deathrattle: Summon a random Beast. Set its stats to 6/6.
- Sprightly Scarab (3/1) [BEAST]: Choose One — Give a Beast +1/+1 and Reborn; or +4 Attack and Windfury.
- Technical Element (5/6) [ELEMENTAL/MECH]: Magnetic. Can Magnetize to both Mechs and Elementals.
- Timecap'n Hooktail (1/4) [DRAGON/PIRATE]: Whenever you cast a Tavern spell, give your minions +1 Attack.
- Waveling (5/1) [ELEMENTAL]: Deathrattle: After the Tavern is Refreshed this game, give a random minion in it +4/+4.
- Wheeled Crewmate (6/3) [PIRATE]: Deathrattle: Reduce the Cost of upgrading your team's Taverns by (1).
- Wildfire Elemental (6/3) [ELEMENTAL]: After this attacks and kills a minion, deal excess damage to an adjacent minion.

---

### Tier 4 (59 minions)

- Abyssal Bruiser (1/1) [NAGA]: Divine Shield. Has +1/+1 for each Tavern spell you've cast this game.
- Banana Slamma (3/6) [BEAST]: After you summon a Beast in combat, double its Attack.
- Bigwig Bandit (4/6) [PIRATE]: Rally: Get a random Bounty.
- Blade Collector (3/2) [PIRATE]: Also damages the minions next to whomever this attacks.
- Bream Counter (4/4) [MURLOC]: While this is in your hand, after you play a Murloc, gain +4/+4.
- Deepwater Chieftain (3/2) [MURLOC]: Battlecry and Deathrattle: Get a Deepwater Clan.
- Determined Defender (5/5): Taunt. Deathrattle: Give adjacent minions +5/+5 and Taunt.
- Diremuck Forager (5/6) [MURLOC]: Start of Combat: When you have space, summon the highest-Attack minion from your hand for this combat only.
- Dual-Wield Corsair (2/4) [PIRATE]: Whenever you spend 5 Gold, give two friendly Pirates +3/+3.
- Egg of the Endtimes (0/5): After this is in your hand for 2 turns, choose a Tier 6 Dragon to hatch into.
- En-Djinn Blazer (5/5) [ELEMENTAL]: Battlecry: After the Tavern is Refreshed this game, give a random minion in it +8/+8.
- Enchanted Sentinel (3/5) [MECH]: Magnetic. Your Tavern spells give an extra +1/+1.
- Eternal Tycoon (2/6) [UNDEAD]: Avenge (5): Summon an Eternal Knight. It attacks immediately.
- Fearless Foodie (2/4) [QUILBOAR]: Choose One — Your Blood Gems give an extra +1/+1 this game; or Get 4 Blood Gems.
- Feisty Freshwater (6/4) [ELEMENTAL]: Deathrattle: You and your teammate each gain two free Refreshes.
- Flaming Enforcer (4/5) [ELEMENTAL/DEMON]: At the end of your turn, consume the highest-Health minion in the Tavern to gain its stats.
- Friendly Geist (6/3) [UNDEAD]: Deathrattle: Your Tavern spells give an extra +1 Attack this game.
- Geomagus Roogug (4/6) [QUILBOAR]: Divine Shield. Whenever a Blood Gem is played on this, this plays a Blood Gem on a different friendly minion.
- Grave Narrator (2/7) [UNDEAD]: Avenge (3): Your teammate gets a random minion of their most common type.
- Heroic Underdog (1/10): Stealth. Rally: Gain the target's Attack.
- Hired Ritualist (3/6) [QUILBOAR]: Once per turn, after a Blood Gem is played on this, gain 2 Gold.
- Humon'gozz (5/5): Divine Shield. Your Tavern spells give an extra +1/+2.
- Hunting Tiger Shark (3/5) [BEAST]: Battlecry: Discover a Beast.
- Ichoron the Protector (3/1) [ELEMENTAL]: Divine Shield. Whenever you play an Elemental, give it Divine Shield until next turn.
- Imposing Percussionist (4/4) [DEMON]: Battlecry: Discover a Demon. Deal damage to your hero equal to its Tier.
- Incubation Researcher (2/8) [DRAGON]: Avenge (4): Get a random Chromadrake.
- Leyline Surfacer (4/6) [ELEMENTAL]: Battlecry and Deathrattle: Get an Arcane Absorption.
- Lovesick Balladist (3/2) [PIRATE]: Battlecry: Give a Pirate +1 Health. (Improved by each Gold you spent this turn!)
- Malchezaar, Prince of Dance (5/4) [DEMON]: Two Refreshes each turn cost Health instead of Gold.
- Mama Mrrglton (6/3) [MURLOC]: Battlecry: Give your other Murlocs +3 Attack. (Improved by each Mrrglton you played this game!)
- Mantid King (3/3): After your team Passes, randomly gain Venomous, Taunt, or Divine Shield until next turn.
- Marquee Ticker (1/5) [MECH]: At the end of your turn, get a random Tavern spell.
- Mirror Monster (4/4) [ALL]: When you buy or Discover this, get an extra copy and Pass it.
- Monstrous Macaw (5/4) [BEAST]: Rally: Trigger your left-most Deathrattle (except this minion's).
- Papa Mrrglton (3/6) [MURLOC]: Battlecry: Give your other Murlocs +3 Health. (Improved by each Mrrglton you played this game!)
- Persistent Poet (2/3) [DRAGON]: Divine Shield. Adjacent Dragons permanently keep Bonus Keywords and stats gained in combat.
- Plaguerunner (4/2) [UNDEAD]: Deathrattle: Your Undead have +2 Attack this game, wherever they are. (+4 if this died outside combat!)
- Private Chef (5/4) [NAGA]: Spellcraft: Choose a minion. Get a different random minion of its type, then Pass it.
- Prized Promo-Drake (1/1) [DRAGON]: Start of Combat: Give your Dragons +4/+4.
- Prosthetic Hand (3/1) [UNDEAD/MECH]: Magnetic. Reborn. Can Magnetize to Mechs or Undead.
- Redtusk Thornraiser (1/6) [QUILBOAR]: At the end of your turn, get a Blood Gem that also gives a Quilboar Reborn.
- Refreshing Anomaly (4/5) [ELEMENTAL]: Battlecry: Gain 2 free Refreshes.
- Rimescale Priestess (3/3) [NAGA]: Spellcraft: Get a random Tavern spell that gives stats.
- Roving Sailor (7/3) [PIRATE]: Battlecry: Give a friendly minion +2/+2. (Improved by each Tavern spell you cast this turn!)
- Rylak Metalhead (5/3) [BEAST]: Taunt. Deathrattle: Trigger the Battlecry of an adjacent minion.
- San'layn Scribe (4/4) [UNDEAD]: Has +4/+4 for each of your team's San'layn Scribes that died this game (wherever this is).
- Seafloor Recruiter (3/5) [NAGA]: Rally: Cast Chef's Choice on the minion to the right.
- Shadowdancer (4/2) [DEMON]: Taunt. Deathrattle: Get a Staff of Enrichment.
- Shifty Snake (6/1) [BEAST]: Deathrattle: Your teammate gets a random Deathrattle minion.
- Sin'dorei Straight Shot (3/4): Divine Shield. Windfury. Rally: Remove Reborn and Taunt from the target.
- Stomping Stegodon (3/3) [BEAST]: Rally: Give your other Beasts +4 Attack and this Rally.
- Tavern Tempest (2/2) [ELEMENTAL]: Battlecry: Get a random Elemental.
- Tortollan Blue Shell (3/6): If you lost your last combat, this minion sells for 5 Gold.
- Trigore the Lasher (9/3) [BEAST]: Whenever another friendly Beast takes damage, gain +2 Health permanently.
- Tunnel Blaster (3/7): Taunt. Deathrattle: Deal 3 damage to all minions.
- Waverider (2/8) [NAGA]: Spellcraft: Give a minion +2/+2. If it's a Naga, also give it Windfury until next turn.
- Woodland Defiler (5/6) [DEMON]: At the end of your turn, add a Fodder to your next 3 Refreshes.
- Wyvern Outrider (2/8) [BEAST]: Whenever this takes damage, gain a free Refresh. (3 times per turn.)
- Zesty Shaker (6/7) [NAGA]: Once per turn, when a Spellcraft spell is played on this, get a new copy of it.

---

### Tier 5 (56 minions)

- Air Revenant (3/6) [ELEMENTAL]: After you spend 7 Gold, cast Easterly Winds.
- Ashen Corruptor (6/6) [DEMON]: After your hero takes damage, rewind it and give minions in the Tavern +1/+1 this turn.
- Bazaar Dealer (4/6) [DEMON]: One Tavern spell each turn costs Health instead of Gold to buy.
- Bile Spitter (1/10) [MURLOC]: Venomous. Rally: Give another friendly Murloc Venomous.
- Brann Bronzebeard (2/4): Your Battlecries trigger twice.
- Brazen Buccaneer (4/4) [PIRATE]: At the end of your turn, give your left-most Pirate +2/+2. Repeat for each card you played this turn.
- Cataclysmic Harbinger (6/10): At the end of your turn, get a copy of the last Tavern spell you cast.
- Catacomb Crasher (6/10) [UNDEAD]: Whenever you would summon a minion that doesn't fit in your warband, give your minions +2/+1 permanently.
- Charging Czarina (6/2) [MECH]: Divine Shield. Whenever you cast a Tavern spell, give your minions with Divine Shield +3 Attack.
- Cousin Errgl (5/5) [MURLOC]: At the end of your turn, get a Mama Mrrglton or a Papa Mrrglton.
- Darkcrest Strategist (4/5) [NAGA]: Spellcraft: Get a random Tier 1 Naga. (Improves each turn!)
- Darkgaze Elder (5/6) [QUILBOAR]: Whenever you spend 6 Gold, this plays a Blood Gem on all your Quilboar.
- Divine Sparkbot (4/2) [MECH]: Taunt. Divine Shield. Deathrattle: Get a Sanctify.
- Draconic Warden (7/4) [DRAGON]: Battlecry and Deathrattle: Get a random Chromadrake.
- Drakkari Enchanter (1/5): Your end of turn effects trigger twice.
- Drustfallen Butcher (2/9) [UNDEAD]: Avenge (4): Get a Butchering.
- Felfire Conjurer (6/5) [DEMON/DRAGON]: At the end of your turn, your Tavern spells give an extra +1/+1 this game.
- Firelands Fugitive (5/7) [ELEMENTAL]: Battlecry: Get a Conflagration.
- Gem Smuggler (4/5) [QUILBOAR]: Battlecry: This plays 2 Blood Gems on all your other minions.
- Glowscale (4/6) [NAGA]: Taunt. Spellcraft: Give a minion Divine Shield until next turn.
- Hog Watcher (5/5): Battlecry: Get a Blood Gem that also gives a Quilboar Divine Shield.
- Hot-Air Surveyor (4/8) [QUILBOAR]: Blood Gems played from your hand cast an extra time.
- Iridescent Skyblazer (3/8) [BEAST]: Whenever a friendly Beast takes damage, give a friendly Beast other than it +3/+1 permanently.
- Kangor's Apprentice (3/6): Deathrattle: Summon plain copies of your first 2 Mechs that died this combat.
- Lantern Lava (5/5) [ELEMENTAL]: Get a plain copy of the first Elemental you sell each turn (except Lantern Lava).
- Leeroy the Reckless (6/2): Deathrattle: Destroy the minion that killed this.
- Living Azerite (5/5) [ELEMENTAL]: Whenever you cast a Tavern spell, give Elementals in the Tavern +3/+3 this game.
- Lurking Leviathan (3/8) [BEAST]: Whenever you summon a Beast, give it +2 Attack and improve this permanently.
- Maelstrom Emergent (2/7) [NAGA]: Your Tavern spells cast an extra time in combat.
- Magnanimoose (5/2) [BEAST]: Deathrattle: Summon a copy of a minion from your teammate's warband. Set its Health to 1 (except Magnanimoose).
- Man'ari Messenger (9/6) [DEMON]: Battlecry: Minions in your team's Taverns have +1/+1 this game.
- Mrglin' Burglar (10/8) [MURLOC]: After you play a Murloc, give a friendly minion and a minion in your hand +5/+4.
- Nalaa the Redeemer (5/7): Whenever you cast a Tavern spell, give a friendly minion of each type +3/+2.
- Nightmare Par-tea Guest (3/3) [ALL]: Battlecry and Deathrattle: Get a Misplaced Tea Set.
- Primalfin Lookout (3/2) [MURLOC]: Battlecry: If you control another Murloc, Discover a Murloc.
- Proud Privateer (8/8) [PIRATE]: Your Bounties cast twice.
- Ring Bearer (3/8) [NAGA/DRAGON]: Whenever 2 friendly minions attack, cast Shiny Ring.
- Rodeo Performer (3/4): Battlecry: Discover a Tavern spell.
- Scrap Scraper (6/5) [MECH]: Deathrattle: Get a random Magnetic Mech.
- Selfless Sightseer (6/2) [DRAGON]: Battlecry: Increase your team's maximum Gold by (1).
- Sewer Lord (4/6) [BEAST]: Deathrattle: Summon two Sewer Rats that summon 2/3 Turtles with Taunt.
- Shipwrecked Rascal (5/4) [PIRATE]: Battlecry and Deathrattle: Get a random Bounty.
- Sinrunner Blanchy (4/4) [UNDEAD/BEAST]: Reborn. This is Reborn with full stats and Bonus Keywords.
- Skeletal Strafer (6/6) [UNDEAD]: At the end of your turn, give your minions +1/+1. Avenge (2): Improve this permanently.
- Spiked Savior (8/2) [BEAST]: Taunt. Reborn. Deathrattle: Give your minions +1 Health and deal 1 damage to them.
- Storm Splitter (5/5) [NAGA]: Once per turn, after you Pass a Tavern spell, get a new copy of it.
- Support System (4/5) [MECH]: At the end of your turn, give a minion in your teammate's warband Divine Shield.
- Three Lil' Quilboar (3/3) [QUILBOAR]: Deathrattle: This plays 3 Blood Gems on all your Quilboar.
- Tichondrius (3/6) [DEMON]: After your hero takes damage, give your Demons +3/+2.
- Titus Rivendare (1/7): Your Deathrattles trigger an extra time.
- Tranquil Meditative (3/8) [NAGA]: Spellcraft: Your Tavern spells give an extra +1/+1 this game.
- Twilight Broodmother (7/4) [DRAGON]: Deathrattle: Summon 2 Twilight Hatchlings. Give them Taunt.
- Twisted Wrathguard (4/4) [DEMON]: After you sell a minion, add a Fodder to your next Refresh.
- Void Pup Trainer (6/6) [DEMON]: Battlecry: Give minions in the Tavern from Tier 3 and below +4/+4 this game.
- Well Wisher (6/6): Spellcraft: Pass a different non-Golden minion.
- Wintergrasp Ghoul (5/3) [UNDEAD]: Deathrattle: Get a Tomb Turning.

---

### Tier 6 (36 minions)

- Balinda Stonehearth (6/6): Your spells that target friendly minions cast twice.
- Batty Terrorguard (6/2) [DEMON]: After you cast a Tavern spell, another friendly Demon consumes a minion in the Tavern to gain its stats.
- Bristlebach (3/10) [QUILBOAR]: Avenge (2): This plays 2 Blood Gems on all your Quilboar.
- Choral Mrrrglr (6/6) [MURLOC]: Start of Combat: Gain the stats of all the minions in your hand.
- Consummate Conqueror (9/7) [DEMON]: Whenever a minion is consumed, give minions in the Tavern +1/+1 this turn.
- Dark Dazzler (4/7) [DEMON]: After your teammate sells a minion, gain its stats. (Once per turn.)
- Dastardly Drust (5/3) [PIRATE]: Whenever you get a Pirate, give your minions +2/+1. Give Golden ones +4/+2 instead.
- Deathly Striker (8/8) [UNDEAD]: Avenge (4): Get a random Undead. Deathrattle: Summon it from your hand for this combat only.
- Earthsong Shaman (4/5) [QUILBOAR]: Windfury. At the end of your turn, play a Blood Gem on all your minions. Repeat for each Bonus Keyword this has.
- Elemental of Surprise (8/8) [ELEMENTAL]: Divine Shield. This minion can triple with any Elemental.
- Eternal Summoner (8/1) [UNDEAD]: Reborn. Deathrattle: Summon 1 Eternal Knight.
- Falling Sky Golem (4/2) [MECH]: Divine Shield. Has +4/+2 for each Deathrattle you've triggered this game (wherever this is).
- Famished Felbat (9/5) [DEMON]: At the end of your turn, your Demons each consume a minion in the Tavern to gain its stats.
- Fire-forged Evoker (8/5) [DRAGON]: Start of Combat: Give your Dragons +1/+1. Improves permanently after you cast a Tavern spell.
- Forsaken Weaver (3/8) [UNDEAD]: After you cast a Tavern spell, your Undead have +1 Attack this game (wherever they are).
- Goldrinn, the Great Wolf (8/8) [BEAST]: Deathrattle: For the rest of this combat, your Beasts have +8/+8.
- Groundbreaker (5/4) [NAGA]: After you play a Naga, gain +1/+1. (Improved by every 4 spells you've cast this game!)
- Ignition Specialist (8/8) [DRAGON]: At the end of your turn, get 2 random Tavern spells.
- Junk Jouster (8/7) [MECH]: After you Magnetize a minion, give your minions +6/+6.
- Kalecgos, Arcane Aspect (4/12) [DRAGON]: After you trigger a Battlecry, give your Dragons +2/+2.
- Loyal Mobster (6/5) [QUILBOAR]: At the end of your turn, this plays a Blood Gem on all your teammate's minions.
- Magicfin Mycologist (4/8) [MURLOC]: Once per turn, after you buy a Tavern spell, get a 1/1 Murloc and teach it that spell.
- Moonsteel Juggernaut (6/6) [MECH]: At the end of your turn, get a 6/6 Magnetic Satellite and improve this.
- Nightbane, Ignited (16/8) [UNDEAD/DRAGON]: Taunt. Deathrattle: Give 2 different friendly minions this minion's Attack.
- One-Amalgam Tour Group (6/7) [ALL]: Whenever you play a card, give friendly minions of its Tier or lower +2/+1.
- P-0UL-TR-0N (8/8) [MECH/BEAST]: Avenge (4): Gain Divine Shield and attack immediately.
- Primitive Painter (3/8) [MURLOC]: After you play a card from Tier 3 or below, give your Murlocs +1/+2.
- Rabid Panther (4/8) [BEAST]: After you play a Beast, give your Beasts +3/+3 and deal 1 damage to them.
- Ruthless Queensguard (3/3) [NAGA]: Battlecry, Deathrattle, and Rally: Cast Queen's Command.
- Ship Jumper (6/6) [PIRATE]: Deathrattle: Summon a 1/1 Sky Pirate and give it this minion's Attack. It attacks immediately.
- Ship Master Eudora (10/5) [PIRATE]: Deathrattle: Give your minions +8/+8. Golden ones keep it permanently.
- Sky Admiral Rogers (4/6) [PIRATE]: After you spend 10 Gold, get a random Bounty.
- Tidemistress Athissa (6/7) [NAGA]: Whenever you cast a spell, give all your Naga +1/+1 permanently.
- Transport Reactor (1/1) [MECH]: Magnetic. Has +1/+1 for each time your team has Passed this game (wherever this is).
- Ultraviolet Ascendant (6/3) [ELEMENTAL]: Start of Combat: Give your other Elementals +3/+2. (Improves after you play an Elemental!)
- Vinespeaker (7/8) [QUILBOAR]: After a friendly Deathrattle minion dies, your Blood Gems give an extra +1 Attack this game.

---

### Tier 7 (13 minions)

- Captain Sanders (9/9) [PIRATE]: Battlecry: Make a friendly minion from Tier 6 or below Golden.
- Champion of Sargeras (10/10) [DEMON]: Battlecry and Deathrattle: Minions in the Tavern have +5/+5 this game.
- Futurefin (7/13) [MURLOC]: At the end of your turn, give this minion's stats to the left-most minion in your hand.
- Highkeeper Ra (6/6): Battlecry, Deathrattle, and Rally: Get a random Tier 6 minion.
- Obsidian Ravager (7/7) [DRAGON]: Rally: Deal damage equal to this minion's Attack to the target and an adjacent minion.
- Polarizing Beatboxer (5/10) [MECH]: Whenever you Magnetize to a different minion, it also Magnetizes to this.
- Sandy (1/1): Start of Combat: Transform into a copy of your teammate's highest-Health minion.
- Sanguine Champion (18/3) [QUILBOAR]: Battlecry and Deathrattle: Your Blood Gems give an extra +1/+1 this game.
- Sea Witch Zar'jira (4/5) [NAGA]: Spellcraft: Choose a different minion in the Tavern to get a copy of.
- Stalwart Kodo (16/32) [BEAST]: After you summon a minion in combat, give it this minion's maximum stats. (3 times per combat.)
- Stitched Salvager (16/4) [UNDEAD]: Start of Combat: Destroy the minion to the left. Deathrattle: Summon an exact copy of it. (Except Stitched Salvager.)
- Stone Age Slab (5/10) [ELEMENTAL]: After you buy a minion, give it +10/+10 and double its stats. (Once per turn.)
- The Last One Standing (12/12) [ALL]: Rally: Give a friendly minion of each type +12/+12 permanently.

---

## Trinket System (Season with Trinkets)

Trinkets are permanent items offered at fixed turns during a game. Each player receives one Lesser and one Greater Trinket per game (unless anomalies/hero powers change this).

- **Lesser Trinkets** (cost 1–3): offered around Turn 5
- **Greater Trinkets** (cost 4+): offered around Turn 8–9

Source: HearthstoneJSON `type: BATTLEGROUND_TRINKET`. 116 Lesser + 61 Greater = 177 unique trinkets.

### Key Lesser Trinkets (cost 1–3)

- Alliance Keychain [2]: The first time a friendly minion dies each combat, give its maximum stats to a random friendly minion.
- Artisanal Urn [1]: Your Undead have +3 Attack.
- Beetle Band [1]: Avenge (5): Summon a 2/2 Beetle. Give it Taunt.
- Bird Feeder [2]: Avenge (2): Give your minions +1/+1.
- Ceremonial Sword [2]: Whenever a friendly minion attacks, give it +4 Attack.
- Charging Staff [1]: At the end of each turn, give your minions with Divine Shield +7 Attack.
- Chromatic Tear [3]: Get all 5 Chromadrakes.
- Cursed Crystal [1]: After the Tavern is Refreshed, give its minions +2/+2 this turn.
- Dazzling Dagger [2]: Your minions have +1 Attack. (Improved by every 4 spells you've cast this game!)
- Deathly Phylactery [2]: Discover a Deathrattle minion. Your first Deathrattle each combat triggers an extra time.
- Eclectic Shrine [1]: Start of Combat: Give a friendly minion of each different type +3/+2. Improve this permanently.
- Electrode Attractor [3]: Magnetic Mechs cost (1). The Tavern offers an extra one whenever it is Refreshed.
- Electromagnetic Device [2]: Discover a Magnetic Mech. Whenever a friendly minion is Magnetized, give it +3/+3.
- Enigmatic Headstone [3]: At the end of each turn, your Undead have +1 Attack this game (wherever they are).
- Exquisite Dishware [3]: At the end of each turn, get a random minion of each different type you control.
- Feral Talisman [2]: Your minions have +2/+1.
- Fountain Pen [1]: Your Elementals that give stats grant an extra +4/+2.
- Gem Donation [1]: The first minion you sell each turn plays its Blood Gems on the 3 highest-Tier minions in the Tavern.
- Goblin Wallet [1]: At the end of each turn, increase your maximum Gold by 1.
- Gold Pendant [2]: Make a random friendly minion from Tier 3 or below Golden. At the start of each turn, repeat this.
- Heart of the Forest [3]: Your Tavern spells give an extra +1/+1. After you cast 5 Tavern spells, improve this.
- Hoggy Bank [1]: Start of Combat: Give your Quilboar "Deathrattle: Get a Blood Gem."
- Innkeeper's Hearth [1]: Discover a minion of your Tier. Set its stats to 12/12.
- Jarred Frostling [3]: Start of Combat: Give 2 friendly Elementals "Deathrattle: Summon a Flourishing Frostling."
- Kodo Leather Pouch [3]: After you buy a card, give two random friendly minions +2/+1.
- Lorewalker Scroll [3]: Whenever you cast a spell on a minion, give it +2/+1.
- Marvelous Mushroom [2]: Your Tavern spells give an extra +1/+1. At the start of each turn, improve this.
- Peacebloom Candle [1]: The first 2 Tavern spells you buy each turn are free.
- Protective Ring [2]: Start of Combat: Give 3 random friendly Pirates Divine Shield.
- Reinforced Shield [2]: Whenever you summon a minion, give it Divine Shield. (5 times per combat.)
- Replica Cathedral [2]: Your first spell each turn casts an extra time.
- Tiger Carving [3]: Whenever a friendly minion takes damage, give a random friendly minion +2 Attack permanently.
- Training Certificate [3]: Start of Combat: Double the stats of your two lowest-Attack minions.
- Twin Sky Lanterns [1]: When you have space, summon a copy of the first minion you summon each combat.
- Unholy Sanctum [3]: After you trigger a Deathrattle, give your right-most minion +2/+2 permanently.
- Valorous Medallion [2]: Start of Combat: Give your minions +2/+2.
- War Drum [2]: One Battlecry each turn triggers two extra times.

### Key Greater Trinkets (cost 4+)

- All-Purpose Kibble [5]: Whenever a friendly Beast attacks, give it +2 Attack and permanently improve this.
- Auric Offering [4]: At the end of each turn, give your left-most minion +3/+2. Repeat for each friendly Golden minion.
- Azsharan Statuette [4]: Get 3 random Spellcraft spells. At the start of each turn, get 3 more.
- Baleful Incense [4]: Start of Combat: Give your left- and right-most Undead Reborn.
- Bewitched Ribbon [5]: Whenever you cast a spell, give your minions +1/+1 permanently. (+3/+3 while in combat!)
- Blood Amulet [5]: After you trigger a Deathrattle, this plays a permanent Blood Gem on 3 random friendly minions.
- Boom Controller [7]: When you have space, summon an exact copy of your first Mech that died each combat.
- Bronze Timepiece [5]: Start of Combat: Give each of your minions Health equal to half its Attack.
- Charm of Generosity [5]: After your team Passes, give your minions +2 Attack.
- Common Thread [4]: Pass a copy of the first card you buy each turn.
- Coral Spear [4]: Whenever you cast a Spellcraft spell, cast Might of Stormwind.
- Designer Eyepatch [7]: You only need 2 copies of a Pirate to make it Golden.
- Electromagnetic Device [5]: Discover 2 Magnetic Mechs. Whenever a friendly minion is Magnetized, give it +4/+4.
- Emerald Dreamcatcher [7]: Start of Combat: Set your Dragons' Attack to the highest in your warband.
- Faerie Dragon Scale [6]: Whenever a friendly Dragon attacks, give it Divine Shield. (3 times per combat.)
- Fang Anklet [4]: Start of Combat: Your Beasts have +1/+1 this combat. After you summon a Beast, improve this permanently.
- Gilnean Thorned Rose [4]: Avenge (3): Give your minions +3/+3 permanently and deal 1 damage to them.
- Holy Mallet [4]: Start of Combat: Give your left and right-most minions Divine Shield.
- Ironforge Anvil [4]: Start of Combat: Triple the stats of your minions with no type.
- Jewelry Box [4]: Get 3 Blood Gems that give a Quilboar Taunt, Divine Shield, or Reborn. At the start of each turn, repeat this.
- Karazhan Chess Set [5]: Start of Combat: Summon a copy of your left-most minion.
- Kodo Leather Pouch [4]: After you buy a card, give two random friendly minions +4/+4.
- Lorewalker Scroll [4]: Whenever you cast a spell on a minion, give it +7/+7.
- Mechagon Adapter [4]: After a friendly Mech loses Divine Shield, give it Divine Shield. (3 times per combat.)
- Pagle's Fishing Rod [5]: Get a random Tier 7 minion. At the start of each turn, get another.
- Portable Factory [5]: Discover a Tier 4 minion with a type. At the start of each turn, get another copy of it.
- Quilligraphy Set [4]: Avenge (4): Your Blood Gems give an extra +1/+1 this game.
- Summoning Sphere [4]: Start of Combat: Summon a copy of your teammate's highest-Health minion.
- Unholy Sanctum [5]: After you trigger a Deathrattle, give your right-most minion +6/+4 permanently.
- Valdrakken Wind Chimes [7]: Your Start of Combat effects trigger an extra time.
- Valorous Medallion [3]: Start of Combat: Give your minions +6/+6.
- Wizard's Pipe [5]: Whenever you cast a Tavern spell, give your minions with no type +4/+4.
- Yogg-Tastic Pastry [5]: Spin the Wheel of Yogg-Saron. At the start of each turn, spin it again.
