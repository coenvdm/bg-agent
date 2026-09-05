# bg_agent — Development Context Log

---
### SUMMARY: 2026-03-15 through 2026-04-12

**Phase 1 — Dataset & Parser (2026-03-15 to 2026-03-25)**
Built `parse_bg.py` + `collect_dataset.py` to parse Hearthstone power logs into structured JSON (shopping/combat rounds, hero health, shop state, action sequences). Fixed ~10 parser bugs: anonymous sell entities via SubSpell dispatch, ZONE_POSITION guard, board > 7 inflation cap, hero health always-30 (HEALTH vs HEALTH-DAMAGE), round-1 shop race condition, `InconsistentPlayerIdError`/`AssertionError` recovery, anomaly DBID detection, ghost game filtering. Final state: 9 clean games, 100% sell resolution on fresh logs.

**Phase 2 — Behavioural Cloning (2026-03-20 to 2026-03-27)**
Built BC v1 (flat 20-class), then BC v2 (`BGPolicyV2`: type_head + pointer_head) in `explore.ipynb`. Fixed inverted mask formula, hand carry-over across rounds, GT label masking. V2 beats majority-class baseline on val set. Implemented `load_bc_v2_weights` for BC → PPO warm-start; extended BC training pipeline.

**Phase 3 — Core Architecture (2026-03-27 to 2026-03-30)**
Refactored `BGPolicyNetwork` to two-headed output (type + pointer) matching BGPolicyV2. Implemented full game loop end-to-end (8-player BG sim, BGCombatSim with ~20 deathrattles, 7 SOC triggers, Titus, DR chains). Wired PPO transition recording with buffered end-turn flush and terminal placement rewards. Implemented: triple/golden system, magnetic Mechs, smart play positioning, spell handling, EffectHandler (battlecries + sell effects), `tie_prob` in combat sim.

**Phase 4 — Hero Powers + Simulator Fidelity (2026-03-31 to 2026-04-10)**
Full Phase 1 (passive) + Phase 2 (active no-pointer) hero power system: 29 heroes, `hero_definitions.json`, `HeroPowerHandler`, 10 new PlayerState fields, 9 unit tests. Simulator fidelity overhaul (Phases 1–3): on-attack/on-damage auras, `game_buffs` tribe permanents, blood gem bonuses, 30+ sell/battlecry effects, `discover_pending` action masking, Rally/Avenge/end-of-turn mechanics, Khadgar helper, 5 new deathrattles. Added `nbstripout` and `enc_zone` notebook helper. Dropped ShopAnalyzer.

**Phase 5 — Architecture Overhaul + Parallel Training (2026-04-10 to 2026-04-11)**
Redesigned BGPolicyNetwork: d_model=256, 4 layers, 8 heads (~3.5M params), per-token pointer scorers, slot positional encoding, scalar_dim 38→94 (own 24 + all-opp 64 + lobby 6). Implemented synchronous player-action batching (`get_action_batch`). Added CPU parallel training via ProcessPoolExecutor in `train.py` and notebook. Optimized PPO: n_epochs=2→1, batch_size→512, UPDATE_INTERVAL→2, 300s timeout, BrokenProcessPool recovery, `_worker_init` + `set_num_threads(1)`.

**Phase 6 — PPO Stability + Deployment (2026-04-11 to 2026-04-12)**
Fixed NaN divergence: switched to AdamW(weight_decay=1e-4), added NaN/Inf/large-loss mini-batch guard, fixed Embedding init (std=0.02). Fixed stale-mask NaN bug (type/pointer masks cached before state mutation). Expanded scalar_dim 94→98 (added gold, frozen, level_cost, hero_power_used). Added reward shaping: board presence bonus, empty-board penalty, hand penalty, gold efficiency penalty, escalating reroll penalty, KL early stopping (target_kl=0.02), return clipping, AdamW. Set up Dockerfile + `.dockerignore` for vast.ai deployment. Training running on 2× RTX 5060 Ti; reward improved from −3.3 to ~−1.8 before weight divergence was fixed.

---
### 2026-04-13 — Refactor _train_parallel to accept callbacks; simplify notebook
**Files changed:** `train.py`, `explore.ipynb`
**What was done:** Replaced the `args` namespace parameter in `_train_parallel` with explicit keyword arguments (`n_workers`, `seed`, `device`, `checkpoint_path`, etc.) and added two optional callbacks: `on_batch(game_idx, summaries, transitions, elapsed)` fired after every batch, and `on_update(metrics, update_count)` fired after every PPO update. Also moved pool rebuilding error handling and timeout into `_train_parallel`. Notebook cell 49 now just defines `_on_batch` and `_on_update` closures (handling plotting, list appending, and checkpoint saving) and calls `_train_parallel` directly — the worker loop, snapshot management, and PPO update trigger no longer live in the notebook.
**Current state:** `_train_parallel` is the single implementation of the parallel training loop. The notebook is a thin orchestration layer: config, callbacks, and plots only.
**Open questions / next steps:**
- Run notebook training cell to verify callbacks fire correctly and plots update as before.
---
### 2026-04-13 — Sync notebook and promote training constants to module level
**Files changed:** `train.py`, `explore.ipynb`
**What was done:** Moved `N_HEURISTIC_SLOTS`, `SNAPSHOT_EVERY`, and `MILESTONE_EVERY` from local variables inside `_train_parallel` to module-level constants in `train.py` so the notebook can import them. Updated `explore.ipynb` cells 46/47/49: cell 46 imports all constants from `train.py` (removing the local `N_PLAYERS = 8` duplicate); cell 47 fixes `gamma=0.99` → `0.997`; cell 49 removes the local `SNAPSHOT_EVERY` definition, adds `n_policy_slots` computation, replaces the old single `opp_sd = snapshot_pool.sample()` pattern with `sample_n` + heuristic sentinel, and adds milestone snapshot support.
**Current state:** Notebook and train.py are in sync. All training constants have a single source of truth in train.py.
**Open questions / next steps:**
- Run training from notebook to verify no import errors and worker processes start correctly.
---
### 2026-04-13 — Dockerfile and vast.ai deployment setup
**Files changed:** `Dockerfile`, `.dockerignore`
**What was done:** Created a Dockerfile based on `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` with all project dependencies and a `.dockerignore` to exclude weights, logs, and notebooks from the image. Discussed vast.ai template setup, SSH key auth for VSCode Remote-SSH, and Git LFS as the recommended approach for tracking `.pt` checkpoints across instances.
**Current state:** Dockerfile is committed and pushed. SSH key auth is configured for the vast.ai instance. Model weights are not yet tracked — Git LFS setup is pending.
**Open questions / next steps:**
- Set up Git LFS for `*.pt` checkpoint tracking so weights persist across instances
- Decide on checkpoint sync strategy (Git LFS vs Hugging Face Hub vs rclone)
---
### 2026-04-13 — Fix board-fill degenerate policy via reward shaping
**Files changed:** `env/game_loop.py`
**What was done:** Removed the flat `+0.10 * board_size` reward from `_end_of_turn_reward` — it was causing the agent to fill the board with weak minions and never sell, since the dense per-slot reward dominated all sparse combat signals. Also moved `FINAL_PLACEMENT_REWARD` from end-of-game to the moment of elimination: when a player dies their placement is immediately known, so the reward now fires on the same transition that has `done=True`, shortening the credit assignment gap significantly. Surviving players still receive their placement reward at game end as before.
**Current state:** `_end_of_turn_reward` no longer rewards board presence; empty-board penalty, hand penalty, and gold efficiency penalty remain. Eliminated players get their placement signal immediately; no double-counting.
**Open questions / next steps:**
- Re-run training to verify the board-fill policy no longer emerges.
- Monitor whether the empty-board penalty alone is sufficient to encourage buying, or if a weaker board-quality signal is still needed.
- Consider whether `FINAL_PLACEMENT_REWARD` magnitudes need retuning now that the signal fires earlier.
---
### 2026-04-13 — Increase PPO discount factor to reduce placement reward decay
**Files changed:** `agent/ppo.py`
**What was done:** Raised `gamma` from `0.99` to `0.997` in `PPOConfig`. With ~120 steps per game, `gamma=0.99` discounted the final placement reward to ~30% of face value by the time it reached early decisions, making per-round combat signals systematically louder. At `0.997` the same reward retains ~70% of its value.
**Current state:** Discount factor is `0.997`; `gae_lambda` remains at `0.95`.
**Open questions / next steps:**
- Monitor value loss during training — higher gamma increases return variance and can make the value function harder to fit.
- If training becomes unstable, drop `gae_lambda` from `0.95` toward `0.90` to reduce variance without touching gamma.
---
### 2026-04-13 — Population diversity / league training system
**Files changed:** `train.py`
**What was done:** Added three coordinated changes to break the self-play echo chamber that prevented the agent from discovering leveling strategies. (1) Upgraded `SnapshotPool` with protected milestone snapshots (every 50 PPO updates) that are never evicted, alongside the existing rolling buffer; added `sample_n(n)` for per-slot independent sampling. (2) Added `HeuristicAgent` — a scripted leveling-focused opponent that permanently occupies one opponent slot per game; it uses `build_type_mask` as its validity oracle and sets `supports_batching = False` to opt out of batched inference. (3) Updated `_worker_run_game` to accept a per-slot `opp_sds` list (dict/`"heuristic"`/None) with policy-network deduplication; updated `_train_parallel` to compose `opp_sds = sample_n(5) + ["heuristic"]` each batch.
**Current state:** Every game now contains one permanent heuristic leveling opponent and five independently-sampled historical policy opponents. Milestone snapshots preserve behavioral diversity across long training runs.
**Open questions / next steps:**
- Run training and check logs for `LEVEL_UP` action frequency — should increase within a few hundred games.
- The heuristic forces sequential shopping (no batched forward pass) — monitor throughput, expect ~1.5–2× slowdown per game.
- If leveling is still not discovered, consider increasing `N_HEURISTIC_SLOTS` to 2.
---
### 2026-04-17 — Add CARDS.md card pool reference document
**Files changed:** `CARDS.md`
**What was done:** Added CARDS.md containing all 264 active Battlegrounds pool minions across 7 tiers, scraped from HearthstoneJSON API. This file is referenced from CLAUDE.md as the authoritative card listing for use when working on `bg_card_pipeline.py`, the symbolic layer specs, or any card-specific logic.
**Current state:** CARDS.md is now tracked in the repo alongside CLAUDE.md. No code changes were made.
**Open questions / next steps:**
- Regenerate CARDS.md after each patch by re-running the HearthstoneJSON scraper filtered on `isBattlegroundsPoolMinion: true`.
- Apply the pending `env/tavern_pool.py.rej` patch (injects `card_id` into drawn card dicts) manually — the hunk was rejected and needs to be applied by hand.
---
### 2026-04-17 — Season refresh: update card pool to 270 minions + trinket system
**Files changed:** `CARDS.md`, `bg_card_pipeline.py`, `bg_card_definitions.json`
**What was done:** Scraped the new season's card pool from HearthstoneJSON API. Updated CARDS.md from 264 → 270 minions with all tier/stat/text changes (major new cards: Chromadrake cycle, Bounty Pirate package, Mrrglton Murloc duo, etc.; removed: all Volumizers, Anub'arak, Bird Buddy, Young Murk-Eye, Rampager, and ~30 others). Added Trinkets section (177 trinkets: 116 Lesser, 61 Greater). Fully replaced TIER_CARDS embedded dict in bg_card_pipeline.py. Updated MULTIPLIER_CARDS (removed young_murk-eye, added balinda_stonehearth/hot-air_surveyor/maelstrom_emergent) and AURA_CARDS (removed shore_marauder/lord_of_the_ruins/mechagnome_interpreter, added tidemistress_athissa/one-amalgam_tour_group). Extended fetch_hearthstone_json to also pull BATTLEGROUND_TRINKET cards and build a trinket list in the JSON output. Pipeline now outputs 270 cards with 0 API drift.
**Current state:** bg_card_definitions.json is freshly generated and synced with the live API (270 minions, 213 trinkets). CARDS.md is the authoritative per-card reference for the symbolic layer.
**Open questions / next steps:**
- Anomalies are out this season (confirmed via API — no BATTLEGROUND_ANOMALY cards in pool).
- Trinkets in HearthstoneJSON have no Lesser/Greater pool flag yet — currently filtered by cost (1-3 = Lesser, 4+ = Greater); verify against in-game trinket shop once live data is available.
- Update symbolic layer DEATHRATTLE_SPECS/AURA_SPECS/TRIGGER_SPECS for new cards (especially Chromadrake cycle, Floating Watcher, Kangor's Apprentice).
- CLAUDE.md Key Multiplier and Key Aura tables need updating to reflect new cards.
---
### 2026-04-17 — Notebook kernel metadata update (no code change)
**Files changed:** `explore.ipynb`
**What was done:** Opening explore.ipynb in a different Python environment (3.9.5 vs 3.12.12) caused Jupyter to update the kernelspec display name and version metadata. No cells or code were modified.
**Current state:** Notebook is functionally identical; only kernel metadata differs.
**Open questions / next steps:**
- No action needed. Kernel version mismatch won't affect execution if the required packages are installed.
---

---
### 2026-04-17 — Simulator mechanics pass: trinkets, battlecries, Spellcraft, aura wiring, combat fixes

**Files changed:** `symbolic/effect_handler.py`, `symbolic/shop_analyzer.py`, `symbolic/combat_sim.py`, `env/player_state.py`, `env/game_loop.py`, `env/trinket_handler.py` (new)

**What was done:** Comprehensive mechanics pass covering the five largest simulator gaps. Added 15+ missing battlecries (King Bagurgle, Mama/Papa Mrrglton, Kalecgos passive, Gem Smuggler, Sanguine Champion, Orc-estra Conductor, Highkeeper Ra, Draconic Warden, En-Djinn Blazer, River Skipper, Patient Scout, Sun-Bacon Relaxer). Implemented Spellcraft system: `on_buy()` grants spell tokens to hand for Spellcraft cards (Deep Blue Crooner, Reef Riffer, Private Chef, Tranquil Meditative); `_cast_spell()` applies effects when spells are played. Created full trinket system: new `TrinketHandler` with offer/select/decline lifecycle (rounds 4/8), `PlayerState` trinket fields, and game_loop integration. Wired `aura_dependency_score` from `BoardFeatures` into `ShopAnalyzer` synergy and scaling estimates. Added `magnetic_bonus_atk/hp` informational fields to `CombatMinion` and end-of-turn handlers for `skeletal_strafer` and `earthsong_shaman`.

**Current state:** Mechanical coverage increased from ~45% to ~70%. Trinket system is wired end-to-end (effect dispatch depends on `trinket_effect` dict in card_defs — effects will silently no-op until card_defs are enriched with that field). Spellcraft is functional as an approximation (spell granted on buy rather than at time of purchase).

**Open questions / next steps:**
- Enrich `bg_card_definitions.json` with `trinket_effect` dicts for each trinket (currently the handler will no-op all effects)
- ~36 battlecries still unimplemented (lower priority: Chromadrake spell-power cycle, Leyline Surfacer, complex discover chains)
- Kalecgos passive (`ps._kalecgos_active`) fires once per battlecry but isn't reset between rounds — add reset in round setup
- Spellcraft "this_combat" buffs use `perm_atk_bonus` as approximation; should use a separate combat-only buff channel
- Validate trinket offer timing against current BG patch (round 4/8 may have changed)
---

---
### 2026-04-18 — Trinket data population: trinket_effect dicts + full handler wiring

**Files changed:** `bg_card_definitions.json`, `env/trinket_handler.py`, `env/game_loop.py`, `train.py`

**What was done:** Populated all 213 trinkets in `bg_card_definitions.json` with `card_id` (slug + tier initial, e.g. `shadowy_elixir_l`) and a `trinket_effect` dict parsed from each card's text field. Effects are categorized into 21 types including fully-live ones (`stat_buff_all`, `stat_buff_tribe`, `stat_buff_low_tier`, `gold_gain`, `armor`, `level_cost_reduction`, `gold_per_round`, `max_gold_per_round`, `end_of_turn_buff_all/leftmost/tribe`, `start_of_combat_buff_all/tribe`) and deferred-but-labeled ones (`avenge`, `combat_trigger`, `complex`, `discover`, etc.). Extended `TrinketHandler` with `_tribe_match`/`_buff_minions` helpers, all new `_apply_on_equip` branches, and two new hooks: `apply_on_round_end` and `apply_on_combat_start`. Fixed `load_card_defs` in `train.py` to merge the trinkets list into the returned card_defs dict so `TrinketHandler` can find them by card_id. Wired `apply_on_round_end` at END_TURN and freeze, and `apply_on_combat_start` before `firestone.simulate` in `game_loop.py`.

**Current state:** Trinket system is fully end-to-end: 152 lesser and 61 greater trinkets in their respective pools, effects fire at the correct hooks. ~67 trinkets have live stat/economy effects; remaining 146 are labeled with descriptive types that log at debug level and no-op pending per-type handler implementation.

**Open questions / next steps:**
- Implement `combat_trigger` effects (Whenever/After hooks — requires tracking attack events and minion deaths during the combat sim)
- Implement `avenge` counters (track death count per combat, fire threshold callback)
- `start_of_combat` buff is applied to `attack`/`health` directly (not `perm_*`) — those buffs are lost after the sim snapshot; consider a separate `combat_atk_buff` field on MinionState
- Validate `card_id` slugs against actual Hearthstone card IDs if/when BG JSON source is re-fetched
- Kalecgos passive reset between rounds still unaddressed (carried over from previous session)
---

---
### 2026-04-18 — CONTEXT.md append method: bash heredoc to avoid full-file reads
**Files changed:** `CLAUDE.md`
**What was done:** Updated Step 1 of the end-of-session checklist to instruct Claude to append to CONTEXT.md via a Bash heredoc (`cat >>`) instead of using the Read+Edit tools. This prevents the entire CONTEXT.md from being loaded into context as the file grows large.
**Current state:** CLAUDE.md checklist now enforces append-only bash writes; hook script unchanged and working correctly.
**Open questions / next steps:**
- Monitor that future sessions correctly use the bash append method.
---

---
### 2026-04-18 — Trinket agent integration: observation + masks + scalar dims
**Files changed:** `agent/policy.py`, `env/game_loop.py`, `train.py`, `explore.ipynb`
**What was done:** Wired trinket state into the agent's perception. Added `trinket_offer_pending` guard to `build_type_mask` (BUY + END_TURN only during offer) and `build_pointer_mask` (shop slots 0-2 valid during offer). Updated `_get_observation` to display pending trinket offer cards in shop zone (mirroring discover_pending pattern) and to expose 2 new economy dims: `n_equipped_trinkets/2` and `trinket_offer_pending` flag. Expanded `SCALAR_DIM` from 98→100 and updated all three `scalar_dim=98` hardcodes in `train.py` to `100`. Refreshed notebook cells 43/44/47 to reflect the 100-dim layout and new SCALAR_LABELS (including economy + trinket labels).
**Current state:** Agent can now observe its trinkets and respond to trinket offer screens; the policy network, PPO training code, and notebook are all aligned on SCALAR_DIM=100.
**Open questions / next steps:** Pre-existing `TavernPool.draw` bug (string tier comparison) blocks full `BattlegroundsGame.reset()` integration tests — fix card_tier typing in `tavern_pool.py`; consider add `trinket_rarity` as a feature dim in the card encoder so the agent distinguishes lesser vs greater trinkets in the shop zone.
---

---
### 2026-04-18 — Add conda environment.yml
**Files changed:** `environment.yml`
**What was done:** Created a conda environment YAML to reproduce the Python 3.9.5 environment. Scanned all source files for imports and cross-referenced the installed pip list to include only the packages actually used by the codebase: torch, numpy, hearthstone, hslog, requests, tqdm, and rich.
**Current state:** environment.yml is ready at the repo root. Recreate with `conda env create -f environment.yml && conda activate bg-agent`.
**Open questions / next steps:**
- CUDA variant of torch may be needed for GPU training (current yml installs CPU torch 1.11.0).
- hslog and hearthstone are version-pinned; verify they remain compatible after any HS patch update.
---

---
### 2026-08-29 — Season refresh: card pool update to 275 minions + 258 trinkets
**Files changed:** `bg_card_pipeline.py`, `bg_card_definitions.json`, `CARDS.md`, `CLAUDE.md`
**What was done:** Fetched the live HearthstoneJSON API and rebuilt the card pool from scratch (not merged with stale data, since naive merging doesn't drop retired cards). Pool changed 270 → 275 minions: 108 added, 103 removed, 59 changed stats/text. Rewrote `_clean_api_text` to correctly join HearthstoneJSON's line-wrapped `<b>` keyword tags with periods (e.g. "Magnetic. Divine Shield. Taunt.") while preserving mid-sentence line-wraps and hyphenated word-wraps without corrupting them — verified against the full 275-card set with zero remaining artifacts. Added detection for a brand-new BG mechanic, **Activate (N)** — a repeatable once-per-turn minion ability costing N Gold, seen on 16 minions this patch — via `detect_activate_cost()`, a new `trigger_type` value, and an `activate` keyword flag; it is not yet wired into any handler. Reviewed and pruned `MULTIPLIER_CARDS` (dropped `hot-air_surveyor`, `maelstrom_emergent` — both retired) and `AURA_CARDS` (dropped 7 retired entries: `twilight_watcher`, `hardy_orca`, `iridescent_skyblazer`, `junk_jouster`, `geomagus_roogug`, `tidemistress_athissa`, `one-amalgam_tour_group`) — confirmed the remaining/removed entries by checking pool membership, and confirmed the regex heuristic in `detect_is_aura` already auto-catches nearly all "whenever...give/gain...friendly" phrasings so the explicit set only needs cards with non-matching phrasing. Regenerated `bg_card_definitions.json` and `CARDS.md` fully from the fresh pool. Updated CLAUDE.md's Key Multiplier table (dropped long-retired Young Murk-Eye, added Balinda Stonehearth) and Key Aura table (all 6 listed cards were retired; replaced with 6 currently-live examples), and added "Activate (N)" to the Mechanics Glossary.
**Current state:** `bg_card_pipeline.py --fetch` now round-trips cleanly (0 new cards merged, confirming embedded TIER_CARDS exactly matches the live API). `bg_card_definitions.json` and `CARDS.md` are in sync at 275 minions / 258 trinkets. `symbolic/board_computer.py` and `symbolic/combat_sim.py` still contain dead lookups for retired card_ids (`shore_marauder`, `lord_of_the_ruins`, `hardy_orca`) — harmless (those ids will never match a live card again) but not cleaned up.
**Open questions / next steps:**
- Implement the new Activate (N) mechanic end-to-end: needs an action-space entry (or extension of HERO_POWER-style handling), a per-turn-once usage gate, gold cost payment, and per-card effect dispatch in `effect_handler.py`/`game_loop.py`. Currently these 16 minions' abilities are inert.
- Update `DEATHRATTLE_SPECS`/`AURA_SPECS`/`TRIGGER_SPECS` in the symbolic layer for the 108 newly added cards (mechanical coverage will have dropped from the ~70% reported last season since a third of the pool turned over).
- Remove the dead retired-card branches in `board_computer.py`/`combat_sim.py` (`shore_marauder`, `lord_of_the_ruins`, `hardy_orca`) during the next symbolic-layer pass.
- Trinket count also changed (213 → 258) — trinket_effect dicts populated last season for the old 213 were not re-verified against the new set; re-run that population pass for trinkets that are new this patch.
---

---
### 2026-08-29 — Activate (N) mechanic, symbolic-layer cleanup, and trinket data rebuild
**Files changed:** `agent/policy.py`, `env/game_loop.py`, `env/player_state.py`, `env/tavern_pool.py`, `env/trinket_handler.py`, `symbolic/board_computer.py`, `symbolic/combat_sim.py`, `symbolic/effect_handler.py`, `bg_card_pipeline.py`, `bg_card_definitions.json`, `CARDS.md`
**What was done:** Follow-up on the four items flagged after the season card-pool refresh.
1. **Activate (N) mechanic** implemented end-to-end: added action type 8 ("activate") to the policy's action space (`N_ACTION_TYPES` 8→9, reusing the board-zone pointer slot like SELL), a `MinionState.activate_cost`/`activated_this_turn` pair (reset each round alongside `hero_power_used`), a `type_action == 8` branch in `game_loop._step` that pays the Gold cost and dispatches `EffectHandler.on_activate`, and mask logic in `build_type_mask`/`build_pointer_mask` gating on affordability and once-per-turn use. Implemented 13 of the 16 Activate minions functionally (Suspicious Prisonguard, Decoy Conjurer, Breakout Mastermind, Hired Mount, Private Investigator, Dead Bellringer, Drone Duplicator, Living Prison, Soulkeeping Jailer, Deft Deserter, Tyrael, plus approximated Clever Castaway/Fruit Vendor); Lurking Lionfish, Cagey Conjurer, and Sky-hatch Runaway are documented no-ops since they need infra this codebase doesn't have yet (per-attacker combat targeting, a generic castable Tavern-spell pool, and triggering Rally outside combat, respectively). `load_bc_v2_weights`'s BC warm-start path is unaffected — it already no-ops for the current d_model=256 architecture before touching type_head shapes.
2. **Symbolic layer cleanup**: removed dead branches for cards retired this season (`twilight_watcher`, `hardy_orca`, `iridescent_skyblazer`, `trigore`, `lord_of_the_ruins`, `shore_marauder`, `prized_promo_drake`) from `combat_sim.py` and `board_computer.py`'s `_KNOWN_AURA_CARDS`; added a real combat-time implementation for Cage Gnawer (on-attack Beast aura, parallel to the existing Roaring Recruiter pattern) since it's a genuine combat trigger. The other 4 cards in the current `AURA_CARDS` set (Timecap'n Hooktail, Charging Czarina, Gunpowder Courier, Torrential Ruiner) are buy-phase spell-cast/gold-spend triggers, not combat triggers — they remain estimate-only in `_KNOWN_AURA_CARDS` (which, discovered along the way, is actually dead code already — `_compute_auras` never reads it, using a board-power-delta instead). Did **not** attempt full `DEATHRATTLE_SPECS`/battlecry coverage for the other ~100 new minions from the season refresh — that's a much larger, open-ended task matching the scale of the "15+ missing battlecries" and "mechanics pass" sessions from 2026-04-17/18, not a single follow-up item.
3. **Trinket data regression fixed**: last session's `--fetch` regenerated `bg_card_definitions.json`'s trinkets via the pipeline's bare `build_trinket_list` (name/cost/tier/text only), silently destroying the `card_id`/`trinket_effect` enrichment the 2026-04-18 session had populated by hand — `train.load_card_defs` merges trinkets into `card_defs` by `card_id`, so with that field missing, **zero** of the 258 trinkets were reachable and `TrinketHandler`'s pools were empty. Fixed by making the enrichment a durable part of the pipeline: `build_trinket_list` now generates a collision-free `card_id` (`slug_l`/`slug_g`, discovered via the API's `spellSchool` field — LESSER_TRINKET/GREATER_TRINKET — which also turned out to be the *correct* tier signal; the old cost<=3 heuristic mis-tiered several trinkets, e.g. Bird Feeder's Greater variant costs 2 Gold) and a `trinket_effect` dict via a new rule-based `parse_trinket_effect()` classifier (ordered regex rules covering the ~12 mechanically-live effect types `TrinketHandler` already knows how to execute, falling back to a labeled `{"type": "complex"}` no-op — 21 of 258 trinkets classify as live, matching TrinketHandler's existing implemented-type coverage; the rest are correctly deferred, not misclassified, per spot-check).
4. **Found and fixed a second, independent pre-existing bug while integration-testing the above**: `TrinketHandler.__init__` filtered its lesser/greater pools by a `trinket_rarity` field that no code anywhere ever sets (the pipeline has always used `tier`) — trinket pools were empty even before yesterday's regression. Fixed to read `tier` (guarded with `isinstance(..., str)` since minions use an int tier 1-7 and trinkets use "lesser"/"greater"). Also fixed the CONTEXT.md-flagged "`TavernPool.draw` string tier comparison" bug at its root: `TavernPool._build_pool` was iterating over *all* of `card_defs` (minions + merged-in trinkets) without filtering, so once trinkets became reachable via fix #3 they entered the shop-draw pool with a string tier and crashed `draw()`'s `card_tier < tier` int comparison. Fixed by skipping non-int-tier entries in `_build_pool`.
**Current state:** All modules import cleanly; `build_type_mask`/`build_pointer_mask` verified directly for the new activate type; `EffectHandler.on_activate` verified directly for several cards (Suspicious Prisonguard, Tyrael, Living Prison); `TrinketHandler` verified end-to-end (129 lesser / 129 greater pool, offer/select works, effect dict attached). Three full 8-player, 25-round games (`BattlegroundsGame.run_game(agents=None)`, random legal actions) completed without error across 3 seeds, exercising the new activate action, trinket offers, and the symbolic-layer changes together.
**Open questions / next steps:**
- The 3 undone Activate cards (Lurking Lionfish, Cagey Conjurer, Sky-hatch Runaway) need, respectively: per-attacker combat targeting outside the normal attack-resolution loop, a generic Tavern-spell-cast-from-pool system, and a way to fire a Rally effect pre-combat — none of which exist yet.
- `DEATHRATTLE_SPECS`/`AURA_SPECS`/`TRIGGER_SPECS`-equivalent coverage for the ~100 other new minions from the season refresh is still open; mechanical coverage has dropped from the ~70% reported after the 2026-04-18 mechanics pass since roughly a third of the pool turned over.
- Only ~8% of trinkets (21/258) have a live mechanical effect; the rest are correctly labeled (`avenge`, `combat_trigger`, `spellcraft`, `discover`, `round_start_effect`, `complex`) but inert. Implementing `combat_trigger`/`avenge` handlers (tracking attack/death events during the combat sim) was flagged as a to-do back in the 2026-04-17 mechanics-pass session too and remains the highest-value next slice given how many trinkets fall in those buckets.
- `_KNOWN_AURA_CARDS` in `board_computer.py` is unused dead code (`_compute_auras` never reads it) — either wire it in or delete it next time it's touched.
- Never verified whether the `[x]`/`$92`-style numeric placeholders that show up garbled in a handful of trinket texts (e.g. "Get a random 92") are fixable from the HearthstoneJSON API data at all, or need per-card manual overrides — cosmetic only, left as-is.
---

---
### 2026-08-29 — Notebook audit: fix crash from last session's Activate action-space change
**Files changed:** `explore.ipynb`, `agent/ppo.py`
**What was done:** Audited `explore.ipynb` against the codebase after the previous session added action type 8 ("activate", `N_ACTION_TYPES` 8→9). Found and fixed a real crash: the notebook's `ACTION_NAMES` lists (training-loop cell and the `RecordingAgent`/trace-inspection cells) were still hardcoded to 8 entries, so `ACTION_NAMES[type_idx]` would `IndexError` the first time the policy sampled ACTIVATE — confirmed via a real (non-random-action) PPO training smoke test that ACTIVATE does get sampled and trained on in normal play (49 times in a 16-game run). Fixed both `ACTION_NAMES` lists and the 8-color plotting palette (needed a 9th). Also found and fixed a subtler correctness bug while auditing the checkpoint-resume cell: `PPOTrainer.load_checkpoint` silently no-ops (by design) when a checkpoint's architecture doesn't match — which is exactly what happens to any checkpoint saved before this action-space change, since its `type_head` is `[8, 256]` against the current `[9, 256]`. The notebook's resume cell had no way to detect this and would print `"Resumed: steps=0, updates=0"` for what was actually a silent fresh start. Made `load_checkpoint` return a bool (loaded vs skipped) and updated the notebook to route a skip through the same delete-and-reinit path already used for other checkpoint failures. Also updated cell 38's architecture-description markdown, which was already stale before this session (said 94-dim scalar context / 8 action types; current is 100-dim / 9 types), and two `[8]`-bool docstring comments in `agent/ppo.py`. Ran three verification passes: a 4-game and a 16-game real PPO training loop (multiprocessing workers, real backprop, not random actions) via `train._train_parallel` — both completed cleanly with reasonable-looking entropy (~1.7-1.9, no collapse) and finite losses.
**Current state:** `explore.ipynb`'s training and trace-inspection cells (46-56) are consistent with the current 9-action-type, 100-dim-scalar, 275-card-pool codebase. Did not touch the reward-shaping constants (`BOARD_SHAPE_ALPHA/GAMMA`, penalty weights, `FINAL_PLACEMENT_REWARD`) — those were extensively and deliberately tuned across 5 commits on 2026-04-19 and nothing in this session's testing surfaced a new problem with them; revisiting them would need real multi-thousand-game training data, not available in this (CPU-only, no GPU) environment.
**Open questions / next steps:**
- **Any existing `bg_agent_ppo.pt` checkpoint from before this action-space change (e.g. on a GPU training box) is architecturally incompatible and will now be discarded on next load** — training will restart from scratch, not resume. This is expected/handled gracefully, not a bug, but worth knowing before the next training run.
- The reward-shaping system has had a lot of very recent firefighting (board-fill exploit, empty-board credit assignment, phi_board cross-turn baseline exploit — all 2026-04-19). Whether it currently produces a *good* strategy rather than just a non-degenerate one is unverified — needs an actual multi-thousand-game run (the notebook's `N_GAMES=5000` training cell) plus a look at the now-fixed `RecordingAgent` trace cells (51-56) to see what strategy actually emerges.
- Cell 42's hand-drawn architecture diagram (matplotlib boxes) is stale on dimensions unrelated to this session's change (shows scalar context [38] and a 3-layer/128-dim Transformer; actual is [100] and 4-layer/256-dim) — left as-is since fixing it means redrawing box/arrow coordinates, not a text edit, and it doesn't affect training.
---

---
### 2026-08-29 — Create the bg-agent conda environment + Jupyter kernel
**Files changed:** `environment.yml`
**What was done:** No `bg-agent` conda environment existed locally yet (only `base` and an unrelated `mtdiff` env) despite `environment.yml` documenting one since 2026-04-18. Building it surfaced two real gaps in that file: it was missing `matplotlib` and `ipykernel` entirely, even though `explore.ipynb` depends on both (matplotlib for every plot cell, ipykernel to be selectable as a Jupyter kernel at all) — the 04-18 session's "scanned all source files for imports" approach evidently didn't scan the notebook. Added both. Then hit a real incompatibility building on this machine: `torch==1.11.0`'s prebuilt CPU wheel fails to import here (`libtorch_cpu.so: cannot enable executable stack as shared object requires: Invalid argument`) — a known class of problem with old (2022-era) prebuilt libtorch binaries on newer hardened kernels; it's not fixable by config, only by using a newer torch build. Bumped `torch` 1.11.0→2.2.2 and `numpy` 1.21.6→1.26.4 (kept `python=3.9.5` — that pin was fine). This also resolves the standing discrepancy between `environment.yml` (old exact pins) and `requirements.txt` (which already asked for `torch>=2.0.0`/`numpy>=1.24.0`) — the two files were drifted apart; `environment.yml` now points the same direction as `requirements.txt`. Created the env, registered it as a Jupyter kernel (`python -m ipykernel install --user --name bg-agent --display-name "Python (bg-agent)"`), and verified: all pinned deps import, every project module (`agent.policy`, `agent.ppo`, `env.game_loop`, `symbolic.*`, `train`, `bg_card_pipeline`, `parse_bg`) imports cleanly under the env's actual Python 3.9.5 (not just the sandbox's system Python 3.14 used for earlier smoke tests this week), and a real PPO training run (`train._train_parallel`, multiprocessing workers, real backprop) completes without error inside this env.
**Current state:** `bg-agent` conda env exists locally with a working torch/numpy/matplotlib stack and is registered as the "Python (bg-agent)" Jupyter kernel — `explore.ipynb` can be pointed at it (Kernel → Change Kernel in Jupyter/VS Code) and will have every import the notebook needs. `environment.yml` is the accurate, working spec (recreate anytime with `conda env create -f environment.yml`).
**Open questions / next steps:**
- This local env is CPU-only and is for notebook exploration; it is unrelated to the Dockerfile/vast.ai GPU training setup, which should keep using a CUDA-matched torch build for actual training runs — don't conflate the two.
- `torch==2.2.2` pulled in NVIDIA CUDA wheel dependencies (`nvidia-cublas-cu12` etc.) even for local CPU-only use, since PyPI's default `torch` package bundles them; harmless but adds noticeable install size/time. A CPU-only index (`--index-url https://download.pytorch.org/whl/cpu`) would avoid that if install size becomes a concern.
- `requirements.txt` and `environment.yml` now agree in spirit (both want modern torch/numpy) but are still two separate files that could drift again — consider consolidating to one source of truth next time either is touched.
---

---
### 2026-08-29 — Add missing pandas dependency to environment.yml
**Files changed:** `environment.yml`
**What was done:** User selected the newly-created `bg-agent` kernel in `explore.ipynb` and hit `ModuleNotFoundError: No module named 'pandas'`. My env-creation session had only checked the training-loop cells (46-56) for imports, missing that the dataset-exploration cells (0-24) use `pandas` for DataFrame-based board/hero stats. A full scan of every code cell's import statements confirmed pandas was the only gap. Added `pandas==2.2.3` to `environment.yml` and installed it directly into the existing env (`pip install pandas==2.2.3` inside `bg-agent`) rather than rebuilding from scratch.
**Current state:** `bg-agent` env now has every package `explore.ipynb` imports across all 57 cells, verified by regex-scanning the full notebook JSON for import statements rather than spot-checking cells.
**Open questions / next steps:** None outstanding for the environment itself.
---

---
### 2026-08-29 — Ran the full notebook end-to-end in bg-agent env; fixed a real syntax error, diagnosed the rest
**Files changed:** `explore.ipynb`
**What was done:** User reported a `KeyError` running `explore.ipynb` and asked me to actually execute the notebook (not just read it) using the new `bg-agent` kernel, fix it, and check for further errors. Installed `nbconvert`/`nbclient` into the env and ran the notebook headlessly (`jupyter nbconvert --to notebook --execute`) with `N_GAMES` temporarily overridden to 6 in a throwaway copy (the real `N_GAMES=5000` training cell can't finish in a test pass — nbconvert's per-cell timeout only fires on true silence, and this cell keeps printing progress, so the very first attempt sat unresponsive until manually killed; never left that override in the committed file). First test run was executed from a copy placed in `/tmp/.../scratchpad/`, which produced a wave of `ModuleNotFoundError: No module named 'agent'/'symbolic'` — a false alarm caused by testing from the wrong directory: several cells rely on `sys.path.insert(0, ".")` or the kernel's default cwd, both of which only resolve correctly when the notebook actually lives in the repo root. Re-ran from a copy placed in the repo root itself, which is representative of how the user actually runs it, and got a much smaller, real error set. Found and fixed one genuine bug: cell 44 (the "100-Dim Scalar Context Vector" heatmap) had an unescaped literal newline inside a single-quoted Python string (`ax.set_title('...\nSampled Rounds\n(...)')` with a real newline instead of `\n`), which is a `SyntaxError` — likely introduced when that cell's title was last edited for the 98→100 scalar-dim change and never actually re-run since. The other 15 errors (cells 3, 4, 7, 8, 10, 12, 16, 20, 22, 24, 26, 27, 31, 41, 44-after-syntax-fix) all trace to one non-code cause: this environment has no `data/Hearthstone_*.json` (the dataset `collect_dataset.py` produces from captured matches), so `games = []`, and every dataset-exploration/BC-v2 cell from cell 3 through the "PPO Policy Network" section assumes at least one real game and fails a few cells downstream of the actual problem with a confusing `KeyError`/`NameError`/`IndexError` rather than a clear message at the source. Added a loud, actionable warning right where `games` loads (cell 1) instead of leaving it a silent "Loaded 0 games" — it now tells the reader exactly what's missing and that the PPO training section further down doesn't need this dataset at all.
**Current state:** Verified via a full headless execution (with the small-N_GAMES override, repo-root copy, then deleted): cells 45-56 — the actual PPO training loop, checkpoint resume, and `RecordingAgent` trace/inspection cells — run with **zero errors**, covering the real fix target ("an agent that can actually learn a feasible strategy"). Cells 0-44 (real-game dataset exploration + legacy BC v2 pretraining) will run cleanly too, but only once `data/Hearthstone_*.json` files exist; that's a data-availability gap, not a notebook bug, and is now clearly signposted at the point it first matters instead of surfacing as a cryptic KeyError several cells later.
**Open questions / next steps:**
- If the user has real captured-match data elsewhere (a different `data/` location, or files not yet copied into this checkout), pointing `DATA_DIR` at it would let cells 3-44 be verified for real rather than just reasoned about from the "0 games" case.
- No further code-level notebook bugs are known at this point across either the dataset-exploration or PPO sections.
---

---
### 2026-08-30 — Diagnosed and fixed why no Battlegrounds games were being captured
**Files changed:** none in the repo (`data/` is gitignored; the actual fix is a file outside the repo, on the Hearthstone install itself — see below)
**What was done:** User asked to find the game-extraction mechanism and make sure it captures the latest games. Confirmed `collect_dataset.py` + `parse_bg.py` (using `hslog.LogParser`) are intact and working — this session runs in WSL, and the real Hearthstone install is reachable at `/mnt/c/Program Files (x86)/Hearthstone/`. Three recent session folders exist under its `Logs/` dir (2026-08-28, 08-29, and 08-30 — today), but **none contain a `Power.log`** — only `Hearthstone.log`, `GameNetLogger.log`, etc. `parse_bg.py` reads `Power.log` specifically. Root cause: no `log.config` file existed in the Hearthstone install directory, so the client was never told to emit Power-level logging at all — this has apparently been the case since at least 2026-08-28 (all three recent sessions are affected), possibly since whatever produced the old March 2026 dataset backup found at `/mnt/c/Users/gebruiker/Desktop/back-up/bg-dataset/data/` (14+ files, not yet reviewed or imported into this repo). Created `/mnt/c/Program Files (x86)/Hearthstone/log.config` enabling `[Power]` and `[Bob]` file logging (LogLevel=1, FilePrinting=true) — the minimal set `parse_bg.py` needs (Power is required; Bob carries supplementary Battlegrounds tavern/lobby data). Ran `collect_dataset.py --logs-dir "/mnt/c/Program Files (x86)/Hearthstone/Logs" --output-dir data` to confirm the pipeline itself runs cleanly end-to-end: it correctly reports 0 games (all 3 sessions skipped, no Power.log present) rather than erroring.
**Current state:** The extraction mechanism is confirmed functional but currently has nothing to extract. `log.config` is now in place for future sessions, but **Blizzard only reads log.config at client startup** — it does not apply retroactively to the 3 existing sessions (which have no Power.log and never will) or to a currently-running client. The user needs to fully quit and relaunch Hearthstone once; only Battlegrounds games played *after* that restart will produce a `Power.log` that `collect_dataset.py` can parse.
**Open questions / next steps:**
- After the user next restarts Hearthstone and plays a BG game or two, re-run `collect_dataset.py --logs-dir "/mnt/c/Program Files (x86)/Hearthstone/Logs"` (from WSL; the script's `DEFAULT_LOGS_DIR` is a raw `C:/...` path that only resolves as-is on native Windows Python, not WSL bash — always pass `--logs-dir` explicitly here).
- There's an old, unreviewed dataset backup at `/mnt/c/Users/gebruiker/Desktop/back-up/bg-dataset/data/` (Hearthstone_2026_03_13 through at least 2026_04_01, 14+ session files) from a previous project location/setup. Not copied into this repo's `data/` — the user didn't ask for old data, just "the latest games," and mixing in months-old sessions (predating this season's card-pool refresh) without being asked seemed presumptuous. Worth asking the user whether they want it merged in for a larger BC-v2 training set.
---

---
### 2026-08-30 — Add GreedyPlayAgent scripted opponent; sync CLAUDE.md reward-shaping docs
**Files changed:** `train.py`, `CLAUDE.md`
**What was done:** Added `GreedyPlayAgent` (train.py) — a naive scripted opponent that places every hand card immediately, buys the first affordable shop minion, and only ever sells to make room for a strictly higher-tier minion offered in the shop (compares shop tiers against the weakest-tier board minion when the board is full). Wired it into the training pipeline as opponent slot type `"greedy"`: added `N_GREEDY_SLOTS=1`, handled the `"greedy"` entry in `_worker_run_game`, and updated `_train_parallel`'s per-batch opponent composition to `4 policy snapshots + 1 heuristic + 1 greedy` (previously 5 + 1 heuristic). Also rewrote the `## Reward Shaping` section of `CLAUDE.md`, which still documented an old/simplified reward formula (flat `round_reward` + no hand/gold/board-shape terms) that had drifted from the actual implementation in `env/game_loop.py` (`compute_round_reward`, `_end_of_turn_reward`, potential-based `_apply_board_shape`, `FINAL_PLACEMENT_REWARD` timing) — the section now points to `game_loop.py` as source of truth and summarizes the real four-component reward.
**Current state:** Training opponent pool is now `HeuristicAgent` (leveling anchor) + `GreedyPlayAgent` (naive buy/play anchor) + 4 sampled historical snapshots, filling the 6 non-training seats each game. `CLAUDE.md`'s reward shaping docs match `game_loop.py`.
**Open questions / next steps:**
- Run a short training batch to confirm `GreedyPlayAgent` picklability/sequential-path behavior under `ProcessPoolExecutor` (only smoke-tested `get_action` directly against fabricated `PlayerState`-like objects, not through the full worker/game loop).
- Consider whether `GreedyPlayAgent` should also be usable as a `StaticAgent`-style eval opponent outside of training (e.g. for a fixed benchmark ladder), not just as a training-pool anchor.
---

---
### 2026-08-30 (2) — Fix asymmetric clipping that biased PPO toward over-selling
**Files changed:** `env/game_loop.py`
**What was done:** User reported the trained agent sells noticeably often. Traced it to `_apply_board_shape`'s two call sites in `step_shopping`: SELL added the raw potential-based shaped reward `α·(γ·Φ(s')−Φ(s))` uncapped (despite a `# SELL: keep negative` comment that the code never enforced), while PLACE clipped the same term to `max(0.0, ...)`. Φ(s) is a 30-trial Monte Carlo win-probability estimate, so it's noisy — PLACE could never register a bad outcome (worst case 0) while SELL's noise landed on both sides, occasionally paying out a positive reward for selling purely from simulation variance. This also broke the policy-invariance guarantee of potential-based shaping (Ng et al. 1999), which only holds when the shaping term is applied symmetrically — the clip was quietly injecting a hand-made bias rather than just speeding up learning. Removed the `max(0.0, ...)` clip from PLACE so both actions now receive the raw, unclipped `_apply_board_shape` result, restoring the theoretical guarantee that shaping doesn't change the optimal policy, only the speed of learning it. Also updated the now-inaccurate `# SELL: keep negative` comment.
**Current state:** Both PLACE and SELL apply identical, unclipped potential-based board shaping. Verified via three full 8-player, 25-round random-action games (seeds 1-3) completing without error post-change; did not run a real training pass (CPU-only, no GPU in this environment) to confirm the sell-frequency behavior actually drops — that needs a real multi-thousand-game run.
**Open questions / next steps:**
- Should be verified with an actual PPO training run (the notebook's `N_GAMES` training cell) and a look at `RecordingAgent` trace cells to confirm sell frequency actually decreases, not just that the code no longer contains the asymmetry.
- The empty-board penalty (-0.30) and hand penalty (-0.08/card) were left untouched — if over-selling persists after this fix, the hand penalty forcing cards out of hand (which requires selling first once the board is full) is the next suspect to look at.
- `_board_win_prob`'s reliance on a potentially stale `next_opponent_id` snapshot (set once per round, can lag a turn or two behind the opponent's real board) is a separate source of Φ noise/bias not addressed here.
---

---
### 2026-08-30 (2) — Rebalance training opponent-pool slot mix
**Files changed:** `train.py`
**What was done:** Changed the fixed opponent-slot counts from 1 `HeuristicAgent` + 1 `GreedyPlayAgent` + 4 sampled historical snapshots to 2 + 2 + 2, as a starting point for the two scripted anchors ([[greedy-play-agent]]) now that both exist. `N_HEURISTIC_SLOTS` and `N_GREEDY_SLOTS` are each `2`; `n_policy_slots` derives from the remainder (`N_OPP_SLOTS - N_HEURISTIC_SLOTS - N_GREEDY_SLOTS = 2`), so no other code needed to change.
**Current state:** Each game's 6 non-training seats are now 2 sampled historical snapshots + 2 `HeuristicAgent` + 2 `GreedyPlayAgent`.
**Open questions / next steps:**
- This mix is a starting point, not a tuned value — revisit once training data shows whether 2 scripted anchors each is too much (crowds out historical self-play diversity) or the right balance.
---

---
### 2026-08-30 (3) — Track per-agent-type win rate over training
**Files changed:** `train.py`, `agent_stats.py` (new), `explore.ipynb`
**What was done:** Added agent-identity tagging through the opponent-pool pipeline so every finished game records which agent occupied each seat: `train_current` (live policy), `heuristic`, `greedy`, or `snapshot_uN`/`milestone_uN` (a frozen historical policy tagged by the PPO update it was frozen at). `SnapshotPool.add`/`sample_n` now carry `(state_dict, tag)` pairs instead of bare state dicts. `_worker_run_game` builds a `pid -> label` map and returns it in the game summary; `_train_parallel` appends one JSONL row per player per game to `data/agent_stats.jsonl` (via new `_append_agent_stats`), including `total_steps` as the restart-safe "training progress" x-axis (the `game` field resets every call so it's not safe to plot against). The single-process (`--workers 1`) path logs the same way with everyone labeled `train_current`. Added `agent_stats.py` with `load`/`summary_table`/`rolling_winrate`/`plot_family_winrates` to aggregate the log by agent family and plot rolling win rate vs. the 1/8 baseline. Wired a matching cell into `explore.ipynb` right after the existing PPO-training-loss plot.
**Current state:** Verified end-to-end with a short `--workers 2 --no-firestone` smoke run — labels (`train_current`/`heuristic`/`greedy`) and rows land correctly in the JSONL log, and `agent_stats.py`'s summary table / rolling plot both render against it. Snapshot-tag dispatch (`(sd, tag)` vs the `"heuristic"`/`"greedy"` sentinels vs `None`) was verified in isolation since the smoke run was too short to populate the snapshot pool (needs 10+ PPO updates).
**Open questions / next steps:**
- No automated test covers `_worker_run_game`'s label assignment directly (only manual smoke-tested) — consider a unit test if this logic gets touched again.
- `agent_stats.py`'s `_family()` collapses all individual snapshot tags into two buckets (`rolling_snapshot`/`milestone_snapshot`) for chart readability; the raw per-tag `label` column is still in the DataFrame if finer-grained drill-down (e.g. win rate of one specific milestone) is ever needed.
- `data/agent_stats.jsonl` grows unbounded across sessions; no rotation/pruning implemented yet.
---

---
### 2026-08-30 (3) — Annealed board-stats potential to give a low-variance early-training signal
**Files changed:** `env/game_loop.py`, `train.py`
**What was done:** Investigated why the trained checkpoint sells so often, going beyond the earlier clipping-asymmetry fix. Ran the actual `bg_agent_ppo.pt` checkpoint through 24 headless instrumented games (real policy, not random actions) and found the problem is more severe than "sells too often": average board size at END_TURN across the whole game is only 1.81 minions (dropping to ~1.25 by rounds 7-8), and the SELL:PLACE ratio approaches ~1:1 in the mid-game — the board never really accumulates. Ruled out "forced by a full hand/board" as the cause: only 1.2% of sells happen with a full board; the other 98.8% are discretionary (avg board size at sell time 3.21, avg hand size 1.02 — plenty of room). Root-cause read: the dense, certain, same-turn penalties (hand `-0.08`/card, gold `-0.05*gold`) are easy for PPO to learn quickly, while "a bigger board wins combat" is a delayed, 8-player-self-play-noisy signal on top of the already-noisy 30-trial MC win-probability estimate used for board-shape reward — early in training (this checkpoint is ~20% through its budget, reward still ~-5, value loss still visibly unstable) the easy signal dominates.

Discussed three options to reduce that noise: (1) blend a deterministic board-stats potential into the shaping term, weighted small and annealed to 0 over training: (2) just increase `BOARD_SHAPE_TRIALS` for a less noisy but still-synergy-aware win-prob estimate; (3) do nothing and wait for more training. User explicitly flagged the risk that a stats-based term could overcorrect into a *new* exploit later in training (buy-the-biggest-numbers, ignoring synergy/tribal package value) — the same failure mode as the original 2026-04-13 `+0.10*board_size` board-fill bug, just one level more sophisticated. Implemented option (1) specifically *because* of that risk: `BattlegroundsGame` gained a `shape_stats_weight` constructor param (default `0.0`, fully backward compatible) and two new methods — `_board_stats_potential` (a bounded `[0,1)` saturating function of total effective board stats via the symbolic layer's existing `_board_power` helper, reused rather than reimplemented) and `_board_potential` (blends it with the existing `_board_win_prob` MC estimate at weight `shape_stats_weight`, falling back to pure win-probability at weight 0). Both `_apply_board_shape` and the per-round `ps.phi_board` reset now go through `_board_potential` so the blend is consistent everywhere Φ is computed (no phantom jump at round start). `train.py` computes a per-batch anneal weight from `ppo_trainer.total_steps` (`_stats_weight_for_step`: linear decay from `BOARD_SHAPE_STATS_WEIGHT_INIT=0.25` to 0 over `BOARD_SHAPE_STATS_ANNEAL_STEPS=250_000` steps — both explicitly flagged as untuned initial guesses) and threads it through the worker task tuple (`current_sd, opp_sds, seed, stats_weight`) into `_worker_run_game`'s `BattlegroundsGame(...)` construction. `BOARD_SHAPE_STATS_SATURATION=30.0` sets where the stats potential reads 0.5 (also untuned).

User also floated annealing proxy-reward terms in general as a pattern worth reusing later, not just for this one term — noted here but not generalized into a framework now (kept to a small `_stats_weight_for_step` helper, not a registry), per the usual scope-to-the-task approach.
**Current state:** All changes verified: module imports cleanly; `BattlegroundsGame` runs full random-action games at `shape_stats_weight` 0.0/0.25/1.0 without error; the real parallel-training path (`_train_parallel`, multiprocessing workers, real PPO updates, not random actions) runs cleanly end-to-end with the new task-tuple plumbing. Not yet validated against real multi-thousand-game training data — we don't yet know whether this actually reduces the near-empty-board behavior or by how much; that requires an actual training run, which hasn't happened yet in this session (see open questions).
**Open questions / next steps:**
- Needs a real training run (ideally restarted from scratch, not resuming `bg_agent_ppo.pt`, to isolate this change's effect) with `avg board size at END_TURN` and `SELL:PLACE ratio` tracked per-round — the same instrumentation used for this session's diagnosis — as the actual success metric, not just the aggregate reward curve.
- `BOARD_SHAPE_STATS_WEIGHT_INIT`, `BOARD_SHAPE_STATS_ANNEAL_STEPS`, and `BOARD_SHAPE_STATS_SATURATION` are all first-guess constants with no empirical basis yet — expect to retune once real data comes in, same as the other board-shape constants that went through several rounds of exploit-driven fixes in April.
- If board size still doesn't recover, `BOARD_SHAPE_TRIALS` (currently 30, ~0.5ms/trial) is the next lever to try — a less noisy pure win-probability estimate, with no new proxy-metric gaming surface at all.
- The per-instrumented-run finding (board size ~1.8, sell≈place mid-game, 98.8% of sells happen on a non-full board) should be re-checked against a fresh checkpoint once training resumes — the checkpoint used for this session's diagnosis has ambiguous provenance (a kernel-restart/interrupt investigation earlier this session couldn't fully establish whether it trained under the pre- or post-clipping-fix code).
---

---
### 2026-08-31 — Fix real GAE bug: advantages bootstrapped across unrelated trajectories
**Files changed:** `agent/ppo.py`, `train.py`
**What was done:** After the annealed board-stats-potential fix (previous session) produced a fully-completed 5000-game fresh-training run with board size *worse* than before (0.97 avg vs. 1.81 baseline) and a reward curve completely flat at ~-5 for the entire run with `value_loss` oscillating in the same 0.1-0.3 band across all 909 PPO updates — never trending down — dug into why the value function wasn't converging at all, rather than assuming it just needed more time or another reward-shaping tweak. Found a real bug in `RolloutBuffer.compute_advantages` ([agent/ppo.py](agent/ppo.py)): GAE was computed by walking the entire rollout buffer as if it were one continuous trajectory (`next_value = self.transitions[t+1].value`), but the buffer actually contains **multiple players' and multiple games' transitions concatenated together** with no `done=True` marker between them. Two concrete sources: (1) `N_TRAIN_PLAYERS=2` share one `ppo_trainer.buffer` per game, and the *sequential* shopping-phase loop (`game_loop.py`'s `for ps in alive_players: ...`) appends one player's entire turn, then the next player's entire turn, back-to-back, all `done=False` — confirmed this sequential path (not the batched one) is what *always* runs in real training, because `_agents_support_batching` requires every seat to support batching and `HeuristicAgent`/`GreedyPlayAgent` (always present, 2 slots each) explicitly opt out; (2) `_train_parallel` merges all 6 workers' games into one shared buffer before calling `update()`, so different *games* get concatenated too. At every one of these boundaries, GAE was bootstrapping off a value estimate from a completely unrelated player/game state — not noisy, actually wrong — which would prevent the value function from ever converging regardless of any reward-shaping fix, since a large fraction of its training targets were structurally corrupted, not just noisy (more data doesn't average away a wrong target that's wrong the same way in every game).

Fixed by tagging every `Transition` with a `traj_id` (identifying which (game, player) trajectory it belongs to) and rewriting `compute_advantages` to group transitions by `traj_id`, running the standard reverse-time GAE recursion independently within each group's own chronological order rather than raw buffer-adjacency order. `PPOAgent` (train.py) now takes a `game_uid` param (the per-game seed, falling back to `uuid.uuid4()` when seed is None) and derives `self.traj_id = (game_uid, player_id)`, threaded through `record_transition`/`record_transition_precomputed` → `PPOTrainer.collect_transition`/`store_transition` → `Transition.traj_id`. Updated all three `PPOAgent(...)` construction sites (`run_one_game`, and both branches inside `_worker_run_game`) to pass `game_uid=seed` (or `game_idx` for `run_one_game`, which has no seed param). `StaticAgent`/`HeuristicAgent`/`GreedyPlayAgent` don't need this — their `record_transition` methods are no-ops, they never contribute to the buffer.

Verified three ways: (1) numeric regression test — single-trajectory input (all same `traj_id`) produces bit-identical output to the old flat algorithm; (2) a synthetic interleaved two-trajectory buffer produces results matching each trajectory computed fully in isolation (the objectively correct answer); (3) confirmed the *old* algorithm on that same interleaved data produces a *different* (wrong) answer, proving the bug was real, not just theoretical. Also ran the real parallel-training path (`_train_parallel`, multiprocessing workers, real PPO updates, warm-up phase) for 3 updates with no errors, and directly logged `traj_id` counts per batch to confirm distinct trajectories are actually being tagged correctly in practice (16 distinct ids for 4 games × 4 training-agent slots each during warm-up, matching the expected composition when the snapshot pool is still empty).

The notebook's own duplicate `_Agent` class (`explore.ipynb` cell 48) has the same theoretical bug but was NOT patched — confirmed the notebook's real parallel-training loop (cell 49) imports and uses `train.py`'s actual `PPOAgent`/`_worker_run_game`/`_train_parallel`, so this fix already covers it; cell 48's local `_Agent` is only used by the cell-55 single-game trace/debug tool, not real training.
**Current state:** GAE now correctly isolates trajectories; regression tests and a short real-training smoke test pass. **Not yet validated with a real multi-thousand-game training run** — the checkpoint from the previous (buggy) run is still sitting in `bg_agent_ppo.pt` / `checkpoint_backups/`; this fix changes the *meaning* of `compute_advantages` enough that resuming from a checkpoint trained under the old buggy GAE onto the new correct GAE is a legitimate approach (weights are still weights either way) but a from-scratch comparison run would give a cleaner read on how much this actually matters.
**Open questions / next steps:**
- Run another fresh (or resumed) training run and check whether `value_loss` actually trends downward now, and whether board size / sell-behavior recovers — this is the real test of whether this bug explains today's earlier findings or was one contributing factor among several.
- `run_fresh_training.py` (repo root, untracked, gitignored implicitly via nothing — should NOT be committed) is a temporary verification script from this session; delete once no longer needed for testing.
- Given this bug likely predates every training run this project has ever done (it's in the original GAE implementation from Phase 6, 2026-04-11/12), any prior conclusions drawn from training curves or learned behavior (e.g. earlier "the agent learned to sell too much" characterization) should be revisited with some skepticism — they were observed under a policy trained on structurally corrupted advantage estimates the whole time.
---

---
### 2026-08-31 (2) — Stop misreporting fully-skipped PPO updates as a perfect 0.0 loss
**Files changed:** `agent/ppo.py`
**What was done:** While watching the GAE-fix verification run, noticed `total_loss=0.0000 value_loss=0.0000` exactly (not just low) on 5 of the first 100 updates. Traced to `PPOTrainer.update`: the KL-early-stop check (`if approx_kl > cfg.target_kl: break`) fires *before* loss is computed or appended to `epoch_metrics`, so if the very first mini-batch of the very first epoch already exceeds `target_kl=0.02`, the whole update breaks immediately with empty metric lists — and the final aggregation (`float(np.mean(v)) if v else 0.0`) silently reported that as a perfect 0.0 loss instead of "no gradient step happened." The collected batch of games (a full worker-batch's worth of games) still gets discarded via `self.buffer.clear()` either way — that part is an intentional PPO safety mechanism (don't train on stale importance weights), not something to change — but reporting it as 0.0 hid how often it was happening (~5% of updates in early testing) and looked like unusually good training rather than wasted data.
**Current state:** `update()` now reports `float("nan")` instead of `0.0` when zero mini-batches actually ran, logs a warning naming the update number, and returns an additional `n_minibatches` count in the metrics dict for any caller that wants to check directly. Verified: `log_update_metrics`'s `%.4f` formatting handles NaN fine (prints "nan", no crash); a short real `_train_parallel` smoke test confirms the new `n_minibatches` key is present and callers relying on `metrics.get(...)` are unaffected. This is a reporting-only fix — no training math changed — so it was safe to make without interrupting the already-running GAE-fix verification run (which won't pick it up until restarted, since the running processes already have the old module loaded).
**Open questions / next steps:**
- Didn't investigate *why* KL spikes past target_kl on the very first mini-batch ~5% of the time — could be worth a look once the GAE fix's impact is clearer, since it might be a downstream symptom of the same kind of noisy-advantage issue, or something else (return normalization interacting with small buffers, etc).
- Should re-check the skip rate on a future run now that it's visible instead of hidden behind a fake 0.0 — if it's still ~5%, that's still 1 in 20 update batches contributing nothing, worth deciding whether to address (e.g. lower the learning rate, or check the first mini-batch's KL before committing to it) rather than just log it.
---

---
### 2026-08-31 (3) — Fix reward chart mixing training-agent reward with scripted/frozen opponents'
**Files changed:** `explore.ipynb` (cell 49 `_on_batch`), `run_fresh_training.py` (this session's verification script)
**What was done:** User asked, while watching the GAE-fix verification run, whether the plotted "Training Reward" is really only from the training agents. Checked and it wasn't: `GameResult.final_rewards` ([game_loop.py:1218](env/game_loop.py#L1218)) has one entry per seat — all 8 players, including `HeuristicAgent`/`GreedyPlayAgent` (always 2 slots each) and frozen historical-snapshot opponents, not just the `PPOAgent`-controlled training seats. Both the notebook's cell 49 and this session's `run_fresh_training.py` computed the plotted per-game reward as an unconditional `np.mean(list(summary['final_rewards'].values()))` — averaging in scripted/frozen agents whose reward never changes over training, diluting whatever real trend the training policy has. This bug predates today's session; it's been in the notebook's cell 49 since that chart was written, not something introduced by today's other changes.

Fixed by filtering to seats labeled `'train_current'` in `summary['agent_labels']` (already computed and returned by `_worker_run_game`, just never consumed for this purpose) before averaging, in both files, with a safe fallback to the old all-seats average if labels are ever missing. Since the already-completed ~700-game GAE-fix verification run only stored the pre-averaged (contaminated) scalar per game — not the per-seat breakdown — its data can't be retroactively corrected, so that run's reward curve is unreliable evidence either way (though the checkpoint's *weights* are still real GAE-fixed weights, preserved at `checkpoint_backups/bg_agent_ppo_gae_fix_contaminated_reward_metric.pt` in case they're useful later). Stopped that run and restarted fresh (again from scratch, not resumed) so the new run's reward curve is clean from game 1.

Editing `explore.ipynb` directly via `NotebookEdit` wasn't possible — the tool requires a full prior `Read`, and even after stripping all embedded PNG outputs (cells 27, 42, 49 — matplotlib architecture diagrams and the live-training chart) the notebook's 57+ cells of real source code alone exceeded the 25,000-token read limit. Edited cell 49's `source` list directly via the same JSON read/write approach already used to clear its stale output, verified with `ast.parse` before saving. Confirmed cell 49's fix is syntactically clean; did not restore the cleared cell 27/42 diagram outputs (cosmetic only, regenerate on next re-run of those cells) or attempt to touch the notebook's own `_Agent` class (cell 48) since — per the 2026-08-31 GAE-fix session note — it's only used by the cell-55 trace/debug tool, not the real training loop cell 49 actually uses.
**Current state:** Both the notebook and the standalone verification script now report training-agent-only reward. A fresh run (checkpoints moved to `checkpoint_backups/`, `bg_agent_ppo.pt` etc. now empty/absent at repo root) is running in the background with all three of today's fixes active: symmetric board-shape clipping, annealed stats-potential blend, per-trajectory GAE, NaN-safe update reporting, and the corrected reward metric.
**Open questions / next steps:**
- This is now the third fresh restart today (clip-symmetry fix → GAE fix → reward-metric fix) — once this run has a meaningful sample, that's the one whose reward curve and board-size instrumentation should actually be trusted for judging whether today's fixes worked.
- Consider: should `game_lengths` (whole-game property, not per-seat) also be filtered somehow, e.g. to games where a training agent survived past round N? Left as-is for now since game length isn't an averaged per-agent reward and doesn't have the same contamination issue — flagging only in case it turns out to matter later.
- The cleared cell 27/42 diagram outputs will just look blank until those cells are next re-run; not a functional issue, purely cosmetic.
---

---
### 2026-08-31 (4) — Fix HeuristicAgent/GreedyPlayAgent stalling on discover/trinket offers
**Files changed:** `train.py`
**What was done:** User asked for a targeted audit of bugs caused by the heuristic/greedy scripted-opponent addition specifically (given it already caused the always-sequential-path issue found earlier today). Found a second, real one: `HeuristicAgent`/`GreedyPlayAgent.get_action` pick their BUY pointer by iterating `ps.shop` directly. But during a discover event or trinket offer, `step_shopping` interprets that same pointer against `ps.discover_pending` or the trinket offer list instead — shorter, unrelated lists that `ps.shop` is never swapped to reflect (only the *encoded observation tensor* built in `_get_observation` gets swapped for trinket/discover, per `game_loop.py:1466-1475`; `ps.shop` itself stays whatever it was). Real agents (`PPOAgent`/`StaticAgent`) are protected because their action sampling is constrained by `build_pointer_mask`, which *does* correctly restrict to the valid discover/trinket range (`agent/policy.py:656-666`) — but the scripted agents never call it, they just compute their own index and bypass masking entirely.

Confirmed the failure mode concretely: built a `PlayerState` with a 7-slot shop (best-tier minion at index 5) and a 3-item `discover_pending`; the old `HeuristicAgent` buy-highest-tier logic computes `choice_idx=5`, outside the valid 0-2 range. Both `step_shopping`'s discover branch and `TrinketHandler.select` are bounds-checked (no crash), but silently no-op on out-of-range input — so the offer never resolves, the state never changes between calls, and the agent repeats the same invalid pick until the sequential shopping loop's `max_actions=30` cap force-ends the turn without the agent ever having chosen END_TURN. `GreedyPlayAgent` mostly escapes by luck (it grabs the first occupied shop slot, almost always index 0, which is usually in-range too) but isn't protected by design either. Since `HeuristicAgent` occupies 2 of 8 seats in every training game, this means the scripted "population diversity" anchor has likely been silently freezing for a full turn every time it hit a discover or trinket event since it was introduced (2026-04-13) — degrading it as a stable training opponent this whole time.

Fixed by checking `ps.trinket_offer_pending or ps.discover_pending` first in both classes' `get_action`, short-circuiting to `(0, PTR_SHOP_OFF + 0)` before reaching the normal `ps.shop`-based logic — index 0 is always valid when either is pending (both lists are non-empty by construction whenever the flag is set). Verified: reproduced the old out-of-range failure directly (`choice_idx=5` for the discover example above), confirmed the fix produces `choice_idx=0` (always in-range) for both agents in both discover and trinket scenarios, and ran a real 3-update parallel-training smoke test with no errors.

While investigating, also noticed (but did NOT fix, flagging only) that both the sequential and batched shopping-phase loops share a `max_actions=30` cap ([game_loop.py:1090](env/game_loop.py#L1090), [game_loop.py:1338](env/game_loop.py#L1338)) with no fallback if an agent never calls END_TURN within it — the player's `end_turn_buffers` entry for that round simply never gets set, meaning that round's combat/placement reward never reaches a recorded transition (though `cumulative_rewards` bookkeeping for final placement is unaffected, only the per-transition PPO training signal for that specific round is lost). This isn't specific to the heuristic/greedy change (both code paths have had it since Phase 5, and it applies to any agent type, including an early/highly-exploratory PPOAgent), and the scripted agents' normal (non-discover/trinket) decision chains are comfortably bounded well under 30 actions by gold/hand/board caps, so this wasn't the active bug here — just a related, lower-priority edge case worth knowing about.
**Current state:** Both scripted opponents now resolve discover/trinket offers immediately instead of stalling. Verified with unit-level reproduction + fix confirmation and a real training smoke test; not yet re-verified with a full multi-thousand-game run (the actively-running verification run from earlier today predates this fix, same as it predates nothing else in this session — see open questions).
**Open questions / next steps:**
- Should decide whether to restart the currently-running verification run (started after the reward-metric fix, currently accumulating a clean sample) yet again to include this fix, or let it continue and treat this as a smaller, separate confound to reason about later. This is the fourth potential restart point today — diminishing returns on restarting for every single fix found; likely worth batching remaining fixes before the next restart rather than restarting on each one.
- The `max_actions=30` no-END_TURN edge case above is worth a real look at some point (e.g., force an END_TURN on cap-out, or at least log a warning so it's visible when it happens) but is deliberately deferred, not fixed, in this session.
- Did not find any other heuristic/greedy-specific issues in this pass: `_append_agent_stats`'s per-agent-type win-rate tracking correctly uses `agent_labels`; `SnapshotPool`/opponent-slot composition math (`n_policy_slots = N_OPP_SLOTS - N_HEURISTIC_SLOTS - N_GREEDY_SLOTS = 2`) is non-negative and correct; `_worker_run_game`'s policy-snapshot deduplication cache doesn't interact with the heuristic/greedy branches (separate `elif` arms, never reach the cache code path).
---

---
### 2026-08-31 (5) — Fix real PPOAgent (not just scripted agents) sampling invalid pointers during discover/trinket
**Files changed:** `agent/policy.py`
**What was done:** Continued the bug audit after the HeuristicAgent/GreedyPlayAgent discover/trinket fix, specifically checking whether the *real* trained policy could have the same class of problem, not just the scripted opponents. It does. `build_pointer_mask(ps, -1)` — the "full occupancy, all zones" mask `PPOAgent.get_action` passes for its first, type-agnostic forward pass ([train.py:184](train.py#L184), also used by the batched shopping path at [game_loop.py:1380](env/game_loop.py#L1380) and the notebook's local `_Agent` class) — only checked `trinket_offer_pending`/`discover_pending` inside the `type_idx == 0` branch, not the separate `type_idx == -1` branch. So when either offer is pending, `get_action` correctly forces `type_idx=0` (BUY) via `type_mask`, but the pointer distribution it samples from was intersected with the *unrestricted* raw `ps.shop` occupancy (up to 7 slots) instead of the actual offer's range (~3 slots) — `step_shopping` ignores `ps.shop` entirely while a trinket/discover offer is pending (same underlying fact as the scripted-agent bug), so any sampled pointer outside the real offer's range silently no-ops.

This is more serious than the scripted-agent version: it directly degrades the *real* training signal, not just an opponent's behavior. Every discover/trinket event gives the policy roughly a `(len(offer)/n_occupied_shop_slots)` chance (~3/7 ≈ 43% typical) of sampling a wasted, no-op action instead of resolving the offer — recorded as a real transition where the state visibly didn't change despite an action being taken, which is at best noisy and at worst actively confusing for the value function to fit (another concrete contributor to the value-loss-never-converges pattern investigated earlier today, on top of the GAE bug).

Fixed by making the `type_idx == -1` branch check `_trinket_pending`/`_discover` for its shop-zone portion the same way the `type_idx == 0` branch already did, before falling through to raw occupancy — board/hand zone marking is unchanged (harmless either way, since `get_action`'s zone-intersection step only keeps the sampled type's own zone regardless). Verified directly: built player states with a 7-slot shop and 3-item discover/trinket pending, confirmed `build_pointer_mask(ps, -1)` now marks exactly slots 0-2 as valid (previously marked all 7), and confirmed the normal (no offer pending) case is unaffected (still marks true raw occupancy). Ran a clean 4-update real training smoke test afterward with no errors.
**Current state:** Both the scripted-agent-specific bug (train.py) and this deeper root-cause version (agent/policy.py, affecting real PPOAgent/StaticAgent/batched-path sampling too) are now fixed. This was found via the user explicitly asking for a broader "make sure there are no other bugs" pass rather than stopping at the first (narrower) fix — worth remembering as a pattern: the first bug found in an area is not necessarily the only one.
**Open questions / next steps:**
- This is now five real fixes in one day (clip symmetry, annealed stats blend, per-trajectory GAE, NaN-safe reporting, reward-metric contamination, scripted-agent discover/trinket stall, and now this policy-level pointer-mask gap) — the currently-running verification run predates all of the last three. Given the accumulating list, the next restart should batch all of today's fixes together rather than restarting per-fix again.
- Did not find further instances of this exact class of bug (checked `get_action_batch` and the `type_idx==0/1/2/8` branches directly — only `-1` had the gap, since it's the only branch that was written before trinket/discover offers existed and never got updated when they were added).
---

---
### 2026-08-31 (6) — Expand run_fresh_training.py's live chart to 10 panels
**Files changed:** `run_fresh_training.py` (temporary verification script, not committed)
**What was done:** User asked for more live-tracked stats beyond reward/loss/length/board-size, to stop needing ad-hoc instrumented scripts for questions we kept re-asking manually all session. Added six more, all computed from data already flowing through `_on_batch`/`_on_update` (no new plumbing into the actual training pipeline needed): sell:place ratio (20-game rolling window, directly answers "is it still selling too much" without eyeballing text percentages), avg placement vs. `heuristic`/`greedy` baselines (from `agent_labels`+`placements`, arguably the most direct "is it actually getting good" signal), skipped-update rate (surfaces the ~5% fully-discarded-update issue found earlier today), policy entropy, LEVEL_UP rate (ties to the weak-economy concern flagged earlier), and max weight magnitude (divergence early-warning, present in the original notebook's design but missing from this script). Chart is now a 2x5 grid: row 1 = training-internals health (reward, losses, entropy, max weight, skip rate), row 2 = strategy/game-shape (game length, board size, sell:place, level-up rate, placement-vs-baselines).
**Current state:** Verified with a real 36-game/3-update smoke test — all 10 panels render correctly, including the windowed panels correctly showing empty (not crashing) when there isn't yet enough data for their window (20 games / 10 updates). History persistence (`data/fresh_training_history.json`) extended to cover all new series so a resume doesn't lose them.
**Open questions / next steps:**
- Not yet applied to the actual live run (same as every fix today — the running process has the old script in memory; needs a restart to pick this up).

---
### 2026-08-31 (6) — Fix CUDA fork crash in ProcessPoolExecutor when training on GPU
**Files changed:** `train.py`
**What was done:** Rented a vast.ai GPU instance (GTX 1660, contract 49382151) via the `vastai` CLI to run `run_fresh_training.py` remotely. First launch crash-looped every batch with `RuntimeError: Cannot re-initialize CUDA in forked subprocess` — `_train_parallel`'s `ProcessPoolExecutor` used the default `fork` start method, which is fine on CPU-only local runs but breaks the moment `device="cuda"` and the parent process has already touched CUDA before spawning workers. Fixed by building a `multiprocessing.get_context("spawn")` and passing it as `mp_context=` to both `ProcessPoolExecutor` constructions in `_train_parallel` (the initial pool and the one rebuilt after a worker error).
**Current state:** Training is running cleanly on the rented instance (contract 49382151, `ssh vast-bg-agent`) inside a `tmux` session named `train`, 6 workers, fresh start (`N_GAMES=5000`). A background sync loop on this machine pulls `data/training_progress.png` and `data/fresh_training_history.json` from the instance every 20s into the local `data/` dir so the chart updates live like the old local notebook runs. Local repo's own `bg_agent_ppo*.pt` weights were left untouched — this run's checkpoints only exist on the remote instance so far.
**Open questions / next steps:**
- Remote checkpoints (`bg_agent_ppo.pt` / `_backup.pt` / `_best.pt`) need to be pulled back down (or the run left to finish) before they're usable locally.
- Instance is billed continuously (~$0.048/hr on $7.85 credit) until stopped — remember to `vastai stop instance 49382151` or destroy it when done.
- The same fork-vs-CUDA issue would resurface for any other `ProcessPoolExecutor`/multiprocessing spot that touches CUDA in the parent before forking — worth a quick audit if more GPU-parallel code gets added.
---

---
### 2026-08-31 (7) — Fix run_fresh_training.py re-executing itself in every spawned worker
**Files changed:** `run_fresh_training.py` (untracked one-off script, not committed), plus board-size/sell-place/level-rate chart panels reworked to per-update mean+std+trend (matching the reward panel style) instead of raw per-game points.
**What was done:** The CUDA-fork fix from session (6) switched `ProcessPoolExecutor` to `spawn`, which re-executes the launching script (`run_fresh_training.py`) in every child process during bootstrap. The script had no `if __name__ == "__main__":` guard, so each of the 6 workers redundantly re-ran the *entire* setup on every spawn — building a fresh `BGPolicyNetwork` on CUDA, reloading the 40MB checkpoint, re-reading the card defs JSON. Combined with the pool-rebuild-on-error retry loop already in `train.py`, this cascaded into `OSError: [Errno 24] Too many open files` and killed the training process (and its tmux session) after ~240 games. Fixed by wrapping the entire body of the script (everything after imports/logging setup) in `if __name__ == "__main__":` — module-level constants used as function default-argument values (`UPDATE_INTERVAL`, `UPDATE_TREND_WINDOW`) stay correctly ordered before the defs that use them since indentation doesn't change execution order. Also removed now-dead `_rolling_ratio`/`GAME_WINDOW` after switching those two chart panels to per-update aggregation.
**Current state:** Training relaunched on the vast.ai instance (contract 49382151) inside tmux session `train`, resumed cleanly from the last checkpoint (steps=32,917, right where the crash left off). Verified via `/proc/<pid>/fd` count that the worker process file-descriptor count is no longer climbing. `run_fresh_training.py`'s `_save_chart()` now plots `board_size`, `sell:place ratio`, and `LEVEL_UP rate` the same way as the reward/length panels: one point per PPO update (mean of the games that fed it) with a ±1 std shaded band and a rolling trend line, instead of one raw point per game.
**Open questions / next steps:**
- History entries for updates 1-20 (saved before this session's chart-schema change) don't have the new per-update board/sell-place/level-rate series — those panels will just start populating fresh from here rather than backfilling.
- Same fd-exhaustion risk would resurface for any other spawn-based multiprocessing entry script in this repo that lacks a `__main__` guard — worth double-checking before adding more of these one-off training scripts.
- Instance still billing continuously; remember to stop/destroy contract 49382151 when the run is done.
---

---
### 2026-08-31 (8) — Wire up and fix the per-update chart aggregation added in session (7)
**Files changed:** `run_fresh_training.py` (untracked one-off script, not committed)
**What was done:** Session (7)'s chart rework left three bugs: (1) `_append_update_stat` was defined but never called for board size/sell:place/level-rate, and `_on_batch` never appended to the new `sell_place_ratios`/`level_rates` per-game lists — so those three panels, and the std bands on reward/length, silently stayed empty forever (confirmed via a local smoke test on the `bg-agent` conda env before touching the remote box again). (2) Once wired up and redeployed, the resumed run crashed for real: `update_reward_avg` (45 entries, carried over from the pre-fix history) and `update_reward_std` (2 entries, since std-tracking didn't exist when the older history was written) hit `ax.fill_between` with mismatched array lengths and killed the tmux session. Fixed `_plot_avg_trend_std` to shade only the tail where both series actually overlap (`n = min(len(std), len(avg))`) instead of assuming equal length. (3) While verifying, noticed `ppo_losses`/`ppo_values` were never included in `_save_history`/the resume block at all (pre-existing, unrelated to this session) -- they silently reset to empty on every restart, which is why the PPO Losses panel kept showing only a handful of points while every neighboring panel showed the full history. Fixed the same way as the other series. Also converted the "Placement vs Baselines" panel from a per-game 10-game rolling window to the same per-update mean+trend style as everything else (no std shading there -- 3 overlapping bands on one axes was worse than the lines alone).
**Current state:** All fixes verified locally first this time -- a fresh-start smoke test, then a resume-through-deliberately-legacy-history smoke test (stripped `_std` and `ppo_losses`/`ppo_values` fields to reproduce exactly what the real resumed checkpoint looks like) -- before touching the remote instance again. Redeployed to contract 49382151; confirmed via the training log that multiple PPO updates fire cleanly post-relaunch. All 10 chart panels now render with real per-update data.
**Open questions / next steps:**
- Board size/sell:place/level-rate/ppo_losses/ppo_values std bands and full history will only be as long as the number of updates since this session's redeploy, since resumed history genuinely lacks that data further back -- this is expected, not a bug.
- Lesson for next time: smoke-test any change to this script locally (the `bg-agent` conda env has all deps, CPU-only) with a tiny `N_GAMES`/`N_WORKERS` override *and* a resume pass through deliberately-stale history, before redeploying to the billed GPU instance -- both bugs this session would have been caught by that in about a minute of local runtime instead of a burned GPU cycle each.
---

---
### 2026-08-31 (9) — Migrate training to a faster GPU, learning the hard way about fractional vast.ai hosts
**Files changed:** `run_fresh_training.py` (untracked one-off script, not committed)
**What was done:** Investigated whether training was GPU-bound; found the GTX 1660 was actually well-utilized (70-95% GPU, ~73% CPU across 24 vCPUs), just weak hardware (~4.8 TFLOPS, no Tensor Cores). User asked to speed things up within a ~$0.32/hr (€0.30) budget and to tune batch_size for better GPU utilization on a stronger card. Bumped `batch_size` 512→1024 (no AMP added -- this codebase has a documented history of hard-won PPO numerical-stability fixes, not worth the risk for a utilization gain).

First attempt: rented an RTX 3090 (contract 49389903, offer with 80 nominal vCPUs) -- got stuck loading the base Docker image for 26+ minutes, destroyed it. Second RTX 3090 (contract 49392734) booted fine, but `N_WORKERS=16` produced *worse* throughput than the original 6-worker setup (53-71s/batch vs the old 7-16s/batch). Root cause: vast.ai's advertised `cpu_cores` (72) and even its own `cpu_cores_effective` (20) were both misleading -- the container's actual `/sys/fs/cgroup/cpu.max` quota was only 8.64 real cores, less than the original dedicated 24-vCPU box. No worker count on that host could recover the old throughput; the fractional/shared nature of the host was the real constraint, not something `N_WORKERS` could fix.

Learned to filter vast.ai offers by `cpu_cores_effective == cpu_cores` (ratio 1.0) to find genuinely dedicated (non-fractional) hosts before renting. Found an RTX 4060 Ti (contract 49396123) with 32 dedicated cores confirmed via the same cgroup check (~30.7 real cores) -- more real CPU than the original box, ~21.7 TFLOPS (4.5x the GTX 1660), $0.118/hr. Set `N_WORKERS=8`, `UPDATE_INTERVAL=16` (2x, preserving the documented ratio). Migrated the checkpoint (had to retry once -- rsync hit a transient "No data available" read error mid-write on `bg_agent_ppo.pt`, resolved by retrying a few seconds later) and confirmed steady-state throughput of ~8s/8-game-batch (~0.98 games/sec) vs. the original ~0.60 games/sec -- a genuine 63% speedup, over 7 clean PPO updates with no errors.
**Current state:** Training running on the RTX 4060 Ti instance (contract 49396123, `ssh vast-bg-agent` -- SSH config repointed) in tmux session `train`, resumed from the same checkpoint lineage (steps ~384k+). Both prior instances (GTX 1660 contract 49382151, fractional RTX 3090 contract 49392734) were destroyed to stop double-billing. Chart-sync loop restarted, `data/training_progress.png` updating live again.
**Open questions / next steps:**
- When renting vast.ai instances for CPU-bound workloads in future, check `/sys/fs/cgroup/cpu.max` immediately after boot (before doing any other setup) rather than trusting the listing's `cpu_cores`/`cpu_cores_effective` fields -- both can overstate real capacity on fractional/shared hosts.
- Should re-run the placement-vs-baselines / entropy / board-size analysis again once enough updates have accumulated on this instance, since the batch_size/worker-count change is itself a training-dynamics change worth accounting for when reading trend lines.
- Instance still billing continuously ($0.118/hr on remaining vast.ai credit); remember to stop/destroy contract 49396123 when the run is done.
---

---
### 2026-08-31 (10) — Add vast-ai-training skill capturing this session's GPU-rental lessons
**Files changed:** `.claude/skills/vast-ai-training/SKILL.md` (new)
**What was done:** Packaged the operational knowledge from sessions (5)-(9) -- renting vast.ai instances, migrating a running training job, and the GPU speedup work -- into a project skill so future sessions don't have to rediscover the same failure modes. Covers: filtering offers by `cpu_cores_effective/cpu_cores` ratio to find genuinely dedicated (non-fractional) hosts instead of trusting the advertised vCPU count; checking `/sys/fs/cgroup/cpu.max` immediately after boot as the only trustworthy real-CPU-quota signal; when to give up on a stuck Docker image pull vs. being patient; the fork-vs-spawn CUDA multiprocessing crash and the `if __name__ == "__main__":` guard needed to stop spawn from re-executing a launch script's setup code in every worker; the transient rsync "No data available" error on a checkpoint mid-write; and a monitoring/cleanup checklist (stable SSH alias reused across migrations, quiet background sync loops, stopping the old instance before destroying it).
**Current state:** Skill file committed under `.claude/skills/vast-ai-training/`. Training itself continues unaffected on the RTX 4060 Ti instance (contract 49396123).
**Open questions / next steps:**
- Skill hasn't been exercised end-to-end from a fresh session yet -- next time a vast.ai instance is needed, confirm the skill actually gets picked up and the checklist holds up in practice, then refine if anything was missing.
---

---
### 2026-08-31 (11) — Add potential-based leveling reward (tier-shape)
**Files changed:** `env/game_loop.py`, `env/player_state.py`, `CLAUDE.md`
**What was done:** User asked why the agent still sells a lot even under board-shape reward. Traced (via a live checkpoint trace of 684 real SELL actions) that the agent isn't cannibalizing its board — 0 golden minions sold, avg tier sold 1.68, median board-shape reward for a sell was exactly 0.0 (93.7% within ±0.02) — it's discarding a steady stream of cheap tier-1/2 filler that barely moves the win-prob estimate either way. Root cause: `LEVEL_UP` had zero direct reward anywhere in the code (verified no `reward +=` in that branch and no hidden reward via `hero_handler.on_tavern_upgrade`), so spending gold on PLACE (immediate board-shape payout) strictly dominated spending the same gold on LEVEL_UP (zero payout, only a delayed multi-round benefit via better future shop rolls) — consistent with `GreedyPlayAgent`, which hardcodes "level first" with no reward involved at all, already beating the trained policy's placement. Added a second potential-based shaping term mirroring `_apply_board_shape`'s pattern: `_tier_potential(ps) = min(1.0, tavern_tier / _expected_tier_for_round(round_num))` (a simple round→tier curve, tier2 by round 3, tier3 by round 5, ... tier6 by round 10, untuned), applied only on a successful LEVEL_UP via `_apply_tier_shape` with `TIER_SHAPE_ALPHA = 0.10`. The `min(1.0, …)` cap and per-round `ps.phi_tier` reset (mirroring `ps.phi_board`) mean leveling past the round's curve pays nothing extra and the term can't be farmed — deliberately guarding against reproducing the same kind of degenerate-policy failure the old flat `+0.10 * board_size` term caused (see 2026-08-30 entry).
**Current state:** Smoke-tested with the live checkpoint over 6 full games (batched=False, mixed StaticAgent/Heuristic/Greedy seats) — no crashes, 128 level-ups logged, rewards bounded in [0, 0.05], correctly zero when leveling happens past curve (e.g. tier 3→4 at round 4 when curve already expects tier 3 by round 5... expects tier 3 at round 5 specifically, so an early tier-4 push at round 4 still capped at Φ=1.0 from reaching tier 3 already). `CLAUDE.md`'s Reward Shaping section updated to document this as component 4, renumbering `FINAL_PLACEMENT_REWARD` to 5; also fixed a stale claim there that PLACE was clipped in board-shape (it isn't, since the 2026-08-30 clip-symmetry fix — the doc just never got corrected). Deployed to the live vast.ai run (contract 49396123) by syncing the changed files and resuming (not restarting from scratch) from the current checkpoint, since this is an additive reward term rather than a change to the meaning of existing returns/advantages (unlike the GAE fix).
**Open questions / next steps:**
- `TIER_SHAPE_ALPHA = 0.10` and the `_expected_tier_for_round` curve are both untuned guesses — watch `level_rate` *together* with `board_size` and `train_current` placement after this redeploy (not `level_rate` alone) to confirm it's closing the leveling gap rather than overcorrecting into "level compulsively, never field a board."
- Because this was a resume rather than a from-scratch restart, the pre/post comparison for whether this change worked should be read from the update-index boundary at the redeploy point in `data/fresh_training_history.json`, not the whole run's trend.
- If `level_rate` doesn't move at all post-redeploy, the `_expected_tier_for_round` curve or `TIER_SHAPE_ALPHA` magnitude are the next things to revisit, per the same reasoning used to diagnose the original zero-reward gap.
---

---
### 2026-08-31 (11) — Replace MC win-probability board-shape potential with a deterministic, quality-weighted stats proxy
**Files changed:** `env/game_loop.py`, `train.py`
**What was done:** Session (10)'s full-run analysis found reward and placement-vs-baseline plateauing then regressing right around update ~106/312 -- which turned out to coincide almost exactly with `BOARD_SHAPE_STATS_ANNEAL_STEPS=250_000`, the point where the existing (but previously always-annealed-away) deterministic `_board_stats_potential` scaffold fully faded out in favour of the noisy 30-trial Monte Carlo win-probability estimate. In the same window, sell:place ratio climbed toward 1.0 while board size shrank and LEVEL_UP rate collapsed -- consistent with the policy learning to farm noise in the MC estimate (churn sell/place actions hoping for a lucky upward sample) rather than genuinely improving the board, once the deterministic anchor was gone. Investigated and ruled out several other NaN/instability hypotheses along the way (tier-shape reward math, both action masks, aura-dependency division) before landing on this diagnosis via the anneal-timing correlation.

Fix: enriched `_board_stats_potential` with keyword bonuses (Divine Shield/Taunt/Reborn/Windfury, +`BOARD_STATS_KEYWORD_BONUS` each) and a tribal-synergy bonus (`BOARD_STATS_SYNERGY_BONUS` when >=4 same tribe, mirroring `CLAUDE.md`'s "synergistic" threshold) on top of the existing `_board_power`-based raw stats sum -- still quality-weighted (a board of seven 1/1s scores far below a board of a few strong minions), not count-weighted, specifically so hoarding weak minions never outscores upgrading. Removed the anneal-to-0 schedule (`_stats_weight_for_step`, `BOARD_SHAPE_STATS_WEIGHT_INIT=0.25`/`BOARD_SHAPE_STATS_ANNEAL_STEPS=250_000`) entirely and replaced it with a fixed `BOARD_SHAPE_STATS_WEIGHT = 1.0` in `train.py` -- the deterministic proxy is now the permanent board-shape signal, not early-training scaffolding. Added a symmetric short-circuit in `_board_potential` (`if w >= 1.0: return self._board_stats_potential(ps)`) so the MC combat sim is never even called when its result would be multiplied by zero.

Validated with a standalone sanity-check script (not committed, scratch-only) calling the real `_board_stats_potential` method against synthetic board states before deploying: confirmed a garbage-hoard board scores well below a strong board despite more minions, confirmed "sell weak -> place strong" nets a real potential increase over the starting board (not just a cost), confirmed hoarding more weak minions doesn't match an upgrade's payoff, and confirmed keyword/tribal-synergy bonuses apply correctly.
**Current state:** Deployed to the RTX 4060 Ti instance (contract 49396123). Previous tier-shape run's final checkpoint backed up to `checkpoint_backups/bg_agent_ppo_pre_board_stats_proxy_final_steps731858.pt` (and `_best_...`/history json) before a fresh restart, matching the project's established scratch-restart convention for reward-shaping changes. First 3 updates confirmed clean (no NaN, no crash). Unplanned bonus: removing the MC sim call from the hot path (every PLACE/SELL action, all self-play) roughly **4x'd** self-play throughput (~0.98 -> ~3.9 games/sec) as a side effect of the reward-quality fix, not a separate optimization.
**Open questions / next steps:**
- This is a genuinely new run (fresh weights) -- not directly comparable update-for-update to the tier-shape run's chart, since both the reward signal and the throughput changed. Let it accumulate a comparable number of updates (~300+) before drawing conclusions about whether the regression pattern is actually fixed.
- BOARD_STATS_KEYWORD_BONUS=3.0 and BOARD_STATS_SYNERGY_BONUS=5.0 are untuned initial guesses (same honesty as the existing BOARD_SHAPE_STATS_SATURATION=30.0) -- revisit once real training data shows whether they're well-scaled relative to raw board power.
- Going to shape_stats_weight=1.0 removes ALL win-probability grounding from board-shape reward -- the deterministic proxy can't see keyword *interactions* (e.g. a Divine Shield minion positioned to protect a Taunt) the way a simulated combat can. Worth watching whether the policy's actual combat performance (not just the proxy score) tracks placement improvement, since that's the one thing this change can't directly verify.
- The still-not-diagnosed NaN-in-evaluate_actions bug from the tier-shape run's skip-rate spikes (session 10) is unrelated to this change and remains open -- worth revisiting if skip-rate spikes recur under the new reward.
---

---
### 2026-08-31 (12) — Fix broken policy learning: PPO optimiser, reward invariance, honest eval, 3x throughput
**Files changed:** `agent/ppo.py`, `agent/policy.py`, `env/game_loop.py`, `env/player_state.py`, `train.py`, `run_fresh_training.py`
**What was done:** Diagnosed why the previous 312-update/731k-step run produced ZERO improvement: `update_train_plc_avg` was flat at 4.58 (worse than both scripted baselines: heuristic 4.48, greedy 3.68) while mean reward improved -4.21 -> -3.75, and `max_w` sat at 1.005 with entropy *rising* 1.44 -> 1.70. Five root causes, all fixed:

1. **Starved optimiser (dominant).** `lr=3e-5` + `batch_size=1024` + ~2,345 transitions/update = 3 minibatches x 4 epochs = **12 optimizer steps per update**, ~3,700 gradient steps for the whole run. Now `lr=2.5e-4`->5e-5 annealed, `batch_size=256`, `clip_eps=0.2`, `entropy_coef=0.015`->0.004 annealed.
2. **KL early-stop discarded whole updates.** A signed, high-variance estimator `mean(old_lp-new_lp)` broke the *minibatch* loop and then all remaining epochs; 8/264 updates applied zero gradient steps. Replaced with Schulman's non-negative k3 estimator, checked only BETWEEN epochs, with epoch 0 always completing.
3. **Value/reward scale mismatch.** Returns were re-normalised with a fresh per-batch mean/std, so `V(s)` lived in normalised units while GAE mixed it with RAW rewards -- the baseline baselined nothing. Replaced with a persistent Welford running return scale (`ret_std`): stored values are always raw-reward units, value targets are `returns/ret_std`. Persisted across checkpoints.
4. **NaN root cause found (the open bug from session 10/11).** `evaluate_actions` built `Categorical(logits=ptr_logits)` over the FULL batch; rows with an all-False pointer mask (non-pointer action types) softmax to NaN, and `NaN * 0.0 == NaN` then poisoned log_probs/entropy, making the guard discard the entire minibatch. Fixed with mask sanitisation before every `Categorical` plus `torch.where` instead of multiply-by-zero.
5. **Shaping was not policy-invariant.** `phi_board`/`phi_tier` reset every round, so the agent was PAID for rebuilding board strength but never CHARGED when it was lost -- a one-sided ratchet. Replaced with ONE `ps.phi` telescoping across the whole episode, paid at every reward-emitting point (all 9 action types + post-combat) with `SHAPE_GAMMA=0.997` matching `PPOConfig.gamma`. Verified: any cyclic action sequence now sums to ~0.

Also measured the reward decomposition for the first time: dense terms totalled **-5.38/game vs -0.375 from FINAL_PLACEMENT_REWARD (~17x)**, and the single largest term was the **hand penalty at -3.41/game** -- larger than combat outcome. Removed the flat +0.1/round survival bonus (unconditional passive income) and applied `DENSE_REWARD_SCALE=0.30` to the rest; dense/placement ratio is now 5.3x.

Added `evaluate_policy()`: one deterministic (argmax) policy seat vs seven fixed scripted opponents, parallelised across a spawn pool, seeded by game index so results are identical at any worker count (verified). This is the un-gameable progress metric. Also surfaced `explained_var`, `clip_frac`, `approx_kl`, `n_minibatches`, `lr`, `ret_std` per update.

Benchmarked worker config on the actual host instead of inheriting values from the old fractional 3090 box: `N_WORKERS=8` on CUDA gave 2.79 games/sec, **26 CPU workers gives 8.64 games/sec (3.1x)**. `_train_parallel`'s `device` arg only reaches `_worker_init`, so workers run CPU while the trainer stays on CUDA. Eval: 87.4s sequential -> 18.1s at 20 workers for 64 games. Both required `ulimit -n 65536` (>12 workers dies with `OSError: [Errno 24] Too many open files`).
**Current state:** Fresh run launched on the RTX 4060 Ti (contract 49396123) in tmux `train`, logging to `training_learnfix.log`. `N_GAMES=150000`, `UPDATE_INTERVAL=26`, `ANNEAL_STEPS=26M`, `EVAL_EVERY=100`/`EVAL_N_GAMES=128`/`EVAL_WORKERS=20` (~12% eval overhead). Previous run's weights+history preserved as `checkpoint_backups/prelearnfix_*`.
**Open questions / next steps:**
- Primary success criterion: `eval_mean_placement` vs 7 greedy must drop meaningfully below 4.5. Secondary, free and much lower-noise: `train_plc - greedy_plc` from the fixed scripted seats in every training game (was +0.80 -> +0.91, i.e. getting *worse*, over the previous run).
- Watch `explained_var` (was ~0.0 at init -- if it stays near 0 after ~50 updates the value function is still broken) and `clip_frac` (healthy 0.05-0.30; the smoke test showed 0.67 on update 1 of a random net, which the KL guard correctly caught).
- `DENSE_REWARD_SCALE=0.30`, `BOARD_POTENTIAL_WEIGHT/TIER_POTENTIAL_WEIGHT` (0.67/0.33) and the keyword/synergy bonuses are all still untuned guesses.
- Combat in this simulator does not mutate `ps.board` (win/loss is an aggregate win-prob estimate), so Φ cannot yet be charged for minions actually dying; the post-combat shaping call is wired and will activate automatically if combat is ever extended to model per-minion death.
- The `_train_parallel` docstring still describes `device` as "torch device string for the main process", which is wrong -- it only sets the workers' device.
---

**Correction to the entry above (same session):** `max_w` / the `max_ws` history series is the maximum absolute *parameter* value of the policy network (`max(p.abs().max() for p in policy.parameters())`), NOT the PPO importance-sampling ratio. The initial diagnosis text above described it as an importance weight; that reading was wrong. It does not change the conclusion or any fix: the "starved optimiser" finding rests on the independent arithmetic (lr=3e-5 x batch_size=1024 x ~2,345 transitions/update = 12 optimizer steps per update), and the real importance-ratio health metric (`clip_frac`) simply did not exist before this session. Post-fix it reads 0.14-0.29 (healthy) with `explained_var` climbing 0.0 -> ~0.25 and `n_minibatches` 15-20/update, which is the actual evidence the optimiser is no longer starved. For reference the parameter-magnitude series also moved ~50x faster after the fix (1.0096 -> 1.0148 in 12 updates, vs 1.0000 -> 1.0055 across 264 updates before).
---

---
### 2026-09-01 — Correct SHAPE_ALPHA sizing after telescoping made the shaping signal negligible
**Files changed:** `env/game_loop.py`
**What was done:** The first run under the new telescoping potential went 500 updates (~13,000 games) with eval placement flat at chance (4.46/4.41/4.87/4.32/4.50 vs 7 greedy) while `sell:place` sat at 0.875 and `level_rate` at 2%. The PPO diagnostics were healthy by then (`explained_var` 0.67, `clip_frac` 0.06, full 4 epochs, `approx_kl` well under target), which ruled out the optimiser and pointed at reward-signal strength.

Root cause: `SHAPE_ALPHA` was left at 0.20, inherited from the old `BOARD_SHAPE_ALPHA` -- but that constant was calibrated for a potential that RESET EVERY ROUND, i.e. paid out ~12x per game. Making the potential telescope correctly across the whole episode without rescaling alpha shrank the aggregate shaping signal by roughly the number of rounds: total telescoped magnitude was bounded to +-0.20 against `FINAL_PLACEMENT_REWARD` spanning +-4.0, about 5% of the objective -- far too weak to densify credit assignment, which is the entire purpose of the term. And because churn is only reward-NEUTRAL under telescoping (nothing pays for it, but nothing charges for it either), there was no gradient discouraging buy/sell cycling.

Raised `SHAPE_ALPHA` 0.20 -> 1.5, sized so a single good placement (Φ moves ~0.07 per minion, measured) is worth ~0.10 immediately -- comparable to the per-round combat reward (+0.15 win / -0.09 loss). Measured over 80 real player-games: shaping/placement ratio went from ~0.019 to 0.139, i.e. into the "densify, don't swamp" band. This is safe by construction: the Ng et al. invariance result holds for ANY alpha, so scaling cannot reintroduce a farmable term -- every cyclic sequence still telescopes to ~0 regardless of magnitude. That is exactly what the earlier correctness work bought.
**Current state:** Fresh run on contract 49396123 in tmux `train`, logging to `training_alpha15.log`; the alpha=0.20 run archived as `checkpoint_backups/alpha020_*`. At update 500/5769 (8.7%) every behavioural indicator is better than the alpha=0.20 run at the same point: `level_rate` 0.058 vs 0.022, `sell:place` 0.733 (stable) vs 0.871 (rising), `board_avg` 3.045 (rising) vs 2.263 (falling), `explained_var` 0.424 vs 0.266, entropy 1.172 vs 1.378. The free in-game skill gap (train_plc - greedy_plc) is improving monotonically 0.927 -> 0.715, vs 1.053 -> 0.88 (noisy, ended worse) for alpha=0.20.
**Open questions / next steps:**
- Eval placement is still ~4.5 (chance). With SE ~0.18 at n=128 it cannot yet resolve the ~0.2 improvement the free gap (averaged over ~3,300 games/segment) already shows, so this is not yet evidence of failure -- but it IS the criterion, and it has to move.
- Do NOT keep retuning at <10% of the run. Two restarts have already cost ~13k games each; the honest reading is that 13k games is simply small for this problem.
- If the free gap stalls above ~0.5 by update ~2500, the next lever to consider is `DENSE_REWARD_SCALE` (0.30 currently weakens the combat win/loss signal to +0.15/-0.09), not alpha again.
- `_train_parallel`'s docstring still wrongly describes `device` as the main-process device; it only sets the workers' device.
---

---
### 2026-09-01 — ROOT CAUSE: every minion in the game had attack=0/health=0, making combat outcomes constant and the task unlearnable
**Files changed:** `env/player_state.py`, `env/game_loop.py`, `agent/card_encoder.py`, `symbolic/board_computer.py`, `symbolic/firestone_client.py`, `symbolic/effect_handler.py`, `symbolic/hero_handler.py`, `symbolic/shop_analyzer.py`, `symbolic/combat_sim.py`, `env/triple_system.py`
**What was done:** After the SHAPE_ALPHA fix still left eval placement flat at chance through update 2000 (~52,000 games; eval slope +0.033/1000 updates, i.e. slightly WORSE), tested whether the reward was misaligned by correlating per-game reward against placement: `corr = -0.973`, `r^2 = 0.948`, actually BETTER than a pure placement reward (-0.970). That refuted the "dense shaping drowns the objective" hypothesis and forced the search lower in the stack.

Measured Φ_board directly against board size over real games instead of reasoning about it. Result: Φ was **identical to 4 decimal places (0.0323) for board sizes 1-6**, spanning only [0, 0.038] rather than the documented [0, 1].

Root cause: `bg_card_definitions.json` and `TavernPool` use `base_atk`/`base_hp`; `MinionState` declares `attack`/`health`; every dict→MinionState conversion read `d.get("attack", 0)`. **Every minion in the game was created 0/0.** The same bug was duplicated at FIVE conversion sites (`game_loop._dict_to_minion`, `effect_handler._dict_to_minion` — so all battlecry Discover/tribe-draw effects, `hero_handler._hp_galakrond`/`_hp_infinite_toki`, `triple_system.check_and_process_triple`), plus four consumers reading stats off ambiguous dicts (`card_encoder` feat[0]/[1], `board_computer._board_power`/`_compute_combat_stats`, `firestone_client._heuristic_estimate`, `shop_analyzer._estimate_card_power`).

Three consequences, all confirmed by measurement:
1. `_board_power` sums attack+health and returns `max(total, 1.0)` -- with all-zero stats it ALWAYS returned the 1.0 floor, so Φ = 1/(1+30) = 0.0323 constant. That floor is what disguised a hard zero as a plausible-looking number for multiple sessions.
2. `card_encoder`'s first two features (attack, health) were **always 0**, so the policy network could not distinguish a 1/1 from a 6/6.
3. **Worst:** `train.py`'s `_worker_run_game`/`_worker_run_eval_game` -- the functions real parallel training and eval actually dispatch -- hardcode `FirestoneClient(mock_mode=True)`, whose heuristic is a board-power ratio. With both boards at power 0 it returned `max(0.05, min(0.95, 0/1e-9))` = **0.05 constant for every combat in every game**, independent of anything the policy did. Combat outcomes were arbitrary, so placement was pinned at chance and the task was literally unlearnable. This invalidates the placement numbers from every prior session, not just this one.

Fix: one `minion_stats(d) -> (attack, health)` helper in `env/player_state.py`, disambiguating by KEY PRESENCE (not value, so a genuine 0-attack minion isn't misread), returning base stats only so `perm_*`/`game_*` bonuses are never double-counted. Applied at all 9 sites. Also fixed `_heuristic_estimate` to include buff bonuses (previously ignored, inconsistent with `_board_power`).

Verified independently of the subagent: `minion_stats` resolves both dict shapes; `_board_power` 84.0 for 7x6/6 vs 4.0 for 2x1/1; combat win_prob strong-vs-weak **0.95**, weak-vs-strong **0.05**, **mirror 0.50** (all three were 0.05 before). Φ vs board size: corr **0.243 -> 0.912**, Φ now 0.129 (1 minion) -> 0.558 (7 minions). Zero-stat board minions 96.4% -> 0%.
**Current state:** Fresh run launched on contract 49396123 in tmux `train`, logging to `training_statfix.log`. Reward constants deliberately UNCHANGED (`SHAPE_ALPHA=1.5`, `DENSE_REWARD_SCALE=0.30`) so this change can be attributed cleanly. Prior runs archived as `checkpoint_backups/statbug_*` and `checkpoint_backups/alpha020_*`.
**Open questions / next steps:**
- This is the first run in which the agent can actually see minion stats AND combat responds to board strength. Everything before it was measuring noise.
- Now that Φ spans ~0.13-0.56 instead of a constant 0.032, `SHAPE_ALPHA=1.5` gives ~0.15 shaped reward per minion placed -- close to the sizing intent. Re-check once real learning appears; it may now be too strong rather than too weak.
- `_worker_run_game`/`_worker_run_eval_game` hardcode `mock_mode=True`, ignoring the `use_firestone`/`--no-firestone` flag entirely (that flag only affects the sequential path). The mock heuristic is a power ratio, not real combat -- worth deciding explicitly whether training should use the real 200-trial sim now that stats are meaningful. O(1) vs MC cost tradeoff.
- `_board_power`'s `max(total, 1.0)` floor was kept as a division guard but is exactly what masked this bug for multiple sessions -- treat any suspiciously constant symbolic quantity as a data bug until proven otherwise.
- Methodological lesson: two full runs and two reward-tuning cycles were spent treating a DATA bug as a REWARD bug. The correlation test that refuted my own hypothesis is what redirected the search; measuring a quantity's actual empirical range (rather than trusting its documented range) should come before tuning any constant that multiplies it.
---

---
### 2026-09-01 — First run with real minion stats: agent decisively beats both scripted baselines
**Files changed:** none (results/teardown only)
**What was done:** Ran the stat-fixed build for 137 PPO updates / 3,562 games / 1.52M steps, then stopped and destroyed the vast.ai instance (contract 49396123) at the user's request. This was the first run in which the agent could see minion stats AND combat responded to board strength, so it is the first result in the project that measures anything real.

**Results.** `EVAL @ update 100: mean_placement=3.25, top1=0.49, top4=0.70` vs 7 fixed GreedyPlayAgents -- against 4.50 (pure chance) in every single prior run this session. top1=0.49 vs a 0.125 chance baseline. In-game placement by quarter (lower=better): train 3.790 -> 2.464 -> 2.531 -> 2.962 (best 2.46), heuristic 3.519 -> 5.179, greedy 5.554 -> 6.766. The trained agent now beats BOTH scripted baselines decisively.

Note that GreedyPlayAgent COLLAPSED from 3.68 (pre-fix) to 6.77 (post-fix). That is the expected consequence of the fix, not a regression: "buy everything, never sell" only looked strong while combat ignored board quality; once combat depends on real stats, greedy fills its board with junk and loses. Any cross-session comparison of baseline placement numbers is invalid across the stat fix.

Behavioural degeneracies resolved WITHOUT further reward tuning, which is the cleanest evidence the earlier churn was a symptom of the data bug rather than a reward-shaping problem: `board_size` 2.26-3.0 -> **5.30** minions at END_TURN, `sell:place` 0.87 -> **0.52** (peaked ~0.9 early then fell on its own). PPO health held throughout: `explained_var` ~0.8, `clip_frac` 0.15-0.25, and **skipped-update rate flat 0%** for the whole run (the NaN bug is fully closed).
**Current state:** Instance DESTROYED, billing stopped, 0 instances remaining. Artifacts pulled and verified locally before teardown: `checkpoint_backups/statfix_bg_agent_ppo.pt` (1,515,848 steps / 137 updates) and `_best.pt` (781,620 steps / 64 updates), both confirmed loadable; `data/fresh_training_history.json`; `data/training_progress.png`; `checkpoint_backups/training_statfix.log`. Earlier archived runs (`alpha020_*`, `statbug_*`) were left on the destroyed host deliberately -- they were all produced under the stat bug (constant 0.05 win_prob) and are scientifically worthless.
**Open questions / next steps:**
- **Game length is pinned at the cap.** All 3,562 games ran EXACTLY 40 rounds (min=max=40, zero variance). Games never resolve by elimination -- placement is decided at the round cap, not naturally. This is also the sole cause of the 8x throughput collapse (11.8 -> 1.3 games/sec; 178 -> 427 steps/game) and therefore of the 31h projected runtime. Investigate why nobody dies: likely combat damage is too low to eliminate players, or elimination isn't wired to the new win_prob. **Fix this before any long run** -- it triples cost and trains the agent in an unrepresentative regime.
- `ACTIVATE` rose to 29% of actions (was 3-7%). Could be legitimate gold usage or an exploit; check whether it is a once-per-turn-per-minion loop being farmed.
- `ANNEAL_STEPS=26_000_000` was sized for 178 steps/game; actual is ~427, so lr/entropy would hit their floor ~40% through a 150k-game run. Resize to ~64M (or fix game length first, which changes the number again).
- `_worker_run_game`/`_worker_run_eval_game` still hardcode `FirestoneClient(mock_mode=True)`, ignoring the `use_firestone` flag. The mock is a board-power ratio, not real combat. Now that stats are meaningful, decide explicitly whether training should use the real 200-trial sim.
- Reward constants (`SHAPE_ALPHA=1.5`, `DENSE_REWARD_SCALE=0.30`) were deliberately left untouched across the stat fix for clean attribution. With Φ now spanning ~0.13-0.56 instead of a constant 0.032, re-check whether 1.5 is now too STRONG rather than too weak.
---

---
### 2026-09-01 — Investigation: why games always hit the 40-round cap, and why ACTIVATE is spammed
**Files changed:** none (investigation only)
**What was done:** Two diagnostics requested after the first stat-fixed run.

**1. Games always run exactly 40 rounds.** Players DO die (`run_game` correctly breaks at `len(alive_players) <= 1`), just far too slowly: 4 of 8 dead by round 40, survivors sitting at 11-22 health. Root cause is the mock combat heuristic in `symbolic/firestone_client.py:_heuristic_estimate`:
`expected_damage_dealt = win_prob * 5.0`, `expected_damage_taken = loss_prob * 5.0`.
**Damage is capped at 5 and never scales with tavern tier or board strength.** Measured over 4 games: mean damage 1.12/combat, max 5, loss rate 46.1% (win 49.9%, tie 4.0%) -- about 1.1 health lost per round against 40 starting health, so natural elimination lands right at ~36 rounds. Total damage per game is 342 vs the 320 needed to wipe the field, which is why every game lands on the cap with zero variance. Real BG damage is `tavern_tier + sum of surviving enemy minion tiers` and ESCALATES (~2-5 early, 15-25 late); that escalation is what ends real games in 12-18 rounds. `step_combat` already has the right shape in its fallback (`max(1, ps.tavern_tier + len(opp.board))`) but it only fires when the rounded expected damage is 0.

**2. ACTIVATE is 92.5% no-ops -- a masking bug in the BATCHED path only.** Measured over 2 full games driven by PPOAgents (the real training path):

| action | attempts | valid pointer | no-op |
|---|---|---|---|
| BUY/SELL/PLACE/REROLL/FREEZE/LEVEL/HERO_PWR/END_TURN | 1379 | 1379 | 0.0% |
| ACTIVATE | 957 | 72 | **92.5%** |

**37.9% of ALL training actions are no-ops**, essentially all ACTIVATE. Root cause at `env/game_loop.py:1663-1667`: the batched path passes `build_pointer_mask(_s, -1)` -- the **type-agnostic full occupancy mask across all three zones** -- as the sampling mask to `get_action_batch`, then computes the CORRECT type-specific mask afterwards at line 1680 for storage. The sequential path (`game_loop.py:1559`, `ptr_mask = build_pointer_mask(ps, chosen_type)`) does it correctly, which is exactly why deterministic eval showed 5 activations across 3 games while training logs showed ACTIVATE at 29% of actions. ACTIVATE is uniquely affected because its valid set is tiny (minions with `activate_cost > 0`, unused this turn, affordable -- typically 0-1 of 7 slots) while occupancy marks every occupied slot in shop+board+hand; BUY/SELL/PLACE coincide with occupancy within their own zone so they measure 0% no-ops.

Why the policy *seeks* no-ops: a no-op costs no gold, changes no state, and does not end the turn, but it does advance one step of `gamma=0.997`. Spamming ~100 of them discounts the pending `_end_of_turn_reward` penalties (hand + unspent gold) by `0.997^100 ~= 0.74`. The agent learned to **stall in order to discount its own penalties** -- a classic discounting exploit. It also inflates step count (427/game), compounding the throughput collapse.

**Second, subtler consequence -- a PPO correctness bug.** The stored `log_prob` comes from `get_action_batch` under the OCCUPANCY mask, but the stored `pointer_mask` is the TYPE-SPECIFIC one. At update time `evaluate_actions` recomputes log-probs under the stored mask, so the importance ratio compares two different distributions even at zero policy change. For ACTIVATE that is roughly `log(1/1)` vs `log(1/15)`, a ratio near 15 that is hard-clipped every update. This biases PPO on every pointer-type action (BUY/SELL/PLACE/ACTIVATE ~= 60-70% of actions), not just ACTIVATE.

Also noted while reading: `Suspicious Prisonguard` is a **tier-1** card with `activate_cost=1` whose effect is a permanent +3/+3 -- 1 gold per turn into permanent stats, available from round 1. Over a 40-round game that is up to +120/+120 for 40 gold. Legitimate for the agent to favour, but it is a balance outlier that the over-long games amplify; worth revisiting after the round-length fix rather than before.
**Current state:** No code changed. Instance remains destroyed. Findings only.
**Open questions / next steps:**
- Fix the batched pointer mask first -- it is an unambiguous bug, it fixes a PPO correctness issue, and it removes 38% of wasted actions (which also speeds training). Requires either a two-pass sample (type first, then build the type-specific pointer mask, as the sequential path does) or extending `get_action_batch` to take per-type pointer masks. Whichever is chosen, the mask used for SAMPLING must be the same one STORED in the transition.
- Then fix damage scaling so games end naturally in ~12-18 rounds, which also restores most of the lost throughput.
- Re-check `SHAPE_ALPHA`/`DENSE_REWARD_SCALE` only AFTER both, since both change the action count and episode length that those constants were implicitly sized against.
- Consider whether the flat `_end_of_turn_reward` penalties should be restructured so stalling is never attractive even if a free no-op reappears (e.g. charge them per-round rather than at END_TURN).
---

---
### 2026-09-01 — Fix both throughput/correctness bugs: pointer masking (both paths) and combat damage scaling
**Files changed:** `agent/policy.py`, `env/game_loop.py`, `train.py`, `symbolic/firestone_client.py`

**Fix 1 — pointer mask, made correct by construction.** Both action-selection paths sampled the pointer under the type-AGNOSTIC occupancy mask (`build_pointer_mask(ps, -1)`) and then separately recomputed the type-SPECIFIC mask for storage. Design applied to both `get_action_batch` and `get_action`: an optional `ptr_mask_fn` callable invoked AFTER the type is sampled, and — critically — the method now RETURNS the mask it actually used, which the caller stores. "Sampled mask == stored mask" is now structurally guaranteed rather than a convention two call sites must independently maintain (maintaining it by convention is exactly how this bug arose). `get_action` went from a 4-tuple to a 5-tuple; all three callers updated (`PPOAgent`, `StaticAgent`, `EvalAgent` in train.py).

**Important process note: the first attempt fixed the wrong path.** The batched-path agent asserted real training uses the batched path because `_agents_support_batching` is True when every agent is a PPOAgent/StaticAgent. That is wrong: the real composition is 2 PPOAgent + 2 StaticAgent + `N_HEURISTIC_SLOTS=2` + `N_GREEDY_SLOTS=2`, and HeuristicAgent/GreedyPlayAgent both set `supports_batching = False` (train.py ~367, ~451), so real games run the SEQUENTIAL path until enough scripted seats are eliminated. Its verification looked clean only because it tested an all-PPOAgent game -- exercising the path it had just fixed rather than the one training uses. Caught by running an independent joint verification on the REAL seat mix, which still showed 88.6% ACTIVATE no-ops. **Lesson: verify on the real configuration, not a convenient one; a green check on the wrong population is worse than no check.**

**PPO impact was real but with a different mechanism than first assumed.** In the sequential path `PPOTrainer.collect_transition` recomputes `log_prob` via `evaluate_actions` under the stored mask, so stored-log_prob/stored-mask are tautologically consistent -- meaning the naive consistency check passes vacuously. The actual damage is that the ACTION was drawn from a broader distribution than the one stored, so the taken action often had ~zero probability under its own stored mask. Measured by degenerate-log_prob rate (`stored log_prob < -15`) on pointer-type transitions: **66.0% -> 0.0%**. In the batched path the mechanism was the one originally described (stored log_prob computed under a different mask): `max |diff| = inf` with 10 mismatched rows -> 0 mismatches across 2089 transitions.

**Fix 2 — combat damage scaling.** `_heuristic_estimate` returned a flat `win_prob * 5.0` / `loss_prob * 5.0`, capped at 5 and never scaling, so nobody died and every game ran out the 40-round clock. Replaced with the real BG formula -- winner's own tavern tier + tier-sum of the winner's surviving minions -- verified against `symbolic/combat_sim.py`'s own `CombatSide.win_damage`. Since the heuristic cannot simulate survivors it scales the winner's pre-combat tier-sum by a dominance fraction derived from win/loss prob (`MIN_SURVIVE_FRAC=0.15`, `MAX_SURVIVE_FRAC=1.0`, `DOMINANCE_RANGE=0.45`). `player_tier`/`opp_tier` are now threaded from `simulate()` (they were received but silently dropped). Attribution is commented explicitly at both sites: `expected_damage_dealt` uses the PLAYER's tier/board, `expected_damage_taken` the OPPONENT's.

**Joint verification on the real training mix (4 PPOAgent + 2 Heuristic + 2 Greedy, 8 games, statfix checkpoint), run independently of the subagents:**

| metric | before | after |
|---|---|---|
| ACTIVATE no-op rate | 88.6% | **0.0%** |
| TOTAL no-op rate | 37.9% | **0.00%** |
| game rounds | 40/40 every game | **15-25, mean 20.5, std 3.4** |
| games ending before the cap | 0/8 | **8/8** |
| actions/game | ~1035 | **409** |
| placements span 1-8 | yes | yes |

Smoke-tested the real parallel training path end-to-end afterwards (spawn workers, PPO updates firing, no NaN, `evaluate_policy` working at n_workers>1).
**Current state:** All three fixes committed. No instance running.
**Open questions / next steps:**
- **The scripted baselines are structurally crippled, which inflates how good our results look.** `GreedyPlayAgent` contains NO LEVEL_UP logic at all -- it sits at tavern tier 1 for an entire game -- and `HeuristicAgent` caps at `tavern_tier < 4`. This is why damage plateaus ~7 instead of 15-25 and games land at ~20 rounds instead of the real-BG 12-18 (with an aggressively-levelling agent the same formula gives mean 18.5 rounds and clean 3.19 -> 9.92 -> 11.26 escalation). **It also means the headline `EVAL 3.25 vs 7 greedy` result is measured against opponents permanently stuck at tier 1** -- before the stat fix that handicap was invisible (all minions were 0/0 so tier was irrelevant), which is exactly why greedy scored 3.68 then and collapsed to 6.77 after. The 4.5 -> 3.25 improvement against a fixed opponent is genuine learning, but the absolute bar is low. Give GreedyPlayAgent/HeuristicAgent real levelling before trusting any absolute placement claim.
- Reward constants (`SHAPE_ALPHA=1.5`, `DENSE_REWARD_SCALE=0.30`) were sized against the OLD regime of 427-step, 40-round games. Episode length has roughly halved and 38% of actions disappeared, so both should be re-measured (not re-guessed) before the next long run.
- `ANNEAL_STEPS` needs resizing again: at ~409 actions/game the old 178-steps/game assumption and the later 427 estimate are both wrong.
- Throughput should recover substantially (2.5x fewer steps per game); re-benchmark `N_WORKERS`/`UPDATE_INTERVAL` on the next rented host rather than reusing the 26/26 tuned for the old regime.
---

---
### 2026-09-01 — Re-measure reward balance and ANNEAL_STEPS after the masking/damage fixes (no changes needed)
**Files changed:** none (measurement only)
**What was done:** The previous entry flagged that `SHAPE_ALPHA`, `DENSE_REWARD_SCALE` and `ANNEAL_STEPS` had all been sized against the OLD regime (40-round games, ~38% no-op actions) and needed re-checking after the pointer-mask and damage fixes roughly halved episode length. Measured rather than re-guessed. **Conclusion: leave all three as they are.**

**Reward balance** (8 games, real seat mix, statfix checkpoint, per player-game):
| term | mean abs |
|---|---|
| potential shaping | 0.314 |
| end-of-turn dense | 1.674 |
| final placement | 2.125 |

`shaping/placement = 0.148` -- still inside the 0.1-0.3 "densify, don't swamp" band it was sized for (it was 0.139 in the old regime), so `SHAPE_ALPHA=1.5` remains correct. `dense/placement = 0.788` -- dense terms are meaningful but not dominating, so `DENSE_REWARD_SCALE=0.30` is fine too.

**ANNEAL_STEPS -- the interesting one.** `progress = total_steps / anneal_steps` uses `PPOTrainer.total_steps`, which counts transitions from the **2 training seats only**, NOT shopping actions across all 8 seats. Earlier notes in this log quoted 178/427 steps-per-game figures that conflated the two. Measured properly via `_train_parallel` with the real seat composition:

| policy | steps/game | rounds/game | implied ANNEAL_STEPS @150k games |
|---|---|---|---|
| fresh/untrained | 84.3 | 29.4 | 12.6M |
| trained (statfix ckpt) | 247.8 | 21.7 | 37.2M |

Steps/game is strongly **policy-dependent and rises as the agent improves**: a weak policy is eliminated early and stops collecting transitions, while a stronger one survives more rounds per game (note games get SHORTER, 29.4 -> 21.7 rounds, while transitions per game nearly TRIPLE). So the correct value for a 150k-game run lies somewhere in 12.6M-37M depending on how strong the policy becomes, and the current **26,000,000 sits squarely in that range** -- the anneal will complete somewhere between ~70% and ~140% of the run, which is acceptable. **Acting on the untrained measurement alone would have set it to 12.6M, which would have been wrong** -- the first number measured was from the least representative policy.
**Current state:** No code changed; all three constants confirmed appropriate for the new regime. HEAD is `080c345`. No vast.ai instance running.
**Open questions / next steps:**
- Unchanged and still the most important: **the scripted baselines cannot level** (`GreedyPlayAgent` has no LEVEL_UP logic at all and stays at tavern tier 1; `HeuristicAgent` caps at `tavern_tier < 4`). This caps late-game damage at ~7 instead of 15-25, holds games at ~20-22 rounds instead of the real-BG 12-18, and -- most importantly -- means the `EVAL 3.25 vs 7 greedy` headline is measured against permanently tier-1 opponents. Fix the baselines before trusting any absolute placement claim.
- When re-measuring steps/game in future, always use `PPOTrainer.total_steps` under the real seat mix, and measure with a policy of representative strength -- the quantity roughly triples between an untrained and a trained policy.
- Re-benchmark `N_WORKERS`/`UPDATE_INTERVAL` on the next rented host; 26/26 was tuned for the old 427-step regime and throughput should now be substantially better (~2.5x fewer steps per game).
---

---
### 2026-09-01 — Give scripted baselines real levelling; the 3.25 headline was mostly an artifact of tier-1 opponents
**Files changed:** `train.py`, `env/game_loop.py`

**What was done.** `GreedyPlayAgent` had NO LEVEL_UP logic at all (permanently tavern tier 1) and `HeuristicAgent` hard-capped at `tavern_tier < 4`. Both now level on a shared curve. Extracted `expected_tier_for_round(round_num)` to a module-level function in `env/game_loop.py` (the `BattlegroundsGame` method now delegates to it) so there is exactly ONE definition, imported by train.py rather than copy-pasted. Added `_scripted_should_level(ps)` shared by both agents: level only when genuinely behind curve, below the tier-6 cap, AND `len(ps.board) >= SCRIPTED_MIN_BOARD_TO_LEVEL (=3)` -- the board guard exists because naive levelling can produce an agent that pours all its gold into tiers and fields an empty board, which would be WORSE than the tier-1 version. Each agent's distinguishing behaviour is preserved: greedy still buys first-affordable left-to-right and never sells except for a strictly higher-tier upgrade; heuristic still buys highest-tier and sells its weakest when full.

**Verification (12 games, 4 Heuristic/4 Greedy population, before via `git stash`):**
| metric | before | after |
|---|---|---|
| tavern tier — heuristic | 4 / 4 | **6 / 6** |
| tavern tier — greedy | 1 / 1 | **6 / 6** |
| board size — heuristic | 7.0 | 7.0 |
| board size — greedy | 7.0 | 6.75 |
| n_rounds | 29.0 (20-37) | **19.75 (16-25)** |
| damage 1-5 / 6-10 / 11-15 / 16+ | 3.04 / 7.13 / 7.05 / 7.24 (flat) | **2.97 / 8.52 / 14.06 / 14.87 (rising)** |

Board size did not collapse, confirming the guard works. Damage now escalates into the real-BG 15-25 band, confirming the earlier plateau was the opponents and not the damage formula. Head-to-head, 4 new-style vs 4 old-style agents: new **3.67** vs old **5.33** mean placement -- decisive, which was the load-bearing acceptance test.

**The important consequence -- our headline result was flattered.** Re-evaluated the `statfix` checkpoint (64 games each, deterministic, independent of the subagent):
| opponent | mean placement | top1 |
|---|---|---|
| 7 greedy (OLD, tier-1 forever) | 3.25 | 0.49 |
| 7 greedy (NEW, levels to 6) | **4.406** | 0.141 |
| 7 heuristic (NEW) | 3.047 | 0.469 |

**Against a competent greedy the agent is at 4.41, barely better than chance (4.5).** The 3.25 headline was largely an artifact of opponents permanently stuck at tier 1. Fair caveat: this checkpoint was also TRAINED against those crippled baselines (2 heuristic + 2 greedy seats), so evaluating it against competent ones is partly distribution shift rather than purely "the agent is weak" -- but the benchmark is honest now either way. Note the new greedy is now the STRONGER of the two baselines (big board + on-curve tier beats heuristic's sell-weakest behaviour in this sim), which inverts their former ordering.
**Current state:** All baseline changes committed. No instance running. **Every placement number recorded in this log before this entry is measured against the old crippled baselines and is NOT comparable to anything measured after it.**
**Open questions / next steps:**
- A fresh training run is now warranted: it would be the first trained AND evaluated against competent opponents, on top of correct minion stats, working combat, correct action masking and escalating damage.
- Game length landed at 19.75 rounds vs the 12-18 real-BG target. Remaining limiter is the tier-6 cap plus starting-health totals in mock combat; not tuned further deliberately.
- Pre-existing (NOT introduced by this change, verified): `GreedyPlayAgent` can oscillate sell->buy and hit the `max_actions=30` per-turn cap in ~0.4-0.5% of turns, because it sells its weakest minion expecting to buy the shop's higher-tier "upgrade" but its buy rule then picks the first AFFORDABLE card instead. Harmless today (the engine drops the unfinished turn gracefully) but worth fixing if greedy is ever used as a serious reference.
- `evaluate_policy`'s default `opponent="greedy"` now measures against the stronger of the two baselines, which is the right default.
---

---
### 2026-09-01 — Throughput investigation: vast.ai "cores" are hyperthreads; rolling dispatch; migrate to a 48-physical-core host
**Files changed:** `train.py`, `run_fresh_training.py`, `.claude/skills/vast-ai-training/SKILL.md`
**What was done:** Started as "are we using the instance well?" and turned into three findings, two fixes and a migration.

**1. The GPU is idle BY DESIGN — not a tunable.** `WORKER_DEVICE='cpu'`: all self-play workers run policy inference on CPU; the GPU is touched only for brief PPO updates. Self-play is >90% of wall-clock, so the project has been paying for a GPU that does almost nothing. This is correct for a d_model=256/4-layer net doing single-sample inference, but it means GPU choice is nearly irrelevant to throughput and CPU is everything.

**2. THE BIG ONE — `cpu_cores` counts HYPERTHREADS, not physical cores.** The old host (contract 49528923) advertised 32 `cpu_cores`, reported `cpu_cores_effective=32` (ratio 1.0, genuinely dedicated) and a cgroup quota of 30.7 — but `lscpu` showed an `Intel Xeon E5-2697A v4` = **16 physical cores**, 2 threads/core. Nothing in the vast.ai listing or in `/sys/fs/cgroup/cpu.max` revealed this, and the skill file explicitly told us `cpu.max` was "the ONLY number to trust" (now corrected). Measured sweep on that box, which is textbook for the shape:

| N_WORKERS | pre-refactor | post-refactor |
|---|---|---|
| 12 | — | 1.56 |
| 16 | 1.94 | 1.96 |
| 20 | 1.96 | 2.03 |
| 26 (was live) | 2.01 | 2.16 |
| 30 | 2.19 | 2.36 |
| 36 / 44 | — | 2.27 / 2.36 (plateau, ~5% run-to-run noise) |

Near-linear to one worker per PHYSICAL core, then a ~10-20% hyperthreading tail, then flat. `N_WORKERS=26` was not "tuned" so much as oversubscribing hyperthreads to mask barrier stalls.

**This also retires the 8.64 games/sec figure as a target.** It was measured on a DIFFERENT host AND under the pre-fix engine (stat bug -> boards of 2.3-3.0 minions, baselines stuck at tier 1). With correct stats, working combat damage and levelling baselines, boards now run 5.3-7.0 minions vs tier-6 opponents, so each round costs far more even though games are shorter (~20 rounds vs 40). **The simulation got heavier when it got correct.** Do not cite 8.64 again.

**3. Fixed a synchronization barrier in `_train_parallel`.** It dispatched games in cohorts of `n_workers` via `pool.map` and blocked until ALL returned — every worker that finished early idled until the cohort's straggler finished, and game length varies (15-25 rounds, std ~3.4). Replaced with rolling dispatch (`pool.submit` + `wait(FIRST_COMPLETED)`, keeping N in flight and refilling a slot the instant one frees). Effect: CPU utilization **72-80% -> 93-94%**, throughput **+8%** at 26-30 workers (gain grows with worker count, as a straggler penalty should). Preserved: the spawn `mp_context`, update/checkpoint boundary arithmetic, `on_batch` grouping (now completion-order), and the `sd`/`sd_stale` reclone. Non-obvious trap handled explicitly: converting `pool.map(timeout=)` to `wait(FIRST_COMPLETED)` silently loses hang detection (as long as some other worker finishes, a permanently-hung task is never noticed and leaks a slot forever), so each future carries a submit timestamp and is age-checked every poll.

**Ruled out along the way, by measurement rather than argument:** torch thread oversubscription (`_worker_init` already sets `set_num_threads(1)`); state-dict pickling per task (13.89MB but only 16.5ms => ~3% of a core at 2 games/sec); and main-process serialization — measured **0.11 cores for the main process vs 28.3 cores aggregate for workers**, i.e. cleanly compute-bound with no serial ceiling, which is what justified scaling out. Note `ps aux` %CPU is a LIFETIME AVERAGE and read 78% for that same main process — it was inflated by earlier GPU-update and eval phases and nearly sent the investigation down the wrong path. Use `/proc/<pid>/stat` deltas for instantaneous load.

**4. Found (and shipped) an unreachable-dead-code bug in `HeuristicAgent`.** Local `train.py` had UNCOMMITTED work that never got committed or deployed — the deployed tree matched HEAD `5229507` exactly. Verified statically why it matters: HEAD's sell branch is gated on `valid(1) and len(ps.board) >= 7 and valid(0)`, but the buy branch ABOVE it returns whenever `valid(0)` and the shop holds any minion — so **the sell branch was unreachable and the heuristic baseline never sold**, and it bought before placing, clogging its hand once the board filled. Same category as last session's tier-1 levelling bug: a baseline weaker than intended, which flatters the agent's numbers. The WIP reorders to Level -> Place -> Sell -> Buy, sells only when the shop is strictly better, and ranks by EFFECTIVE (buffed) power via `_effective_power` rather than base stats (a minion sitting on +30/+30 read as its base 1/1 and got nominated as "weakest").

**5. Migrated hosts and restarted from scratch.** Old contract 49528923 destroyed. New: **contract 49546957, Xeon Gold 5418Y (2 sockets x 24 = 48 PHYSICAL cores), RTX 5000 Ada 32GB, 251GB RAM, $0.401/hr**, quota 92.16. Verified with `lscpu` BEFORE deploying. Measured sweep there (trained checkpoint, per the "measure with a representative-strength policy" lesson):

| N_WORKERS | games/sec |
|---|---|
| 24 | 3.97 |
| 48 | 4.61 |
| 64 | 4.79 |
| 80 | 4.89 |
| 90 | **5.03** |

**~2.2x the old host** (not the 3x that 3x-the-cores predicted). Per-worker throughput is ~35% better (newer core), but scaling is sub-linear well below 48 physical cores — 24->48 workers buys only +16%. Most likely all-core clock throttling (2.0GHz base / 3.8GHz boost) plus dual-socket NUMA, not a software bottleneck. Config now `N_WORKERS=90`, `UPDATE_INTERVAL=90` (invariant preserved; ~22.5k transitions/update vs ~6.5k, so ~1,667 updates over the run instead of 5,769), `EVAL_WORKERS=48`. `N_GAMES`, `ANNEAL_STEPS`, `EVAL_EVERY`, `EVAL_N_GAMES` deliberately unchanged.
**Current state:** Fresh run launched on contract 49546957 in tmux `train`, logging to `training_48core.log`, N_GAMES=150000 from scratch (no checkpoint present at launch — confirmed). Both fixes verified ACTIVE in the running process via `inspect.getsource`. At update 12 / 1,080 games health is normal for a fresh init and improving: `explained_var` 0.000->0.164, `clip_frac` 0.572->0.342, `avg10` -3.97->-3.27, and steps/game ~86.6 (matches the documented ~84.3 for an untrained policy — throughput will FALL toward the benchmarked 5.03 as the policy strengthens and survives longer). Previous run archived and verified loadable as `checkpoint_backups/cleanrun_*` (2,237,567 steps / 482 updates; best 1,691,362 / 348). Only one instance billing.
**Open questions / next steps:**
- **Credit is tight: $5.14 with the run projected at ~8.5h / ~$3.40.** It fits, but with little margin — top up before relying on the full 150k games plus any re-runs.
- Sub-linear scaling above ~24 workers on the new host is unexplained (all-core clock throttle vs. dual-socket NUMA). Worth one experiment with `numactl` pinning workers per socket before assuming 48-core boxes are inherently only ~2x.
- `ACTIVATE` is 0-1% of actions in the fresh run vs 8-12% in the previous one. Most likely just a fresh policy with a near-empty board (few minions with `activate_cost>0` to act on), but confirm it rises as boards fill — if it stays ~0 there may be a masking regression hiding.
- `clip_frac` opened very high (0.57) and is falling; normal for a random init, but if it doesn't settle under ~0.3 the 3.5x larger PPO batch from `UPDATE_INTERVAL=90` may want a smaller `clip_eps` or lr.
- This run is the first trained AND evaluated against baselines that both level AND sell correctly, so its placement numbers are not comparable to ANY earlier entry in this log.
- `.claude/check_context_log.sh` and `.claude/settings.json` carry uncommitted changes that predate this session and were left untouched.
---

---
### 2026-09-01 (addendum) — Post-launch: `queue_factor` fix, and a correction to the "no serial ceiling" claim
**Files changed:** `train.py`
**What was done:** Measuring the live run after launch exposed something the earlier benchmark structurally could not see.

**Correction to the previous entry.** It records "main 0.11 cores vs workers 28.3 cores => cleanly compute-bound with no serial ceiling." Two caveats: (a) that measurement came from `benchmark_workers.py`, which ran with `update_interval=10**9` — i.e. **PPO updates disabled** — so it measured dispatch overhead only and excluded update cost entirely; (b) a later attempt to re-measure on the live run reported "main = 0.00 cores", which was simply the WRONG PID (it matched the `tmux` wrapper, not the python process). The real main process was at **326% CPU (3.26 cores)**. Lesson, twice over in one session: verify the PID you are sampling, and never generalize a benchmark's resource profile to production when the benchmark deliberately disabled a major phase of the work.

**The real finding: the box was 46% idle.** With `N_WORKERS=90` / `UPDATE_INTERVAL=90`, workers held only 44.0 cores and main 3.3, i.e. ~47 of 92. The GPU was busy in 23% of samples and per-group wall-times swung 3.2s / 11.5s / 15.3s for the same 90 games. Cause: while `ppo_trainer.update()` runs (now a 22.5k-transition update, ~3.5x bigger after `UPDATE_INTERVAL` 26->90), the main loop stops harvesting completed futures and refilling slots, so workers finish and idle. Rolling dispatch fixed the *cohort barrier* but the pipeline still drains whenever the main process blocks.

**Fix: `queue_factor` (default 2.0) in `_train_parallel`.** Keep `ceil(n_workers * queue_factor)` tasks SUBMITTED rather than exactly `n_workers`. `ProcessPoolExecutor` queues submitted tasks internally and a worker pulls the next one the instant it finishes, with no main-process involvement — so a backlog keeps workers fed straight through a main-process stall. `queue_factor=1.0` reproduces the old behavior exactly. Two subtleties handled: the stall age-check now includes queue wait time so its threshold is scaled by `queue_factor`; and the error/timeout rebuild path re-primes to the in-flight target rather than `n_workers`. Documented tradeoff: a queued game runs with the weights current at submit time, so at `queue_factor=2.0` with `UPDATE_INTERVAL == n_workers` it is at most ~1 update stale — normal off-policyness for PPO.

**Result: partial.** Workers went **44.0 -> 58.5 cores (+33%)**, but vmstat idle only moved **46.0% -> 44.5%**. So the update stall was real but is NOT the whole story, and roughly 45% of the box is still unexplained idle. Stopped optimizing there rather than keep restarting a healthy run for diminishing returns.
**Current state:** Run resumed (not fresh-restarted — this was a scheduling change with no effect on learning semantics) from the update-143 checkpoint and is healthy: `steps` continued at 2,152,269, `best_avg10=1.799` preserved. **First honest eval landed: `EVAL @ update 100: mean_placement=3.95, top1=0.27, top4=0.62` vs 128 games of competent greedy** (chance = 4.50 / 0.125). For reference the previous session's `statfix` checkpoint scored 4.406 against this same levelling greedy, so ~9,000 games into a from-scratch run this build has already passed it. NOTE: because this was a resume, `_train_parallel`'s game counter restarted at 0, so the run will play 150,000 MORE games from that point; `ANNEAL_STEPS` is driven by `total_steps` which persisted, so annealing is unaffected.
**Open questions / next steps:**
- **~45% of the box is still idle and unexplained.** Candidates: the update stall is longer than the 1-game backlog covers (try `queue_factor=3.0`); per-game fixed overhead (every task ships a **13.89MB** policy state dict that each worker must unpickle, ~9.4ms, plus per-game component construction) which bites hardest while games are short; or memory-bandwidth/NUMA limits on this dual-socket box. Measure before changing anything.
- **Highest-value next optimization: stop shipping the state dict with every task.** Have workers cache weights keyed by a version counter and send `None` when unchanged — with `UPDATE_INTERVAL=90` the weights change once per 90 games, a ~90x reduction in state-dict traffic (currently ~125MB/s of pickling at 9 games/sec).
- Sub-linear scaling above ~24 workers on this host is still unexplained; try `numactl` socket pinning before assuming 48-core boxes are only ~2x.
- Credit was $5.14 at migration. Watch it — top up before relying on a long run.
- `benchmark_workers.py` exists only on the remote host and measures self-play with updates DISABLED. If it is ever reused, either enable updates or state the exclusion prominently, since that exclusion is what hid this bug.
---

---
### 2026-09-01 (close) — Instance shut down; the 45%-idle question is still OPEN
**Files changed:** none (state correction + findings only)
**What was done:** Corrects the two entries above, whose "Current state" lines claim a run is live on contract 49546957. **That is no longer true — the user shut the instance down.** No instances running, nothing billing. Credit went $5.14 -> $4.75 across the whole session (both hosts).

**What was lost and what survived.** The run's WEIGHTS are gone: the background monitor synced `data/training_progress.png` and `data/fresh_training_history.json` every few minutes but NOT `bg_agent_ppo.pt`, so the from-scratch lineage trained on the 48-core host (through ~205 PPO updates) cannot be resumed. Only ~30 minutes of training, so the cost is small, but **the process gap is real: the monitoring recipe in the vast-ai-training skill syncs the chart only, and it was followed without adding checkpoint sync.** Skill updated accordingly. Surviving locally: `data/fresh_training_history.json` (18,270 games of reward history, 205 PPO update records) and the chart. `checkpoint_backups/cleanrun_*` (the PRE-migration run, 2,237,567 steps / 482 updates) is intact and verified loadable — it is now the newest usable checkpoint, but note it was trained against the OLD HeuristicAgent that never sold.

**The one result worth keeping from the lost run:** `EVAL @ update 100: mean_placement=3.953, top1=0.273, top4=0.62` over 128 games vs competent (levelling + selling) greedy, chance being 4.50 / 0.125. The previous session's `statfix` checkpoint scored 4.406 against that same baseline, so ~9,000 games into a from-scratch run this build had already passed it. That is the first number in this project measured against baselines that both level AND sell correctly.

**The 45%-idle question — still unresolved, do not treat the earlier entries as having answered it.** The anchor fact: the OLD 16-core host reached **93-94%** CPU after the rolling-dispatch fix, while the new 48-core host sat at **~55%** running the same code. Four things changed simultaneously (3x cores, 30->90 workers, a 3.5x larger PPO update from `UPDATE_INTERVAL` 26->90, and ~4x the games/sec through one main process), so nothing is cleanly attributed.
- **Leading hypothesis — per-game fixed cost scaling with throughput.** Every task ships the full **13.89MB** policy state dict: 32MB/s at the old host's 2.3 games/sec, but **125MB/s** at the new host's 9.0 games/sec, plus ~1,500 transitions/sec unpickled and pushed through `buffer.add()`. `ProcessPoolExecutor` does all of this in TWO single threads in the main process (the `multiprocessing.Queue` feeder pickles outgoing tasks; the manager thread unpickles results), both largely GIL-bound. Main measured 3.26 cores, but the GIL-bound share cannot exceed ~1 core regardless of thread count — the remainder is GPU/memcpy time that releases the GIL. Saturating those threads starves workers, which shows up as idle. Predicts the effect worsens as throughput rises, which matches.
- **`queue_factor` was real but not dominant.** It lifted workers 44.0 -> 58.5 cores (+33%), so update-stall starvation existed — but idle barely moved, so it was not the main term. The earlier entry implied a completed diagnosis; it was partial.
- **A measurement to distrust.** The "44.5% idle after the fix" reading was taken shortly after a restart, while 90 workers were still spawning and importing torch. Contaminated. The trustworthy figure is the 46% measured at update 68 on a warm run.
- **Probably NOT the cause:** NUMA / memory bandwidth. 90 processes each with a ~14MB weight working set on a dual-socket box certainly slows workers, and is the better explanation for the **sub-linear scaling** (24->48 workers bought only +16%) — but a memory-stalled core still counts as BUSY, so it cannot produce idle.
- **The experiment that would settle it (~10 min, do this first next time):** instrument `_train_parallel` to accumulate wall-clock into three buckets — blocked in `wait()`, merging results, and inside `ppo_trainer.update()` — and sample the main process PER-THREAD via `/proc/<pid>/task/*/stat`. If one thread pins near 1.00 core, the GIL/serialization hypothesis is confirmed and the state-dict fix below is the answer. This attributes time directly instead of inferring it from aggregate idle.

**Measurement lessons this session kept re-teaching (all three bit at least once):**
1. `ps aux` %CPU is a LIFETIME AVERAGE, not current load — it read 78% for a process using 0.11 cores, and was also used for the unreliable "58.5 cores" worker figure. Use `/proc/<pid>/stat` deltas.
2. **Verify the PID you sample.** A `pgrep -f 'python3 -u run_fresh_training.py'` matched the *tmux* wrapper whose command string contained that text, reporting 0.00 cores for a process actually at 326%.
3. **Never generalize a benchmark's resource profile to production when the benchmark disabled a phase of the work.** `benchmark_workers.py` ran with `update_interval=10**9` (PPO updates OFF). That is what produced the "main 0.11 cores, no serial ceiling" claim and is precisely what hid the idle-core problem.
**Current state:** No instances running, nothing billing, credit $4.75. HEAD has the rolling-dispatch refactor, `queue_factor`, the HeuristicAgent sell/ordering fix, and config tuned for a 48-physical-core host (`N_WORKERS`/`UPDATE_INTERVAL=90`, `EVAL_WORKERS=48`) — **all of which are committed but now UNVALIDATED at length, since the run that would have validated them was only ~30 minutes long.**
**Open questions / next steps:**
- **Before any new long run, sync checkpoints as well as charts.** The monitor loop must rsync `bg_agent_ppo.pt`/`_best.pt` periodically, not just the PNG and history JSON.
- **Highest-value optimization remains: stop shipping the 13.89MB state dict with every task.** Workers should cache weights keyed by a version counter, with the task sending `None` when unchanged — at `UPDATE_INTERVAL=90` weights change once per 90 games, roughly a 90x cut in state-dict traffic. This is also the direct test of the leading idle hypothesis.
- On re-rent, `N_WORKERS=90`/`UPDATE_INTERVAL=90` were benchmarked with updates DISABLED and are therefore not validated end-to-end. Consider whether a smaller `UPDATE_INTERVAL` (shorter stalls) beats the larger PPO batch, and re-measure with updates ON.
- Sub-linear scaling above ~24 workers is still unexplained; try `numactl` socket pinning before assuming 48-core boxes only deliver ~2x.
- The next run restarts from either scratch or `checkpoint_backups/cleanrun_*` — but that checkpoint predates the HeuristicAgent fix, so resuming from it means the agent's early training saw a baseline that never sold. A from-scratch run is the cleaner lineage.
---

---
### 2026-09-01 (relaunch) — Weights broadcast kills main-process serialization; new run on an i9-14900KF
**Files changed:** `train.py`, `run_fresh_training.py`
**What was done:** Implemented the two things the previous entry flagged as next steps, then relaunched.

**1. Per-phase instrumentation (measure before tuning).** `_train_parallel` now accumulates wall-clock into four buckets — `wait` (blocked in `concurrent.futures.wait`), `dispatch` (task construction + `pool.submit`), `merge` (result unpickle + buffer merge + stats + log), `update` (inside `ppo_trainer.update()`) — and logs one `phase:` line per `on_batch` boundary. This exists because aggregate CPU idle told us the box was half empty but never WHICH phase serialized.

**2. Weights are broadcast, not unicast — and a correction to the previous entry's recommendation.** That entry proposed caching weights in workers by version for "a ~90x reduction in state-dict traffic". **That reasoning was wrong:** with `UPDATE_INTERVAL == N_WORKERS` each worker handles only ~1 game per weight version, so a version cache would have hit ~never. The real inefficiency was that the main process UNICAST a 13.89MB `state_dict` into all N tasks per update (~1.26GB per update at N=90), pickled in ProcessPoolExecutor's single GIL-bound queue-feeder thread. Fix: write the weights ONCE per version to `/dev/shm` (atomic `torch.save` to `.tmp` + `os.replace`), and have tasks carry only `(path, version)`. **Measured task payload: 13,887,728 -> 92 bytes.** Retention keeps the last 3 versions because the `queue_factor` backlog means an in-flight task can reference a superseded version; a worker that cannot load its version raises rather than silently training on wrong weights. A one-entry worker cache is kept only to make the occasional repeat free — it is explicitly NOT where the win comes from.

**It worked, and the instrumentation proves it.** On the new host the `phase:` lines read `dispatch=0.0% merge=0.0-0.3%` — main-process serialization is no longer measurable. The remaining split is `wait` 68-90% (workers doing useful parallel work) and `update` 9-31%. Note this does NOT retroactively confirm the old 45%-idle diagnosis, since the host also changed; it does confirm that main-process serialization is no longer a candidate going forward.

**3. New host — and the divide-by-two heuristic is ALSO wrong.** Contract **49567627: Intel Core i9-14900KF, $0.108/hr**. vast.ai reports `cpu_cores=32`, and the "physical = cpu_cores/2" rule from the earlier entry would say 16 — but this is a HYBRID chip (8 P-cores with HT + 16 E-cores without), and `lscpu` reports `Core(s) per socket: 24`. **Only `lscpu` (Core(s) per socket x Socket(s)) is reliable; neither `cpu_cores` nor halving it is.** Single socket (so no dual-socket NUMA penalty this time), 6.0GHz boost, 31GB RAM, RTX 4060 Ti 8GB.

**Benchmark re-run WITH PPO UPDATES ENABLED** (`update_interval == n_workers`, production-equivalent) — the previous sweep used `update_interval=10**9` and was therefore not comparable to a real run:

| N_WORKERS | games/sec (updates ON) |
|---|---|
| 16 | 3.13 |
| 24 | **3.18** (one per physical core) |
| 32 | 2.81 (past physical cores; also ~0.5GB/worker against 31GB RAM) |

**Value comparison:** the 48-physical-core Xeon Gold host measured 5.03 games/sec but with updates OFF (realistically ~3.9 with them on) at $0.401/hr, versus 3.18 at $0.108/hr here — roughly **3x better throughput per dollar despite half the physical cores**, because these cores are far faster. Config set to `N_WORKERS=24`, `UPDATE_INTERVAL=24` (~6k transitions/update, back in the previously-validated PPO batch regime, 6,250 updates across the run), `EVAL_WORKERS=24`. `N_GAMES`, `ANNEAL_STEPS`, `EVAL_EVERY`, `EVAL_N_GAMES` unchanged.
**Current state:** Fresh from-scratch run live on contract 49567627 in tmux `train`, logging to `training_i9.log`, no checkpoint present at launch (confirmed). Healthy through the first 6 updates: `best_avg10=-3.669` (normal random init), batches 1.8-2.4s for 24 games. Credit was $4.75 at launch; at $0.108/hr that is ~44h of runway, versus ~11h on the previous host. **The monitor now syncs the CHECKPOINT (to `checkpoint_backups/live_bg_agent_ppo.pt`, slower cadence, distinct filename, with the mid-`torch.save` retry) as well as the chart and history** — the gap that lost the previous run.
**Open questions / next steps:**
- The `update` phase is 9-31% of wall-clock and is now the largest non-`wait` term. Whether shrinking `UPDATE_INTERVAL` helps is unclear a priori: total gradient work is roughly invariant to batch size, so smaller/more-frequent updates may just redistribute the same cost. Measure with the `phase:` lines before changing it.
- This is the first run with correct baselines (levelling AND selling), correct pointer masking, escalating damage, real minion stats, rolling dispatch and broadcast weights all at once. Its placement numbers are the first that should be treated as a real baseline; nothing earlier in this log is comparable.
- Watch that `clip_frac` settles under ~0.3 and `explained_var` climbs, as in the previously-validated ~6k-transitions/update regime.
- `benchmark_v2.py` (updates ON) supersedes `benchmark_workers.py` (updates OFF). Only the former exists on this host; do not resurrect the latter.
---

---
### 2026-09-01 (correction) — "chance = 4.5" is WRONG for evaluate_policy, and eval indices are not comparable across UPDATE_INTERVAL changes
**Files changed:** none (correction to earlier entries in this log)
**What was done:** The i9 run's first eval read `EVAL @ update 100: mean_placement=7.18, top1=0.01, top4=0.11` — alarming at first glance, and it arrived right after the weights-broadcast change, so it was investigated as a possible regression. **It is not a bug.** Two separate mistakes in how earlier entries framed these numbers:

**1. `chance = 4.50` is the wrong null for this eval.** Earlier entries (including this session's) compare `evaluate_policy` results against "chance 4.50 / top1 0.125". That null only holds if all 8 seats are equally skilled. `evaluate_policy(opponent='greedy')` puts **ONE** agent against **SEVEN** GreedyPlayAgents — and since the 2026-09-01 baseline fix, greedy is genuinely strong (it holds 2.8-3.4 in-game placement across the training population). An untrained agent facing 7 competent greedy bots should therefore land near **7-8, not 4.5**. So 7.18 is roughly the UNTRAINED baseline for this eval, not evidence of a broken run. Corollary, and the more important half: the previous run's **3.95** was a substantially stronger result than this log credited — scoring below 4.5 against 7 competent greedy means beating the average greedy seat, not merely edging out chance.

**2. Eval indices are NOT comparable across runs when UPDATE_INTERVAL changes.** `EVAL_EVERY` counts UPDATES, not games. At `UPDATE_INTERVAL=90` update 100 was 9,000 games; at `UPDATE_INTERVAL=24` it is **2,400 games**. The 7.18-vs-3.95 comparison was therefore between points 3.75x apart in experience. This run reaches a like-for-like comparison around update 375. **Whenever `UPDATE_INTERVAL` changes, restate eval milestones in GAMES before comparing anything.**

**Weights broadcast verified healthy** (this was the real worry, since the change was new): `/dev/shm` version files advance continuously (v115->v117->v119 over 20s), exactly 3 retained, older ones reclaimed, each 13,887,447 bytes. Workers are receiving fresh weights every update. Independently corroborated by the in-game signal, which could not improve if workers were stuck on frozen weights.

**In-game learning signal at 2,520 games (all 8 seats, lower = better):**
| quarter | train | heuristic | greedy |
|---|---|---|---|
| Q1 | 6.384 | 2.196 | 2.813 |
| Q2 | 5.554 | 2.282 | 3.034 |
| Q3 | 5.389 | 2.486 | 3.359 |
| Q4 | 5.610 | 2.558 | 3.375 |

The agent improves 6.38 -> ~5.5 while BOTH scripted baselines degrade in lockstep — it is taking placements from them, which is the signature of real learning rather than a metric artifact. It is still well behind both baselines at this stage. Mean game length 19.6 rounds, matching the ~19.75 expected post-baseline-fix.
**Current state:** Run continues on contract 49567627, healthy, no intervention taken.
**Open questions / next steps:**
- Q4 shows a slight regression (5.389 -> 5.610) over ~26 updates. Probably noise at this sample size; if it persists past update ~200 it is worth a look.
- **When quoting any `evaluate_policy` number in future, state the opponent AND the untrained baseline for that opponent.** "Beats chance" is meaningless for a 1-vs-7 eval.
---

---
### 2026-09-02 — Training run stopped as converged; instance destroyed; final policy saved
**Files changed:** none (teardown only)
**What was done:** Assessed the i9-14900KF run (contract 49567627) for convergence before stopping, per the user's request to stop it once converged. Checked the actual trend data rather than eyeballing the chart:
- `eval_mean_placement` (vs fixed competent greedy, the honest un-gameable metric) has oscillated in a noisy 1.9-2.6 band for the last ~1,400 updates (updates 4300-5700+) with no directional trend.
- Dense per-update reward trend over the last 500 updates was flat-to-slightly-declining (1.54 -> 1.32 in 100-update chunks), not still climbing.
- `lr` (5.65e-05, floor 5e-05) and `entropy_coef` (0.0044, floor 0.004) were both essentially at their annealed floor.
- `best_avg10` (3.844) had not improved across the last ~225 updates checked.

All four signals agreed: converged, not still improving. Stopped training cleanly at update 5766 (138,360+ games of the planned 150,000, steps=25,157,367) by waiting for a fresh checkpoint-save log line before killing the tmux session, to avoid interrupting a `torch.save` mid-write. Pulled and verified all three checkpoints load correctly:
- `bg_agent_ppo.pt` — final, steps=25,157,367, updates=5766
- `bg_agent_ppo_best.pt` — steps=21,460,679, updates=4850 (best_avg10 stopped improving after this point, so this predates the final checkpoint by ~900 updates)
- `bg_agent_ppo_backup.pt` — steps=25,153,507, updates=5765

Destroyed the vast.ai instance (contract 49567627). $3.02 credit remaining.
**Current state:** No vast.ai instances running, nothing billing. All checkpoints, `data/fresh_training_history.json`, and `data/training_progress.png` preserved locally. This run is the first one in the project trained AND evaluated under a fully correct engine (real minion stats, correct pointer masking, escalating combat damage, levelling+selling scripted baselines, fixed optimiser, telescoping reward invariance, honest fixed-opponent eval) — every earlier session's placement numbers are not comparable to this one.
**Open questions / next steps:**
- `bg_agent_ppo_best.pt` (update 4850) and the final `bg_agent_ppo.pt` (update 5766) are close but not identical -- since `eval_mean_placement` was noisy-flat rather than cleanly monotonic in that window, it's not obvious the "best" checkpoint by `best_avg10` is actually the strongest by the more trustworthy eval metric. Worth a direct head-to-head eval between the two before picking one to build on.
- The run stopped at 138,360/150,000 games (92.2%) -- short of the originally planned N_GAMES, but the convergence evidence (flat eval, flat-to-declining reward, annealed-out lr/entropy) suggests the remaining ~11,600 games were unlikely to move it further.
- No further work has been done on this checkpoint (no BC warm-start, no further reward tuning) -- it reflects exactly the fixes documented across the 2026-09-01 sessions and nothing else.
---

---
### 2026-09-02 — Real combat sim wired in, REORDER action added, opponent pool fixed; new run launched
**Files changed:** `env/game_loop.py`, `env/player_state.py`, `agent/policy.py`, `train.py`, `run_fresh_training.py`, `CLAUDE.md`, `tools/live_graph.py` (new), `tools/sync_from_vast.sh` (new)

**Analysis of the converged run first.** Eval vs 7 greedy went 7.18 -> ~2.1 by update ~1300 (31k games) and then sat flat in a 1.9-2.6 band for the remaining 107k games. **78% of that run bought nothing measurable.** PPO itself was healthy throughout (clip_frac 0.12, approx_kl 0.026, explained_var flat ~0.63) -- the learner was fine, it had run out of task.

**ROOT CAUSE 1 -- all training combat was a stat-sum coin flip, and the entire symbolic layer was disconnected from the reward.** Every training/eval path built `FirestoneClient(mock_mode=True)`, so combat resolved via `_heuristic_estimate`: `win_prob = sum(atk+hp)_mine / (sum mine + sum theirs)`. Measured, on equal-stat-sum boards:

| matchup | real BGCombatSim | mock |
|---|---|---|
| Taunt+Divine Shield vs plain | 1.00 | 0.50 |
| Reborn vs plain | 1.00 | 0.50 |
| 7 small vs 4 big | 0.00 | 0.50 |

The mock returns 0.50 for all of them -- it was not teaching a wrong lesson, it was teaching *nothing* about keywords, positioning, or the count/quality tradeoff. Meanwhile `symbolic/combat_sim.py` (1,355 lines, full turn-by-turn sim) was **dead code** in training. Fixed: `step_combat` now uses a dedicated `BGCombatSim(n_trials=COMBAT_SIM_TRIALS=8)`, behind `use_real_combat=True` (set False for A/B). Cost is negligible (0.15ms/call at 1 trial, 0.78ms at 8, ~80 combats/game) and **throughput went UP, not down** -- real combat ends games in ~15.8 rounds instead of 20.5, which more than pays for the sim.
- Robustness verified independently: `_run_one_trial_fast` threw **0 exceptions in 3,000 trials** over the real 275-card pool (its `except -> tie` swallow is not hiding anything).
- Both sims are now seeded FROM THE GAME SEED. Leaving them unseeded would have taken RNG from OS entropy and silently made `evaluate_policy` non-reproducible -- the pinned per-game seeds exist precisely so eval points are comparable. Game-level determinism verified: identical placements across 3 runs at a fixed seed.

**ROOT CAUSE 2 -- no positioning existed.** `PLACE` appended to the board; there was no REORDER (CLAUDE.md documented a `REORDER(from,to)` that was never built). Under the mock this was irrelevant. Under the real sim it is worth a **mean 0.55 swing in (win_prob - loss_prob)** over non-blowout matchups (median 0.45, max 1.09; one matchup went -0.18 -> +0.91 by reordering the same 5 minions). Added action type 9 `REORDER(board_idx)` = move-to-front: one pointer, and BFS over all 5,040 orderings of a 7-minion board confirms **every permutation is reachable in at most n-1 moves**. `REORDER_BUDGET_PER_TURN = 6` is exactly that worst case, so the budget never blocks a reachable arrangement while still bounding stalling. Board slot 0 is never a valid target, so every REORDER is a real state change -- both guards exist because a costless repeatable action is a `gamma=0.997` discounting exploit (the ACTIVATE failure mode from 2026-09-01). `N_ACTION_TYPES` 9 -> 10, so **all prior checkpoints are incompatible**; this had to go in before the run, not after.

**ROOT CAUSE 3 -- the opponent pool got WEAKER as the run progressed.** `MILESTONE_EVERY=50` milestones were never evicted and sampling was uniform over `rolling + milestones`: at update 5766 that was **115 milestones vs 20 rolling = 85% of self-play opponents drawn from an ever-growing pile of ancient checkpoints**, plus 1.6GB of state_dicts pinned in the main process. Fixed with `MILESTONE_CAPACITY=10` + `P_RECENT=0.75` recency-weighted sampling. Milestones are THINNED, not FIFO-evicted, by removing the most redundant interior entry (smallest neighbour gap). **The first thinning implementation was wrong and the verification caught it**: repeated `older[0::2]` halving retained `u50, u5200, u5450 ... u6000` -- oldest plus a cluster at the end with a 5,150-update hole. Spacing has to be computed, not approximated by slicing. Now: `50, 850, 1450, 1950, 2650, 3100, 3650, 4300, 5050, 6000` (gaps 450-950), median sampled opponent 80 updates behind, ~20% genuinely old.

**ROOT CAUSE 4 -- the weights broadcast only ever covered the current policy.** `_make_task` still embedded two raw snapshot `state_dict`s per task: **27.8 MB/task, 667 MB/update** -- twice the traffic the earlier broadcast fix removed. Snapshots now go through the same content-addressed `/dev/shm` file store (refcounted, atomic `.tmp`+`os.replace`, reclaimed on eviction and at run end). **Measured: 27,775,546 -> 197 bytes/task, and zero /dev/shm leaks.** Workers load by stable `snapshot_id` through a bounded 8-entry LRU.
- **The old instrumentation could not have seen this.** `t_dispatch` wrapped only `pool.submit()`, which returns as soon as the item is queued; pickling happens later in the pool's single GIL-bound feeder thread and surfaced as `t_wait`. The previous session's "main-process serialization is no longer measurable" was a **false negative, not a finding**. Tasks are now pickled explicitly and timed, and the `phase:` line reports bytes/task and MB/update.

**Eval was measuring one saturating axis with a moving seed.** Now three series: `eval_mean_placement` (greedy, key and semantics preserved), `eval_heur_mean_placement`, and `eval_ref_mean_placement` vs 7 frozen copies of the policy at update 500 -- an opponent that keeps discriminating after the scripted bars saturate. Two fixes found while verifying:
- The reference opponent originally used `StaticAgent`, which SAMPLES; measured over 4 repeats of a fixed-seed 16-game eval that produced 5.75/6.13/6.25/6.31 -- ~0.6 placement of pure noise on the metric that is supposed to be un-gameable. `EvalStaticAgent` is now deterministic (and opts out of batching, or EvalAgent's argmax is silently lost). Re-verified: identical across 4 runs, and sequential == parallel exactly.
- `seed=ppo_trainer.total_steps` meant **every eval point sampled a different set of games**. The policy never trains on eval games so a fixed seed cannot be overfitted; `EVAL_SEED=12345` removes the between-point variance that was most of the old 1.9-2.6 oscillation.
- Eval also stopped pickling a full state_dict into each of its n_games tasks (~1.8GB/eval at n=128); it broadcasts through a file like training does.

**Run sizing, measured on the target host (i9-14900KF, 24 physical cores, contract 49669406) with updates ON.** 3.79 games/s at 16 workers, 5.77 at 24. The bench's apparent "best = 32 workers" was an artifact of tying `update_interval` to `n_workers` (nw=32 did half as many updates); with `update_interval` held fixed, 16 workers (3.97 g/s) beats 32 (3.28 g/s). A steps/game spread of 63.8-172.6 across configs looked like data loss but is **seed-driven variance, not loss** -- transitions/game were flat within each run (148->146->187->144 by quarter) and depend on how long training seats survive. `UPDATE_INTERVAL` deliberately decoupled from `N_WORKERS` and set to 48: at the new ~125 steps/game that restores ~6,000 transitions/update, the same PPO batch size the previous run actually ran at (keeping 24 would have quietly shrunk it to ~2,100).

**Current state:** Fresh 200,000-game run **live** on contract 49669406 in tmux `train`, logging to `training.log`. Confirmed at update 34+ with zero errors: entropy ~1.75, `best_avg10` -3.69 -> -1.26, board size 0.58 -> 5.23, 125.6 steps/game, ~6,030 transitions/update (exactly the target). $3.02 credit at $0.1081/hr is ~28h of runway; the run needs ~12h of self-play. Local `tools/live_graph.py` serves an auto-refreshing dashboard at **http://127.0.0.1:8420** (downsamples the ~44MB history to a ~110KB series file), fed by `tools/sync_from_vast.sh` in local tmux `bgsync`; graph runs in local tmux `bggraph`. Both are in tmux specifically because `nohup`/`setsid` did NOT survive session teardown.
**Open questions / next steps:**
- **REORDER has no dense reward signal.** The board-shape potential is order-invariant (a stat sum), so a REORDER pays exactly 0 shaped reward and can only be learned through delayed combat outcomes. It will learn slowly. A positional potential term would speed it up, and potential-based shaping cannot change the optimum -- but a wrong heuristic would slow learning, so measure before adding one.
- First eval lands at update 50; the reference opponent freezes at update 500. Watch whether `eval_ref_mean_placement` keeps moving after greedy/heuristic saturate -- that is the whole reason it exists.
- Board size at END_TURN is worth watching now that combat is real: under the mock the agent settled at 4.1/7, but the real sim rewards stat CONCENTRATION at equal stat sum, so 4.1 may have been fine. This run's number is the first meaningful one.
- The scripted baselines do not position their minions at all, so real combat may make them weaker relative to the agent than they should be. If the agent clears them quickly, they need positioning logic before they are a fair bar.
---

---
### 2026-09-02 (same day, follow-up) — REORDER discount-stalling exploit: measured, fixed, run restarted
**Files changed:** `env/game_loop.py`
**What was done:** The REORDER budget shipped earlier the same day bounded the discount-stalling exploit but did NOT remove the incentive, and the agent walked straight to the bound. Caught by inspecting the live run's action mix.

**The measurement.** Over the first 51 updates of the run:

| | first fifth | last fifth |
|---|---|---|
| REORDER | 2.4% | **24.9%** |
| END_TURN | 12.0% | **5.0%** |

REORDER reached **4.82 of the 6-per-turn budget (80% of cap)** while END_TURN more than halved -- turns were getting LONGER. A board changes by ~1-2 minions per turn, so ~5 move-to-front ops per turn is not plausibly positioning.

**The arithmetic.** REORDER costs no gold, so it is a free action, and a free action in a discounted MDP is a stalling device: it advances a `gamma=0.997` step, so an agent expecting NEGATIVE return improves its objective purely by delaying the bad news. Gain per free step is `(1-gamma)*|V| = 0.003*|V|`, up to **~0.012/step** as V approaches the -4.0 placement floor. The only existing counterweight was the potential-shaping discount drag, `SHAPE_ALPHA*Phi*(SHAPE_GAMMA-1) ~= -0.0023/step` at Phi=0.5 -- about **5x too small**. At 4.82 reorders/turn over ~18 rounds that is ~87 extra steps/game, discounting terminal rewards by an extra ~23%.

**The fix.** `REORDER_COST = 0.03`, charged on each REORDER that actually applies (invalid ones cannot be sampled -- the mask excludes slot 0, an exhausted budget, and boards under 2 minions -- so the cost cannot be dodged by aiming at a no-op). Sizing: 2.5x the worst-case per-step stalling gain, so the gradient is unambiguous; and small against what a real reposition is worth (board order swings win-loss by ~0.55, roughly a full placement step ~= 1.0 of FINAL_PLACEMENT_REWARD), so genuine positioning still pays ~30x over. This deliberately breaks strict potential-shaping policy-invariance -- it is a real action cost, not a potential difference, and that is the point: the free action has to stop being free.

Verified mechanically: consecutive REORDERs now return **-0.0316** each (= -0.03 cost + -0.0016 shaping drag), consistently, and the cost is not charged on a slot-0 no-op or when the budget is exhausted. NOTE on verification method: the first `_apply_potential_shaping` call after `reset()` re-baselines `ps.phi`, so the first action of a turn carries a one-off correction -- an initial test that measured the very first action read the cost as absent. Warm up before measuring shaped rewards.

**Restarted the run** at ~3,000/200,000 games (cheap) rather than letting a policy already 25% committed to stalling continue. Fresh run confirmed clean at update 2: REORDER=1%, END_TURN=26%, zero errors.
**Current state:** Run live on contract 49669406, tmux `train`. Local sync (`bgsync`) and live graph (`bggraph`, http://127.0.0.1:8420) in local tmux -- note `nohup`/`setsid` do NOT survive Claude-session teardown, tmux does.
**Open questions / next steps:**
- **NOT yet confirmed.** REORDER at 1% by update 2 is not evidence -- the previous run also started at 2.4% and took ~50 updates to reach 24.9%. Re-check the action mix around update 50-100 and confirm REORDER settles well below the 6/turn cap.
- The direct redundancy diagnostic (REORDERs spent per turn vs the BFS-minimum needed to reach the order the turn ended on) was written but **never produced a result** -- it was killed along with the tmux session during the restart. The case above rests on the action-rate trend plus the arithmetic, not on a measured productive-vs-redundant split. Worth completing if the exploit recurs.
- If a cost of 0.03 turns out to suppress genuine positioning (watch whether REORDER collapses to ~0 AND board-order quality stays poor), the better fix is the principled one: treat REORDER as taking zero time (per-transition `gamma=1.0` in GAE), which removes the stalling incentive at its source without taxing useful repositioning. That needs a per-transition gamma in `RolloutBuffer.compute_advantages`.
- ACTIVATE is at 0% in the new run (was climbing to 4.1%); it has a real gold cost so it should not stall, but worth a glance later.
---

---
### 2026-09-02 (addendum) — REORDER_COST verified working; the rising trend is positioning, not stalling
**Files changed:** none (measurement only)
**What was done:** Resolved the "NOT yet confirmed" item from the previous entry. At update 57 -- the same window where the uncosted run reached 24.9% REORDER -- the costed run reads:

| metric | old (no cost) | new (REORDER_COST=0.03) |
|---|---|---|
| REORDER share of actions | 24.9% | **13.0%** |
| reorders / shopping turn | 4.82 | **1.72** |
| % of the 6/turn budget cap | 80% | **29%** |
| actions / shopping turn | 20.0 | **13.2** |

REORDER was still trending UP (0.16 -> 0.51 -> 0.91 -> 1.07 -> 1.72 per turn), which on its own is ambiguous: it could be slower stalling. **The discriminator is the reorder:place ratio.** A newly PLACEd minion is appended to the END of the board, so correcting its position costs ~1 move-to-front -- meaning legitimate positioning should scale WITH placement and sit below 1.0, whereas stalling should rise INDEPENDENTLY of it.

| window | places/turn | reorders/turn | ratio |
|---|---|---|---|
| 3 | 1.55 | 0.98 | 0.63 |
| 4 | 2.00 | 1.25 | 0.63 |
| 5 | 2.48 | 1.63 | **0.66** |

The ratio has **flattened at ~0.65** while both rates climb together -- the agent is filling its board faster and repositioning ~2 of every 3 new minions. The old uncosted run's final window gives 24.9/13.7 = **1.82** by the same arithmetic, i.e. nearly two reorders per minion placed and still climbing. So the rise is board-building, not step-padding, and REORDER_COST=0.03 did not over-suppress positioning either (it did not collapse to ~0).
**Current state:** Run healthy at update 69, zero errors. First eval landed: **greedy=5.38, heuristic=5.97** at update 50 (~1,200 games) against an untrained baseline of ~7.2-8.0. `best_avg10` crossed 0 (+0.038); board 5.42, 17.2 rounds/game.
**Open questions / next steps:**
- The reorder:place RATIO (not the REORDER rate alone) is the metric to watch for this exploit. A rising REORDER share is expected and fine while the ratio stays below ~1.0; the alarm condition is the ratio climbing.
- The direct redundancy diagnostic (`_reorder_waste.py`: reorders spent per turn vs the BFS-minimum needed for the order the turn ended on) is still slow and has not produced a number. The ratio analysis above supersedes it in practice, but the direct measurement would be strictly better evidence if this comes up again.
- Reference eval still `n/a` until the snapshot freezes at update 500.
---

---
### 2026-09-02 (addendum 2) — Direct redundancy measurement completed: 0% redundant REORDERs
**Files changed:** none (measurement only)
**What was done:** Completed the direct diagnostic left unfinished in the previous two entries (`_reorder_waste.py`), run against the live policy at update 63. Per shopping turn it records the board order at the turn's first REORDER and at END_TURN, then BFS-computes the MINIMUM move-to-front ops needed to get between them; anything spent beyond that minimum was spent to advance gamma, not to position.

```
shopping turns with >=1 REORDER: 4
  REORDERs SPENT : 6  (1.50/turn)
  minimum NEEDED : 6  (1.50/turn)
  REDUNDANT      : 0  (0.0%)
```

Every REORDER was necessary to reach the ordering the turn actually ended on -- no cycling. The 1.50/turn also matches the 1.72/turn read off the aggregate action rates.

**Read this with its limitation.** n=4 turns / 6 reorders is a SMALL sample. The diagnostic can only score turns where board MEMBERSHIP is unchanged between the first reorder and END_TURN (otherwise "minimum moves to reach this permutation" is undefined), and that filter drops most turns, since the agent typically places or sells after reordering. So it is a clean measurement on a narrow slice, not an independent substitute for the reorder:place ratio. The two agree from different angles: the ratio shows reorders scaling WITH placements (~0.65, flat), this shows the ones that occur are not wasted.
**Current state:** Unchanged -- run healthy, REORDER_COST=0.03 confirmed working by two independent lines of evidence.
**Open questions / next steps:**
- If this needs re-measuring at scale, relax the filter: score turns with membership changes by comparing only the SURVIVING minions' relative order, which would make most turns usable instead of ~1 in 10.
---

---
### 2026-09-02 (evening) — Adaptive opponent pool + economy retune; run restarted
**Files changed:** `train.py`, `run_fresh_training.py`, `env/game_loop.py`, `CLAUDE.md`

**Why.** A traced game of the u350 policy (see the artifact / `data/game_trace.json`) showed it winning 1st place with 40/40 HP via a single tier-1 card: buy `Suspicious Prisonguard` (`Activate (1): Give another minion +3/+3`), pump one `Flighty Scout` 3/3 -> 36/36, and from round 9 take exactly two actions per turn (ACTIVATE, FREEZE) while banking all 10 gold and holding a 4/7 board. Two root causes, both fixed.

**Fix 1 -- the board potential was pinned at its ceiling, so developing paid nothing.** `BOARD_SHAPE_STATS_SATURATION` (the effective-stats value at which Φ reads 0.5) was **30.0**, its own comment admitting an untuned guess. Real late-game boards measure ~110, so Φ sat near its asymptote from ~round 8 and every further purchase earned ≈0 shaped reward. Raised to **60.0** after measuring the real board-value distribution by round. This was the deeper cause -- not that holding gold was cheap, but that spending it was worthless.

**Fix 2 -- late-game idle gold was priced at almost nothing.** `gold_scale` faded linearly to a 0.2 floor and stayed flat there from round 13 to the end, so a full 10-gold purse cost `0.015*10*0.2 = -0.03`/turn against `WIN_REWARD = 0.15`. Replaced with a **V-shaped** schedule: the early fade is unchanged (saving to level is sometimes correct), then it ramps back up at `GOLD_SCALE_LATE_RAMP=0.13`/round to `GOLD_SCALE_LATE_CEIL=1.5`. Idle gold now costs ~0.19/turn at round 21, i.e. more than a combat win is worth -- which is the correct sign late.
NOTE on units: `GOLD_PENALTY_COEF` is already `DENSE_REWARD_SCALE`-multiplied to **0.015**; the `0.05` in the source is pre-scale. An earlier analysis quoted the unscaled figure and overstated the penalty 3.3x. Always quote the scaled value.

**Balance re-verified (the hard constraint).** CONTEXT records dense terms once running ~17x `FINAL_PLACEMENT_REWARD` and drowning out the objective, and Fix 2 makes a dense term materially larger. Measured over 12 games on the real seat mix: **|dense|/|placement| = 0.62 mean, 0.48 median** -- the terminal objective still dominates comfortably.
The first attempt at this measurement returned a suspiciously exact ratio of 1.00; the cause was that `BattlegroundsGame._accumulated_rewards` is **dead code** (initialised at game_loop.py:698, never written), so reading it silently yielded 0.0 and made the ratio tautological. The real per-player total is `GameResult.final_rewards` (game_loop.py:1760). Worth deleting `_accumulated_rewards` at some point -- it is a live trap.

**Fix 3 -- adaptive opponent pool.** The scripted baselines were exhausted at 8% of the run: eval vs greedy went 5.38 (u50) -> **1.42 with top1=0.81** (u450), while the PREVIOUS run's entire 4,400-update plateau sat at ~2.1. Four of six opponent slots were bots the agent beats almost always, scheduled to stay that way for ~92% of the run -- the same wasted-compute failure that made the last run useless after 31k games.
`_train_parallel` now takes an optional `opponent_mix_fn()` consulted at TASK-CREATION time (default `None` reproduces the old fixed mix exactly), with a defensive clamp so a bad callable can never desync `opp_sds` from `opp_pids`. `AdaptiveOpponentMix` in run_fresh_training.py drives it off the **fixed-opponent eval only** -- in-game placement is confounded because the snapshot pool co-evolves, so "everyone improved together" is indistinguishable from "nobody improved". Two-threshold hysteresis: reduce at eval < 2.0 for 2 consecutive points, restore at > 3.0 for 2 consecutive.
Reduced mix is **1 greedy + 0 heuristic + 5 snapshots**, deliberately not 0 scripted: greedy is the only agent in the pool that builds a WIDE board (6.75-7.0 minions), which is exactly the archetype that should punish the narrow carry board. Self-play is auto-curricular in DIFFICULTY but not in STRATEGIC DIVERSITY -- every snapshot plays the current policy's style, so dropping all scripted seats would remove the only non-self behaviour and risk polishing the monoculture rather than refuting it.
Verified independently of the subagent: trigger logic passes 8/8 synthetic cases (no flip on a single noisy point, no oscillation inside the dead zone, non-consecutive lows ignored, NaN/None skipped rather than treated as data, correct round-trip). Subagent additionally confirmed real seat-label counts: default mix gives 2 heuristic + 2 greedy every game; reduced gives 0 heuristic + 1 greedy + 5 snapshots once the pool has entries.

**Current state:** Fresh 200,000-game run **live** on contract 49669406, tmux `train`, healthy at update 7 with 0 errors (entropy ~1.6, `best_avg10` -3.57 -> -3.33). Local sync in tmux `bgsync`, live graph in tmux `bggraph` at http://127.0.0.1:8420. Credit ~$2.7.
**Open questions / next steps:**
- **Watch `opponent_mix` against the eval curve.** The mix is recorded on the same x-axis as `eval_updates` precisely so a metric change right after a switch is attributable rather than coincidence.
- **Watch for loss spikes.** The pre-restart run showed isolated `total_loss` spikes (7.8, 1.7, 509) that recovered within one update, contained by `max_grad_norm=0.5`. Larger reward magnitudes from the gold ramp are exactly the condition that could make those worse -- if they stop self-correcting, that is the cause to check first.
- The Prisonguard line may still be strong even after these fixes. If the agent re-converges on it, the next lever is the card itself or a baseline that punishes narrow boards -- not more reward tuning.
- `_accumulated_rewards` is dead code and misleading; delete it.
---

---
### 2026-09-02 (addendum) — Measurements behind the economy retune, and a determinism nuance
**Files changed:** `CLAUDE.md` (accurate rewrite of the Reward Shaping section), `tools/trace_game.py` (kept)
**What was done:** The economy subagent's full report arrived after its `env/game_loop.py` work had already been committed in `eb80b35` (shared working directory; its edits were swept into that commit). Its independently-derived constants matched what was deployed EXACTLY -- same `BOARD_SHAPE_STATS_SATURATION = 60.0`, same V-shaped gold schedule -- which is a genuine independent corroboration rather than an agreement by construction.

**The measurement behind SATURATION=60** (30 games, real seat mix, frozen update-449 checkpoint, 3,474+ per-player-round samples). Board `value` (effective atk+hp + keyword/synergy bonuses) for the trained population:
| p25 | p50 | p75 | p90 | p99 |
|---|---|---|---|---|
| 26 | 71 | 121 | 162 | 249 |

By round, median value: **57 (r8), 103 (r12), 131 (r16), 164 (r20)**. Median leftover gold hit **9/10 by round 10** and stayed there, confirming the hoarding was systemic and not a quirk of the one traced game. At the old `30.0` the distribution was already past half-saturated at its 25th percentile; `60.0` puts Φ=0.5 near both the round-8 median (57) and the population median (71), stretching the useful gradient from p25=0.30 to p90=0.73. The functional form `v/(v+S)` was kept -- the problem was the constant, not the shape.
Notably the degenerate traced board (one pumped minion, 4/7 board, round 18) computes to **value=137 -- squarely mid-distribution, not an outlier.** Its problem was never low board value; it was that the potential could not SEE any difference up there.

**Gold-penalty counterfactual, on identical recorded trajectories** (sidesteps the reproducibility caveat below): mean per-game gold penalty roughly doubles, 0.216 -> 0.447; the worst hoarders go 0.77 -> 2.96 (~4x). Still small against the ±4 placement span. My own independent balance check agreed: |dense|/|placement| = 0.62 mean / 0.48 median.

**Determinism nuance -- two findings that look contradictory but are not.** The subagent reports the engine "isn't fully seed-reproducible even after seeding torch/random/numpy per-game". Earlier this session I verified the opposite: identical placements across 3 runs at a fixed seed, and `evaluate_policy` bit-identical across 4 repeats with sequential == parallel. Both are correct because they test different paths: my tests used **EvalAgent (deterministic argmax) + scripted bots**, whereas the subagent measured the **training mix containing PPOAgent/StaticAgent, which SAMPLE** from the policy distribution -- reproducible only if every worker's torch RNG state matches at every sampling point, which across a parallel pool it does not. **The property that matters for the honest metric -- `evaluate_policy` reproducibility -- is verified and holds.** Do not "fix" training-path nondeterminism on the strength of the subagent's note without re-reading this.

**Rounds/game:** 21.2 before vs 21.5 after the change (range 14-30 / 14-32), with ~15% of games exceeding 25 rounds in BOTH -- unchanged, so not a regression from this work. Worth noting it is above the real-BG 12-18 band though, and higher than the 15.3-15.8 measured with an untrained policy: a stronger policy survives longer and the lobby takes longer to resolve.

**CLAUDE.md** was rewritten to match the code. Verified every claim against source before accepting it: the flat `+0.1` survival bonus is genuinely removed; all quoted coefficients match the DENSE_REWARD_SCALE-multiplied values (WIN_REWARD 0.15, LOSS_PENALTY -0.09, damage coefs 0.015, rank 0.045, hand 0.024); and the split `_apply_board_shape`/`_apply_tier_shape`/`phi_board`/`phi_tier` scheme the old doc described no longer exists in code (`_apply_potential_shaping` does; `_apply_board_shape` does not -- the only surviving mentions are historical comments).
**Current state:** Run live and healthy on contract 49669406. No further code changes.
**Open questions / next steps:**
- `~15%` of games exceeding 25 rounds is a pre-existing engine characteristic worth a look independently of this work.
- `_accumulated_rewards` is still dead code (see the previous entry) -- delete it.
---

---
### 2026-09-03 — Mid-run check at 49%: economy fix worked; gold ramp is mistimed
**Files changed:** none (measurement only)
**What was done:** Traced the live policy (update ~2050) and compared against the u350 degenerate trace.

**The economy fix WORKED.** Board size by round, current policy vs the old degenerate one:
```
rnd:        1  2  3  4  5  6  7  8  9 10 11 12 13 14
u2050:      0  1  2  3  4  4  4  4  4  6  7  7  7  7     <- full board by r11, held
u350:       0  1  2  2  2  2  2  2  4  4  4  4  4  4     <- 4/7 forever
```
It now buys AND sells to upgrade (r11 is a 10-action turn: ACTIVATE x3, SELL x2, BUY x2, PLACE x2, FREEZE) where the old policy did nothing but ACTIVATE+FREEZE for ten consecutive rounds. Game length 18 -> 14 rounds. Still 1st, still 40/40 HP.

**CORRECTION to an in-session misread.** The aggregate `update_board_avg` reads 3.95 and I initially reported board width as "essentially unchanged". That was wrong: the whole-game mean averages over the early ramp, so a game reaching 7 still means ~4.3. **Late-game board width is the metric; the whole-game mean is not.** ACTIVATE doubling 8.3% -> 16.9% is likewise not the degenerate engine reasserting itself -- it now happens alongside a full board and active shop churn.

**REAL REMAINING PROBLEM -- the gold ramp is mistimed relative to game length.** Leftover gold by round: `r9=5, r10=4, r11=3, r12=7, r13=9, r14=9`. It spends down mid-game then re-banks late. Cause: `gold_scale` is at its 0.2 FLOOR at round 13 and only reaches `GOLD_SCALE_LATE_CEIL=1.5` by round **23** -- but this game ended at round **14**, so the ramp's teeth are in a round range the game never reaches and late idle gold still costs ~0.03/turn, exactly the value the fix was meant to correct.
The ramp was sized against a measured ~21-round median taken from the u449 checkpoint. **A stronger policy ends games FASTER** (it eliminates opponents sooner), so the ramp slides out of range precisely as the agent improves -- the schedule is chasing a target that moves away from it. Indexing it to an absolute round number is the design error.
Fix for the NEXT run (not worth a third restart at 49% with the primary fix working): index the ramp to something that co-moves with game state -- players remaining, or a fraction of expected remaining game length -- rather than `round_num`.

**Run health at 49%** (update 2053/4167, 98,424 games, 0 errors): adaptive pool switched (2,2) -> (0,1) at update 300 as designed. Evals -- greedy floored at 1.2-1.7 (top1 ~0.9) for 1,600 updates and no longer discriminating; heuristic 1.5-2.4; **vs-frozen-u500-self 4.34 -> 2.97 and still moving**, which is exactly the axis it was added to supply. `explained_var` 0.60 -> 0.73 (last run plateaued at 0.63). REORDER 4.3% of actions with a reorder:place ratio of 0.31, far below the 1.0 alarm.
**Open questions / next steps:**
- Gold ramp indexing (above) -- the one concrete carry-over for the next run.
- Games run 14-19 rounds now; the earlier "~15% exceed 25 rounds" figure came from a weaker checkpoint and should be re-measured against the current policy before acting on it.
- The trace tool's `gold` column is the round's GRANTED gold (`_gold_for_round`), NOT leftover -- leftover has to be read from the last action's `gold: 'X->Y'` field. Easy to misread; it briefly led me to think the agent was still hoarding all 10.
---

---
### 2026-09-03 — Gauntlet eval + Bradley-Terry ratings; resumed from checkpoint
**Files changed:** `train.py`, `run_fresh_training.py`

**Why.** Self-play makes in-game placement structurally useless: with 8 seats drawn from one policy family the mean is FORCED to 4.5, so "everyone improved together" and "nobody improved" are the same number. The fixed-opponent evals solve that but each is a fixed bar that eventually gets cleared -- greedy already has (1.2-1.7, top1 ~0.9 for 1,600 updates, measuring nothing). A single frozen reference just moves the wall.
Also measured: the existing reference eval was **noise-dominated**. At `EVAL_REF_GAMES=32` the placement SE is ~0.4 against a total trend of ~0.94 over 1,800 updates.

**The gauntlet.** Seat the current policy against `GAUNTLET_SIZE=7` DIFFERENT frozen checkpoints spanning the run, chosen evenly spaced (always keeping oldest and newest). Key structural point: **a BG lobby seats 8, so ONE game is already a full 8-way comparison** -- 28 pairwise outcomes per game. Comparing n agents pairwise in a 2-player game costs O(n^2) matches; here it costs O(n) games. That is what makes rating-fitting affordable at eval cadence. References roll forward (`GAUNTLET_EVERY=300`), so unlike a single reference the comparison set does not saturate.

**Ratings.** `fit_gauntlet_elo` fits Bradley-Terry over those pairwise outcomes and reports on the Elo scale, anchored so successive evals are comparable.
- **First implementation was wrong and the test caught it.** Plain gradient ascent with a fixed step in Elo units DIVERGED on a synthetic set with known strengths -- ratings of +-2000 and the ordering scrambled -- because the gradient scales with the number of comparisons while the step size did not. Replaced with the standard MM/Zermelo iteration (`p_i <- (W_i + prior) / sum_j n_ij/(p_i+p_j)`), which is parameter-free and converges monotonically. Verified: recovers the true ordering of 7 synthetic references exactly, and ratings are monotone in true strength (+100 -> +1261 as strength rises 2 -> 14).
- **Anchor bug, also caught by test:** anchoring on `min()` of path strings sorts lexicographically, so `"ref_u1200" < "ref_u300"` picked the wrong anchor -- and worse, the anchor would CHANGE as references were added, making successive Elo values incomparable, which defeats the entire purpose of anchoring. Now parses the update number out of the filename.
- Caveat recorded in the docstring: Elo assumes TRANSITIVITY, and self-play readily produces cyclic strength that a scalar cannot represent. **A rising rating alongside a flat gauntlet placement is the signature of that.** Balduzzi et al. 2018 ("Re-evaluating Evaluation") is the Nash-averaging treatment for the cyclic case.

**Cost, measured not guessed.** A gauntlet lobby runs a neural forward pass for ALL 8 seats where the scripted evals run 1 -- roughly **8x per game**. At the 96 games/50 updates I first wrote, that projected to ~25% of total run wall-clock. Cut to `EVAL_GAUNTLET_GAMES=32` on its own `GAUNTLET_EVAL_EVERY=200` cadence: 32 games still yields 896 pairwise outcomes, ample for 8 ratings. Also raised `_W_EVAL_CACHE_MAX` 4 -> 9: it was sized for "one eval needs at most two distinct networks", true for the scripted evals but wrong for the gauntlet where every seat is a different checkpoint, so the LRU thrashed and reloaded from disk nearly every game.

**Resumed rather than restarted**, per the user's instruction and because the gauntlet is **eval-only** -- it does not touch the reward function or the action space, so the policy, optimiser state and history all stay valid. Verified continuity: `Resumed this run: steps=21142995, games=114312, updates=2382`.
Before stopping the old process a verified checkpoint backup was secured (`checkpoint_backups/preGauntlet_u2369_bg_agent_ppo.pt`, loads with model+optimizer) -- the sync loop had already pulled it, which is exactly the failure mode that loop exists for.
**Current state:** Live on contract 49669406, resumed at update 2382, 0 errors. First gauntlet eval fires at update 2400.
**Open questions / next steps:**
- The gauntlet is thin at first on a resumed run: it seeds from the single existing `bg_agent_ppo_reference.pt` (u500) and accumulates one new reference per 300 updates, so it reaches a full 7-checkpoint spread around update ~4200. Early gauntlet numbers compare against fewer distinct selves and should be read as provisional.
- Watch gauntlet placement AND Elo together -- divergence between them is the cyclic-strength tell described above.
- Still unfixed from the previous entry: the gold ramp is indexed to absolute round number and its teeth (r13-23) fall outside games that now end at ~14.
- Trap for later: `pkill -f <pattern>` matched this session's own wrapping shell and killed it mid-sequence. Use precise patterns or pgrep-then-kill by PID.
---

---
### 2026-09-03 (evening) — Flattened the unspent-gold penalty: round-indexed schedule was built on a false premise
**Files changed:** `env/game_loop.py`, `CLAUDE.md`

**Why.** User observed that gold cannot be banked across turns in this game (`ps.gold` is unconditionally overwritten by `_gold_for_round(round_num)` every round — verified, no carry-over/interest anywhere in the codebase, matching real Battlegrounds rather than TFT/Underlords). The V-shaped `_gold_penalty_scale` schedule's early/mid-game leniency (fading 1.0 → 0.2 by round 13) was justified in the code comments as "saving across 1-2 turns to hit a tier-5/6 level-up spike is sometimes correct" — but that's not possible here: next turn's gold is a fixed function of round number regardless of what's held back, and `ps.level_cost` decays on its own each round independent of spending. So there is no round at which leftover gold is anything but pure waste, and a flat coefficient is the more correct model, not a simplification.

The schedule also had an independent, previously-flagged bug: its late-game ramp (round 13 → `GOLD_SCALE_LATE_CEIL=1.5` by round 23) was sized against a ~21-round median from an older checkpoint, but as the policy improves games get *shorter* — the 2026-09-02 mid-run entry already caught this in a live trace (game ended round 14, ramp's teeth start at round 13, barely engaged). A flat coefficient has no round dependence, so this mistiming can't recur as the policy keeps improving.

**Measurement, not a guess.** Wrote a throwaway script (monkeypatched `BattlegroundsGame._end_of_turn_reward` to record every `(round_num, gold)` pair, no permanent instrumentation left behind) and replayed 24 games on the real seat mix (4 policy-driven seats sharing the live checkpoint `checkpoint_backups/live_bg_agent_ppo.pt` via `EvalAgent` + 2 `HeuristicAgent` + 2 `GreedyPlayAgent`, seeded). Collected 2,436 end-of-turn events across games running 15-21 rounds (mean 17.2). Candidate flat scales against the SAME trajectories:

| schedule | total cost (2,436 events) | ratio to old |
|---|---|---|
| OLD (V-shaped) | 29.28 | 1.00 |
| flat 0.2 | 11.97 | 0.41 |
| **flat 0.5** | **29.93** | **1.02** |
| flat 1.0 | 59.87 | 2.04 |
| flat 1.5 | 89.80 | 3.07 |

`GOLD_PENALTY_SCALE = 0.5` chosen because it reproduces the old schedule's aggregate magnitude almost exactly on real trajectories — fixes the false premise and the mistiming bug without also gambling on an unvalidated new magnitude. Gold-events skew low already (median leftover = 0, p90 = 7) — this checkpoint mostly isn't hoarding, so the fix matters mainly for the tail and for future checkpoints, not as a wholesale reward-magnitude change.

**Balance re-verified** (same 24 games, 192 player-game rows): mean|dense_plus_shaping| = 0.823 vs mean|placement_reward| = 2.125 per player-game → ratio 0.39, comfortably under the ~0.6-0.8 band prior sessions treated as "meaningful but not dominating." At flat 0.5, a full 10-gold purse costs -0.075/turn (half of `WIN_REWARD=0.15`) at every round, not just late.

Removed `_gold_penalty_scale`/`GOLD_SCALE_FADE_ROUND`/`GOLD_SCALE_FLOOR`/`GOLD_SCALE_LATE_RAMP`/`GOLD_SCALE_LATE_CEIL` entirely (no other callers anywhere in the codebase — grepped to confirm). `CLAUDE.md`'s reward-shaping section (`_end_of_turn_reward`) and the balance-mandate paragraph updated to match.

**Current state:** Change is a pure reward-shaping edit, not yet trained against — no live run affected (no instance currently running). Smoke-tested: a scripted-only game (4 Heuristic + 4 Greedy) runs end-to-end post-edit with no errors.
**Open questions / next steps:**
- Not yet validated by an actual training run — this is a measured-on-trajectories sizing, not an outcome measurement. Worth checking `level_rate`/board_size/placement together after the next run, per CLAUDE.md's standing guidance on retuning economy incentives.
- The `GOLD_PENALTY_SCALE=0.5` measurement used one checkpoint (`live_bg_agent_ppo.pt`, ~u2369+). Gold-hoarding behavior may differ across checkpoint strength; re-measure if a future policy shows the old hoarding pattern again.
---

---
### 2026-09-03 — Ghost fights, Rao-Blackwellised combat reward, damage-coef retune
**Files changed:** `env/game_loop.py`, `env/matchmaker.py`, `CLAUDE.md`
**What was done:** Three changes to the combat reward path, each sized by
measurement on 40 games with the real greedy/heuristic seat mix before being
committed.
(1) **Ghosts are fought for real.** `step_combat` short-circuited every ghost
matchup to an automatic win with zero damage — a free `WIN_REWARD` on 7.5% of
all pairings (3.92 per lobby-game, rounds 9–16, ~16.6-round mean game).
`Matchmaker.get_ghost` had been written for exactly this and was never called
from anywhere. Ghosts now resolve to the **most recently eliminated** player
(user's call; uniform-random left a round-8 corpse in the pool all game) and
run through `BGCombatSim` normally. `pair_players` returns a real dead
player's id; `-1` now means only "nobody has died yet". Also fixed the
announcement path so `next_opponent_id` points at the ghost — it was `None`,
which encoded an *empty* opponent board, so simulating ghost fights without
this would have added lethal risk while keeping the agent blind to it.
(2) **Rao-Blackwellised combat reward.** `compute_round_reward` takes an
`outcome_dist` and pays the outcome/damage terms at their expectation over the
combat distribution instead of the single sampled roll. Free — `BGCombatSim`
already runs `COMBAT_SIM_TRIALS = 8` full combats and `step_combat` was
collapsing them to one coin flip purely to label the reward. Dynamics stay
sampled deliberately.
(3) **Damage coefficients retuned in opposite directions**, both having
measured at 0.0% of reward variance. `DAMAGE_TAKEN_COEF` 0.05 → 0.6 raw
(0.015 → 0.18): it telescopes, so its *lifetime* maximum was 0.015 against a
placement span of 8 — inert, not weak. `DAMAGE_DEALT_COEF` 0.05 → 0: it does
not telescope (unbounded positive income) and triple-counts `WIN_REWARD`,
`RANK_DELTA_COEF` and `board_potential`.
Also fixed dead code in `Matchmaker`: the ghost-recipient filter tested
`last_round_pairs.get(pid) == -1`, but that dict is built with `if b != -1`,
so it was always empty and always fell through to uniform choice; now avoids
giving the same player a ghost twice in a row.

**Measurements (40 games each, greedy/heuristic mix):**
- ghost frequency before fix: 3.92/lobby-game, 7.5% of pairings, round p50=12
- ghosts after fix: 93% win / 4% loss / 3% tie, mean `win_prob` 0.919, 13.7
  damage on losses, board visible to recipient in 93% of shop actions
- Rao-Blackwell: unbiased (mean gap +0.0008); removes 22.6% of per-combat
  reward variance but only **2.3% of total return variance** — 76.5% of
  combats are already near-decisive (`win_prob<=0.125` or `>=0.875`) and
  placement variance (sd 2.50) dwarfs combat variance (sd 0.63) by design
- per-round reward variance decomposition: outcome 44.6%, rank 18.4%,
  damage_taken 0.0%, damage_dealt 0.0% (rest is outcome/rank covariance)
- dense/placement ratio 0.22 → 0.25 (band 0.6–0.8), mean dense −0.154

**Honest correction:** the ghost fix was initially framed as "the agent is
told its board doesn't matter in the window where board strength decides
placement." The measurement does not support that. Ghosts really are
near-certain wins (dead boards are weak *by selection*), so the old auto-win
approximated a ~92%-win fight; expected `WIN_REWARD` moves 0.150 → ~0.138.
It is a correctness fix, not a large behavioural lever. The substantive
change in this batch is the damage-coefficient retune.

**Current state:** Restarted on vast.ai contract 49669406 (ssh9.vast.ai:29406)
from `bg_agent_ppo.pt` at updates=2549 / steps=22,894,433 / games=122,280.
Verified after restart: 0 exceptions, `update_count` continuing correctly
(2549 → 2558), steps 23.02M. Pre-restart checkpoint backed up locally as
`checkpoint_backups/preRB_u2541_bg_agent_ppo.pt` (loads clean, u2544). Local
tmux `bgsync` + `bggraph` still running.

**Open questions / next steps:**
- The critic was trained on the OLD reward scale (damage term 12x smaller,
  damage_dealt present). Expect a value-loss transient and a visible
  discontinuity in the reward series at update 2549 — do not misread it as a
  regression. Watch `eval_greedy`/gauntlet, which no shaping term can inflate.
- Gauntlet still has only 2 references (`ref_u500`, `ref_u2400`); the 5.22
  placement anomaly is still unexplained and still needs a fuller spread.
- `_accumulated_rewards` in `env/game_loop.py` is still dead code (initialised,
  never written) — delete it.
- The `RANK_DELTA_COEF` term is 18.4% of per-round reward variance and cannot
  be Rao-Blackwellised cheaply (it depends on the whole lobby's joint
  trajectory). Unaddressed.
- Placement variance (sd 2.50) dominates total return variance ~16:1 over the
  dense terms. The lever there is a better critic, not reward engineering.
---

---
### 2026-09-03 (addendum) — Gauntlet history x-axis was misaligned on resume
**Files changed:** `run_fresh_training.py`
**What was done:** Found while reading the first post-restart metrics: the
gauntlet's two real data points were being charted ~2350 updates too far
left. `eval_gauntlet_placement`/`eval_gauntlet_elo` were added to the code
*after* this run had already accumulated 47 eval points, so on resume they
loaded with length 7 against `eval_updates`' 54, and both the trainer-side PNG
and `tools/live_graph.py` pair them positionally with
`zip(eval_updates, series)` — which silently pairs from the LEFT. The 5.22
measured at update 2400 was drawn at update 50; the 4.19 measured at 2600 was
drawn at 250. The append path was never wrong (both lists are appended in
lockstep during a run); only the resume path was. Fixed by left-padding every
eval-aligned series to `len(eval_updates)` on resume, which also covers any
future late-added eval series. Verified on restart: "left-padded
eval_gauntlet_placement with 47 None to align with eval_updates (54)".

**Note on the metric itself:** the gauntlet number was never wrong, only its
x-position — so the 5.22-at-u2400 anomaly stands as a real measurement, and
the u2600 reading of **4.19 (elo +79)** is a genuine improvement on it,
landing inside the ~4.0-4.2 triangulated expectation.

**Current state:** restarted from `bg_agent_ppo.pt` at updates=2731 /
steps=24,519,279 / games=130,992. 0 exceptions.

**Open questions / next steps:**
- Next gauntlet fires at update 2800; confirm it charts at the right x.
- Gauntlet still has only 2 references; a full 7-checkpoint spread is what
  makes the Elo meaningful.
---

---
### 2026-09-03 (addendum 2) — Reroll penalty flattened; two factual corrections
**Files changed:** `env/game_loop.py`
**What was done:** The flat gold penalty (this session, earlier) and the
escalating reroll penalty were fighting each other. With a full board and no
affordable buy, reroll is the only gold sink -- but the old escalating cost
(0.015 base, +0.015/reroll past 2) priced even the FIRST reroll as a net loss
against holding the gold (holding 1 gold costs 0.0075; the old reroll cost
alone was 0.015). Measured live on the current checkpoint: 93% of round-13+
end-of-turns banked >=5 gold (mean 7.47) with buy/reroll/freeze all legal and
only 3.2/30 actions used that turn -- the policy was correctly solving the
reward as written, not failing to converge.

Per explicit user spec ("rerolling should give lower cost than floating gold
overall, except for the freeze-to-afford-a-minion case, which should stay
fine"): REROLL_PENALTY_BASE 0.015 -> 0.003 scaled, REROLL_PENALTY_STEP -> 0
(escalation removed). FREEZE needed no change -- it has never carried a
penalty beyond the ordinary flat gold charge.

**Verified rigorously, not just argued, against the user's follow-up worry
that this could let the agent get stuck only rerolling:** a plain reroll
changes ps.shop only, never ps.board or ps.tavern_tier, so Phi(s) is
EXACTLY unchanged by it. That makes the existing potential-shaping "discount
drag" term (already derived for REORDER_COST) an exact per-reroll tax of
0.003 + 0.0045*Phi, which is <= the 0.0075 gold-hold cost at EVERY reachable
state (Phi in [0,1] always), not just on average -- so reroll is never
reward-optimal as a substitute for a real BUY (BUY's own potential payout is
unbounded by this tax), only as a substitute for hoarding gold when nothing
is worth buying.

**Two errors caught by the user and corrected in the same session, both
recorded in the code comment now:**
1. Claimed ps.max_gold was "fixed, non-extendable" at 10. Wrong -- it's a
   per-player field; trinket_handler.py's max_gold_increase/max_gold_per_round
   effects (Bob's Tip Jar +4, Goblin Wallet +1/turn) raise it at runtime,
   capped only at min(20, ...). Didn't change the safety argument (only
   needed SOME finite ceiling), only the stated number.
2. While fixing (1), confirmed a real, separate symbolic-layer gap, flagged
   inline and left unfixed (out of scope tonight): Snare Trapper (T4) and
   Selfless Sightseer (T5) are MINIONS with "increase maximum gold" battlecry
   text, but bg_card_pipeline.py only ever parses that text inside
   parse_trinket_effect()/_TRINKET_RULES -- a trinket-only parser -- so no
   code path emits the effect for a minion card. Goblin Wallet and Bob's Tip
   Jar are genuinely Trinkets and ARE correctly wired; these two minions are
   not.

**Current state:** deployed and restarted on vast.ai contract 49669406 from
bg_agent_ppo.pt at updates=2934/steps=26,535,285. Verified after restart: 0
exceptions, checkpoint advancing normally (u2940, 26.7M steps at last check).

**Open questions / next steps:**
- Watch the training log's REROLL/BUY action-share and board_size over the
  next several hundred updates to confirm the flattened cost doesn't shift
  reroll share up at the expense of BUY in practice (the reward-design
  argument above says it shouldn't, but that's a design guarantee, not yet
  an observation under the new incentive).
- Missing minion battlecry wiring (Snare Trapper, Selfless Sightseer), flagged
  above as out of scope for tonight, was fixed concurrently by another active
  session (bg-agent-dc) directly in symbolic/effect_handler.py's on_play()
  while this session was still writing this entry -- hand-wired by name
  rather than extending the trinket regex parser to minions (minion
  battlecries already dispatch by name there), with Snare Trapper's Choose
  One approximated as a 50/50 RNG pick, consistent with this file's existing
  approximation for other unimplemented-choice cards (Clever Castaway). Left
  for that session to test and commit; not folded into this session's commit.
- Also answered: why gauntlet/7-past-selves eval runs far less often than
  greedy/heuristic. A gauntlet game runs 8 neural forward passes (policy +
  GAUNTLET_SIZE=7 other checkpoints) vs 1 for greedy/heuristic (policy + 7
  cheap scripted agents) -- ~8x cost/game. An earlier version at greedy's
  cadence (96 games/50 updates) measured ~25% of total run wall-clock, hence
  the throttle to 32 games/200 updates. No code change requested.
---

---
### 2026-09-03 (addendum 3) — Snare Trapper / Selfless Sightseer max-gold wiring fixed
**Files changed:** `symbolic/effect_handler.py`, `env/game_loop.py`
**What was done:** Closed the gap flagged (but deliberately left unfixed) in
addendum 2: Snare Trapper (T4) and Selfless Sightseer (T5) are minions whose
battlecry text claims to raise maximum Gold, but `bg_card_pipeline.py` only
ever parses "increase your maximum gold" inside
`parse_trinket_effect()`/`_TRINKET_RULES`, a trinket-only parser with no hook
for minions. Rather than extending that regex parser to minions, wired both
directly into `symbolic/effect_handler.py`'s `on_play()` name-keyed dispatch,
matching how every other minion battlecry in that file is handled (e.g.
`shellcollector`'s gold gain). Selfless Sightseer applies its +1 max Gold
unconditionally (scaled by Brann's `times`, same convention as the rest of
the file); no Duos/team concept exists in this engine, so "your team's
maximum Gold" just means the caster. Snare Trapper is a Choose One
("Get a random Quilboar; or Increase your maximum Gold by 1"), and no Choose
One decision mechanic exists anywhere in this codebase -- every other Choose
One card (Crater Miner, Intrepid Botanist, Sly Infiltrator, Sprightly Scarab,
Fearless Foodie, Veteran Brigand, Thorned Trailblazer) is likewise
unimplemented. Building agent-facing choice infra (a new decision point, plus
policy/masking changes) for one card was judged out of scope for this fix, so
the missing choice is approximated with a 50/50 RNG pick between the two
branches, following the same "no infra exists, approximate it" convention
already used elsewhere in `on_play`/`on_activate` (e.g. Clever Castaway's
Discover approximation). Both branches are otherwise fully real: the gold
branch reuses the same `min(20, ps.max_gold + ...)` cap as Bob's Tip Jar /
Goblin Wallet; the Quilboar branch reuses `_bc_draw_tribe` at
`tier=ps.tavern_tier`. Verified with a standalone script constructing
`PlayerState`/`MinionState`/`EffectHandler` directly: Selfless Sightseer
raises `max_gold` 10->11 and respects the 20 cap; Snare Trapper's gold branch
(RNG forced) raises it 10->11; its Quilboar branch (RNG forced, no
`tavern_pool`) correctly leaves `max_gold` unchanged and no-ops safely.
**Current state:** Both cards now genuinely raise `ps.max_gold` when played.
The stale "out of scope, flagged for a separate pass" comment block in
`env/game_loop.py` (originally added in addendum 2, just above the reroll
cost formula) has been rewritten to describe the fix instead of the gap.
**Open questions / next steps:**
- The Choose One mechanic itself remains entirely unimplemented for all 8
  affected Quilboar/Dragon cards; Snare Trapper's 50/50 RNG approximation
  papers over that for max-gold purposes only. A real fix would need a new
  agent-facing decision point (likely modeled on `discover_pending`'s
  pause-shopping pattern) and is a materially larger change than this one.
- No formal test suite exists in this repo to add regression coverage to;
  verification here was a one-off manual script, not a committed test.
---

---
### 2026-09-03 (addendum 4) — Flagged: accurate "choosing" mechanic, deferred to a future Opus session
**Files changed:** `symbolic/effect_handler.py` (comment only, no logic change)
**What was done:** User asked for accurate implementation of every "choosing"
effect (Choose One battlecries, and target selection for buffs/effects that
are real player decisions in Hearthstone) but explicitly wants it flagged now
and implemented later in a dedicated session with Opus, not built today.
Audited the current CARDS.md pool to make the flag concrete rather than
vague, and recorded it as a module-level comment at the top of
`symbolic/effect_handler.py`. Three buckets identified:
- **Choose One battlecries** (pick one of two whole effects): no mechanic
  exists at all. 7 affected minions (Crater Miner, Intrepid Botanist, Sly
  Infiltrator, Sprightly Scarab, Fearless Foodie, Snare Trapper, Veteran
  Brigand) plus Thorned Trailblazer (a meta-card on top of the mechanic) plus
  two token sources (Bramble Tunneler's Rally, both Glass of Perspective
  trinkets) that grant "a random Choose One card" needing a real payload.
- **Real target-choice effects** currently resolved by RNG or a fixed
  heuristic instead of the agent: Mind Muck (heuristic: highest-ATK Demon),
  Suspicious Prisonguard and Tyrael (both `_buff_random`/`rng.choice` on
  "another minion" -- not actually random in the real card text), Clunker
  Junker (not implemented at all), and 5 Spellcraft spells whose generic
  `buff_one` handling in `_cast_spell` picks `self._rng.choice(ps.board)`
  instead of a real target.
- Confirmed genuinely-random effects ("Get a random X") are correctly RNG
  already and are explicitly out of scope.
Also surfaced, separately, while auditing: several on_play name-keyed
branches (waxridertogwaggle, deflectobot's battlecry, masterofrealities,
gemsmuggler, murozond, goldgrubber-on-sell) don't match any card in the
current 275-card CARDS.md pool -- likely drift from a retired/renamed card.
Harmless dead code, not fixed now, flagged in the same comment for a
separate pass.
**Current state:** No behavior changed. The reusable pattern for a real
agent choice already exists (`ps.discover_pending` + mask-to-BUY-only +
candidates encoded into shop slots 0-2, see `env/game_loop.py`'s "Discover
in progress" handling and `agent/policy.py`'s `_discover` masking) and is
the documented starting point for whoever picks this up.
**Open questions / next steps:**
- Implement a generalised "pending choice" mechanism (pause flag + mask
  override + observation encoding) that extends the discover pattern to
  also carry a Choose-One's two *effects* as options, not just minion
  candidates -- this is the actual scoped-out work, intended for a future
  session using Opus.
- Separately: confirm and prune the stale on_play dispatch branches noted
  above once the choosing work is underway (unrelated but adjacent).
---

---
### 2026-09-03 (addendum 5) — Per-round action-mix tracking (round-bucket breakdown)
**Files changed:** `agent/ppo.py`, `train.py`, `run_fresh_training.py`
**What was done:** User asked how to track which actions are chosen during
training, specifically a per-turn breakdown of the most common actions.
Existing infra already tracked the FLAT action mix (per-game and per-update,
`action_type_game_rates` / `update_action_rate_avg`, plotted as a stacked
panel and printed each batch) — what was missing was breaking that mix down
by round number, since no code path carried round number onto a stored
transition at all.
- `agent/ppo.py`: added `Transition.round_num: Optional[int] = None`, a
  metadata-only field following the exact precedent `traj_id` already set
  (not consumed by the network). Threaded through both `collect_transition`
  and `store_transition`.
- `train.py`: `PPOAgent.record_transition` / `record_transition_precomputed`
  now read `round_num` off `obs["player_state"].round_num` (the observation
  dict already carries the live `PlayerState` — see `game_loop.py`'s
  `_get_observation`) and pass it through. No observation/network shape
  change; round_num was deliberately NOT added to `scalar_context`, since
  that would touch the policy network's input and needs no new information
  the network doesn't already have room for use of round_num elsewhere.
- `run_fresh_training.py`: added `_round_bucket()` (width-4 buckets:
  R1-4/R5-8/R9-12/R13-16/R17-20/R21+, catch-all top bucket -- median game
  length ~15-21 rounds per CLAUDE.md, so raw per-round tracking would be
  sparse/noisy late-game). Added `round_bucket_action_game_rates` (per-game,
  mirrors `action_type_game_rates` with an extra bucket dimension) and
  `update_round_bucket_rate_avg` (per-update aggregate, same rolling-window
  pattern as every other `_avg`/`_std` pair here), both persisted to history
  JSON with the same string-keyed nested-dict convention already used for
  `action_type_game_rates`/`update_action_rate_avg`, and resumable from an
  older history file that predates this key (`.get(..., {})` throughout).
  Added a compact console line (`by-round (top3, last30g): ...`) printed
  alongside the existing cumulative/recent~200 action-mix lines, and a
  heatmap panel (round bucket x action type, most recent update) in the
  previously-empty `axes[1][6]` slot of the training dashboard PNG.
**Verified:** matplotlib isn't installed in this sandbox (training runs on
vast.ai, per the vast-ai-training skill) so the heatmap panel wasn't
rendered end-to-end here. Everything else was: a standalone script built a
tiny real `BGPolicyNetwork`+`PPOTrainer` and confirmed `round_num` survives
`collect_transition`/`store_transition` into the buffer; a stub trainer
confirmed `PPOAgent.record_transition`/`record_transition_precomputed`
correctly extract `round_num` from `obs["player_state"]` (and degrade to
`None` without crashing when `player_state` is absent, e.g. terminal
transitions for eliminated players); the bucket-edge-case math
(`_round_bucket` at 1/4/5/8/9/.../40/None) and the per-game/per-update
NaN-safe aggregation logic were both re-verified against expected values in
isolation; and the history JSON's nested-dict round-trip (including
resuming from an old history file missing the new key entirely) was
confirmed to restore exactly. The matplotlib panel itself is straightforward
`imshow`/`text` usage matching this file's existing chart idioms, but has
not been visually inspected.
**Current state:** Fully wired but not yet exercised by an actual training
run (this session doesn't have a live run to attach to). The next
`run_fresh_training.py` run (fresh or resumed) will start populating the new
series automatically.
**Open questions / next steps:**
- Watch the first live run's console `by-round` line and the dashboard
  heatmap to confirm the panel renders sensibly (font sizes / cell count
  were chosen by eye, not measured against the actual saved PNG).
- Not done, out of scope for this ask: no changes to `tools/build_dashboard.py`
  (the separate untracked web-artifact dashboard) -- it wasn't asked for, and
  it's the user's own in-progress work; the history JSON schema stays
  compatible with it (only new keys added) if that dashboard is extended
  later.
---

---
### 2026-09-03 (addendum 6) — Round-bucket action mix wired into the live localhost dashboard
**Files changed:** `tools/live_graph.py`
**What was done:** User asked whether addendum 5's per-round action-mix
tracking would show up at the local live dashboard (http://127.0.0.1:8420,
served by `tools/live_graph.py`, fed by `tools/sync_from_vast.sh`) and
update live. Answer required checking: that script is a SEPARATE, hand-rolled
inline-SVG dashboard with its own `build_series()` extracting an explicit
whitelist of history-JSON keys -- it does NOT automatically pick up new keys
just because they exist in `data/fresh_training_history.json`. It does
already embed the full `run_fresh_training.py`-generated PNG verbatim
(`data/training_progress.png`, copied into `data/live/` on every rebuild) in
a collapsed "Trainer-side PNG" `<details>` section, so the round-bucket
heatmap panel added to that PNG in addendum 5 was ALREADY going to appear
there automatically, live, every rebuild -- just tucked inside a
click-to-expand section, not as a first-class native chart matching the
page's own theme/tooltips/hover like every other panel.
Added that native treatment: `build_series()` now also extracts
`update_round_bucket_rate_avg` into a `round_buckets` list (one entry per
bucket with data so far, each `{label, x, series}` in the same shape as the
existing flat `actions` block -- a JSON array rather than a dict specifically
so bucket order can't be disturbed by JS's numeric-string-key reordering
quirk). The front-end `render()` function appends one small chart card per
round bucket right after the existing flat "Action mix" card, reusing the
existing `card()`/`lineChart()` SVG primitives verbatim (top-3 action types
by mean share + a folded "other", exactly mirroring the flat panel's own
folding logic) -- no new charting code, no new CSS, just the same pattern
looped per bucket. Buckets with no data yet (e.g. R21+ early in a run) are
omitted rather than rendered empty.
**Verified:** monkeypatched the module's `HISTORY`/`CHART_PNG`/`LIVE_DIR`
path constants to a scratch directory (never touched the real, gitignored
`data/fresh_training_history.json` this machine already has from a prior
run) and called `build_series()` directly against synthetic history data:
confirmed all 6 buckets appear with correctly-shaped `{label, x, series}`
entries when fully populated; confirmed graceful, crash-free degradation
against an old history file missing the key entirely (`round_buckets: []`);
confirmed a realistic partial-run shape (early buckets fully populated, a
late bucket with only some updates, R21+ never reached with all-empty
per-action lists) correctly omits the untouched bucket and doesn't crash,
with output JSON-serializable in every case. The inherited x/y length-
mismatch behaviour when different action types within one bucket have
different real series lengths (same simplification the pre-existing flat
`actions` block already has -- one shared `x` sized to the longest series)
was confirmed non-fatal but not fixed, since fixing it would mean changing
the existing flat panel's behaviour too and wasn't asked for. The JS side
was reviewed by hand against the working flat-panel block it mirrors
(no `node` available in this sandbox to actually execute/lint it).
**Current state:** `tools/live_graph.py --build` (or `--serve`) run against a
real synced history file will now render up to 6 additional native "Action
mix — round RX-Y" cards on the grid, alongside the pre-existing embedded-PNG
copy of the same data. Not yet observed against a live run from this session.
**Open questions / next steps:**
- Watch the page against a real live run to confirm the 6 extra cards read
  well in the existing grid layout (`grid-template-columns:
  repeat(auto-fit,minmax(400px,1fr))` should just reflow them in, but this
  wasn't visually confirmed).
- `tools/build_dashboard.py` (the separate untracked web-artifact dashboard)
  still hasn't been touched -- same reasoning as addendum 5: not asked for,
  and it's the user's own in-progress work.
---

---
### 2026-09-03 (addendum 7) — Action-mix charts show all 10 action types, no "other" bucket
**Files changed:** `tools/live_graph.py`
**What was done:** User found the action-mix panels unclear because lower-
frequency action types were folded into a single "other" line, hiding what
was actually in it. Consulted the `dataviz` skill rather than improvising:
its non-negotiables say a >8th categorical series is never a generated hue —
it must fold to "Other", split into small multiples, or use composite
encoding — and folding is exactly what the user was objecting to. Chose
composite encoding (color + dash) over small multiples, since the request
was specifically to see every action type on one chart, not spread across
more cards.
Expanded the page's palette from the 3-slot "all-pairs-safe" subset it was
using to the reference palette's full 8-slot categorical set (blue, orange,
aqua, yellow, magenta, green, violet, red — `--s4` through `--s8` added to
all three theme blocks: bare `:root`, the `prefers-color-scheme: dark` media
block, and `:root[data-theme="dark"]`). The file's own comment had
mis-scoped the earlier 3-slot choice as "the all-pairs-safe subset" — that
restriction is for scatter/small-multiples forms; this file's charts are
lines with a legend, which validate against the full 8 slots per
`references/palette.md`. Added a fixed action-index -> {color, dash} map
(`ACTION_STYLE`, `S8`), assigned once in `ACTION_TYPE_NAMES` order and never
re-ranked by current value (color follows the entity, not its rank, per the
skill's own non-negotiables — the OLD top-3-by-mean-this-update logic
violated that: a given action's color could change over time as rankings
shifted). ACTIVATE and REORDER (indices 8-9) reuse slots 1-2 with a dashed
stroke. Rewrote both the flat "Action mix" card and the per-round-bucket
cards (added last session) to build all 10 series directly via a shared
`actionSeries()` helper — no top-3/other folding logic left anywhere.
Also capped direct end-of-line text labels at <=4 series (the skill's own
"<=4 are also direct-labeled" rule) since 10 overlapping end-labels would
have been its own clutter problem; the end-marker DOT stays on every line as
a cheap anchor, and the legend (now dash-aware: a dashed swatch for the two
composite-encoded series) plus the existing hover tooltip carry full
identification for the rest.
**Verified:** Python file parses; restarted the local `bggraph` tmux session
(this time via `tmux kill-session` directly, not Ctrl-C into a non-shell
pane — avoiding the mistake from the previous restart) and confirmed via
curl: the new JS (`ACTION_STYLE`/`S8`/`actionSeries`) and all 8 `--s4`..`--s8`
CSS variables in both themes are being served; `series.json`'s action names
in both the flat block and all 6 round-bucket blocks match the new JS
`ACTION_NAMES` array exactly (BUY/SELL/PLACE/REROLL/FREEZE/LEVEL/HERO_PWR/
END_TURN/ACTIVATE/REORDER), so every name-keyed lookup resolves. Could not
visually render or screenshot the page (no browser/`node` in this sandbox,
same limitation as the previous session) — the palette values themselves are
copied verbatim from the skill's pre-validated reference instance
(`references/palette.md` states this exact 8-hue order passes every adjacent-
pair gate in both light and dark), so only the JS wiring was newly at risk,
and that was reviewed by hand plus confirmed via the data-shape checks above.
This file is local-only tooling (serves the localhost dashboard from synced
history data) and does not run on the remote training instance, so no
remote redeploy was needed for this change.
**Current state:** http://127.0.0.1:8420 now shows every action type
distinctly on the Action Mix panels (flat + all 6 round-bucket cards), fixed
colors/dash per action, no "other" bucket anywhere in this file.
**Open questions / next steps:**
- Actually look at the rendered page (a real browser, not curl) to confirm
  the 10-series chart reads well at a glance and the dash pattern is visible
  at the line's stroke-width — this was verified structurally, not visually.
---

---
### 2026-09-04 — Real choosing mechanics, Hearthstone-fidelity dynamics fixes, pointer-network policy
**Files changed:** `agent/policy.py`, `env/player_state.py`, `env/game_loop.py`, `env/triple_system.py`, `symbolic/effect_handler.py`, `symbolic/combat_sim.py`, `train.py`, `run_fresh_training.py`, `tools/live_graph.py`, `tools/trace_game.py`, `CLAUDE.md`

**What was done:**

*1. Real choices (the deferred `choosing_mechanic` work).* Added `PendingChoice`
to `PlayerState` plus two action types — `10 CHOOSE_TARGET` (board pointer) and
`11 CHOOSE_OPTION` (shop pointer) — each with its own scorer head. Legal only
while `ps.choice_pending` is set, and then the only legal type. All seven
"Choose One" minions now work (six previously did *nothing* when played; Snare
Trapper was a 50/50 coin flip); Choose-One branches are rendered as pseudo-cards
in the shop zone so the existing 44-dim encoder describes them ("+4 Attack and
Windfury" -> attack=4 + windfury bit). Every "choose a minion" effect (Mind Muck,
Suspicious Prisonguard, Tyrael, targeted Spellcraft, Blood Gem) now asks the
agent instead of calling `rng.choice`; Clunker Junker went from unimplemented to
working. Thorned Trailblazer implemented as a per-turn charge that applies BOTH
branches. Choices QUEUE (Brann doubles a Choose One into two independent
choices; Sprightly Scarab's branch raises a nested target choice). A choice with
one legal target is applied immediately rather than burning an action, and
`_force_resolve_choices` guarantees a choice can never be dropped by exhausting
the action budget. Triple rewards became a real Discover (they used to take
`candidates[0]` unconditionally) and their minion now gets keywords/activate_cost
from card_defs (it arrived with no Taunt/Divine Shield/Activate before).

*2. Model.* `forward()` now returns pointer logits as ONE `[B, N_ACTION_TYPES, 24]`
stack indexed by action type instead of parallel tensors that three call sites
each had to select between with their own `if type == 8` — that shape makes the
sample/evaluate-mismatch bug class (hit twice, see 2026-08-31/09-01) structurally
unrepresentable. Added a **pointer-network query term**: each of six scorer roles
gains a scaled dot-product against a per-role query emitted from `[CLS || scalar]`,
zero-initialised so it is an exact no-op at init. `num_layers` 4 -> 6. Added
`POLICY_ARCH`/`make_policy()` because seven call sites spelled the architecture
out by hand and a single disagreeing number does NOT raise — `load_checkpoint`
catches it, warns, returns False, and training silently restarts from zero.

*3. Hearthstone fidelity (audited, all silent divergences).* Tavern upgrade no
longer rerolls the shop or clears `frozen`, and the new tier's cost starts at
full base (it started at `base-1`, and round-start decay took another 1, so
levelling was permanently a gold cheap from the turn after every upgrade).
FREEZE is a shop toggle that no longer ends the turn. Combat first-attacker is
now the side with MORE minions (was a coin flip) — the largest behavioural
change, since board width is valuable in BG partly *because* it buys the first
attack, and the coin flip meant nothing ever taught the policy to go wide.
Start-of-combat order follows the same precedence. Loss-damage fallback fixed
(was the LOSER's tier + a minion COUNT; now the opponent's tier + summed
surviving minion tiers, matching `win_damage`). `round_history` now records both
sides of each pairing (it recorded one, so any statistic from it sampled half
the fights). End-of-turn costs are now charged on the forced turn-end when the
30-action budget runs out — otherwise burning the budget dodges the gold/hand
penalties entirely.

*4. Metrics/dashboard.* Added unspent gold at END_TURN, combat win rate, damage
taken per combat, tavern tier (plotted against board size), choice events/game,
and a Choose-One branch-0 share **collapse detector** — pinned at 0.00/1.00
means the choice head is not discriminating, which is exactly the pathology
ACTIVATE showed while sharing SELL's scorer. Dashboard extended to 12 action
types.

**Current state:** All gates pass — log-prob sampling/evaluation consistency
(worst 1.4e-05 over 1256 real transitions with every scorer deliberately
randomised to a different scale), choice-resolution unit tests (effect lands on
the pointed slot, Tyrael overwrites rather than adds, nested Choose One -> target
works, single-target shortcut does not pause), dynamics gates (upgrade keeps the
shop, cost curve 5/4/3/2/1 then 7/6, freeze does not end the turn, wider board
always first, equal boards 0.52 coin flip), a full PPO update, and a complete
shrunken training run end-to-end including the resume path. Rented vast.ai
instance 49799906 (i9-14900KF, 24 physical cores, RTX 4060 Ti, $0.1347/hr).

**Open questions / next steps:**
- This is a FRESH run, not a resume: the 29.5M-step checkpoint was trained on a
  materially different MDP (free shop refresh on level, coin-flip attack order,
  no choices) and a different architecture. Old checkpoints will not load.
- Expect `level_rate` to FALL vs pre-2026-09-04 runs — levelling is now a gold
  more expensive and no longer comes with a free refresh. That is the intended
  correction, not a regression.
- Watch the branch-0 collapse detector once `choice_events` accumulates;
  CHOOSE_* are rare actions (~0.3/game) so they need many games to show signal.
- Frozen shops still top up to `shop_size(tier)` next round. Left as-is
  deliberately — real-BG behaviour here was not confirmed with enough certainty
  to justify changing it.
- Account credit was $9.96 at rental time (~74h at this instance's rate).
---

---
### 2026-09-05 — Deep debug: width-blind potential caused a 155-Elo regression; fresh restart
**Files changed:** `env/game_loop.py`, `run_fresh_training.py`, `CLAUDE.md`, `agent/ppo.py`, `train.py` (last two = previous session's uncommitted bootstrap/KL fix, committed here)

**What was done:**

*Diagnosis.* The 2026-09-04 run had been REGRESSING for ~1,000 updates and no
metric in the log showed it. Gauntlet Elo (anchored to a fixed oldest ref, so
comparable over time): u400 `+92` -> u800 `+299` (peak) -> u1200 `+256` ->
u1600 `+148` -> u1800 `+144`. Placement vs the frozen reference went 1.81 ->
3.16 over the same span. The greedy/heuristic evals showed nothing because
they saturated at update ~350 (top1 0.92 from there on).

*Root cause 1 (the big one): `_board_stats_potential` was blind to board
WIDTH.* It was a flat sum of attack+health, so one 40/40 scored `Phi=0.571`
and seven 6/6s scored `Phi=0.562` — indistinguishable, while the 2026-09-04
first-attacker fix had just made width MORE valuable. Its docstring defended
the flat sum as guarding against hoarding 1/1s, but left the opposite
degenerate board unguarded, and that is the one the policy found. Instrumented
6 games on the live u1806 checkpoint (PPO seat only): **135 of 150 choices
(90%) were Suspicious Prisonguard's "+3/+3 to another minion"**, 22.5/game;
gold went **39.9% to REROLL, 24.0% to ACTIVATE (the pump), 13.6% to buying
minions**; board size sat at **4.42/7 and never filled** (p90 5.05). Per gold
the pump paid 0.0129 dPhi vs BUY's 0.0098 — the agent was correct to pump, the
potential was wrong. Consistent with the run-level metrics: `choice_events`
climbed 15 -> 64/game, combat win rate 0.558 -> 0.486, game length 24.8 ->
19.8 rounds. Note the 2026-09-02 `BOARD_SHAPE_STATS_SATURATION` 30->60 change
AMPLIFIED this: it was diagnosed from a Prisonguard-pumped board, and
desaturating made each pump pay more.

*Root cause 2: `bg_agent_ppo_best.pt` was frozen at update 309.* It was
selected on `mean(game_rewards[-10:])`, whose sd is ~0.8; `best_avg10=3.452`
was a lucky window at game 14,808 and stood for 1,500 updates. The run's
actual best weights (u800) were never saved anywhere.

*Root cause 3 (already fixed last session, uncommitted until now): bootstrap
rows in the policy loss.* At 0.3% of rows and batch 256, ~54% of ALL
minibatches contained a row with importance ratio ~5e8 carrying a
placement-sized advantage; after `max_grad_norm=0.5` that single row set the
direction of the whole step. Live for updates 1-1548.

*Fixes.* (a) `value` is now `BOARD_STATS_MINION_SCALE * sum_i (atk_i+hp_i)**
BOARD_STATS_MINION_EXPONENT` — concave **per minion, then summed** (never over
the board total, which would be width-blind the same way). EXP=0.7/SCALE=3.0
were fitted on 624 real end-of-turn boards so Phi's MEDIAN is unchanged (value
p50 34.0 -> 59.8, Phi p50 0.362 -> 0.499 at SAT=60): the shape changes, the
magnitude does not, so the validated dense-vs-placement balance carries over
and `BOARD_SHAPE_STATS_SATURATION` stays 60.0. Keyword/synergy bonuses
rescaled 3->5 and 5->9 purely to hold their existing share of `value` under
the 1.76x median shift. (b) `bg_agent_ppo_best.pt` is now selected on gauntlet
Elo (`best_elo`, persisted in history); `best_avg10` is still tracked but
selects nothing. (c) Committed the bootstrap/KL fix. (d) CLAUDE.md updated:
the board_potential spec, plus a new "Progress Metrics & Model Selection"
section recording which metrics are and are not trustworthy.

**Verified:** New ordering on equal-total-stat boards is 1x40/40 `0.518` <
2x20/20 `0.569` < 4x10/10 `0.620` < 7x6/6 `0.652`, with 7x1/1 at `0.362`
(still below the single 40/40, so the anti-hoarding property survives);
BUY-and-place now pays 3.17x a +3/+3 pump (was 1.62x), which per gold is 1.49x
in BUY's favour where it used to be 1.3x in the pump's. Phi stays in [0,1) at
7x400/400 (`0.974`). Smoke gates pass on a real 3-game/364-transition rollout:
potential shaping stays inside its telescoping bound on every game, bootstrap
rows are tagged and reach `policy_w` as zeros (6/364), and a full PPO update
runs with finite KL. Deployed with md5 parity checked on all four files.

**Current state:** The 2026-09-04 run is archived (remote
`archive/run_2026-09-04/`, local `checkpoint_backups/run_2026-09-04_regressed/`
including `peak_elo_ref_u900.pt`). A FRESH run is training on vast.ai
49799906 in tmux `train`; local sync loop `bgsync` restarted. Early sign is
right: from random init BUY is 12-15% of actions vs 8% in the regressed run.

**Open questions / next steps:**
- **REROLL was NOT retuned and is the #1 thing to watch.** It was 32% of
  actions and 40% of gold in the old run and is 31% from random init now.
  Deliberately left alone so the potential fix can be attributed cleanly:
  reroll spam may be a SYMPTOM of buying being underpriced. If `update_board_avg`
  does not climb well above the old 4.42 by ~update 400, revisit
  `REROLL_PENALTY_BASE`. Any escalation must stay under the 0.0075/gold
  end-of-turn holding cost at every reachable count (max 10/turn) to preserve
  the 2026-09-03 invariant — that caps a per-reroll step at ~0.0006.
- Watch `update_board_avg` (target: well above 4.42, toward filling 7),
  `update_choice_avg` (should NOT climb toward 38-64/game again) and gauntlet
  Elo at u400/600/800 against the old run's +92/+299.
- HERO_POWER is 8-10% of actions from random init; 10/10 sampled uses changed
  no observable state in the old checkpoint. Probably an artifact of the
  snapshot not covering `hero_power_used`/hero-specific state, but worth
  confirming that non-`active_noptr` heroes aren't burning a real action for
  nothing.
- `batch=0.1s` / `batch=0.0s` lines appear intermittently in the log for
  24-game batches, which is not physically possible — the batch timer is
  measuring something other than wall-clock for those. Cosmetic, but it makes
  throughput unreadable.
---

---
### 2026-09-05 (b) — Field-relative potential: fixing the sell/upgrade path
**Files changed:** `env/game_loop.py`, `agent/policy.py`, `CLAUDE.md`

**What was done:**

*Diagnosis.* The width fix earlier today worked on its target (Prisonguard pump
down 45%, gold-to-BUY up 43%, board now fills to 7 by round ~10), but the agent
still refused to sell-and-upgrade: from round ~15 its action mix was REROLL +
END_TURN and literally nothing else (measured 8 games: R16 reroll=50,
end_turn=5, everything else zero), with SELL at 2.0% of actions and 43% of all
gold burned on rerolls. Tested two candidate causes and used BGCombatSim as
ground truth for what a board change is actually WORTH. Shaped reward paid per
1.0 of real win probability gained:
    round  6 (add a 20 into an empty slot)   0.103
    round 14 (upgrade a 6 into a 32)         0.061
    round 18 (upgrade a 30 into a 70)        0.026   <- 4x under-priced
A late upgrade worth **+0.445 win probability was paid +0.0114**, barely more
than the 0.009 saved by not rerolling three times, while costing 2 net gold and
3 actions instead of 1. This refutes the "late upgrades are genuinely marginal"
reading I had argued the message before -- the sim says they are worth 31-45
win-probability points. `BOARD_SHAPE_STATS_SATURATION = 60` was a stand-in for
"a typical opponent", but real opponent boards grow ~8 (R1) -> ~190 (R16), so a
fixed constant made Phi saturate exactly when the board is full and upgrading is
the only lever left.

*Fix A (the cause).* The denominator is now the mean board value of the OTHER
alive players, floored at `BOARD_SHAPE_FIELD_FLOOR = 60.0`
(`_field_denominator` / `_refresh_field_values`). Phi becomes a ratio-to-field,
i.e. a Bradley-Terry-shaped win-probability proxy, which is what placement
actually depends on. Snapshotted ONCE per round so Phi stays a deterministic
function of state (a denominator drifting as other seats shop concurrently is
not) and so Phi moves only through the agent's own board within a turn. Self is
excluded from the mean -- including it lets the agent's own improvement inflate
its own denominator, damping the true gradient ~12.5% at n=8. Self-calibrating
by design: it reads the live lobby rather than a fitted per-round curve, so it
cannot go stale as the policy improves (the exact failure mode that killed the
round-indexed gold ramp on 2026-09-02).

*Fix B (credit assignment).* `BOARD_STATS_HAND_WEIGHT = 0.5` counts non-spell
minions in hand. BUY used to pay EXACTLY +0.0000 -- 3 gold for a shaped signal
of "nothing happened", with the whole payoff deferred to PLACE, so BUY read as
pure cost in a per-action advantage. Note this does NOT change the total value
of buy-and-place and cannot (Phi telescopes; measured +0.0188 either way) -- it
is purely smoothing, and it matters because "sell first, hope to buy better" is
a bootstrapping trap: if the policy rarely upgrades, V(post-sell) is low, so
A(sell) is negative, so it never learns to sell.

*Fix C.* `build_type_mask` now enables HERO_POWER only when the hero's
`power_type == "active_noptr"`, matching `step_shopping`'s own condition. 17 of
29 heroes are passive/null; for those the action burned one of the turn's 30
actions and only set `hero_power_used` (measured: 71% of sampled HERO_POWER
uses changed no state).

**Verified:** Calibration spread 4.02x -> 2.32x with late payoff roughly doubled
and the EARLY payoff unchanged (+0.0763 -> +0.0768), so the width fix's
early-game incentive survives intact -- by construction, since the floor equals
the old constant and the field mean only passes 60 around round 6, making this a
strict late-game correction. Phi stays in [0,1) at every extreme tested
(empty board, 7x400/400 against a weak field, 1x1/1 against a huge field).
Dense-vs-placement balance re-measured on the REAL seat mix (12 games / 96
player-games): ratio **0.502**, inside the <~0.6-0.8 band (up from 0.39 -- a real
increase, worth watching). Telescoping bound held on all 12 games; PPO update
runs clean (kl 0.048, 24/3682 bootstrap rows tagged). Added a reset() clear of
`_field_values` after confirming a reused BattlegroundsGame would otherwise
carry game N's endgame lobby into game N+1's initial Phi.

**Current state:** Deployed with md5 parity on all 5 files; training RESUMED
(not restarted) from update 1010 / 13.6M steps, previous checkpoint archived
remotely as `archive/pre_fieldphi_u1010.pt`. Running at update 1011+.

**Operational note:** vast.ai instance 49799906 was found **stopped** by the host
around 10:20 (credit was fine at $4.58, so not a billing stop). Restarted with
`vastai start instance`; workspace, checkpoints and gauntlet refs all survived.
One stale `/dev/shm/bg_snapshot_*.pt` worker error on relaunch, which the pool
rebuild path handled by itself. The local sync loop had died with the tmux
server and has been restarted -- it had synced through update 999, so nothing
was lost.

**Open questions / next steps:**
- **My "Elo is flat-to-down" read from earlier today was WRONG and I should
  not have leaned on it.** It was 2 gauntlet references and 32 games:
  +122 (u400) -> +74 (u600) -> +72 (u800) -> **+194.6 (u1000, new best)**.
  Do not call a trend off the gauntlet until there are >=4 refs.
- Watch, in order: gauntlet Elo (beat +194.6), `update_board_avg` (was 4.62),
  SELL rate (was 2.0% of actions), and the R15+ action mix (was reroll +
  end_turn only). Early post-resume signal is in the right direction on all of
  them (SELL 6->8%, BUY 9->12%, REROLL 33->26%, HERO_POWER 4->1%) but that is
  2 updates on a 200-game window that still contains pre-change data.
- `explained_var` will dip before recovering: the value function was fitted on
  the old Phi and the reward function just changed under it.
- Dense/placement ratio moved 0.39 -> 0.502. Still inside the band but it is the
  second dense change in one day; re-measure before any third.
- Reroll deliberately still NOT retuned. If R15+ is still reroll-only by
  ~update 1400, the cause is not pricing and I should look at whether SELL's
  pointer scorer can even identify the weakest minion.
---

---
### 2026-09-05 (c) — Reverted the field-relative Phi; fixed the instrument instead
**Files changed:** `env/game_loop.py` (reverted), `train.py`, `run_fresh_training.py`, `CLAUDE.md`

**What was done:**

*The field-relative Phi (2026-09-05b) was a regression and is reverted.* 600
updates / 8.8M steps of evidence, every axis worse: board_avg 4.586 -> 4.28-4.38,
choice_avg (the Prisonguard pump) 22.5 -> 28-30, BUY 0.115 -> 0.101-0.108,
SELL 0.083 -> 0.071-0.079, PLACE 0.116 -> 0.101-0.108, train placement
4.045 -> 4.15-4.26, combat winrate 0.492 -> 0.478-0.485. Gauntlet Elo
194.6 -> 188.7 -> 183.1 -> 163.3. The behavioural metrics don't depend on the
gauntlet, so they corroborate the Elo decline independently.

*Why my gates missed it.* The calibration gate measured dPhi for one player's
board change **while holding the field fixed** at a measured constant. In
self-play the field CO-EVOLVES with the agent: as the population's boards grow
the denominator grows with them, damping the board-building signal in a way a
fixed-field test cannot see. I had named this exact risk in the design
discussion ("a relative Phi only rewards out-building others... the absolute
signal is a much denser teacher") and then built a gate that could not test it.

*What the deeper investigation actually found.* Chasing the "agent won't
sell/upgrade" symptom further, three findings, all negative for the reward
hypothesis:
  - **The pointer scorers are near-perfect.** P(sell the 1/1 dud among six
    30/30s) = 1.000 in every board position; on a graded 2/2..40/40 board the
    mass is 0.54/0.46 on the two weakest and 0.00 elsewhere. P(buy the 40/40
    among five 2/2s) = 1.000. Representation is NOT the problem.
  - **The type distribution responds correctly.** On REAL in-game states
    (round>=10, 8 games): full board with a genuine upgrade in the shop ->
    P(SELL)=0.122, P(BUY)=0.058, P(END_TURN)=0.014; full board with no upgrade
    -> P(SELL)=0.017, P(END_TURN)=0.191. SELL responds to shop quality by
    **7.09x**, BUY by 5x. The agent recognises upgrades.
  - **The by-round profile is healthy.** tier 6 by round 12 (curve says 10),
    board 6.4-6.7/7 from round 11 on, hp 36-40 through round 17. The
    "board 4.42/7, never fills" framing from earlier sessions was WRONG -- that
    average is dragged down by rounds 1-5 where the board cannot be full.
  - The self-play league (`SnapshotPool`: rolling + milestone snapshots,
    recency-weighted sampling, thinning) is already sophisticated. Left alone.

*Conclusion: stop changing the MDP; fix the measurement.* Potential-based
shaping provably cannot change the value of the sell/upgrade cycle (Phi
telescopes -- the endpoints fix the sum, measured +0.0188 regardless of hand
weight), so Phi was never the lever for it. Meanwhile the only trustworthy
progress metric was running on 32 games. Demonstrated directly: the SAME policy
at the SAME seed scored mean placement **1.312 at n=32 and 1.562 at n=96**.

*Changes (measurement only, zero MDP change):*
  - `EVAL_GAUNTLET_GAMES` 32 -> 96 (placement SE 0.41 -> 0.23).
  - `GAUNTLET_EVERY` 300 -> 150. GAUNTLET_SIZE is 7 but the pool held only 5
    refs at update 1600 and would not have filled until 2100, so every gauntlet
    so far fitted Bradley-Terry over a different, growing opponent graph -- a
    large part of why successive Elo read 122/74/72/195.
  - Eval budget REALLOCATED, not just increased: `EVAL_GREEDY_GAMES` 64->32 and
    `EVAL_HEUR_GAMES` 32->16 (both documented as saturated and
    non-discriminating) fund `EVAL_REF_GAMES` 32->96. Net +12% eval cost.
  - `evaluate_policy` now returns `placement_se`, and both the EVAL and GAUNTLET
    log lines print `+/-2se`, so these numbers get read as intervals.
  - Kept from 2026-09-05b: the HERO_POWER mask fix (independently verified,
    does not touch Phi). Reverted: field denominator AND hand credit -- hand
    credit was confounded with the field change, and reverting both puts the
    u1000 value function back in the exact MDP it was fitted on.

**Verified:** `load_checkpoint` returns True on the u1000 checkpoint (a False
here silently restarts from zero) and `explained_var` is 0.809 immediately,
confirming the value function is back in its native MDP. Structural asserts
that the width fix is present and the field/hand constants are gone. PPO update
clean (kl 0.0031). `placement_se` verified non-nan and positive at n=32 and 96.
md5 parity on all deployed files.

**Current state:** Training RESUMED from `bg_agent_ppo_best.pt` (update 1000,
13.46M steps, Elo +194.6) -- the checkpoint the Elo-based selection added on
2026-09-05 preserved. That fix paid for itself: under the old avg10 criterion
the peak would have been lost. Regressed run archived remotely at
`archive/run_fieldphi_regressed/`. Gauntlet refs u1200/u1500 (frozen from the
regressed policy) deleted so the new run freezes its own; **ref_u300 is kept as
the Elo anchor, so the new run's Elo is directly comparable to the +194.6 bar.**
History reset to a minimal `{best_elo, best_avg10}` file after verifying all 72
history reads use `.get()` with defaults and none use `hist[...]`.

**Open questions / next steps:**
- **Do not touch Phi again without a co-evolution test.** Any shaping change
  must be validated in a setting where the opponent population moves too, not
  against a frozen field.
- The bar is Elo **+194.6** (anchored to ref_u300, so comparable). First
  gauntlet at update 1200 on 96 games; ignore anything inside +/-2se.
- Reroll still NOT retuned, and I now have evidence AGAINST retuning it: the
  policy rerolls 57% of the time in states where it has identified an upgrade,
  but it also correctly shifts 7x toward SELL in those states. That may be
  legitimate option value at 10 gold, not a defect. Needs a real
  stopping-rule analysis before touching the price.
- update_count (1000+) is offset from the fresh history's index (0+). Cosmetic
  only -- `eval_updates` records true update numbers so eval charts are correct.
---
