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
