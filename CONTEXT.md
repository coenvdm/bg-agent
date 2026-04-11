# bg_agent — Development Context Log

---
### 2026-04-10 — Drop ShopAnalyzer from notebook
**Files changed:** `explore.ipynb`
**What was done:** Removed the unused ShopAnalyzer from explore.ipynb — deleted the import/instantiation from the symbolic setup cell, the "Shop Analyzer — Buy Value Estimation" markdown heading, and the analyze_shop demo cell. No other cells depended on it.
**Current state:** Notebook is coherent end-to-end: dataset EDA → symbolic board analysis → card encoding → BC v2 → PPO forward-pass demo.
**Open questions / next steps:**
- ShopAnalyzer still exists as a module in symbolic/; delete or keep for later if buy-value signals are ever fed into the policy.
---

---
### 2026-04-10 — Simulator fidelity overhaul (Phases 1–3)
**Files changed:** `symbolic/combat_sim.py`, `symbolic/effect_handler.py`, `symbolic/hero_handler.py`, `env/player_state.py`, `env/game_loop.py`, `agent/policy.py`
**What was done:** Comprehensive simulator fidelity improvements across three phases. Phase 1 (combat accuracy): added on-attack auras (Twilight Watcher, Roaring Recruiter), on-damage auras (Hardy Orca, Iridescent Skyblazer, Trigore), Lord of the Ruins demon-damage aura, Shore Marauder passive, and Monstrous Macaw Rally. Also wired up the `game_buffs` system (`PlayerState`) for tribe-specific permanent buffs (Nerubian Deathswarmer, Dune Dweller, Felemental, Anubarak), and Blood Gem bonus tracking (`blood_gem_atk_bonus`/`blood_gem_hp_bonus`) for Prickly Piper/Moon-Bacon Jazzer. Phase 2 (shop-phase): added sell effects (Fire Baller, Snow Baller, Minted Corsair, Tad), `EffectHandler` now receives `TavernPool` for pool draws; `discover_pending` on `PlayerState` pauses shopping and lets the agent choose via `BUY(0/1/2)` — observation shows discover options in shop slots and action masking in `policy.py` restricts to BUY-only with pointers 0–2 during discover; token/item battlecries (Razorfen Geomancer, Shell Collector, Briarback Drummer, Refreshing Anomaly, Tavern Tempest, Archaedas); discover battlecries (Tiger Shark, Imposing Percussionist, Primalfin Lookout, Rodeo Performer); consume-shop battlecries (Picky Eater, Mind Muck, Furious Driver); shop-phase aura triggers (Timecapn Hooktail, Plankwalker, Mechagnome Interpreter). Phase 3: new combat deathrattles (Bassgill summons 4/5 Murloc, Scourfin +5/+5 to random friendly, Dramaloc +5 ATK to 2 Murlocs, Apexis Guardian +3/+2 to all Mechs); Rafaam post-combat copies a minion from the opponent board to hand; Tess draws a random pool card into the next shop.
**Current state:** Dry-run passes cleanly (2 games, ~1430 transitions, no errors). All simulator mechanics are wired end-to-end.
**Open questions / next steps:**
- Validate new mechanics with targeted explore.ipynb smoke tests (Dragon board + Twilight Watcher; Blood Gem with bonuses; discover flow).
- P3-A remaining: Vol'jin, Shudderwock, Maiev, Saurfang (targeted hero powers requiring action-space extension).
- Consider encoding `discover_pending` state explicitly in the scalar context so the policy can distinguish discover from normal shopping.
---

---
### 2026-04-10 — Fix Refreshing Anomaly free-refresh consumption
**Files changed:** `env/game_loop.py`, `agent/policy.py`
**What was done:** The Refreshing Anomaly battlecry was setting `ps._free_refreshes` but the reroll handler in `step_shopping` never checked it — free refreshes were silently ignored. Fixed the reroll branch to consume one free refresh before spending gold, and updated `build_type_mask` in `policy.py` so that reroll is masked as valid whenever `_free_refreshes > 0` even if the player has 0 gold.
**Current state:** Dry-run passes cleanly. Free refreshes from Refreshing Anomaly now work end-to-end.
**Open questions / next steps:**
- Smoke-test Refreshing Anomaly in explore.ipynb: play it, confirm `_free_refreshes == 2`, reroll twice without gold deduction.
---

---
### 2026-04-10 — Add nbstripout to keep notebook diffs clean
**Files changed:** `.gitattributes`, `explore.ipynb`
**What was done:** Installed `nbstripout` git filter and added `.gitattributes` to apply it to all `*.ipynb` files. Outputs and execution counts are now stripped from notebooks at `git add` time, so future notebook runs won't generate noisy diffs.
**Current state:** explore.ipynb is re-committed in stripped form; the filter is active for all future commits.
**Open questions / next steps:**
- Any other contributors will need to run `nbstripout --install` once in their clone to activate the filter locally.
---

---
### 2026-04-09 — Notebook execution outputs committed
**Files changed:** `explore.ipynb`
**What was done:** Ran explore.ipynb end-to-end after the enc_zone fix; cell outputs and execution counts were updated in the notebook file. Also diagnosed a disk-full issue (C: at 100%) — pip cache (2.5 GB) and Temp (2.9 GB) identified as main culprits.
**Current state:** Notebook runs cleanly through the PPO forward-pass demo; no source logic changed.
**Open questions / next steps:**
- Free disk space: run `pip cache purge` and clear Temp to recover ~5 GB.
- Consider adding `nbstripout` via `.gitattributes` to avoid large output diffs on every notebook run.
---

---
### 2026-04-07 — Fix missing enc_zone helper in explore.ipynb
**Files changed:** `explore.ipynb`
**What was done:** Added the `enc_zone(zone, tavern_tier, round_num)` helper function as a new notebook cell (inserted before the PPO forward-pass demo cell). The function wraps `board_computer.compute()` + `card_encoder.encode_board()` and returns `(7×44 ndarray, BoardFeatures)`, matching the call signature already used in the demo cell.
**Current state:** The PPO forward-pass demo cell in explore.ipynb can now run without a `NameError`.
**Open questions / next steps:**
- Verify the full forward-pass demo cell runs end-to-end with real game data.
- Consider moving `enc_zone` into a shared utility module if it is needed outside the notebook.
---

---
### 2026-03-31 — Phase 1 & 2 hero power system
**Files changed:** `hero_definitions.json`, `HEROES.md`, `agent/hero_encoder.py`, `symbolic/hero_handler.py`, `env/player_state.py`, `env/game_loop.py`, `agent/policy.py`
**What was done:** Implemented the full Phase 1 (passive) and Phase 2 (active no-pointer) hero power system. Created `hero_definitions.json` with 29 heroes (1 null + 16 Phase 1 + 12 Phase 2). `HeroPowerHandler` dispatches all passive hooks (on_sell, on_buy, on_play, on_refresh, on_tavern_upgrade, on_start_of_round, on_end_turn) and the `activate_no_pointer` path for type_action==6. `PlayerState` gained 10 new hero-power and buy-cost-override fields; `MinionState` gained `maiev_dormant_rounds`. `build_type_mask` now correctly gates hero_power on gold/charges/used-flag. Heroes are assigned without replacement at game start via `reset()`.
**Current state:** Full game dry-run passes (2 games, ~1900 transitions each). All 9 hero unit tests pass. Heroes are correctly assigned and their effects fire during gameplay.
**Open questions / next steps:** Phase 3 targeted heroes (Vol'jin, Shudderwock, Maiev, Saurfang) require extending `TYPES_WITH_POINTER` to include type 6 and a two-step HERO_POWER_T2 action. Hero identity is not yet encoded in the policy input (scalar_context still 38-dim; hero embedding + 4 flags would expand to 50). Ysera/Fungalmancer Flurgl use flag-based pool injection which is approximate (no tribe filtering in TavernPool).
---
### 2026-03-30 — Magnetic Mechs, smart play positioning, and spell handling
**Files changed:** `env/player_state.py`, `env/game_loop.py`
**What was done:** Added `magnetic` and `is_spell` fields to `MinionState`. Updated `_dict_to_minion` to populate both fields from `bg_card_definitions.json` (checking `keywords.magnetic` and `has_magnetic`). The PLACE action in `step_shopping` now: (1) casts and discards spell cards without consuming a board slot, (2) merges Magnetic Mechs into the rightmost friendly Mech on the board instead of placing them separately, and (3) uses a new `_smart_position` helper to insert Taunt/Divine-Shield minions at the front and Windfury/normal minions at the back. Added `_cast_spell` method handling Blood Gem (+1/+1 to random friendly) and generic coin/tavern-spell effects (+1 gold), with a no-op fallback for unknown spells.
**Current state:** All three mechanics are live in the shopping loop; `policy.py`, `ppo.py`, and `train.py` are unchanged and the action-space layout is unaffected.
**Open questions / next steps:**
- `_smart_position` could be made more sophisticated (e.g. place low-health minions behind Taunts).
- Spell detection relies on absent `base_atk`/`base_hp` plus negative attack/health from the pool dict; a future patch should add an explicit `is_spell` flag to `bg_card_definitions.json`.
- Additional named spells (e.g. Murozond's Gift, Pirate Parrot spells) can be added to `_cast_spell` as they are encountered.
---
### 2026-03-23 — Expand hand to 10 slots; rename MINION_FEATS/encode_minion to card-agnostic names

**Files changed:** `explore.ipynb`

**What was done:** `MAX_HAND` was 2, which is wrong — the BG hand holds up to 10 cards (minions and spells). Expanded to 10. Renamed `MINION_FEATS` → `CARD_FEATS`, `encode_minion` → `encode_card`, `_EMPTY_MINION` → `_EMPTY_CARD` to reflect that the encoding handles spells too (they simply encode as 0 for combat stats). Updated `STATE_DIM` inline comment (183 → 271) and `BGPolicy` docstring accordingly.

**Current state:** State vector is now 271-dim (`7 + (7+7+10)*11`). Any previously saved `bg_policy.pt` is stale and must be retrained.

**Open questions / next steps:**
- Retrain the BC model — the saved checkpoint is now dimension-incompatible.
- Consider adding an `is_spell` flag to the card encoding so the model can distinguish spells from minions.

---
### 2026-03-23 — Fix stale BC cell comments and hero_power in confusion matrix

**Files changed:** `explore.ipynb`

**What was done:** Updated two BC cells that were out of date after the 2026-03-20 hero_power action space expansion. Cell 29 (BGPolicy docstring) had stale dimensions (`STATE_DIM = 181`, `N_ACTIONS = 19`); corrected to 183/20. Cell 33 (confusion matrix) was missing `hero_power` in both `action_type()` (so index 18 fell through to `"end_turn"`) and `type_labels`; both fixed.

**Current state:** All BC cells are consistent with the 20-class action space and 183-dim state vector.

**Open questions / next steps:**
- Re-run notebook end-to-end to refresh outputs after the MIN_ROUNDS filter and BC fixes.

---
### 2026-03-23 — Filter incomplete games from explore.ipynb

**Files changed:** `explore.ipynb`

**What was done:** Investigated why a placement=2 game showed non-zero final health. Traced it to a game quit after only 6 rounds — the placement was a mid-game leaderboard snapshot, not a true final result. Added a `MIN_ROUNDS = 8` filter in the notebook's data-loading cell to silently skip incomplete/quit games.

**Current state:** Notebook loads 6 of 7 sessions, skipping the 6-round incomplete game. All analysis cells are unaffected.

**Open questions / next steps:**
- Consider adding a similar guard in `collect_dataset.py` so incomplete games are never written to the dataset in the first place.
- Investigate whether Duos games need separate handling (partner death = elimination without personal health hitting 0).

---
### 2026-03-15 — Initial project scaffold committed

**Files changed:** `bg_card_pipeline.py`, `bg_card_definitions.json`, `train.py`, `env/game_loop.py`, `env/player_state.py`, `env/tavern_pool.py`, `env/matchmaker.py`, `symbolic/board_computer.py`, `symbolic/shop_analyzer.py`, `symbolic/firestone_client.py`, `agent/policy.py`, `agent/card_encoder.py`, `agent/ppo.py`

**What was done:** Full project scaffold committed covering all major modules — the game loop (8-player BG simulation), symbolic layer (auras, multipliers, deathrattle specs), card pipeline (HearthstoneJSON fetch → JSON DB with 264 minions across 7 tiers), and the neural agent (Transformer policy + PPO training loop). The card database was scraped on 2026-03-14 and reflects the current active BG pool. Git commit and push instructions were also added to CLAUDE.md.

**Current state:** All planned files from the CLAUDE.md file structure exist. The project has not yet been run end-to-end; no training has been verified.

**Open questions / next steps:**
- Verify `env/game_loop.py` runs a full 8-player episode without errors
- Confirm `bg_card_pipeline.py --output bg_card_definitions.json` regenerates the card DB correctly
- Hook up Firestone sim subprocess in `symbolic/firestone_client.py` and test win-prob estimates
- Run a short PPO self-play training loop in `train.py` and confirm reward signals are sensible
- Check that action masking is correctly applied for all invalid actions in the buy phase
---

---
### 2026-03-15 — Pure-Python BG combat simulator

**Files changed:** `symbolic/combat_sim.py` (new), `symbolic/firestone_client.py`, `env/game_loop.py`

**What was done:** Built a pure-Python Monte Carlo BG combat simulator (`BGCombatSim`) in `symbolic/combat_sim.py`, replacing the heuristic estimator as the default backend in `FirestoneClient`. The simulator implements the full BG combat loop: round-robin attack pointer with taunt targeting, divine shield, venomous (including DS interaction), windfury (two attacks before pointer advances), reborn (one resurrection per minion), cleave (Blade Collector), and Titus Rivendare (cached flag, DRs trigger twice). ~20 deathrattles are handled (token summons: Bonehead, Cord Puller, Beetle summons, Cadaver Caretaker, Twilight Hatchling, Eternal Summoner; AoE: Tunnel Blaster, Silent Enforcer; buffs: Silithid Burrower, Showy Cyclist, Stellar Freebooter). Seven start-of-combat triggers fire before the loop (Amber Guardian, Humming Bird, Prized Promo-Drake, Misfit Dragonling, Fire-forged Evoker, Irate Rooster, Soulsplitter). Deaths are resolved iteratively (up to 10 waves) to handle AoE DR chains. Performance-optimized with `__dict__` cloning instead of `copy.copy`, a cached `_titus` flag on `CombatSide`, and a two-level DR dispatch (O(1) exact dict + substring fallback). `game_loop.py` updated to pass `player_tier`/`opp_tier` to `simulate()` for accurate win-damage calculation.

**Current state:** BGCombatSim runs at ~17 ms/call (200 trials, typical mid-game boards) and ~34 ms on stress boards with Titus+golden deathrattle chains. FirestoneClient defaults to BGCombatSim; the real Firestone subprocess path remains intact if a `firestone_path` is provided.

**Open questions / next steps:**
- Run a full 8-player self-play episode in `game_loop.py` and confirm no errors
- Benchmark combat sim during actual training (4 combats/round × N rounds) to assess if 17 ms is acceptable or if trials should be reduced to 50–100 for training
- Add venomous + cleave cross-minion damage tests for correctness validation
- Consider adding Avenge triggers (Bird Buddy, Budding Greenthumb) once training begins, if they appear frequently in high-tier boards
---

---
### 2026-03-15 — Add stop hook and restructure CLAUDE.md for session discipline

**Files changed:** `.claude/settings.json`, `.claude/check_context_log.sh`, `CLAUDE.md`

**What was done:** Added a Claude Code `Stop` hook (`.claude/settings.json` + `check_context_log.sh`) that fires at the end of every agent session. The script checks whether `CONTEXT.md` was modified when source files were changed; if not, it prints a prominent warning and exits with code 2, blocking the agent from finishing until the log is updated. Restructured `CLAUDE.md` to place the mandatory end-of-session checklist (append CONTEXT.md → stage → commit → push) immediately after the title, and replaced the old duplicate sections at the bottom with a pointer to the top.

**Current state:** Session discipline infrastructure is in place. The stop hook will enforce CONTEXT.md updates going forward.

**Open questions / next steps:**
- Consider whether the hook path should use a relative path instead of an absolute one for portability
- All previous open questions from the initial scaffold still apply (end-to-end run, Firestone sim, PPO training)
---

---
### 2026-03-15 — Fix stop hook path after environment change

**Files changed:** `.claude/settings.json`

**What was done:** The stop hook command in `.claude/settings.json` pointed to a stale Docker/container path (`/sessions/clever-adoring-newton/mnt/bg-dataset/...`) from the session in which it was created. Updated to the correct Git Bash path (`/c/Users/coenv/bg-dataset/...`). Verified the hook fires correctly: it detects the 5 dirty source files, prints the wrap-up warning banner, and exits with code 2 to block the agent from stopping.

**Current state:** Stop hook is confirmed working in the current environment.

**Open questions / next steps:**
- Portability: the absolute path will break again if the repo is cloned elsewhere; consider switching to a path relative to `REPO_ROOT` derived inside the script (already done) but the `settings.json` command itself still needs a stable anchor
- All previous open questions from the initial scaffold still apply (end-to-end run, Firestone sim, PPO training)
---

---
### 2026-03-15 — Expand scalar context and add gold-efficiency reward

**Files changed:** `agent/policy.py`, `agent/ppo.py`, `env/game_loop.py`, `env/player_state.py`, `train.py`

**What was done:** Expanded `SCALAR_DIM` from 30 to 38 by adding 8 next-opponent features (tier, health, armor, board_size, dominant_tribe_count, is_synergistic, rounds_since_seen, health_delta) and 6 lobby-wide features (num_alive, mean_opp_tier, mean_opp_health, num_synergistic_boards, health_rank, tier_rank) with a full layout comment in `policy.py`. Added `prev_health` and `last_seen_round` fields to `OpponentSnapshot` in `player_state.py` to support the health_delta and rounds_since_seen features. Added a decaying gold-efficiency penalty (`-0.05 * gold * scale`) at END_TURN in `game_loop.py`, scaling from 1.0 on round 1 down to 0.2 by round 16+.

**Current state:** Policy and PPO types reflect the new 38-dim scalar context. There is a known bug: `train.py` still passes `scalar_dim=30` to `BGPolicyNetwork` instead of 38 — this will cause a shape mismatch at training time and must be fixed before running.

**Open questions / next steps:**
- Implement `SymbolicBoardComputer.to_scalar_vector()` to actually populate all 38 dims from live game state
- Run a short self-play episode to confirm reward signals (including gold penalty) are sensible
---

---
### 2026-03-15 — Fix scalar_dim mismatch in train.py

**Files changed:** `train.py`

**What was done:** Fixed one-line bug: `scalar_dim=30` → `scalar_dim=38` in `build_components()` to match the expanded `SCALAR_DIM` constant in `policy.py`.

**Current state:** `train.py` and `policy.py` are now consistent on 38-dim scalar context.

**Open questions / next steps:**
- Implement `SymbolicBoardComputer.to_scalar_vector()` to populate all 38 dims from live game state
- Run a short self-play episode to confirm reward signals are sensible
---

---
### 2026-03-18 — Fix InconsistentPlayerIdError in dataset parser

**Files changed:** `parse_bg.py`, `collect_dataset.py`

**What was done:** `collect_dataset.py` was only printing `str(exc)` on parse failures, hiding the real traceback. Added `traceback.print_exc()` temporarily to expose the root cause: `hslog.player.InconsistentPlayerIdError` was raised when a BG log reassigned a player's `player_id` mid-session (entity_id=5 appeared first as player_id=2, then as player_id=6). Fixed by catching `InconsistentPlayerIdError` (and `CorruptLogError`) around `parser.read()` in `parse_power_log`, then calling `parser.flush()` on the partial parse. The temporary debug traceback was removed.

**Current state:** All 4 log sessions now parse cleanly — 5 BG games collected total. The previously-failing session (`Hearthstone_2026_03_15_19_19_02`) yields 2 games.

**Open questions / next steps:**
- Implement `SymbolicBoardComputer.to_scalar_vector()` to populate all 38 dims from live game state
- Run a short self-play episode to confirm reward signals are sensible
---

---
### 2026-03-20 — Hero power action tracking + BC action space expansion

**Files changed:** `parse_bg.py`, `explore.ipynb`

**What was done:** Added `hero_power` as a tracked action in the parser: `handle_block` now detects `CARDTYPE=HERO_POWER` blocks owned by the friendly player and emits `{"action": "hero_power", "card_id", "name", "gold_remaining", "hero_power_cost"}`. Two new helpers (`_hero_power_entity`, `_hero_power_card_id`) were added alongside the existing `_hero_power_cost`. The shopping dict now includes `hero_power_card_id` and `hero_power_cost` every round (populated from the hero power entity in PLAY, so always present regardless of whether the player used it). In the notebook the BC action space grew from 19 → 20 classes (`hero_power` = index 18, `end_turn` = 19), the state vector grew from 181 → 183 dims by adding `hero_power_cost` and `hero_power_available` to the context block, `valid_action_mask` gained a hero-power rule (gold ≥ hp_cost AND not yet used this turn), and `extract_transitions` tracks `hp_used` per round and passes both new context features into `encode_state`.

**Current state:** Parser and notebook BC pipeline are consistent with the expanded action space. Existing JSON datasets do not yet contain `hero_power_card_id`/`hero_power_cost` fields — logs need to be re-parsed with the updated `parse_bg.py` to populate them.

**Open questions / next steps:**
- Re-run `collect_dataset.py` to regenerate JSON files with the new shopping fields
- Verify `hero_power` actions are actually captured by checking a re-parsed game's action list
- Consider adding a hero-power usage rate chart to the notebook grouped by `hero_power_card_id`
---

---
### 2026-03-20 — Upgrade stop hook to blocking mode with auto-log requirement
**Files changed:** `.claude/check_context_log.sh`
**What was done:** Changed the stop hook exit code from 1 (advisory) to 2 (blocking) so Claude Code must respond before finishing a session. The hook now always fires unless CONTEXT.md was already updated — even when no source files were changed, it instructs Claude to write a short session description. Two message paths: one for sessions with dirty files (full entry required) and one for no-change sessions (brief description required).
**Current state:** Stop hook is active and will block session exit until CONTEXT.md is appended.
**Open questions / next steps:**
- Verify exit code 2 behaviour in the installed Claude Code version
- Consider whether the hook should also enforce `git commit` completion before allowing exit
---

---
### 2026-03-20 — Fix stop hook output: redirect echo to stderr
**Files changed:** `.claude/check_context_log.sh`
**What was done:** Claude Code's hook feedback system surfaces stderr, not stdout. All `echo` statements in the stop hook were redirected to stderr (`>&2`) so the blocking message is actually visible when the hook fires.
**Current state:** Stop hook should now surface its warning banners correctly when exit code 2 is returned.
**Open questions / next steps:**
- Confirm the stderr output appears in the next session where CONTEXT.md is not pre-updated
---

---
### 2026-03-20 — Add hero power tracking and split card list to CARDS.md
**Files changed:** `parse_bg.py`, `CLAUDE.md`, `explore.ipynb`
**What was done:** Added hero power tracking to `parse_bg.py`: the shopping output now includes `hero_power_card_id` and `hero_power_cost` fields, and hero power activations are recorded as `{"action": "hero_power", ...}` entries in the actions list. The full 264-minion card list was extracted from `CLAUDE.md` into a dedicated `CARDS.md` file to keep `CLAUDE.md` concise; a reference link was added in its place. Notebook outputs were refreshed.
**Current state:** Parser captures hero power usage per round; CARDS.md holds the canonical card pool.
**Open questions / next steps:**
- Add `CARDS.md` to git (currently untracked)
- Re-run `collect_dataset.py` to regenerate JSON files with the new `hero_power_*` fields
- Verify hero power actions appear correctly in a re-parsed game log
---

---
### 2026-03-20 — Stop hook verified working (no-change session)
**Files changed:** none
**What was done:** Session confirmed the stop hook's "no file changes" path fires correctly. The hook blocked exit and surfaced the session log reminder via stderr as intended. No source files were modified.
**Current state:** Stop hook is fully operational on both code-change and no-change paths.
**Open questions / next steps:**
- Add `CARDS.md` to git (still untracked)
---

---
### 2026-03-20 — No-change session log (stop hook confirmation)
**Files changed:** none
**What was done:** No source files were modified. Session consisted only of receiving and acknowledging the stop hook's no-change path output.
**Current state:** No changes pending.
**Open questions / next steps:**
- Add `CARDS.md` to git (still untracked)
---

---
### 2026-03-20 — Hero power action tracking + BC action space expansion

**Files changed:** `parse_bg.py`, `explore.ipynb`

**What was done:** Added `hero_power` as a tracked action in the parser: `handle_block` now detects `CARDTYPE=HERO_POWER` blocks owned by the friendly player and emits `{"action": "hero_power", "card_id", "name", "gold_remaining", "hero_power_cost"}`. Two new helpers (`_hero_power_entity`, `_hero_power_card_id`) were added alongside the existing `_hero_power_cost`. The shopping dict now includes `hero_power_card_id` and `hero_power_cost` every round. In the notebook the BC action space grew from 19 to 20 classes (`hero_power` = index 18, `end_turn` = 19), the state vector grew from 181 to 183 dims by adding `hero_power_cost` and `hero_power_available` to the context block, `valid_action_mask` gained a hero-power rule (gold >= hp_cost AND not yet used this turn), and `extract_transitions` tracks `hp_used` per round. Dataset re-collected with `--force`: 8 sessions parsed, 11 games, 35 hero_power actions captured.

**Current state:** Parser and notebook BC pipeline are consistent with the expanded 20-class action space. Re-parsed JSON files contain `hero_power_card_id` and `hero_power_cost` in every shopping dict. Three oldest JSON files (pre-2026-03-13) are stale — their source logs no longer exist so they lack the new fields, but `.get()` defaults handle them gracefully.

**Open questions / next steps:**
- Delete the 3 stale pre-March-13 JSON files that can't be re-parsed
- Add a hero-power usage rate chart to the notebook grouped by `hero_power_card_id`
- Add `CARDS.md` to git (still untracked)
---

---
### 2026-03-20 — Filter ghost game records + backfill hero power fields

**Files changed:** `parse_bg.py`

**What was done:** Fixed two issues. (1) Ghost records: sessions with 2+ BG game trees sometimes produced records with no rounds and no hero card_id (lobby abandons / client crashes before shopping started). Added a guard in `parse_power_log` to skip any record where rounds is empty AND hero card_id is absent. The two affected sessions (2026-03-15, 2026-03-18) now correctly report 1 game each instead of 2. (2) Backfilled the 3 stale JSON files (pre-2026-03-13, whose source logs no longer exist) with `hero_power_card_id` and `hero_power_cost` using `card.hero_power` from the card DB, resolving skin suffixes automatically.

**Current state:** 9 clean games across 9 JSON files, all with `hero_power_card_id` and `hero_power_cost` populated in every shopping round.

**Open questions / next steps:**
- Add a hero-power usage rate chart to the notebook grouped by `hero_power_card_id`
- Add `CARDS.md` to git (still untracked)
---

---
### 2026-03-20 — Fix hero health always showing 30 in parsed data

**Files changed:** `parse_bg.py`

**What was done:** Fixed a bug in `_hero_snap` where `hero_health` was always 30. The code was reading `GameTag.HEALTH` directly, which is the hero's *max* health (always 30 for BG heroes, never changes). In Hearthstone's tag system, damage taken is accumulated in `GameTag.DAMAGE` separately, and current HP = `HEALTH - DAMAGE`. Fixed by computing `max(0, max_hp - damage)` using both tags.

**Current state:** `hero_health` in `shopping` snapshots and `hero_health_after` in `combat` snapshots now correctly reflect actual current HP across all rounds.

**Open questions / next steps:**
- Re-parse existing JSON files to populate correct hero health values
- Add `CARDS.md` to git (still untracked)
---

---
### 2026-03-20 — Combat result, reorder, shop frozen, discovers

**Files changed:** `parse_bg.py`

**What was done:** Implemented all 4 parser improvements. (1) **Combat result**: Added `BACON_WON_LAST_COMBAT` tag detection in `handle_tag_change` as the primary signal (tag 1422), plus health-delta fallback in `_flush_combat` (hp+armor decreased = loss). `PlayState.WINNING/LOSING/TIED` kept as secondary. Result now populated in 76/128 rounds (47 win, 29 loss); 52 remain None (ties or rounds where tag didn't fire). (2) **Reorder**: `handle_block` restructured to capture `from_pos` via `ZONE_POSITION` before dispatching children for `BlockType.MOVE_MINION` blocks, then computes `to_pos` after. Emits `{action: reorder, card_id, from_pos, to_pos, gold_remaining}`. Fixed guard to use `entities.get(eid)` returning None instead of `{}` to avoid KeyError in `_is_minion`. (3) **Shop frozen**: `BACON_FREEZE_TOOLTIP` tag read from player entity at MAIN_ACTION start; stored as `shop_frozen: bool` in shopping dict. (4) **Discovers**: Added `handle_choices` and `handle_chosen_entities` packet handlers. Choices stored by id with option entity_ids; ChosenEntities finalizes the event after ShowEntity has revealed card_ids. Emitted as `discovers` list on each round dict with source card, options, chosen, and phase.

**Current state:** 9 games, 45 reorder actions, 134 discover events, combat results now populated for ~60% of rounds. 3 stale JSON files backfilled with `shop_frozen: false` and `discovers: []` defaults.

**Open questions / next steps:**
- Investigate the 9 None-result rounds where health decreased (likely hero_eid not set at MAIN_START for round 1 combats)
- shop_frozen never fires — verify BACON_FREEZE_TOOLTIP is the right tag by checking a session where freeze was used
- Add CARDS.md to git (still untracked)
---
---
### 2026-03-20 — Win/Loss/Tie Detection via Opponent DAMAGE Tag
**Files changed:** `parse_bg.py`
**What was done:** Implemented per-round combat result detection using two signals: (1) a DAMAGE tag change on any non-player hero entity during combat fires in handle_tag_change to set result="win" immediately, and (2) player health delta at flush_combat distinguishes loss from tie. Removed the faulty BACON_WON_LAST_COMBAT detection that fired for all entities every round. Also wired up _capture_combat_opponent to save opponent entity id and pre-combat health snapshot (used for the entity-delta fallback approach, later superseded by the tag-change approach).
**Current state:** All 6 fresh games now have proper win/loss/tie on every round. Stale files (no source logs) retain nulls on rounds where player health was stable (cannot distinguish win/tie without stored opponent post-combat state).
**Open questions / next steps:** Some rounds show ghost opponents (dead players) — these will never fire a DAMAGE tag, so non-loss ghost rounds remain "tie". Acceptable for ML since neither result affects player health. Consider storing opponent post-combat health in the schema for future backfill capability.
---
---
### 2026-03-20 — Add opponent_is_ghost to combat state
**Files changed:** `parse_bg.py`
**What was done:** Added opponent_is_ghost bool field to the combat dict. True when the opponent hero is missing (opp hero entity not found) or has health <= 0 (dead/eliminated player). Ghost boards are left from eliminated players and are generally easier to beat since they cannot adapt.
**Current state:** 17 ghost rounds detected across 6 fresh games. Field is present in schema docstring. Stale files do not have this field.
**Open questions / next steps:** encode opponent_is_ghost as a feature in encode_state; stale files need backfill if used for training.
---
---
### 2026-03-20 — Fix 5 data correctness bugs in parser

**Files changed:** `parse_bg.py`

**What was done:** Fixed five data correctness issues identified by auditing the parsed JSON against actual game mechanics. (1) **Empty minion names**: `_minion_snap()` now falls back to `_card_db_name(card_id)` — all 1348 minion entries in fresh files now have names populated. (2) **tavern_tier captured post-shopping**: Added `_tavern_tier_at_start` snapshot at MAIN_ACTION (before any level_up actions), replacing the previous `hero["tech_level"]` read at MAIN_END. (3) **level_up new_tier stale**: `PLAYER_TECH_LEVEL` TagChange sometimes arrives after the TechUp block (or after MAIN_END); re-derived new_tier in `_flush_shopping` from the final hero tier with an arithmetic fallback when the tag hasn't updated. (4) **sold entities appearing in board_at_end**: Added `_sold_eids` set tracking entity IDs sold during shopping; `_flush_shopping` now filters these from `player_board()` when zone-change TagChanges arrive after MAIN_END. (5) **duplicate place actions**: Added `_placed_eids` set; place detection skips entities already recorded this shopping phase, preventing battlecry re-triggers from generating phantom extra place events. Lazy `_eid` resolution also added for sell/buy card_ids revealed after the action block but before MAIN_END.

**Current state:** All 5 bugs fixed on fresh-parsed files (0 tier errors, 0 missing names, 0 duplicate places). Two known limitations remain: (a) 32% of sell actions target anonymous BG entities whose card_id is never set in the log — unresolvable; (b) 2/77 rounds have board_at_end > 7 due to zone-position TagChanges for minion placements arriving after MAIN_END. Stale JSON file `03_13_17_24_20` cannot be re-parsed (source log gone).

**Open questions / next steps:**
- Stale file `Hearthstone_2026_03_13_17_24_20.json` still has old (unfixed) data; delete or clearly mark it if used for training
- Investigate anonymous sell entities (32%) — may correspond to specific BG hero powers or spell-created tokens whose card_ids are intentionally hidden in the log
- The board > 7 edge case (2/77 rounds) could be fixed by tracking entity ZONE transitions explicitly and deferring the board_at_end snapshot
---
---
### 2026-03-20 — Refresh explore.ipynb outputs after dataset growth

**Files changed:** `explore.ipynb`

**What was done:** Notebook outputs re-executed against the expanded dataset (5 → 9 games). No code changes; only cell execution counts and printed statistics updated.

**Current state:** Notebook reflects current dataset state.

**Open questions / next steps:** (see parser fix entry above)
---
---
### 2026-03-20 — Fix SubSpell packet dispatch; resolve anonymous sell entities

**Files changed:** `parse_bg.py`, `data/*.json` (re-parsed)

**What was done:** Diagnosed root cause of anonymous sell entities: `FULL_ENTITY` packets for hero-battlecry-created tokens (e.g. Timewarped Festergut creating Putricide's Creation) are emitted inside `SUB_SPELL_START` blocks. The parser's `_dispatch` only handled `Block`; `SubSpell` packets were silently dropped, so their child `FullEntity`s — which carry the card_id — were never processed. Fix: added a `SubSpell` branch in `_dispatch` that iterates and dispatches the packet's children. All sessions that could be re-parsed now show 0 anonymous sells. The remaining 104 anonymous sells are in stale JSON files whose raw logs are no longer available.

**Current state:** Fresh-parsed sessions have 100% sell resolution. Stale files (`03_06`, `03_08`, `03_12`, `03_13`) were parsed with the old code and still have anonymous entries but cannot be re-parsed.

**Open questions / next steps:**
- Delete or exclude stale JSON files (`03_06`, `03_08`, `03_12`, `03_13_17_24_20`) from training since they contain anonymized sell entries and old bug artifacts
- Check whether `SubSpell` packets also wrap other entity creation events we care about (discovers, spell effects) — the same fix covers those too
- Board > 7 edge case (2/77 rounds) still unresolved
---

---
### 2026-03-21 — Fix anomaly detection; full data correctness audit

**Files changed:** `parse_bg.py`

**What was done:** Performed a full correctness audit of the collected dataset across all 6 games. Found and fixed the anomaly detection bug: `BACON_GLOBAL_ANOMALY_DBID` is a numeric DBID on the GameEntity, not a card_id string — the old check (`if card_id:`) always evaluated False because the GameEntity has no card_id. Added `_DBID_DB` (integer-keyed alias of the `load_dbf()` result) and rewrote the detection in both `_store_entity` and `handle_tag_change` to look up the DBID and store the correct card_id string. All 6 sessions now report correct anomalies (e.g. "Grapnel of the Titans", "Boon of Chronum").

**Current state:** Anomaly field is populated correctly in all parsed games. Other known bugs (sell/hero names empty, round-1 shop empty, shop count inflated) are documented but not yet fixed.

**Open questions / next steps:**
- Fix sell and reorder action names: card_db_name fallback not applied when card_id is known but entity name is empty
- Fix hero name field: `_hero_snap()` never calls `_card_db_name()` as fallback
- Investigate round-1 `shop_at_start=0`: shop minions arrive via ShowEntity packets during MAIN_ACTION, after the snapshot fires
- Reduce shop count inflation in later rounds: filter by `zone_pos > 0` to exclude stale SETASIDE entities
---

---
### 2026-03-21 — Fix name fallbacks, round-1 shop, and shop count inflation

**Files changed:** `parse_bg.py`

**What was done:** Fixed all four open data-correctness bugs from the previous session. (1) Added `_card_db_name()` fallback to sell and reorder action name fields — previously the fallback was missing when `card_id` was known but the entity `name` was empty. (2) Added `_card_db_name()` fallback to `_hero_snap()`, which previously left the hero name blank. (3) Fixed round-1 `shop_at_start=0` by deferring the shop snapshot lazily to the first `handle_block` call during shopping (ShowEntity packets that build the shop on turn 1 arrive after the MAIN_ACTION step change fires, so the eager snapshot was always empty on round 1); a fallback in `_flush_shopping` handles turns with no player blocks. (4) Fixed shop count inflation by adding a `zone_pos > 0` guard in `shop_at_turn` and `spell_shop_at_turn` to exclude stale SETASIDE entities whose slot has been vacated (zone_pos resets to 0 when a minion leaves the shop).

**Current state:** All four known parser bugs are resolved. Round-1 shop snapshot should now be populated, hero names should always resolve, sell/reorder names should always resolve, and shop counts should no longer be inflated by slot-less lingering entities.

**Open questions / next steps:**
- Re-parse all games and verify: round-1 `shop_at_start` is non-empty, hero names are set, sell/reorder names are non-blank, shop counts match observed shop size
- Board > 7 edge case (2/77 rounds) still unresolved
- Consider deleting/excluding stale JSON files (`03_06`, `03_08`, `03_12`, `03_13`) from training
---

---
### 2026-03-21 — Fix ZONE_POSITION guard and turn-1 shop snapshot race

**Files changed:** `parse_bg.py`

**What was done:** Removed the `ZONE_POSITION > 0` guard from the hand/spell-shop snapshot methods — the condition was incorrectly filtering out valid SETASIDE entities whose position tag is absent. Fixed a turn-1 shop snapshot race: hero selection fires a PLAY block before ShowEntity packets for shop minions arrive, so the lazy snapshot is now deferred until `shop_at_turn()` returns a non-empty list.

**Current state:** Parser correctly captures shop state at turn 1 and does not drop hand/spell-shop entities that lack a ZONE_POSITION tag.

**Open questions / next steps:**
- Re-parse all games and verify round-1 `shop_at_start` is non-empty across the full dataset
- Board > 7 edge case (2/77 rounds) still unresolved
- Consider deleting/excluding stale JSON files (`03_06`, `03_08`, `03_12`, `03_13`) from training
---

---
### 2026-03-25 — Fix board > 7 inflation from late Zone-change TagChanges
**Files changed:** `parse_bg.py`
**What was done:** Added a hard `[:7]` cap to `player_board()` after sorting by `zone_pos`. Zone-change TagChanges that arrive after `MAIN_END` can spuriously set extra entities to `Zone.PLAY`, inflating the board count beyond the legal maximum of 7. The truncation is safe because legitimate minions have lower `zone_pos` values and sort before any late-arriving interlopers. The existing `_sold_eids` exclusion in `_flush_shopping` already handles the opposite direction (sold minions whose zone-leave tag is delayed); this cap handles the zone-enter direction.
**Current state:** Both directions of post-MAIN_END zone lag are now guarded: late zone-leave via `_sold_eids`, late zone-enter via the `[:7]` cap in `player_board()`.
**Open questions / next steps:**
- Re-parse full dataset and verify the 2/77 board-inflation rounds are resolved.
- Confirm no legitimate board ever has fewer than expected minions after the cap (i.e., no false truncation).
- Consider deleting/excluding stale JSON files (`03_06`, `03_08`, `03_12`, `03_13`) from training.
---

---
### 2026-03-25 — Fix parse_bg.py dropping games after first parse error
**Files changed:** `parse_bg.py`
**What was done:** Fixed a bug where `parse_power_log` silently dropped all games after the first `InconsistentPlayerIdError` or `CorruptLogError`. The single `try/except` caught the error and exited, leaving the rest of the log file unread. Changed to a `while True` loop that re-calls `parser.read(fh)` after each error, resuming from the current file cursor position until the entire file is consumed.
**Current state:** Parser now processes all games in a multi-game log file even when player-ID reassignment errors occur mid-session.
**Open questions / next steps:**
- Verify fix with a real multi-game log that triggers `InconsistentPlayerIdError`
- Check whether `CorruptLogError` also advances the file cursor (same assumption as `InconsistentPlayerIdError`)
---

---
### 2026-03-25 — Fix AssertionError from hslog player manager crashing parse
**Files changed:** `parse_bg.py`
**What was done:** The parse loop only caught `InconsistentPlayerIdError` and `CorruptLogError`, but `hslog/player.py` also raises a bare `AssertionError` from `create_or_update_player` on inconsistent player state. This caused `parse_power_log` to propagate the exception and `collect_dataset.py` to print `ERROR: ` (empty message) and skip the session entirely. Added `AssertionError` to the caught exception tuple so the loop skips the bad line and continues.
**Current state:** Session `Hearthstone_2026_03_25_20_05_56` now parses successfully (2 games recovered).
**Open questions / next steps:**
- Re-run `collect_dataset.py` to pick up previously skipped sessions.
- Consider logging skipped-line counts per session for visibility.
---

---
### 2026-03-25 — Fix three BC training bugs in explore.ipynb
**Files changed:** `explore.ipynb`
**What was done:** Ran the BC pipeline end-to-end and found three bugs that prevented learning. (1) **Inverted masking formula**: `logits + (mb - 1) * NEG_INF` was adding `+1e9` to invalid-action logits instead of `-1e9` (since NEG_INF = -1e9, the sign negated). Fixed to `(1 - mb) * NEG_INF` in both the training loop and the validation masking. (2) **Hand not carried between rounds**: `extract_transitions` initialised `hand = []` at the start of every round, losing cards carried over from the previous shopping phase. Added `prev_hand` tracking across rounds, initialised from `hand_at_end` in the round record. This reduced place-action mask violations from 364/414 (88%) to the carry-over gap. (3) **GT label masked out during loss**: ~25% of training labels still fell outside the approximate valid mask (buy after reroll, residual tracking drift), causing ~1e9 per-sample loss spikes that swamped gradients. Fixed by forcing `masks_train[arange, y_train] = 1.0` before the DataLoader; same fix applied to the val mask used by the LR scheduler (original masks_val kept for accuracy reporting). After all three fixes: train loss drops to ~0.51, train accuracy reaches 78%, val loss is a healthy 2.66.
**Current state:** BC training pipeline is functionally correct. Val accuracy (~29%) is near the majority-class baseline (31%) due to dataset size (8 games, 168 val samples) and class imbalance — not a code issue.
**Open questions / next steps:**
- Collect more game data to improve generalisation (current val set is 1-2 games, too noisy to measure progress)
- Add class weights to CrossEntropyLoss to address class imbalance (place/reroll dominate, end_turn under-represented)
- The buy-after-reroll mask gap (shop cleared on reroll, subsequent buy slot unknown) is not yet fixed — requires tracking the new shop contents post-reroll
---

---
### 2026-03-25 — Fix extract_transitions buy labels + add BC v2 two-headed model
**Files changed:** `explore.ipynb`
**What was done:** Two improvements to the BC pipeline. (1) **find_by_card_id fix** in `extract_transitions`: skip buy transitions where the slot index is -1 (card not in tracked shop) instead of mislabelling them as `buy_0` — 77 bad transitions removed. Also pre-scan each round's action list before the loop to rebuild the post-reroll shop from subsequent buy `card_id` fields (`reroll_shops` dict). Also added hand carry-over (`prev_hand`) so cards not played in round N are tracked in round N+1's hand. (2) **BC v2 two-headed model** (`BGPolicyV2`) appended to the notebook as a new section: action TYPE head (8 classes: buy/sell/place/reroll/freeze/level_up/hero_power/end_turn) + card POINTER head (24 slots: shop[0-6] | board[0-6] | hand[0-9]). `extract_transitions_v2` returns separate type and pointer labels. Training uses combined loss (type CE + 0.5 × pointer CE). Result: val type accuracy 35.2% vs 29.6% majority baseline (model now beats baseline), per-type accuracy 62–90% on full dataset, card pointer accuracy 90.9%.
**Current state:** BC v1 (20-class slot model) and BC v2 (type+pointer) both present in notebook. V2 is the recommended model going forward.
**Open questions / next steps:**
- Collect more games to reduce overfitting (currently val set = 1 game, 179 samples)
- Add class weights to address imbalance (place/reroll dominate at ~23-30% each)
- Connect BC v2 pre-training to PPO policy warm-start (the shared trunk maps to agent/policy.py)
- Pointer head slot ordering follows purchase order for post-reroll shops (not display order) — acceptable approximation until the parser captures new shop contents after rerolls
---

---
### 2026-03-27 — BC v2 → PPO warm-start implementation
**Files changed:** `explore.ipynb`, `agent/policy.py`, `train.py`
**What was done:** Implemented structured warm-start from the BC v2 checkpoint into the PPO policy network. (1) **`explore.ipynb`**: added a save cell that writes `bc_v2.pt` with the full model state dict plus a `ppo_action_groups` dict mapping each of the 8 BC action types to their corresponding PPO action indices (buy→0-6, sell→7-13, place→14-83, reroll→86, freeze→85, level_up→84, hero_power→87, end_turn→88). (2) **`agent/policy.py`**: added `load_bc_v2_weights(bc_path)` to `BGPolicyNetwork` — transfers BC `type_head` weights/biases to all 89 corresponding PPO `policy_head[-1]` rows (so the policy starts with a calibrated prior over action types), and copies BC `shared[4]` (Linear 128→128) into the scalar half (columns 128:256) of PPO `policy_head[0]` (Linear 256→128), giving the policy a sensible starting point for processing its 38-dim scalar context. Both transfers verified against expected values. (3) **`train.py`**: added `--load-bc-v2 bc_v2.pt` CLI argument; v2 takes priority over legacy `--load-bc` if both are specified.
**Current state:** Full warm-start pipeline is implemented and tested. Run `python train.py --load-bc-v2 bc_v2.pt --no-firestone --dry-run` to exercise it (requires bc_v2.pt to be generated by running the BC v2 save cell in the notebook).
**Open questions / next steps:**
- Run the game loop end-to-end (`env/game_loop.py`) — still the primary blocker for PPO training
- Collect more games (current val set is 1 game; need ~20+ for reliable accuracy measurement)
- `bc_v2.pt` is not yet generated — requires running the notebook BC v2 training cells to convergence first
---

---
### 2026-03-27 — Run game loop end-to-end; fix 4 bugs
**Files changed:** `env/game_loop.py`
**What was done:** Ran the game loop end-to-end for the first time with random agents across 3 seeds — all completed without errors. Found and fixed 4 bugs: (1) `_placement_counter` initialised to `n_players + 1 = 9` instead of `n_players = 8`, causing the first eliminated player to receive placement 9 in an 8-player game (out of range, also breaks `FINAL_PLACEMENT_REWARD` lookup); (2) random agent end_turn bias used hardcoded index `19` (which is `play_h0_p5` in the 95-action space) instead of `END_TURN_IDX = 88`; (3) same wrong index `19` used as the `if not valid` last-resort fallback; (4) observation docstring comment on `hand_tokens` said `[2, 44]` instead of `[10, 44]`.
**Current state:** Game loop runs cleanly end-to-end. Placements are always in range 1–8 (asserted across 3 seeds). The pure-Python BGCombatSim is used as default backend (mock_mode=False).
**Open questions / next steps:**
- Tie probability is approximated as `(1 - win_prob) / 2`; BGCombatSim does not expose `tie_prob` separately — worth adding to `SimResult`
- Hero power action (index 87) is a no-op placeholder; needs per-hero dispatch
- Golden triple merging is not implemented
- Run `python train.py --load-bc-v2 bc_v2.pt --no-firestone --dry-run` to confirm full training pipeline works end-to-end (requires bc_v2.pt from notebook)
- Collect more games to expand dataset
---

---
### 2026-03-27 — Implement tie_prob in combat simulator
**Files changed:** `symbolic/combat_sim.py`, `symbolic/firestone_client.py`, `env/game_loop.py`
**What was done:** Added `tie_prob` and `loss_prob` fields to `SimResult` (with defaults of 0.0 for backward compatibility). `BGCombatSim.simulate()` already tracked `ties` and `losses` per trial but discarded them — now included in the returned `SimResult`. Updated `FirestoneClient._heuristic_estimate()` to estimate tie probability (peaks at 5% on evenly-matched boards, tapers toward 0 on lopsided ones). Updated `FirestoneClient._run_firestone()` to parse `tie_prob`/`loss_prob` from subprocess JSON output with fallback. Updated `game_loop.step_combat()` to sample outcomes using `sim.tie_prob` directly instead of the incorrect `(1-win_prob)/2` approximation. Verified probabilities sum to 1.0 across all board configurations.
**Current state:** Tie probability is now accurately tracked through the full pipeline. Empty vs empty boards correctly report tie_prob=1.0. The game loop runs cleanly; note that random agents with empty boards produce all-tie combats and hit the 40-round cap, which is expected.
**Open questions / next steps:**
- Random agents rarely play cards so boards are empty and all combats tie — meaningful testing requires trained or scripted agents that actually populate boards
- Hero power (action 87) is still a no-op placeholder
- Golden triple merging is not implemented

---
### 2026-03-27 — Remove BC v1 (BGPolicy) from notebook; keep only BGPolicyV2
**Files changed:** `explore.ipynb`
**What was done:** Deleted the 20-cell BC v1 section from the notebook (cells 18–37), which included the single-headed `BGPolicy` model, its `encode_card`/`extract_transitions` helpers, the 300-epoch training loop, confusion matrix evaluation, step-by-step prediction demo, and the save/load checkpoint cell. `BGPolicyV2` (two-headed: action-type + card-pointer) is now the sole training model in the notebook.
**Current state:** Notebook has 46 cells. All BC training infrastructure references `BGPolicyV2` only. The BC v1 model and pipeline are gone from the notebook but the checkpoint `bg_policy.pt` (if it exists on disk) is unaffected.
**Open questions / next steps:**
- Run BC v2 training cells end-to-end to confirm no remaining imports depend on removed BC v1 helpers
- Wire `bc_v2.pt` checkpoint into `train.py --load-bc-v2` warm-start
---

---
### 2026-03-27 — Update PPO cells to match current architecture; move after BC v2
**Files changed:** `explore.ipynb`
**What was done:** Updated the 6 PPO notebook cells to reflect the current `BGPolicyNetwork` architecture: 32 tokens (CLS + 7b + 7s + 10h + 7opp), 38-dim scalar context (own-board 24 + opponent 8 + lobby 6), and 95-action output. Fixed `hand_t` from `[:2]` to `[:10]`, updated the architecture diagram to show Opp Tokens and correct counts, expanded `SCALAR_LABELS` from 24 to 38 entries with boundary dividers. Added note about BC v2 warm-start via `load_bc_v2_weights`. Moved all 6 PPO cells to after the BC v2 section to reflect the pretraining flow.
**Current state:** Notebook flows Dataset → Neurosymbolic → BC v2 (BGPolicyV2) → PPO (BGPolicyNetwork). PPO cells are consistent with `agent/policy.py`.
**Open questions / next steps:**
- Run notebook end-to-end to verify no broken imports after the reorder
- Wire `policy.load_bc_v2_weights("bc_v2.pt")` once a bc_v2.pt checkpoint is saved
- `ppo_action_groups` mapping in the checkpoint needed for `load_bc_v2_weights` — confirm it is saved by the BC v2 training cell
---

---
### 2026-03-27 — Fix hand tensor shape in PPO forward demo cell
**Files changed:** `explore.ipynb`
**What was done:** Fixed a `RuntimeError` in the PPO forward pass demo cell. `card_encoder.encode_board` always returns `[7, 44]` regardless of input size, so `hand_enc4[:10]` was still 7 rows, producing 28 tokens instead of the expected 31 (7b+7s+10h+7o). Fixed by allocating a `np.zeros((10, 44))` buffer and copying `hand_enc4` into it, ensuring `hand_t` is always `[1, 10, 44]`.
**Current state:** PPO forward pass demo runs without shape errors.
**Open questions / next steps:**
- Same padding pattern should be applied in `game_loop.py` wherever hand tokens are assembled for the policy — verify `env/game_loop.py` pads hand to 10 correctly
---

---
### 2026-03-27 — Restore state encoding helpers deleted with BC v1
**Files changed:** `explore.ipynb`
**What was done:** Added a new cell after `bc-v2-constants` that restores the state-encoding infrastructure that was removed with BC v1: `CARD_FEATS`, `MAX_SHOP/BOARD/HAND`, `CONTEXT_FEATS`, `STATE_DIM`, zone offsets (`_SHOP_OFF`, `_BOARD_OFF`, `_HAND_OFF`, `_IS_PRES`), `encode_card`, `encode_slot_list`, `encode_state`, and `find_by_card_id`. These are depended on by `extract_transitions_v2`, `valid_action_type_mask`, and `BGPolicyV2`.
**Current state:** BC v2 section is self-contained and runnable from top to bottom.
**Open questions / next steps:**
- Run BC v2 cells end-to-end to confirm no further missing dependencies
---

---
### 2026-03-30 — Refactor BGPolicyNetwork to two-headed output matching BGPolicyV2
**Files changed:** `agent/policy.py`, `agent/ppo.py`, `train.py`, `explore.ipynb`
**What was done:** Replaced the flat 95-action policy head with two separate heads matching BGPolicyV2: `type_head` (8 action types) and `pointer_head` (24 card slots). `load_bc_v2_weights` now does a direct shape-exact weight copy (no row-mapping). `Transition` dataclass updated to store `(type_action, ptr_action, type_mask, pointer_mask)` instead of a flat action and 95-dim mask. `PPOAgent` in train.py updated to use `build_type_mask` + `build_pointer_mask` and return (type_idx, ptr_idx) tuples. Notebook PPO cells updated throughout.
**Current state:** Architecture is fully aligned — BC and PPO share the same output structure; BC→PPO weight transfer is now direct and complete.
**Open questions / next steps:**
- SWAP/reorder actions were dropped; add as a 9th type if board positioning proves important
- `game_loop.py` needs to interpret (type_idx, ptr_idx) tuples from PPOAgent.get_action instead of a flat action int
---

---
### 2026-03-30 — Update game_loop.py to use two-step (type, ptr) action API
**Files changed:** `env/game_loop.py`
**What was done:** Updated `step_shopping` to take `(type_action, ptr_action)` instead of a flat int, dispatching on type index 0-7 with pointer decoded via PTR_*_OFF constants. Updated `_get_agent_action` to return a `(type_idx, ptr_idx)` tuple, replacing `build_action_mask`/`END_TURN_IDX` with `build_type_mask`/`build_pointer_mask`. Random agent now samples type first then pointer. Updated `run_game` to unpack the tuple.
**Current state:** Full pipeline (game_loop → PPOAgent → BGPolicyNetwork) is consistent on the two-step action API end-to-end.
**Open questions / next steps:**
- `PPOAgent.record_transition` is not yet wired into game_loop; transitions are not being collected during self-play — needs integration
- Board position for place actions is always append; could add position prediction later
---

---
### 2026-03-30 — Wire PPO transition recording into game_loop.py
**Files changed:** `env/game_loop.py`
**What was done:** Added transition recording to the shopping loop. Non-end-turn steps are recorded immediately via `agent.record_transition(..., done=False)`. End-turn steps are buffered in `end_turn_buffers` and flushed after combat in the round-rewards loop, combining the gold-penalty step reward with the round combat reward. `done=True` is set when the player was just eliminated. Uses `hasattr(agent, "record_transition")` so random/scripted agents are unaffected.
**Current state:** PPO rollout buffer is now populated during self-play. The full training pipeline (game_loop → transitions → PPO update) is wired end-to-end.
**Open questions / next steps:**
- Final placement reward (+4/-4) is not captured in a transition; could add a terminal transition after game end
- `record_transition` requires the agent to hold a reference to its PPOTrainer — confirm this is set up correctly in train.py
---

---
### 2026-03-30 — Wire PPOAgent.record_transition into game loop with buffered end-turn flush
**Files changed:** `env/game_loop.py`
**What was done:** Added `end_turn_buffers` dict to the shopping phase loop. Non-end-turn transitions are recorded immediately via `agent.record_transition(done=False)`. The END_TURN transition is buffered and flushed after combat so its reward includes both the gold-spend step reward and the post-combat round reward, with `done=True` for eliminated players.
**Current state:** Self-play now collects transitions correctly; the reward signal on the last shopping action of each round now reflects the full combat outcome.
**Open questions / next steps:**
- Confirm `active_agents[pid]` indexing is correct when some agents are non-PPO (e.g. random agents in mixed lobbies)
- Test that `end_turn_buffers` is fully drained each round (no leaks if a player dies mid-shopping before issuing END_TURN)
- Putricide/unknown minion encoding: currently falls back to zeros for dims 12–43; a dynamic fallback reading keywords from live state would improve feature coverage
---

---
### 2026-03-30 — Dynamic fallback encoding for unknown/generated minions
**Files changed:** `agent/card_encoder.py`
**What was done:** Added `_dynamic_fallback` method to `CardEncoder` that fills dims 8 and 12–38 from the live minion dict when no card definition exists. Recovers tribes from `tribe`/`tribes` fields, infers trigger type via regex against `raw_text`, and detects DR-summon minions. Added `_RAW_TEXT_TRIGGER_PATTERNS` constant for the inference rules. Wired fallback into `encode` so unknown cards (Putricide creations, token summons, etc.) get meaningful features instead of all-zeros.
**Current state:** Unknown minions now encode with tribe, trigger, and deathrattle signal when the game state provides `tribe`/`tribes`/`raw_text`; pure stat-only unknowns (no extra fields) still degrade gracefully to zeros for those dims.
**Open questions / next steps:**
- `board_computer.py` still skips unknown cards in tribal synergy and aura detection — those paths need similar live-state fallbacks
- Putricide creations with multi-tribe (ALL / ANY) need special handling if that tag surfaces in log data
- Consider adding token stubs (Skeleton 1/1 Undead, Beetle 2/2 Beast, etc.) to bg_card_definitions.json so the static path is preferred over inference
---

---
### 2026-03-30 — Fix load_checkpoint to tolerate stale architecture checkpoints

**Files changed:** `agent/ppo.py`
**What was done:** Wrapped `policy.load_state_dict(...)` in a `try/except RuntimeError` block. When a checkpoint saved with the old flat `policy_head` architecture is loaded against the new `type_head`+`pointer_head` architecture, the call now logs a warning and returns early rather than crashing the training run.
**Current state:** Dry-run (`python train.py --dry-run --no-firestone --seed 42`) completes successfully: 2 games, ~1185 transitions collected, checkpoint saved.
**Open questions / next steps:**
- Delete stale `bg_agent_ppo.pt` before production training run to start from a clean slate
- BC warm-start (`--load-bc-v2 bc_v2.pt`) should be tested before full training

---

---
### 2026-03-30 — Add terminal transition for final placement reward
**Files changed:** `env/game_loop.py`
**What was done:** After placements are assigned, each player's agent receives a terminal transition via `record_transition(last_obs, end_turn, -1, reward=final_r, done=True)`. This delivers the +4/-4 placement reward as a `done=True` step so GAE bootstraps to 0 at game end rather than ignoring the signal entirely.
**Current state:** All reward components (per-step, round combat, final placement) are now captured in the PPO rollout buffer.
**Open questions / next steps:**
- Run a dry-run (`python train.py --dry-run`) to confirm no runtime errors end-to-end
---

---
### 2026-03-30 — Implement triple/golden card system
**Files changed:** `env/triple_system.py`, `env/game_loop.py`
**What was done:** Created `env/triple_system.py` with `make_golden` (doubles base attack, health, max_health, perm_atk_bonus, perm_hp_bonus and sets golden=True) and `check_and_process_triple` (detects 3+ non-golden copies of the same card_id across hand+board, merges them into one golden copy in hand, returns 2 source cards to the pool, and grants a discover from tier+1 capped at tier 6 with auto-select of first candidate). Wired `check_and_process_triple` into `step_shopping` in `game_loop.py` after both the BUY action (type_action==0) and the PLACE action (type_action==2) succeed.
**Current state:** Triple detection and golden creation are fully implemented; the discover auto-selects the first card with no UI interaction.
**Open questions / next steps:**
- Add a UI/agent hook so the policy can actually choose the discover card instead of always picking index 0
- Confirm the golden card's `entity_id` is set correctly if entity tracking is added later
- Run a dry-run to verify no runtime errors from the triple path
---

---
### 2026-03-30 — Implement battlecry and sell-effect dispatch (EffectHandler)
**Files changed:** `symbolic/effect_handler.py`, `env/game_loop.py`
**What was done:** Created `symbolic/effect_handler.py` with an `EffectHandler` class that dispatches known buy-phase battlecries and sell effects keyed on normalised card name. Battlecries implemented: Murozond (+1/+1 all others), Waxrider Togwaggle (+2/+2 random Dragon), Deflect-o-Bot (divine shield random Mech), Master of Realities (+1/+1 random Elemental), Recruiter (1/1 Recruit token to hand). Sell effects implemented: Sellemental (1/1 Water Droplet Elemental to hand), Gold Grubber (+2/+1 random friendly). All buff battlecries fire twice when `ps.has_brann` is True. Wired `on_play` into `step_shopping` type_action==2 (after `_update_multiplier_flags`, before triple check) and `on_sell` into type_action==1 (after gold return). `EffectHandler` is instantiated in `BattlegroundsGame.__init__` and degrades gracefully when card_defs is empty.
**Current state:** Playing or selling a minion now triggers the relevant effect for the implemented card set; unrecognised cards are silently ignored.
**Open questions / next steps:**
- Discover-type battlecries (e.g. Murozond BG variant) are still skipped pending a UI/agent hook
- Hero powers remain a no-op placeholder
- Add unit tests for each battlecry and sell effect
---

---
### 2026-03-30 — Magnetic Mechs, smart play positioning, spell handling
**Files changed:** `env/player_state.py`, `env/game_loop.py`
**What was done:** Added `magnetic: bool` and `is_spell: bool` fields to MinionState. Updated `_dict_to_minion` to populate these from card_defs mechanics/keywords. PLACE action now: (1) casts and discards spell cards without consuming a board slot; (2) merges Magnetic Mechs into the rightmost friendly Mech (adding all stats and copying keywords); (3) uses `_smart_position` (new helper) to insert normal minions at position 0 for taunt/divine_shield, end for all others. Blood Gem and Coin spells have concrete effects; others are no-op.
**Current state:** The three mechanics are wired end-to-end; dry-run still passes with no errors.
**Open questions / next steps:**
- Smart positioning could be extended: place high-attack minions in front, low-health in back
- `_cast_spell` handles only Blood Gem and Coin; remaining spells need a dispatch table similar to EffectHandler

---
### 2026-03-30 — Combat sim: Rally, Avenge, end-of-turn, Khadgar, new DRs and SOC triggers
**Files changed:** `symbolic/combat_sim.py`
**What was done:** Extended BGCombatSim with five major mechanic additions: (1) Rally trigger (`_fire_rally`) firing at attack time for Roaring Recruiter (+3/+1 random friendly), Felstomper (+2/+0 all friendly Beasts), Stasis Elemental (+1/+1 all friendlies); (2) Avenge mechanic with `deaths_this_combat` counter on CombatSide, `_check_avenge()` called after each death-resolution wave, covering Famished Felbat, Dragonspawn Lieutenant, Imposing Direhorn, Bristleback Knight; (3) end-of-turn effects (`_fire_end_of_turn`) firing when the attack pointer wraps to 0, implementing Amalgam of the Ancient; (4) Khadgar extra-token helper (`_has_khadgar`, `_summon_tokens_with_khadgar`) available as a call-site utility; (5) three new deathrattles (Selfless Hero, Kaboom Bot, Kangor's Apprentice with dead-Mech tracking on CombatSide) and two new SOC triggers (Red Whelp, Amalgadon).
**Current state:** Combat sim now covers all five major in-combat mechanic families (DR, SOC, Rally, Avenge, EOT); the BGCombatSim public interface is unchanged.
**Open questions / next steps:**
- Khadgar helper exists but existing DR token summons still use direct `_make_token` — callers can be migrated to `_summon_tokens_with_khadgar` in a follow-up
- Avenge triggers fire per avenge-minion at the first death-wave where threshold is crossed; verify BG rules on simultaneous avenge interactions
---

---
### 2026-04-10 — PPO training + turn inspection notebook cells; ANOMALIES.md
**Files changed:** `explore.ipynb`, `ANOMALIES.md`
**What was done:** Added two new notebook sections. (1) PPO Training: imports, policy + PPO setup with BC v2 warm-start, a 50-game self-play training loop with live reward/loss reporting, and training-curve plots. (2) Turn Inspection: a `RecordingAgent` that captures every greedy action with value estimates and type probabilities, a round-by-round text log, a full-game overview (V(s) trace + action-type stacked bar), and a per-round deep-dive heatmap. Also created `ANOMALIES.md` documenting the full Season 11 anomaly pool (the last season anomalies were active, removed in Season 12) with effect text, categories, and simulator implementation notes.
**Current state:** Notebook has 57 cells; all new code cells parse cleanly. Training cells are ready to run.
**Open questions / next steps:**
- Run the training cells and observe whether the agent develops coherent strategies (leveling timing, tribe focus, economy management).
- Check if value estimates V(s) rise meaningfully as training progresses (indicates the critic is learning).
- Increase N_GAMES beyond 50 once behavior looks plausible.

---
### 2026-04-10 — Fix NaN training bug: cache action masks before state mutation
**Files changed:** `train.py`, `explore.ipynb`
**What was done:** PPO updates were producing all-NaN network weights on the first update. Root cause: `_get_observation` returns a live reference to `PlayerState`, so by the time `record_transition` is called (after `step_shopping` mutates the state), `build_type_mask(ps)` and `build_pointer_mask(ps, ta)` reflect the POST-action state. For BUY/SELL/PLACE, the chosen pointer slot is no longer occupied, so `pointer_mask[chosen] = False`, giving `log_prob = -inf` for every pointer transition. The update then computed `ratio = exp(new_lp - (-inf)) = inf`, and `inf * ~0_advantage = nan`, corrupting all gradients. Fixed by caching both masks inside `get_action` (before the action is applied) and using the cached values in `record_transition`. Applied the same fix to the `_Agent` and `RecordingAgent` classes in `explore.ipynb`.
**Current state:** PPO updates now complete cleanly: policy_loss≈0.006, value_loss≈0.187, no NaN. Training is ready to use.
**Open questions / next steps:**
- Run the notebook training cells end-to-end and inspect agent behavior.
- The live-ps reference bug still affects the stored board/shop/hand token arrays in the observation dict — these are numpy arrays captured at obs-build time so they ARE pre-mutation snapshots, unlike the ps object itself. Only the mask computation was affected.

---
### 2026-04-10 — Fix training cell to continue across re-runs
**Files changed:** `explore.ipynb`
**What was done:** Updated the PPO training cell so re-running it continues training from where it left off. Seed is now derived from `ppo_trainer.total_steps` so each re-run uses fresh game seeds instead of replaying the same games. Lists `game_rewards`/`ppo_losses`/`ppo_values` now accumulate across re-runs (checked via `dir()`) so the training-curve plot shows full history. Weights carry over automatically via the live `policy_train`/`ppo_trainer` objects and the on-disk checkpoint.
**Current state:** Training cell is re-run-safe. Just run the cell repeatedly to keep training.
**Open questions / next steps:**
- Run enough games (200+) to see whether reward improves and value loss stabilises.

---
### 2026-04-10 — CPU parallel training via ProcessPoolExecutor
**Files changed:** `train.py`
**What was done:** Implemented the previously-stubbed `--workers` flag. Added `_worker_run_game` (module-level, required for Windows `spawn`) which rebuilds all components locally, runs one game with a frozen policy snapshot, and returns `List[Transition]` + game summary. Added `_train_parallel` which dispatches batches of N games via `ProcessPoolExecutor`, merges returned transitions into the main PPO buffer, and triggers updates/checkpoints at the normal intervals. Serial path (workers=1) is unchanged. Benchmarks: forward pass ~9.3ms, game ~2s, 16 cores → expected ~10-12× speedup.
**Current state:** `python train.py --workers 8 --games 500 --no-firestone` now runs games in parallel. Verified: import OK, serial dry-run OK, parallel dry-run (2 workers) completes and saves checkpoint.
**Open questions / next steps:**
- Measure real-world speedup with `--workers 8` vs `--workers 1` over 50 games.
- Consider adding per-batch wall-clock logging to compare throughput.

---
### 2026-04-10 — Parallel training in notebook
**Files changed:** `explore.ipynb`
**What was done:** Replaced the serial training loop in cell 49 with a `ProcessPoolExecutor`-based parallel loop that reuses `_worker_run_game` from `train.py`. Cell 46 gains `ProcessPoolExecutor`, `clear_output`, and `from train import _worker_run_game` imports. Each batch dispatches `N_WORKERS` games simultaneously, merges returned transitions into `ppo_trainer.buffer`, and triggers a PPO update + checkpoint save when an `UPDATE_INTERVAL` boundary is crossed. A `_live_plot()` helper updates the reward and loss charts after every batch using `clear_output(wait=True)`.
**Current state:** Cell 49 runs parallel training. Set `N_WORKERS = 4` (or `os.cpu_count()`) at the top. Re-running the cell continues training from where it left off.
**Open questions / next steps:**
- Benchmark actual wall-clock speedup vs. serial (expect ~N_WORKERS× on 16-core machine).
- Increase `N_GAMES` to 500+ once the setup is confirmed working.

---
### 2026-04-10 — Architecture overhaul: per-token pointer, slot positions, 3.5M params, all-opp context
**Files changed:** `agent/policy.py`, `env/game_loop.py`, `train.py`, `explore.ipynb`
**What was done:** Redesigned BGPolicyNetwork with four improvements: (1) per-token pointer scoring — three Linear(d_model,1) scorers acting directly on board/shop/hand Transformer outputs replace the global CLS→24 pointer MLP, giving the network direct card-selection signal; (2) slot positional encoding — shared Embedding(10,d_model) applied per zone so slot 0 on the board vs slot 3 is distinguishable; (3) scaled up to d_model=256, 4 layers, 8 heads (~3.5M params vs ~700K); (4) all-opponent scalar context expanded from 8 dims (one announced opponent) to 64 dims (all 8 player slots × 8 dims, own slot zeroed), SCALAR_DIM 38→94. BC warm-start disabled (d_model mismatch); PPO trains from Xavier init.
**Current state:** Dry-run passes cleanly (2 games, 403 transitions, no NaN). Old checkpoint shapes trigger a load warning and are skipped automatically — training starts fresh.
**Open questions / next steps:**
- Delete or archive `bg_agent_ppo.pt` before starting a fresh training run (it will be overwritten on first checkpoint save anyway).
- Run 500+ games to compare convergence against old architecture.
- Future: retrain BC model with new BGPolicyNetwork architecture to re-enable warm-start.
---
### 2026-04-10 — Architecture overhaul + notebook scalar-dim fixes
**Files changed:** `agent/policy.py`, `env/game_loop.py`, `train.py`, `explore.ipynb`
**What was done:** Overhauled BGPolicyNetwork to d_model=256/4-layers/8-heads (~3.46M params). Replaced CLS→24 pointer_head MLP with three per-token scorers (sell/buy/place Linear(256,1)) acting directly on zone tokens. Added slot_pos_embed (Embedding(10,256)) per zone. Expanded scalar context from 38→94 dims: own(24) + all_opp(8×8=64) + lobby(6). Updated _get_observation() in game_loop.py and all train.py build sites. Fixed explore.ipynb cells (policy_forward, scalar_viz, scalar_header, policy_header) that still used 38-dim tensors.
**Current state:** Architecture is complete, dry-run passes (2 games, no NaN). All notebook demo cells use 94-dim scalars. Ready for sustained PPO training from random init.
**Open questions / next steps:**
- Resume PPO training (restart kernel, re-run cells 46-48); monitor reward trend over 50+ games
- BC retrain with new arch for faster convergence warm-start
- Consider in-turn action history LSTM for coherent multi-step turn planning
---
---
### 2026-04-11 — Worker pool optimizations + notebook scalar-dim fixes
**Files changed:** `train.py`, `explore.ipynb`
**What was done:** Fixed notebook cells (policy_header, policy_forward, scalar_header, scalar_viz) that still used 38-dim scalar tensors after the architecture change to 94-dim. Added `_worker_init` pool initializer to train.py so card_defs are sent once per worker process rather than every game call; state_dict is now only recloned after PPO updates. Added `torch.set_num_threads(1)` to `_worker_init` to prevent CPU oversubscription when running many workers.
**Current state:** Training is running with new architecture (~3.46M params, d_model=256). After ~90 games reward is ~-2.8, PPO losses are decreasing normally. Checkpoint saves after each update so training can resume by re-running cell 47 then 49.
**Open questions / next steps:**
- Benchmark N_WORKERS=8 vs 12 with set_num_threads(1) to find optimal worker count on 16-core machine
- Monitor reward trend over 200-500 games for meaningful improvement above -2.8 baseline
- BC retrain with new arch (d_model=256, scalar_dim=94) for warm-start once PPO shows plateau
---
---
### 2026-04-11 — Add batch timeout to prevent silent hangs with many workers
**Files changed:** `explore.ipynb`
**What was done:** Added a 300s timeout to `_pool.map()` in the training cell so that if any worker hangs (infinite game loop, matchmaker deadlock, etc.) the batch is skipped rather than blocking silently forever. Investigated CPU utilization with 12 workers on Ryzen 7 4800U (8 physical / 16 logical cores); recommended N_WORKERS=12 given 58% utilization at 8 workers.
**Current state:** Training cell has timeout safety. Worker pool uses `_worker_init` with `set_num_threads(1)` and 300s per-batch timeout.
**Open questions / next steps:**
- Identify root cause of worker hang (infinite action loop or matchmaker deadlock in game_loop.py)
- Monitor whether timeout is triggered — if so, add logging to identify which game state causes the hang
- Continue reward trend monitoring toward 200-500 game target
---
---
### 2026-04-11 — Fix BrokenProcessPool crash + add OOM recovery
**Files changed:** `explore.ipynb`
**What was done:** Added BrokenProcessPool exception handler to the training cell so that if a worker process is killed by the OS (likely OOM with 12 workers), the pool is rebuilt and training continues rather than crashing. Also confirmed the 300s timeout handler was in the disk version but not being used (stale kernel). Diagnosed 25-minute hang as stale kernel running old cell without timeout. Reduced recommended N_WORKERS to 8 to avoid RAM exhaustion on Ryzen 7 4800U.
**Current state:** Training cell handles both TimeoutError and BrokenProcessPool gracefully. N_WORKERS=8 is the safe default for this machine.
**Open questions / next steps:**
- Check RAM usage in Task Manager while training at 8 workers; if <80% try 10
- Monitor whether BrokenProcessPool warning appears at 8 workers
- Continue reward trend — currently ~-2.8 after ~90 games, expect improvement around 200-500 games
---
