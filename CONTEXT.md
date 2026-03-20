# bg_agent — Development Context Log

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
