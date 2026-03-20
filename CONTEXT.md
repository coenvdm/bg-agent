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
