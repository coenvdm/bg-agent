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
### 2026-03-15 — Add stop hook and restructure CLAUDE.md for session discipline

**Files changed:** `.claude/settings.json`, `.claude/check_context_log.sh`, `CLAUDE.md`

**What was done:** Added a Claude Code `Stop` hook (`.claude/settings.json` + `check_context_log.sh`) that fires at the end of every agent session. The script checks whether `CONTEXT.md` was modified when source files were changed; if not, it prints a prominent warning and exits with code 2, blocking the agent from finishing until the log is updated. Restructured `CLAUDE.md` to place the mandatory end-of-session checklist (append CONTEXT.md → stage → commit → push) immediately after the title, and replaced the old duplicate sections at the bottom with a pointer to the top.

**Current state:** Session discipline infrastructure is in place. The stop hook will enforce CONTEXT.md updates going forward.

**Open questions / next steps:**
- Verify the stop hook fires correctly inside Claude Code by making a test file change
- Consider whether the hook path should use a relative path instead of an absolute one for portability
- All previous open questions from the initial scaffold still apply (end-to-end run, Firestone sim, PPO training)
---
