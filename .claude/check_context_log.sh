#!/usr/bin/env bash
# Stop hook: blocks session completion until CONTEXT.md is updated.
# Exit code 2 forces Claude Code to handle the output before finishing.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTEXT_FILE="$REPO_ROOT/CONTEXT.md"

cd "$REPO_ROOT" || exit 0

# If CONTEXT.md has already been modified or staged this session — we're done.
CONTEXT_DIRTY=$(git status --porcelain "$CONTEXT_FILE" 2>/dev/null)
CONTEXT_STAGED=$(git diff --cached --name-only 2>/dev/null | grep -F "CONTEXT.md")

if [ -n "$CONTEXT_DIRTY" ] || [ -n "$CONTEXT_STAGED" ]; then
  exit 0
fi

# Check whether any tracked source files were modified this session.
DIRTY_FILES=$(git status --porcelain 2>/dev/null | grep -v '^\?' | grep -v 'CONTEXT.md')

if [ -n "$DIRTY_FILES" ]; then
  echo "" >&2
  echo "╔══════════════════════════════════════════════════════════════════╗" >&2
  echo "║  🚫 SESSION BLOCKED — CONTEXT.md not updated                    ║" >&2
  echo "╠══════════════════════════════════════════════════════════════════╣" >&2
  echo "║  Source files were changed but CONTEXT.md has not been updated. ║" >&2
  echo "╚══════════════════════════════════════════════════════════════════╝" >&2
  echo "" >&2
  echo "ACTION REQUIRED: Append a new entry to CONTEXT.md using the format" >&2
  echo "defined in CLAUDE.md (date, files changed, what was done, current" >&2
  echo "state, open questions). Then stage, commit, and push." >&2
  echo "" >&2
  echo "Changed files this session:" >&2
  echo "$DIRTY_FILES" >&2
else
  # No source files changed — nothing to block on, exit cleanly.
  exit 0
fi

# Exit 2 = Claude Code blocks session exit and must respond to this output.
exit 2
