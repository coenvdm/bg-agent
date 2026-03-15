#!/usr/bin/env bash
# Stop hook: warns if CONTEXT.md has not been updated this session.
# Claude Code surfaces this output before the agent finishes — acting as a
# hard-to-ignore reminder to append to CONTEXT.md and commit.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTEXT_FILE="$REPO_ROOT/CONTEXT.md"

# Check if CONTEXT.md has any uncommitted changes (new content appended)
cd "$REPO_ROOT" || exit 0

CONTEXT_DIRTY=$(git status --porcelain "$CONTEXT_FILE" 2>/dev/null)
CONTEXT_STAGED=$(git diff --cached --name-only 2>/dev/null | grep -F "CONTEXT.md")

if [ -n "$CONTEXT_DIRTY" ] || [ -n "$CONTEXT_STAGED" ]; then
  # CONTEXT.md has been modified or staged — good, nothing to warn about
  exit 0
fi

# Check if any tracked source files were modified this session
DIRTY_FILES=$(git status --porcelain 2>/dev/null | grep -v '^\?' | grep -v 'CONTEXT.md')

if [ -z "$DIRTY_FILES" ]; then
  # No source changes at all — no need to log
  exit 0
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ⚠️  SESSION WRAP-UP REQUIRED — CONTEXT.md not updated           ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Source files were changed this session but CONTEXT.md has      ║"
echo "║  not been updated. Before finishing, you MUST:                   ║"
echo "║                                                                  ║"
echo "║  1. Append a new entry to CONTEXT.md (see format in CLAUDE.md)  ║"
echo "║  2. Stage changed source files (not *.pt, data/, __pycache__/)  ║"
echo "║  3. git commit -m 'Session YYYY-MM-DD — <short title>'          ║"
echo "║  4. git push origin master                                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Exit code 2 blocks the agent from stopping and forces it to address this.
exit 2
