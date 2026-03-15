#!/usr/bin/env python3
"""
Watches the Hearthstone Logs directory and automatically parses new BG games
into the data/ folder as soon as a session finishes.

A session is considered "done" when its Power.log hasn't been modified for
STABLE_SECS seconds (default 60). Runs until you press Ctrl+C.

Usage:
    python watch_logs.py
    python watch_logs.py --poll 15 --stable 30
    python watch_logs.py --logs-dir "D:/Hearthstone/Logs" --output-dir ./data
"""

import argparse
import json
import logging
import time
import sys
from pathlib import Path

logging.disable(logging.WARNING)  # suppress hslog "Broken option nesting" warnings

from parse_bg import parse_power_log

DEFAULT_LOGS_DIR = Path("C:/Program Files (x86)/Hearthstone/Logs")
DEFAULT_OUTPUT   = Path(__file__).parent / "data"
POLL_INTERVAL    = 30   # seconds between directory scans
STABLE_SECS      = 60   # seconds without modification → session is done


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _is_stable(log_file: Path, stable_secs: int) -> bool:
    age = time.time() - _mtime(log_file)
    return age >= stable_secs


def _parse_session(session_dir: Path, output_dir: Path) -> bool:
    """Parse session_dir/Power.log and write JSON to output_dir. Returns True on success."""
    log_file     = session_dir / "Power.log"
    session_name = session_dir.name
    out_file     = output_dir / f"{session_name}.json"

    print(f"[parse] {session_name} … ", end="", flush=True)
    try:
        records = parse_power_log(log_file, session_name=session_name)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return False

    if not records:
        print("0 BG games — skipping")
        return False

    out_file.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    n            = len(records)
    round_counts = [len(r.get("rounds", [])) for r in records]
    avg          = sum(round_counts) / n if n else 0
    print(f"{n} game(s), avg {avg:.1f} rounds → {out_file.name}")
    return True


def watch(logs_dir: Path, output_dir: Path, poll: int, stable: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Watching : {logs_dir}")
    print(f"Output   : {output_dir.resolve()}")
    print(f"Poll every {poll}s · stable after {stable}s of no changes")
    print("Press Ctrl+C to stop.\n")

    # last_parsed_mtime[session_name] = mtime of Power.log when we last parsed it
    # Absence = never parsed this session.
    last_parsed_mtime: dict[str, float] = {}

    # Pre-populate from existing JSON files so we don't re-parse old sessions on startup.
    # We don't know the exact mtime they were parsed at, so use -1 as a sentinel
    # meaning "parsed, but don't know mtime — re-check only if file changed."
    for f in output_dir.glob("Hearthstone_*.json"):
        last_parsed_mtime[f.stem] = -1.0

    if last_parsed_mtime:
        print(f"Skipping {len(last_parsed_mtime)} already-parsed session(s).\n")

    try:
        while True:
            sessions = sorted(logs_dir.glob("Hearthstone_*"))

            for idx, session_dir in enumerate(sessions):
                log_file = session_dir / "Power.log"
                if not log_file.exists():
                    continue

                name      = session_dir.name
                is_latest = idx == len(sessions) - 1
                cur_mtime = _mtime(log_file)

                prev_mtime = last_parsed_mtime.get(name)

                # Old sessions that haven't changed → skip forever
                if prev_mtime is not None and not is_latest:
                    continue

                # Already parsed and nothing new written
                if prev_mtime is not None and cur_mtime == prev_mtime:
                    continue

                # File changed (or never parsed) — wait for stability
                if not _is_stable(log_file, stable):
                    if is_latest:
                        age = time.time() - cur_mtime
                        print(f"[wait]  {name} — active ({age:.0f}s since last write, "
                              f"need {stable}s)")
                    continue

                # Stable and new/updated → parse
                ok = _parse_session(session_dir, output_dir)
                if ok:
                    last_parsed_mtime[name] = cur_mtime

            time.sleep(poll)

    except KeyboardInterrupt:
        print("\nWatcher stopped.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Auto-parse new Hearthstone BG sessions into the data/ folder."
    )
    ap.add_argument(
        "--logs-dir", default=str(DEFAULT_LOGS_DIR),
        help=f"Hearthstone Logs folder (default: {DEFAULT_LOGS_DIR})",
    )
    ap.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT),
        help="Where to write JSON files (default: ./data/)",
    )
    ap.add_argument(
        "--poll", type=int, default=POLL_INTERVAL,
        help=f"Seconds between scans (default: {POLL_INTERVAL})",
    )
    ap.add_argument(
        "--stable", type=int, default=STABLE_SECS,
        help=f"Seconds of no modification before parsing (default: {STABLE_SECS})",
    )
    args = ap.parse_args()

    watch(
        logs_dir=Path(args.logs_dir),
        output_dir=Path(args.output_dir),
        poll=args.poll,
        stable=args.stable,
    )


if __name__ == "__main__":
    main()
