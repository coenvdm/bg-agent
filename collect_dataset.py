#!/usr/bin/env python3
"""
Batch dataset collector for Hearthstone Battlegrounds.

Scans all Power.log sessions under the Hearthstone Logs directory,
parses each one with parse_bg.py, and saves the results to data/.

Usage:
    python collect_dataset.py [--logs-dir <path>] [--output-dir <path>]

Default logs dir: C:/Program Files (x86)/Hearthstone/Logs/
Default output:   ./data/
"""

import argparse
import json
import sys
from pathlib import Path

from parse_bg import parse_power_log

# ─── defaults ──────────────────────────────────────────────────────────────
DEFAULT_LOGS_DIR = Path("C:/Program Files (x86)/Hearthstone/Logs")
DEFAULT_OUTPUT   = Path(__file__).parent / "data"


def collect(logs_dir: Path, output_dir: Path, force: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Each session is a dated sub-folder: Hearthstone_YYYY_MM_DD_HH_MM_SS
    sessions = sorted(logs_dir.glob("Hearthstone_*"))
    if not sessions:
        print(f"No session folders found in {logs_dir}", file=sys.stderr)
        return

    total_games = 0

    for session_dir in sessions:
        log_file = session_dir / "Power.log"
        if not log_file.exists():
            continue

        session_name = session_dir.name
        out_file = output_dir / f"{session_name}.json"

        if out_file.exists() and not force:
            print(f"[skip]  {session_name} (already parsed — use --force to re-parse)")
            continue

        print(f"[parse] {session_name} … ", end="", flush=True)
        try:
            records = parse_power_log(log_file, session_name=session_name)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            continue

        if not records:
            print("0 BG games found")
            continue

        out_file.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        n = len(records)
        total_games += n
        round_counts = [len(r.get("rounds", [])) for r in records]
        avg_rounds = sum(round_counts) / n if n else 0
        print(f"{n} game(s), avg {avg_rounds:.1f} rounds -> {out_file.name}")

    print(f"\nDone. Total BG games collected: {total_games}")
    print(f"Output directory: {output_dir.resolve()}")


def load_all(data_dir: Path) -> list:
    """
    Helper: load all parsed game records from data_dir into one flat list.
    Useful for downstream ML scripts.
    """
    all_records = []
    for json_file in sorted(data_dir.glob("*.json")):
        records = json.loads(json_file.read_text(encoding="utf-8"))
        all_records.extend(records)
    return all_records


def print_stats(data_dir: Path) -> None:
    records = load_all(data_dir)
    if not records:
        print("No data found.")
        return

    heroes: dict = {}
    placements: list = []
    round_counts: list = []

    for rec in records:
        hero = rec.get("hero") or {}
        name = hero.get("name") or hero.get("card_id") or "Unknown"
        heroes[name] = heroes.get(name, 0) + 1

        p = rec.get("placement")
        if p:
            placements.append(p)

        round_counts.append(len(rec.get("rounds", [])))

    print(f"Total games:        {len(records)}")
    print(f"Average placement:  {sum(placements)/len(placements):.2f}" if placements else "Average placement: N/A")
    print(f"Average rounds:     {sum(round_counts)/len(round_counts):.1f}")
    print(f"\nTop heroes played:")
    for hero, count in sorted(heroes.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:4d}×  {hero}")


def main():
    ap = argparse.ArgumentParser(description="Collect Battlegrounds dataset from Hearthstone logs.")
    ap.add_argument(
        "--logs-dir",
        default=str(DEFAULT_LOGS_DIR),
        help=f"Path to Hearthstone Logs folder (default: {DEFAULT_LOGS_DIR})",
    )
    ap.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help=f"Where to write parsed JSON files (default: ./data/)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-parse sessions even if already parsed.",
    )
    ap.add_argument(
        "--stats", action="store_true",
        help="Print dataset statistics and exit.",
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir)

    if args.stats:
        print_stats(output_dir)
        return

    collect(
        logs_dir=Path(args.logs_dir),
        output_dir=output_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
