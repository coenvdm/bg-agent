"""Analysis helpers for data/agent_stats.jsonl — per-agent-type win-rate tracking.

train.py appends one row per player per finished game to this file (see
`_append_agent_stats` in train.py): which agent occupied the seat
("train_current", "heuristic", "greedy", "snapshot_uN", "milestone_uN"),
its placement, and the cumulative PPO step count at that point. This module
loads that log and aggregates it into win rates over training progress, so
you can see which agent types win most often and whether the training
policy is actually improving relative to its own past snapshots and the
scripted anchors.

Usage (e.g. in explore.ipynb, after `import pandas as pd`)::

    from agent_stats import load, rolling_winrate, summary_table, plot_family_winrates

    df = load()                      # data/agent_stats.jsonl
    summary_table(df)                # win rate / avg placement per family, overall + recent
    plot_family_winrates(df)         # rolling win-rate-over-time chart
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

N_PLAYERS = 8
BASELINE_WINRATE = 1.0 / N_PLAYERS  # expected win rate if every seat were equally skilled


def _family(label: str) -> str:
    """Collapse individual snapshot tags into a coarser family for plotting.

    Every "snapshot_uN" / "milestone_uN" collapses to one series each so the
    chart stays readable even after hundreds of distinct snapshot tags have
    been logged; the raw `label` column is kept for drill-down.
    """
    if label in ("train_current", "heuristic", "greedy"):
        return label
    if label.startswith("milestone_"):
        return "milestone_snapshot"
    if label.startswith("snapshot_"):
        return "rolling_snapshot"
    return "unknown"


def load(path: str | Path = "data/agent_stats.jsonl") -> pd.DataFrame:
    """Load the JSONL stats log into a DataFrame with `family` and `win` columns.

    Returns an empty DataFrame (with the expected columns) if the file
    doesn't exist yet or has no rows.
    """
    path = Path(path)
    columns = ["game", "total_steps", "timestamp", "pid", "label", "placement", "reward"]
    if not path.exists():
        return pd.DataFrame(columns=columns + ["family", "win"])

    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df.assign(family=[], win=[])

    df["family"] = df["label"].apply(_family)
    df["win"] = (df["placement"] == 1).astype(int)
    return df


def summary_table(df: pd.DataFrame, recent_n: int = 500) -> pd.DataFrame:
    """Per-family win rate and average placement, overall and over the most
    recent *recent_n* rows (by log order, i.e. most recently played).

    Lower avg_placement is better (1=best, 8=worst); win_rate above
    `BASELINE_WINRATE` (1/8) means that family wins more than its "fair
    share" of games.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["games", "win_rate", "avg_placement",
                     "recent_win_rate", "recent_avg_placement"]
        )

    overall = df.groupby("family").agg(
        games=("win", "count"),
        win_rate=("win", "mean"),
        avg_placement=("placement", "mean"),
    )

    recent_df = df.tail(recent_n)
    recent = recent_df.groupby("family").agg(
        recent_win_rate=("win", "mean"),
        recent_avg_placement=("placement", "mean"),
    )

    out = overall.join(recent, how="left").sort_values("win_rate", ascending=False)
    return out.round(3)


def rolling_winrate(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """Rolling per-family win rate, indexed by row order (chronological).

    Row order (not the `game` column, which resets every training-run call)
    is the safe x-axis: the log is append-only and always written in the
    order games actually completed, across restarts and re-runs alike.
    """
    if df.empty:
        return pd.DataFrame()

    pivot = df.assign(_row=range(len(df))).pivot(index="_row", columns="family", values="win")
    min_periods = max(1, min(window, window // 10))
    return pivot.rolling(window, min_periods=min_periods).mean()


def plot_family_winrates(
    df: pd.DataFrame,
    window: int = 200,
    ax: Optional["object"] = None,
) -> None:
    """Plot rolling win-rate-over-time per agent family, with the 1/8 baseline.

    If `train_current`'s curve is trending above the scripted anchors
    (heuristic/greedy) and above its own rolling_snapshot/milestone_snapshot
    curves, the policy is winning more than its past selves — i.e. improving.
    """
    import matplotlib.pyplot as plt

    if df.empty:
        print("No rows in agent_stats log yet — run training first.")
        return

    rolled = rolling_winrate(df, window=window)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    colors = {
        "train_current":      "#2196F3",
        "heuristic":          "#FF9800",
        "greedy":             "#9C27B0",
        "rolling_snapshot":   "#4CAF50",
        "milestone_snapshot": "#F44336",
    }
    for family in rolled.columns:
        ax.plot(rolled.index, rolled[family], label=family,
                 color=colors.get(family), lw=2)

    ax.axhline(BASELINE_WINRATE, color="gray", lw=1, ls="--",
               label=f"1/{N_PLAYERS} baseline")
    ax.set_xlabel(f"Game (log order, {window}-game rolling window)")
    ax.set_ylabel("Win rate")
    ax.set_title("Win rate by agent type over training")
    ax.legend()
    ax.set_ylim(0, max(0.3, rolled.max().max() * 1.1 if not rolled.empty else 0.3))
