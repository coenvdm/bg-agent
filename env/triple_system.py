"""Triple detection and golden card creation for Hearthstone Battlegrounds."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.player_state import MinionState, PlayerState
    from env.tavern_pool import TavernPool


def make_golden(minion: "MinionState") -> None:
    """Upgrade a MinionState to golden in-place: double stats, set golden=True.

    Doubles base attack/health and permanent bonuses. max_health is updated
    to match the new golden health value.
    """
    minion.attack = minion.attack * 2
    minion.health = minion.health * 2
    minion.max_health = minion.max_health * 2
    minion.perm_atk_bonus = minion.perm_atk_bonus * 2
    minion.perm_hp_bonus = minion.perm_hp_bonus * 2
    minion.golden = True


def check_and_process_triple(ps: "PlayerState", tavern_pool: "TavernPool") -> bool:
    """Check if player ps just formed a triple. If so, create the golden and grant discover.

    A triple is three non-golden copies of the same card_id across hand + board
    combined. Two golden copies of the same card do NOT re-triple.

    When a triple is detected:
    - The three source minions are merged into one golden copy (make_golden on
      the first copy found, the other two are returned to the pool).
    - The golden card is placed in hand (not board).
    - The player receives a discover: 3 cards are drawn from tier+1 (capped at
      tier 6). The first candidate is added to hand; the other 2 are returned.

    Modifies ps.hand and ps.board in place, calls tavern_pool.return_cards and
    tavern_pool.draw. Returns True if a triple was processed.
    """
    # Build a flat list of (source, index, zone) for all non-golden minions.
    # zone is "hand" or "board" so we can remove from the right list.
    non_golden_hand = [
        (m, i, "hand") for i, m in enumerate(ps.hand) if not m.golden
    ]
    non_golden_board = [
        (m, i, "board") for i, m in enumerate(ps.board) if not m.golden
    ]
    all_non_golden = non_golden_hand + non_golden_board

    # Count occurrences of each card_id among non-golden minions.
    from collections import Counter
    card_id_counts: Counter = Counter(m.card_id for m, _, _ in all_non_golden)

    # Find the first card_id that has 3+ copies.
    triple_card_id = None
    for card_id, count in card_id_counts.items():
        if count >= 3:
            triple_card_id = card_id
            break

    if triple_card_id is None:
        return False

    # Collect exactly 3 instances of the tripling card_id (preserve order so
    # the golden inherits the stats of the "oldest" copy, i.e. the first found).
    triple_instances = [
        (m, i, zone)
        for m, i, zone in all_non_golden
        if m.card_id == triple_card_id
    ][:3]

    # The first instance becomes the golden; the other two are returned to pool.
    golden_source = triple_instances[0][0]
    sources_to_return = [triple_instances[1][0], triple_instances[2][0]]

    # Remove all three from hand/board. Remove in reverse index order within
    # each zone to preserve indices during deletion.
    hand_indices_to_remove = sorted(
        [i for _, i, zone in triple_instances if zone == "hand"], reverse=True
    )
    board_indices_to_remove = sorted(
        [i for _, i, zone in triple_instances if zone == "board"], reverse=True
    )
    for i in hand_indices_to_remove:
        ps.hand.pop(i)
    for i in board_indices_to_remove:
        ps.board.pop(i)

    # Upgrade the golden source minion in-place.
    make_golden(golden_source)

    # Place the golden card in hand.
    ps.hand.append(golden_source)

    # Return the two non-golden source cards to the pool as plain dicts.
    pool_dicts = []
    for m in sources_to_return:
        pool_dicts.append({
            "card_id": m.card_id,
            "name": m.name,
            "attack": m.attack,
            "health": m.health,
            "tier": m.tier,
        })
    tavern_pool.return_cards(pool_dicts)

    # --- Discover: draw 3 cards from tier+1 (capped at 6) ---
    discover_tier = min(6, ps.tavern_tier + 1)
    candidates = tavern_pool.draw(discover_tier, 3)

    if candidates:
        # Auto-select the first candidate.
        chosen = candidates[0]
        rest = candidates[1:]

        # Convert chosen card to MinionState and add to hand.
        from env.player_state import MinionState
        chosen_minion = MinionState(
            card_id=chosen.get("card_id", chosen.get("id", "")),
            name=chosen.get("name", ""),
            attack=chosen.get("attack", 0),
            health=chosen.get("health", 0),
            max_health=chosen.get("health", 0),
            tier=chosen.get("tier", 1),
        )
        ps.hand.append(chosen_minion)

        # Return unchosen candidates to the pool.
        if rest:
            tavern_pool.return_cards(rest)

    return True
