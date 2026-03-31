"""Hero identity encoding for the RL policy."""
from __future__ import annotations

import json
import os
from typing import Dict, List

_DEFS_PATH = os.path.join(os.path.dirname(__file__), "..", "hero_definitions.json")

NULL_HERO_ID = "TB_BaconShop_HERO_00"

def _load() -> Dict:
    with open(_DEFS_PATH) as f:
        return json.load(f)

_HERO_DEFS: Dict = _load()

N_HEROES: int = 64  # fixed capacity (padded for future additions)

# Map hero_card_id -> dense integer (0 = null/unknown)
HERO_ID_MAP: Dict[str, int] = {
    card_id: entry["hero_id"]
    for card_id, entry in _HERO_DEFS.items()
}

# Map hero_card_id -> full definition dict
HERO_DEF_MAP: Dict[str, dict] = dict(_HERO_DEFS)


def get_hero_id(hero_card_id: str) -> int:
    """Return the dense integer for this hero. Returns 0 for unknown heroes."""
    return HERO_ID_MAP.get(hero_card_id, 0)


def get_hero_def(hero_card_id: str) -> dict:
    """Return the full definition dict for this hero. Falls back to null hero."""
    return HERO_DEF_MAP.get(hero_card_id, HERO_DEF_MAP[NULL_HERO_ID])


def encode_hero_flags(ps) -> List[float]:
    """Return 4 scalar floats encoding hero power state for the policy input.

    Returns [hp_available, hp_cost_norm, counter_norm, charges_norm]
    """
    hdef = get_hero_def(getattr(ps, "hero_card_id", ""))
    used    = getattr(ps, "hero_power_used", True)
    cost    = getattr(ps, "hero_power_cost", 0)
    gold    = getattr(ps, "gold", 0)
    chg     = getattr(ps, "hero_power_charges", -1)
    ctr     = getattr(ps, "hero_power_counter", 0)
    ptype   = hdef.get("power_type", "null")

    hp_available = float(
        not used
        and (chg == -1 or chg > 0)
        and gold >= cost
        and ptype not in ("passive", "null")
    )
    hp_cost_norm   = min(1.0, cost / 10.0)
    counter_norm   = min(1.0, ctr / 20.0)
    charges_norm   = 1.0 if chg == -1 else min(1.0, chg / 10.0)

    return [hp_available, hp_cost_norm, counter_norm, charges_norm]
