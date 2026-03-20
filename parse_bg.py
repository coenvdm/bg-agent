#!/usr/bin/env python3
"""
Hearthstone Battlegrounds dataset parser.

Parses Power.log files and extracts structured per-round data for ML training.

Output schema (one JSON object per game):
{
  "session":    str,
  "game_index": int,
  "hero":       { entity_id, card_id, name, health, armor, tech_level },
  "anomaly":    str | null,
  "placement":  int | null,
  "rounds": [
    {
      "round": int,
      "shopping": {
        "tavern_tier": int, "hero_health": int, "hero_armor": int,
        "gold_at_start": int, "hero_power_card_id": str, "hero_power_cost": int,
        "shop_at_start":  [ {card_id, name, attack, health, …}, … ],
        "board_at_start": [ {card_id, name, attack, health, …}, … ],
        "spell_shop_at_start": [ {card_id, name, cost, tier, golden, zone_pos}, … ],
        "actions": [
          { "action": "buy"|"place"|"sell"|"reroll"|"freeze",
            "card_id": str, "name": str, "gold_remaining": int },
          { "action": "play_spell", "card_id": str, "name": str,
            "gold_remaining": int, "spell_cost": int },
          { "action": "level_up", "new_tier": int,
            "gold_remaining": int, "upgrade_cost": int },
          { "action": "hero_power", "card_id": str, "name": str,
            "gold_remaining": int, "hero_power_cost": int }
        ],
        "board_at_end": [ … ],
        "hand_at_end":  [ … ],
      },
      "combat": {
        "opponent_hero":     { … } | null,
        "opponent_board":    [ … ],
        "player_board":      [ … ],
        "result":            "win"|"loss"|"tie"|null,
        "hero_health_after": int,
        "hero_armor_after":  int,
      }
    }
  ]
}

Key discoveries from log analysis:
  - GameEntity TURN odd  (1,3,5…) = player shopping phase
  - GameEntity TURN even (2,4,6…) = combat phase
  - Most combat turns do NOT emit MAIN_ACTION (step=10) → detect combat at MAIN_START (step=9)
  - Shop minions = CARDTYPE=MINION, CONTROLLER=player_id, ZONE=SETASIDE
    (they accumulate per-round; we snapshot at MAIN_ACTION start)
  - Board minions = CARDTYPE=MINION, CONTROLLER=player_id, ZONE=PLAY
  - Buy action  → BLOCK PLAY with entity card_id containing "DragBuy" or "DragBuy_Spell"
  - Sell action → BLOCK PLAY with entity card_id = "TB_BaconShop_DragSell"
  - Level up    → BLOCK PLAY with entity card_id containing "TechUp"
  - Reroll      → BLOCK PLAY with entity card_id containing "Reroll"
  - Freeze      → BLOCK PLAY with entity card_id containing "Freeze"
  - Place (board) → BLOCK PLAY for a MINION entity with target=0 during shopping
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

from hslog import LogParser
from hslog.exceptions import CorruptLogError
from hearthstone.enums import BlockType, CardType, GameTag, PlayState, Step, Zone

try:
    from hslog.player import InconsistentPlayerIdError
except ImportError:
    # Older hslog versions may not expose this; define a placeholder so the
    # except clause below still compiles but never matches.
    class InconsistentPlayerIdError(Exception):  # type: ignore
        pass
from hearthstone.cardxml import load_dbf

# TEMP_RESOURCES (tag 137) holds bonus coins from hero powers; not present in all
# library versions, so fall back to the raw numeric ID if the attribute is missing.
_TEMP_RESOURCES_TAG = getattr(GameTag, "TEMP_RESOURCES", 137)

# Card database for name and cost enrichment (spell costs aren't in the game-state tags)
_db, _ = load_dbf()
_CARD_DB: dict = {c.id: c for c in _db.values()}
del _db


def _card_db_cost(card_id: str) -> int:
    c = _CARD_DB.get(card_id)
    return c.cost if c is not None else 0


def _card_db_name(card_id: str) -> str:
    c = _CARD_DB.get(card_id)
    return c.name if c is not None else ""

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BACON_HERO_PREFIX  = "TB_BaconShop_HERO"   # old-style hero card IDs
_BG_HERO_PREFIXES  = (BACON_HERO_PREFIX, "BG")  # newer heroes: BG##_HERO_###


def _is_hero_card(card_id: str, tags: dict = None) -> bool:
    """
    Return True if this entity is the player's active BG hero card.
    Uses CARDTYPE tag when available; falls back to card_id pattern.
    """
    if not card_id:
        return False
    # Prefer CARDTYPE == HERO (most reliable)
    if tags is not None:
        from hearthstone.enums import CardType as CT
        ct = tags.get(GameTag.CARDTYPE)
        if ct is not None:
            return ct == CT.HERO
    # Fallback: old-style TB_BaconShop_HERO_* cards
    if card_id.startswith(BACON_HERO_PREFIX):
        return True
    # Newer BG heroes: BGxx_HERO_nnn (no extra suffix)
    # Hero powers end in 'e', enchantments in 'e2', 'pe', 'pe6' etc.
    # Plain hero cards are exactly: BG##_HERO_### or BG##_HERO_###_SKIN_*
    if card_id.startswith("BG") and "_HERO_" in card_id:
        # Reject if it has a 'e' or 'p' suffix pattern (hero power / enchantment)
        base = card_id.split("_SKIN_")[0]  # strip skin suffix first
        if not (base[-1].isdigit() or base.endswith("_G")):
            return False
        return True
    return False

# Partial card-id strings that identify specific shop actions
_BUY_CARDS    = ("TB_BaconShop_DragBuy",)          # DragBuy / DragBuy_Spell
_SELL_CARDS   = ("TB_BaconShop_DragSell",)
_LEVEL_CARDS  = ("TechUp",)                         # TechUp02_Button etc.
_REROLL_CARDS = ("Reroll_Button", "8p_ShopAction_Roll")
_FREEZE_CARDS = ("Freeze", "8p_ShopAction_Freeze")


def _card_matches(card_id: str, patterns) -> bool:
    return any(p in card_id for p in patterns)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tags_to_dict(tag_list) -> dict:
    """Convert hslog's list-of-(tag,value) tuples into a plain dict."""
    result: dict = {}
    for tag, value in (tag_list or []):
        result[tag] = value
    return result


def _resolve_entity(ref) -> Optional[int]:
    if isinstance(ref, int):
        return ref
    if hasattr(ref, "entity_id"):          # PlayerReference
        return ref.entity_id
    if isinstance(ref, str) and ref == "GameEntity":
        return 1
    return None


def _minion_snap(entity: dict) -> dict:
    tags = entity.get("tags", {})
    return {
        "entity_id":     entity["id"],
        "card_id":       entity.get("card_id", ""),
        "name":          entity.get("name", ""),
        "attack":        tags.get(GameTag.ATK, 0),
        "health":        tags.get(GameTag.HEALTH, 0),
        "divine_shield": bool(tags.get(GameTag.DIVINE_SHIELD, 0)),
        "poisonous":     bool(tags.get(GameTag.POISONOUS, 0)),
        "reborn":        bool(tags.get(GameTag.REBORN, 0)),
        "taunt":         bool(tags.get(GameTag.TAUNT, 0)),
        "windfury":      bool(tags.get(GameTag.WINDFURY, 0)),
        "golden":        bool(tags.get(GameTag.PREMIUM, 0)),
        "tier":          tags.get(GameTag.TECH_LEVEL, 1),
        "zone_pos":      tags.get(GameTag.ZONE_POSITION, 0),
    }


def _spell_snap(entity: dict) -> dict:
    tags    = entity.get("tags", {})
    card_id = entity.get("card_id", "")
    return {
        "entity_id": entity["id"],
        "card_id":   card_id,
        "name":      _card_db_name(card_id),
        "cost":      tags.get(GameTag.COST) or _card_db_cost(card_id),
        "tier":      tags.get(GameTag.TECH_LEVEL, 1),
        "golden":    bool(tags.get(GameTag.PREMIUM, 0)),
        "zone_pos":  tags.get(GameTag.ZONE_POSITION, 0),
    }


def _hero_snap(entity: dict) -> dict:
    tags = entity.get("tags", {})
    return {
        "entity_id":  entity["id"],
        "card_id":    entity.get("card_id", ""),
        "name":       entity.get("name", ""),
        "health":     tags.get(GameTag.HEALTH, 0),
        "armor":      tags.get(GameTag.ARMOR, 0),
        "tech_level": tags.get(GameTag.PLAYER_TECH_LEVEL, 0),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-game stateful tracker
# ──────────────────────────────────────────────────────────────────────────────

class BGGameTracker:
    """
    Replays all packets for one Battlegrounds game and builds the dataset record.

    friendly_player_id = PlayerID of the local human player
    dummy_player_id    = PlayerID of the BACON_DUMMY_PLAYER (Bob / AI opponents)
    """

    def __init__(self, friendly_player_id: int, dummy_player_id: int):
        self.friendly_player_id = friendly_player_id
        self.dummy_player_id    = dummy_player_id

        # entity_id → {"id", "card_id", "name", "tags": dict}
        self.entities: Dict[int, dict] = {}
        self._name_to_id: Dict[str, int] = {}

        self.game_turn:     int = 0
        self.current_step: Optional[int] = None

        self.game_entity_eid: int = 1          # updated from CreateGame packet
        self.player_hero_eid:   Optional[int] = None
        self.player_entity_eid: Optional[int] = None  # player entity (holds RESOURCES tags)
        self.anomaly_card_id: Optional[str] = None

        # Round accumulators
        self.rounds:          List[dict]     = []
        self._cur_round:      Optional[dict] = None
        self._actions:        List[dict]     = []
        self._in_shopping:         bool       = False
        self._in_combat:           bool       = False
        self._shop_at_start:       List[dict] = []
        self._spell_shop_at_start: List[dict] = []
        self._board_at_start:      List[dict] = []
        self._gold_at_start:       int        = 0

        # Hero power entity tracking
        self.player_hero_power_eid: Optional[int] = None

        # Combat snapshots
        self._combat_player_board:   List[dict]     = []
        self._combat_opponent_hero:  Optional[dict] = None
        self._combat_opponent_board: List[dict]     = []

        # Track entity creation turn (for shop freshness)
        self._entity_created_turn: Dict[int, int] = {}
        # Track the turn an entity last entered ZONE=PLAY (for opponent freshness)
        self._entity_entered_play_turn: Dict[int, int] = {}
        # Track the latest non-zero COST seen for each player-owned spell entity.
        # The COST tag is set after MAIN_ACTION fires, so the snapshot at step 10
        # always sees cost=0; we accumulate here across the whole game instead.
        self._spell_costs: Dict[int, int] = {}

    # ── entity helpers ────────────────────────────────────────────────────────

    def _entity(self, eid: int) -> dict:
        if eid not in self.entities:
            self.entities[eid] = {"id": eid, "card_id": "", "name": "", "tags": {}}
        return self.entities[eid]

    def _resolve(self, ref) -> Optional[int]:
        eid = _resolve_entity(ref)
        if eid is not None:
            return eid
        if isinstance(ref, str):
            return self._name_to_id.get(ref)
        return None

    # ── state queries ─────────────────────────────────────────────────────────

    def _is_minion(self, e: dict) -> bool:
        return e["tags"].get(GameTag.CARDTYPE) == CardType.MINION

    def _ctrl(self, e: dict) -> int:
        return e["tags"].get(GameTag.CONTROLLER, 0)

    def _zone(self, e: dict) -> int:
        return e["tags"].get(GameTag.ZONE, Zone.INVALID)

    def player_board(self) -> List[dict]:
        result = [
            _minion_snap(e) for e in self.entities.values()
            if self._is_minion(e)
            and self._ctrl(e) == self.friendly_player_id
            and self._zone(e) == Zone.PLAY
        ]
        result.sort(key=lambda x: x["zone_pos"])
        return result

    def shop_at_turn(self, turn: int) -> List[dict]:
        """
        Shop = ctrl=player_id, zone=SETASIDE MINION entities
        created during the previous combat turn (turn-1) or the current turn.
        This filters out stale shop-slot entities from earlier rounds.
        """
        fresh_threshold = max(0, turn - 2)   # only show entities from this turn or last 2
        result = []
        for e in self.entities.values():
            if (self._is_minion(e)
                    and self._ctrl(e) == self.friendly_player_id
                    and self._zone(e) == Zone.SETASIDE):
                created = self._entity_created_turn.get(e["id"], 0)
                if created >= fresh_threshold:
                    result.append(_minion_snap(e))
        result.sort(key=lambda x: x["zone_pos"])
        return result

    def spell_shop_at_turn(self, turn: int) -> List[dict]:
        """
        Spell cards in the shop: CARDTYPE=SPELL, CONTROLLER=player, ZONE=SETASIDE.
        Uses the same freshness window as shop_at_turn to filter stale entities.
        """
        fresh_threshold = max(0, turn - 2)
        result = []
        for e in self.entities.values():
            if (e["tags"].get(GameTag.CARDTYPE) == CardType.SPELL
                    and self._ctrl(e) == self.friendly_player_id
                    and self._zone(e) == Zone.SETASIDE):
                created = self._entity_created_turn.get(e["id"], 0)
                if created >= fresh_threshold:
                    snap = _spell_snap(e)
                    if snap["cost"] == 0:
                        snap["cost"] = self._spell_costs.get(e["id"], 0)
                    result.append(snap)
        result.sort(key=lambda x: x["zone_pos"])
        return result

    def player_hand(self) -> List[dict]:
        return [
            _minion_snap(e) for e in self.entities.values()
            if self._is_minion(e)
            and self._ctrl(e) == self.friendly_player_id
            and self._zone(e) == Zone.HAND
        ]

    def player_hero(self) -> Optional[dict]:
        if self.player_hero_eid is None:
            return None
        e = self.entities.get(self.player_hero_eid)
        return _hero_snap(e) if e else None

    def _hero_power_entity(self) -> Optional[dict]:
        """Return the player's hero power entity, or None if not found."""
        for e in self.entities.values():
            if (e["tags"].get(GameTag.CARDTYPE) == CardType.HERO_POWER
                    and self._ctrl(e) == self.friendly_player_id):
                return e
        return None

    def _hero_power_cost(self) -> int:
        """Return the gold cost of the player's hero power (0 if unknown)."""
        e = self._hero_power_entity()
        if e is None:
            return 0
        cid  = e.get("card_id", "")
        cost = e["tags"].get(GameTag.COST, 0)
        return cost if cost > 0 else _card_db_cost(cid)

    def _hero_power_card_id(self) -> str:
        """Return the card_id of the player's hero power ("" if unknown)."""
        e = self._hero_power_entity()
        return e.get("card_id", "") if e else ""

    def _available_gold(self) -> int:
        """Gold the player can still spend this turn (total allocated minus already used)."""
        if self.player_entity_eid is None:
            return 0
        tags  = self.entities.get(self.player_entity_eid, {}).get("tags", {})
        total = tags.get(GameTag.RESOURCES, 0) + tags.get(_TEMP_RESOURCES_TAG, 0)
        used  = tags.get(GameTag.RESOURCES_USED, 0)
        return max(0, total - used)

    # ── phase transitions ─────────────────────────────────────────────────────

    def _on_turn(self, new_turn: int):
        self.game_turn = new_turn
        if new_turn % 2 == 1:                    # player's shopping turn
            self._cur_round = {
                "round":    (new_turn + 1) // 2,
                "shopping": None,
                "combat":   None,
            }
            self._actions = []

    def _on_step(self, new_step: int):
        self.current_step = new_step

        if new_step == Step.MAIN_START:           # step = 9
            if self.game_turn % 2 == 0:           # even turn = combat
                self._in_combat = True
                self._combat_player_board   = self.player_board()
                self._capture_combat_opponent()

        elif new_step == Step.MAIN_ACTION:        # step = 10
            if self.game_turn % 2 == 1:           # odd turn = shopping
                self._in_shopping    = True
                self._actions        = []
                self._board_at_start = self.player_board()
                self._shop_at_start  = self.shop_at_turn(self.game_turn)
                self._gold_at_start       = self._available_gold()
                self._spell_shop_at_start = self.spell_shop_at_turn(self.game_turn)

        elif new_step == Step.MAIN_END:           # step = 12
            if self._in_shopping:
                self._in_shopping = False
                self._flush_shopping()
            elif self._in_combat:
                self._in_combat = False
                self._flush_combat()

    def _capture_combat_opponent(self):
        """
        Snapshot the opponent board at the start of combat.

        In BG, opponent minions enter ZONE=PLAY under CONTROLLER=dummy_player_id
        at the start of each combat. We only count entities that entered PLAY
        within the last 2 turns (current turn or the prior shopping turn) to
        avoid accumulation of stale entities from previous combats.
        """
        def _opp_minions_at_threshold(threshold):
            result = []
            for e in self.entities.values():
                if not (self._is_minion(e)
                        and self._zone(e) == Zone.PLAY
                        and self._ctrl(e) != self.friendly_player_id
                        and self._ctrl(e) != 0):
                    continue
                entered = self._entity_entered_play_turn.get(e["id"], 0)
                if entered >= threshold:
                    result.append(_minion_snap(e))
            return result

        # Try decreasing freshness windows:
        #  1. Entities that entered PLAY this exact combat turn (cleanest).
        #  2. Entities from the prior shopping turn (late-game show-opponents).
        #  3. Any ctrl≠player entity currently in PLAY (persistent end-game boards).
        fresh_threshold = self.game_turn
        opp_board = _opp_minions_at_threshold(fresh_threshold)
        if not opp_board:
            fresh_threshold = max(0, self.game_turn - 1)
            opp_board = _opp_minions_at_threshold(fresh_threshold)
        if not opp_board:
            # Final fallback: use all non-player PLAY minions (persistent board)
            fresh_threshold = 0
            opp_board = _opp_minions_at_threshold(fresh_threshold)
        opp_board.sort(key=lambda x: x["zone_pos"])

        # Opponent hero: a BaconShop HERO card in PLAY not owned by the player.
        # Heroes persist in PLAY across multiple rounds so we don't apply a
        # freshness filter here — just pick the one matching the opponent controller
        # (most likely the same ctrl=dummy that owns the opponent board minions).
        opp_ctrls = {_minion_snap(e)["entity_id"] and e["tags"].get(GameTag.CONTROLLER)
                     for e in self.entities.values()
                     if self._is_minion(e)
                     and self._zone(e) == Zone.PLAY
                     and self._ctrl(e) != self.friendly_player_id
                     and self._ctrl(e) != 0
                     and self._entity_entered_play_turn.get(e["id"], 0) >= fresh_threshold}
        opp_ctrls.discard(None)

        opp_hero = None
        for e in self.entities.values():
            if (_is_hero_card(e.get("card_id", ""), e.get("tags"))
                    and self._zone(e) == Zone.PLAY
                    and self._ctrl(e) in opp_ctrls):
                opp_hero = _hero_snap(e)
                break

        self._combat_opponent_hero  = opp_hero
        self._combat_opponent_board = opp_board

    def _flush_shopping(self):
        if self._cur_round is None:
            return
        hero = self.player_hero()
        self._cur_round["shopping"] = {
            "tavern_tier":   hero["tech_level"] if hero else 0,
            "hero_health":   hero["health"]     if hero else 0,
            "hero_armor":    hero["armor"]       if hero else 0,
            "gold_at_start":       self._gold_at_start,
            "hero_power_card_id":  self._hero_power_card_id(),
            "hero_power_cost":     self._hero_power_cost(),
            "shop_at_start":       self._shop_at_start,
            "spell_shop_at_start": self._spell_shop_at_start,
            "board_at_start": self._board_at_start,
            "actions":        list(self._actions),
            "board_at_end":   self.player_board(),
            "hand_at_end":    self.player_hand(),
        }

    def _flush_combat(self):
        if self._cur_round is None:
            return
        hero = self.player_hero()
        self._cur_round["combat"] = {
            "opponent_hero":     self._combat_opponent_hero,
            "opponent_board":    self._combat_opponent_board,
            "player_board":      self._combat_player_board,
            "result":            None,
            "hero_health_after": hero["health"] if hero else 0,
            "hero_armor_after":  hero["armor"]  if hero else 0,
        }
        self.rounds.append(deepcopy(self._cur_round))

    # ── packet handlers ───────────────────────────────────────────────────────

    def _store_entity(self, eid: int, card_id: str, tag_list, name: str = ""):
        entity = self._entity(eid)
        if card_id:
            entity["card_id"] = card_id
        tags = _tags_to_dict(tag_list)
        entity["tags"].update(tags)
        if name:
            entity["name"]        = name
            self._name_to_id[name] = eid

        # Record when this entity was first created/initialized
        if eid not in self._entity_created_turn:
            self._entity_created_turn[eid] = self.game_turn

        # Track when entity is initially placed in PLAY zone (via FullEntity/ShowEntity)
        if tags.get(GameTag.ZONE) == Zone.PLAY:
            # Only update if not already tracked, or if this is a fresh creation
            if eid not in self._entity_entered_play_turn:
                self._entity_entered_play_turn[eid] = self.game_turn

        # Detect anomaly
        if eid == self.game_entity_eid and GameTag.BACON_GLOBAL_ANOMALY_DBID in tags:
            if not self.anomaly_card_id and card_id:
                self.anomaly_card_id = card_id

    def handle_create_game(self, packet):
        eid = packet.entity if isinstance(packet.entity, int) else 1
        self.game_entity_eid = eid        # record real game entity ID
        self._store_entity(eid, "", packet.tags)
        for player in (packet.players or []):
            p_ref = player.entity
            p_eid  = p_ref.entity_id if hasattr(p_ref, "entity_id") else int(p_ref)
            p_name = p_ref.name      if hasattr(p_ref, "name")      else ""
            self._store_entity(p_eid, "", player.tags, name=p_name or "")
            if getattr(player, "player_id", None) == self.friendly_player_id:
                self.player_entity_eid = p_eid

    def handle_full_entity(self, packet):
        eid     = packet.entity if isinstance(packet.entity, int) else _resolve_entity(packet.entity)
        card_id = getattr(packet, "card_id", "") or ""
        name    = getattr(packet, "name",    "") or ""
        if eid is not None:
            self._store_entity(eid, card_id, packet.tags, name=name)

    def handle_show_entity(self, packet):
        eid = _resolve_entity(packet.entity)
        if eid is None and isinstance(packet.entity, str):
            eid = self._name_to_id.get(packet.entity)
        if eid is None:
            return
        card_id = getattr(packet, "card_id", "") or ""
        self._store_entity(eid, card_id, packet.tags)

    def handle_change_entity(self, packet):
        eid = _resolve_entity(packet.entity)
        if eid is None:
            return
        card_id = getattr(packet, "card_id", "") or ""
        self._store_entity(eid, card_id, packet.tags)

    def handle_tag_change(self, packet):
        eid = self._resolve(packet.entity)
        if eid is None:
            return

        tag   = packet.tag
        value = packet.value

        entity  = self._entity(eid)
        entity["tags"][tag] = value

        # Game-level step/turn transitions
        if eid == self.game_entity_eid:
            if tag == GameTag.TURN:
                self._on_turn(int(value))
            elif tag == GameTag.STEP:
                self._on_step(int(value))
            return

        # Track non-zero COST on any player-owned entity.
        # CARDTYPE is intentionally not checked here: the COST TagChange often
        # fires before CARDTYPE is set, so filtering by SPELL would miss entries.
        # The dict is only consumed by spell_shop_at_turn which already filters
        # for SPELL entities before applying the override.
        if (tag == GameTag.COST
                and int(value) > 0
                and self._ctrl(entity) == self.friendly_player_id):
            self._spell_costs[eid] = int(value)

        # Identify player's chosen hero when it enters PLAY zone
        if tag == GameTag.ZONE and int(value) == Zone.PLAY:
            self._entity_entered_play_turn[eid] = self.game_turn
            cid = entity.get("card_id", "")
            if (_is_hero_card(cid, entity["tags"])
                    and entity["tags"].get(GameTag.CONTROLLER) == self.friendly_player_id):
                self.player_hero_eid = eid

    def handle_block(self, packet):
        """
        Process a BLOCK and detect player actions from PLAY-type blocks.
        Gold is captured BEFORE children are dispatched so we record the
        gold available to the player at the moment they made the decision,
        not after the cost has been deducted.
        """
        gold_before = self._available_gold() if self._in_shopping else 0

        for child in (packet.packets or []):
            self._dispatch(child)

        # Only analyse PLAY blocks during the shopping phase
        if not self._in_shopping:
            return
        if packet.type != BlockType.PLAY:
            return

        eid     = packet.entity if isinstance(packet.entity, int) else -1
        entity  = self.entities.get(eid, {})
        card_id = entity.get("card_id", "")
        target  = packet.target if isinstance(packet.target, int) else 0

        snap = {"card_id": card_id, "name": entity.get("name", "")}

        # ── Buy (minion) / Play spell ─────────────────────────────────────────
        # Both fire a DragBuy block; the target's CARDTYPE distinguishes them.
        if _card_matches(card_id, _BUY_CARDS) and target:
            t_entity  = self.entities.get(target, {})
            t_ctype   = t_entity.get("tags", {}).get(GameTag.CARDTYPE)
            action_name = "play_spell" if t_ctype == CardType.SPELL else "buy"
            t_card_id   = t_entity.get("card_id", "")
            action      = {
                "action":         action_name,
                "card_id":        t_card_id,
                "name":           t_entity.get("name", "") or _card_db_name(t_card_id),
                "gold_remaining": gold_before,
            }
            if action_name == "play_spell":
                action["spell_cost"] = gold_before - self._available_gold()
            self._actions.append(action)

        # ── Sell ─────────────────────────────────────────────────────────────
        elif _card_matches(card_id, _SELL_CARDS) and target:
            t_entity = self.entities.get(target, {})
            self._actions.append({
                "action":         "sell",
                "card_id":        t_entity.get("card_id", ""),
                "name":           t_entity.get("name", ""),
                "gold_remaining": gold_before,
            })

        # ── Level up ─────────────────────────────────────────────────────────
        elif _card_matches(card_id, _LEVEL_CARDS):
            hero = self.player_hero()
            self._actions.append({
                "action":         "level_up",
                "new_tier":       hero["tech_level"] if hero else 0,
                "gold_remaining": gold_before,
                "upgrade_cost":   gold_before - self._available_gold(),
            })

        # ── Reroll ───────────────────────────────────────────────────────────
        elif _card_matches(card_id, _REROLL_CARDS):
            self._actions.append({
                "action":         "reroll",
                "gold_remaining": gold_before,
            })

        # ── Freeze ───────────────────────────────────────────────────────────
        elif _card_matches(card_id, _FREEZE_CARDS):
            self._actions.append({
                "action":         "freeze",
                "gold_remaining": gold_before,
            })

        # ── Hero power ───────────────────────────────────────────────────────
        elif (entity.get("tags", {}).get(GameTag.CARDTYPE) == CardType.HERO_POWER
              and self._ctrl(entity) == self.friendly_player_id):
            hp_cost = gold_before - self._available_gold()
            self._actions.append({
                "action":          "hero_power",
                "card_id":         card_id,
                "name":            entity.get("name", "") or _card_db_name(card_id),
                "gold_remaining":  gold_before,
                "hero_power_cost": hp_cost,
            })

        # ── Place minion on board ─────────────────────────────────────────────
        # A MINION-type entity "playing" itself (target=0) = placed on board
        elif (self._is_minion(entity)
              and target == 0
              and self._ctrl(entity) == self.friendly_player_id
              and self._zone(entity) == Zone.PLAY):
            self._actions.append({
                "action":         "place",
                "card_id":        card_id,
                "name":           entity.get("name", ""),
                "gold_remaining": gold_before,
            })

    def _dispatch(self, packet):
        ptype = type(packet).__name__
        if ptype == "CreateGame":
            self.handle_create_game(packet)
        elif ptype == "FullEntity":
            self.handle_full_entity(packet)
        elif ptype == "ShowEntity":
            self.handle_show_entity(packet)
        elif ptype == "ChangeEntity":
            self.handle_change_entity(packet)
        elif ptype == "TagChange":
            self.handle_tag_change(packet)
        elif ptype == "Block":
            self.handle_block(packet)
        # Choices / SendChoices / ChosenEntities / MetaData / HideEntity → skipped

    def process_tree(self, packet_tree):
        for packet in packet_tree.packets:
            self._dispatch(packet)

    # ── finalization ──────────────────────────────────────────────────────────

    def finalize(self) -> dict:
        hero = self.player_hero()

        # Primary: read PLAYER_LEADERBOARD_PLACE from the player's hero entity.
        # This tag is updated live as opponents are eliminated and holds the true
        # final placement (1–8) at game end.
        placement = None
        if self.player_hero_eid is not None:
            hero_entity = self.entities.get(self.player_hero_eid)
            if hero_entity:
                placement = hero_entity["tags"].get(GameTag.PLAYER_LEADERBOARD_PLACE)

        # Fallback: check PLAYER_LEADERBOARD_PLACE on the player-type entity
        if placement is None:
            for e in self.entities.values():
                if (e["tags"].get(GameTag.CONTROLLER) == self.friendly_player_id
                        and e["tags"].get(GameTag.CARDTYPE) == CardType.PLAYER):
                    placement = e["tags"].get(GameTag.PLAYER_LEADERBOARD_PLACE)
                    if placement:
                        break

        # Last resort: PLAYSTATE (only distinguishes 1st vs not-1st)
        if placement is None:
            for e in self.entities.values():
                if (e["tags"].get(GameTag.CONTROLLER) == self.friendly_player_id
                        and e["tags"].get(GameTag.CARDTYPE) == CardType.PLAYER):
                    if e["tags"].get(GameTag.PLAYSTATE) == PlayState.WON:
                        placement = 1
                    break

        return {
            "hero":      hero,
            "anomaly":   self.anomaly_card_id,
            "placement": placement,
            "rounds":    self.rounds,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Player detection
# ──────────────────────────────────────────────────────────────────────────────

def _detect_players(packet_tree) -> tuple:
    """Return (friendly_player_id, dummy_player_id)."""
    for packet in packet_tree.packets:
        if type(packet).__name__ != "CreateGame":
            continue
        friendly_pid = None
        dummy_pid    = None
        for player in (packet.players or []):
            tags = _tags_to_dict(player.tags)
            pid  = player.player_id
            if tags.get(GameTag.BACON_DUMMY_PLAYER, 0) == 1:
                dummy_pid = pid
            else:
                friendly_pid = pid
        return (friendly_pid or 1, dummy_pid or 9)
    return (1, 9)


def is_battlegrounds(packet_tree) -> bool:
    for packet in packet_tree.packets:
        ptype = type(packet).__name__
        if ptype == "CreateGame":
            tags = _tags_to_dict(packet.tags)
            if tags.get(GameTag.BACON_BARTENDER_CARD_ID, 0) != 0:
                return True
            for player in (packet.players or []):
                if _tags_to_dict(player.tags).get(GameTag.BACON_DUMMY_PLAYER, 0) == 1:
                    return True
        elif ptype == "FullEntity":
            if "BaconShop" in (getattr(packet, "card_id", "") or ""):
                return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def parse_power_log(log_path, session_name: str = "") -> List[dict]:
    """
    Parse a Power.log file and return a list of Battlegrounds game records.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    parser = LogParser()
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        try:
            parser.read(fh)
        except (InconsistentPlayerIdError, CorruptLogError):
            # BG logs sometimes re-assign player IDs mid-session; flush
            # whatever was parsed before the bad line and continue.
            pass
    parser.flush()

    records: List[dict] = []
    bg_index = 0

    for packet_tree in parser.games:
        if not is_battlegrounds(packet_tree):
            continue

        friendly_pid, dummy_pid = _detect_players(packet_tree)
        tracker = BGGameTracker(friendly_pid, dummy_pid)
        tracker.process_tree(packet_tree)

        result = tracker.finalize()
        # Skip ghost records: BG game tree detected but no gameplay captured
        # (lobby abandon, client crash before shopping started, etc.)
        if not result["rounds"] and not (result.get("hero") or {}).get("card_id"):
            continue

        records.append({
            "session":    session_name or log_path.parent.name,
            "game_index": bg_index,
            "is_bg":      True,
            **result,
        })
        bg_index += 1

    return records


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Parse a Hearthstone Power.log → Battlegrounds JSON dataset."
    )
    ap.add_argument("log",          help="Path to Power.log")
    ap.add_argument("--output","-o", default="-",
                    help="Output file (default: stdout).")
    ap.add_argument("--pretty",     action="store_true",
                    help="Pretty-print JSON.")
    ap.add_argument("--session",    default="",
                    help="Session name tag.")
    args = ap.parse_args()

    records = parse_power_log(args.log, session_name=args.session)
    print(f"Found {len(records)} Battlegrounds game(s).", file=sys.stderr)

    out_str = json.dumps(records, indent=2 if args.pretty else None,
                         ensure_ascii=False)
    if args.output == "-":
        print(out_str)
    else:
        Path(args.output).write_text(out_str, encoding="utf-8")
        print(f"Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
