"""PlayerState and MinionState dataclasses for Hearthstone Battlegrounds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class OpponentSnapshot:
    """Everything we know about a specific opponent, updated after each combat.

    Stored per opponent player_id in PlayerState.opponent_snapshots so the
    policy can look up the announced next opponent's last known state during
    the shopping phase.
    """

    board: List["MinionState"] = field(default_factory=list)  # last seen board (max 7)
    tavern_tier: int = 1
    health: int = 40
    prev_health: int = 40  # health at the snapshot before this one (for delta)
    armor: int = 0
    board_size: int = 0
    dominant_tribe: Optional[str] = None
    dominant_tribe_count: int = 0
    is_synergistic: bool = False  # dominant tribe count >= 4
    last_seen_round: int = 0      # round number when this snapshot was recorded


@dataclass
class MinionState:
    """Represents a single minion on the board, in hand, or in the shop."""

    entity_id: int = 0
    card_id: str = ""
    name: str = ""
    attack: int = 0
    health: int = 0
    max_health: int = 0
    divine_shield: bool = False
    venomous: bool = False   # unified name; parser uses "poisonous"
    reborn: bool = False
    taunt: bool = False
    windfury: bool = False
    golden: bool = False
    tier: int = 1
    zone_pos: int = 0
    perm_atk_bonus: int = 0  # accumulated permanent attack buffs
    perm_hp_bonus: int = 0   # accumulated permanent health buffs
    game_atk_bonus: int = 0  # this-game attack buffs
    game_hp_bonus: int = 0   # this-game health buffs
    magnetic: bool = False   # Magnetic mechanic: merges with rightmost friendly Mech when played
    is_spell: bool = False   # True for spell cards (no board slot used)

    @classmethod
    def from_snap(cls, snap: dict) -> "MinionState":
        """Construct from a parse_bg.py _minion_snap dict.

        The snap dict has keys: entity_id, card_id, name, attack, health,
        divine_shield, poisonous, reborn, taunt, windfury, golden, tier,
        zone_pos.
        """
        return cls(
            entity_id=snap.get("entity_id", 0),
            card_id=snap.get("card_id", ""),
            name=snap.get("name", ""),
            attack=snap.get("attack", 0),
            health=snap.get("health", 0),
            max_health=snap.get("health", 0),  # initialise max_health from current
            divine_shield=bool(snap.get("divine_shield", False)),
            venomous=bool(snap.get("poisonous", False)),  # parser key is "poisonous"
            reborn=bool(snap.get("reborn", False)),
            taunt=bool(snap.get("taunt", False)),
            windfury=bool(snap.get("windfury", False)),
            golden=bool(snap.get("golden", False)),
            tier=snap.get("tier", 1),
            zone_pos=snap.get("zone_pos", 0),
        )

    def effective_attack(self) -> int:
        """Return base attack plus all accumulated attack bonuses."""
        return self.attack + self.perm_atk_bonus + self.game_atk_bonus

    def effective_health(self) -> int:
        """Return base health plus all accumulated health bonuses."""
        return self.health + self.perm_hp_bonus + self.game_hp_bonus


@dataclass
class PlayerState:
    """Full state for one player in a Hearthstone Battlegrounds game."""

    player_id: int = 0
    hero_card_id: str = ""
    health: int = 40
    armor: int = 0
    max_health: int = 40
    gold: int = 0
    max_gold: int = 10
    tavern_tier: int = 1
    level_cost: int = 5
    frozen: bool = False
    board: List[MinionState] = field(default_factory=list)   # max 7
    hand: List[MinionState] = field(default_factory=list)    # max 10
    shop: List[MinionState] = field(default_factory=list)    # max 7
    round_num: int = 1
    alive: bool = True
    placement: Optional[int] = None
    last_damage_taken: int = 0
    last_damage_dealt: int = 0
    last_result: Optional[str] = None  # "win" | "loss" | "tie"
    tribe_buffs: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    has_brann: bool = False
    has_titus: bool = False
    has_drakkari: bool = False
    # Per-opponent snapshots: keyed by opponent player_id, updated after each
    # combat so the policy can inspect the announced next opponent's state.
    opponent_snapshots: Dict[int, OpponentSnapshot] = field(default_factory=dict)
    # Set at the start of each round (before shopping) once pairings are known.
    next_opponent_id: Optional[int] = None

    @property
    def total_health(self) -> int:
        """Effective HP including armor."""
        return self.health + self.armor

    def get_rank(self, all_players: List["PlayerState"]) -> int:
        """Return 1-based rank by total_health descending (1 = highest health)."""
        sorted_players = sorted(all_players, key=lambda p: p.total_health, reverse=True)
        for rank, player in enumerate(sorted_players, start=1):
            if player.player_id == self.player_id:
                return rank
        return len(all_players)

    @classmethod
    def from_round_record(cls, round_rec: dict, player_id: int = 0) -> "PlayerState":
        """Construct a PlayerState from a parsed round record (parse_bg.py output).

        The round_rec is one element of the "rounds" list in the game JSON:
        {
            "round": int,
            "shopping": { "tavern_tier", "hero_health", "hero_armor",
                          "gold_at_start", "board_at_start", "board_at_end",
                          "hand_at_end", "shop_at_start", "actions", ... },
            "combat":   { "result", "hero_health_after", "hero_armor_after",
                          "player_board", "opponent_board", "opponent_hero", ... }
        }
        """
        shopping = round_rec.get("shopping") or {}
        combat = round_rec.get("combat") or {}

        # Health and armor: prefer post-combat values when available
        health = combat.get("hero_health_after", shopping.get("hero_health", 40))
        armor = combat.get("hero_armor_after", shopping.get("hero_armor", 0))

        # Derive damage taken/dealt from combat result
        last_result = combat.get("result")  # "win" | "loss" | "tie" | None
        hero_health_before = shopping.get("hero_health", health)
        hero_armor_before = shopping.get("hero_armor", armor)
        total_before = hero_health_before + hero_armor_before
        total_after = health + armor
        damage_taken = max(0, total_before - total_after)

        # Board, hand, and shop from the shopping phase end snapshot
        board_snaps = shopping.get("board_at_end") or shopping.get("board_at_start") or []
        hand_snaps = shopping.get("hand_at_end") or []
        shop_snaps = shopping.get("shop_at_start") or []

        board = [MinionState.from_snap(s) for s in board_snaps]
        hand = [MinionState.from_snap(s) for s in hand_snaps]
        shop = [MinionState.from_snap(s) for s in shop_snaps]

        # Detect multiplier cards on the board
        board_card_ids = {m.card_id for m in board}
        has_brann = any("BrannBronzebeard" in cid or "TB_BaconUps_800" in cid
                        for cid in board_card_ids)
        has_titus = any("TitusRivendare" in cid or "TB_BaconUps_116" in cid
                        for cid in board_card_ids)
        has_drakkari = any("DrakkariEnchanter" in cid or "TB_BaconUps_090" in cid
                           for cid in board_card_ids)

        return cls(
            player_id=player_id,
            health=health,
            armor=armor,
            gold=shopping.get("gold_at_start", 0),
            tavern_tier=shopping.get("tavern_tier", 1),
            board=board,
            hand=hand,
            shop=shop,
            round_num=round_rec.get("round", 1),
            alive=health > 0,
            last_damage_taken=damage_taken,
            last_result=last_result,
            has_brann=has_brann,
            has_titus=has_titus,
            has_drakkari=has_drakkari,
        )
