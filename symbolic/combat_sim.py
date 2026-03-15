"""
BGCombatSim — pure-Python Hearthstone Battlegrounds combat simulator.

Implements the core BG combat loop:
  - Round-robin attack selection with wrap-around pointer
  - Target selection: taunt preference, random otherwise
  - Divine shield, venomous, windfury, reborn, cleave (Blade Collector)
  - ~20 common deathrattles (token summons, AoE damage, stat buffs)
  - ~7 start-of-combat triggers
  - Titus Rivendare (deathrattles trigger an extra time)
  - Monte Carlo win-probability estimation (200 trials ≈ 2–5 ms)

Not modelled: hero powers, rally, avenge, end-of-turn effects, shop triggers,
blood gems, spellcraft, duos mechanics — none of these fire during combat.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# ── Constants ────────────────────────────────────────────────────────────────

MAX_BOARD     = 7    # maximum minions per side
MAX_TURNS     = 100  # hard cap — prevents infinite loops (e.g. reborn stalemates)
MAX_DR_WAVES  = 10   # max death-resolution iterations per attack
DEFAULT_TRIALS = 200


# ── SimResult (re-exported so callers need not import firestone_client) ──────

@dataclass
class SimResult:
    win_prob:              float
    expected_damage_dealt: float
    expected_damage_taken: float
    trials:                int


# ─────────────────────────────────────────────────────────────────────────────
# CombatMinion
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CombatMinion:
    """Mutable minion state for one combat trial."""

    uid:           int          # unique within a trial; used for identity checks
    name:          str
    card_key:      str          # normalized card_id (for Titus/etc detection)
    name_key:      str          # normalized card name (for DR/SOC dispatch)
    attack:        int
    health:        int
    tier:          int
    tribes:        List[str]    # uppercase e.g. "BEAST"

    divine_shield: bool
    venomous:      bool
    reborn:        bool
    taunt:         bool
    windfury:      bool
    cleave:        bool
    golden:        bool

    # runtime state — mutated during the trial
    dead:              bool = field(default=False, repr=False)
    attacks_this_turn: int  = field(default=0,     repr=False)
    reborn_used:       bool = field(default=False,  repr=False)
    # Stores the UID of the minion that dealt the killing blow (for Leeroy DR)
    killed_by_uid:     int  = field(default=-1,     repr=False)
    # Stitched Salvager stores a clone of its left neighbour at SOC
    _stored_clone: Optional["CombatMinion"] = field(default=None, repr=False)

    # ── helpers ─────────────────────────────────────────────────────────────

    def take_damage(self, amount: int, *, venomous_src: bool = False,
                    killer_uid: int = -1) -> bool:
        """Apply damage. Returns True if divine shield popped (no HP lost)."""
        if amount <= 0:
            return False
        if self.divine_shield:
            self.divine_shield = False
            return True                # shield popped — venomous has no effect
        self.health -= amount
        if venomous_src:
            self.health = min(self.health, 0)  # venomous kills outright
        if self.health <= 0:
            self.dead = True
            self.killed_by_uid = killer_uid
        return False

    def has_tribe(self, tribe: str) -> bool:
        return tribe.upper() in self.tribes

    def is_titus(self) -> bool:
        return "titus" in self.card_key or "tb_baconups_116" in self.card_key

    def make_reborn_copy(self, new_uid: int) -> "CombatMinion":
        """Return a 1-HP copy with Reborn stripped and reborn_used=True.

        Uses __new__ + manual field copy to avoid copy.copy() overhead.
        """
        c = object.__new__(CombatMinion)
        c.__dict__.update(self.__dict__)
        c.uid          = new_uid
        c.health       = 1
        c.divine_shield = False
        c.reborn       = False
        c.reborn_used  = True
        c.dead         = False
        c.attacks_this_turn = 0
        c.killed_by_uid = -1
        c._stored_clone = None
        return c


# ─────────────────────────────────────────────────────────────────────────────
# CombatSide
# ─────────────────────────────────────────────────────────────────────────────

class CombatSide:
    """One player's board for a single combat trial."""

    def __init__(self, minions: List[CombatMinion], tavern_tier: int = 1):
        self.minions:     List[CombatMinion] = list(minions)
        self.tavern_tier: int = tavern_tier
        self._ptr:        int = 0
        self._uid_ctr:    int = max((m.uid for m in minions), default=0) + 1
        # Cached Titus flag — updated in remove() when Titus leaves the board
        self._titus:      bool = any(m.is_titus() for m in minions)
        # Track tribe types that died this combat (for Arid Atrocity)
        self.dead_tribe_types: set = set()

    # ── UID factory ─────────────────────────────────────────────────────────

    def next_uid(self) -> int:
        uid = self._uid_ctr
        self._uid_ctr += 1
        return uid

    # ── Attack pointer ───────────────────────────────────────────────────────

    def pick_attacker(self) -> Optional[CombatMinion]:
        if not self.minions:
            return None
        if self._ptr >= len(self.minions):
            self._ptr = 0
        m = self.minions[self._ptr]
        return m

    def advance_ptr(self) -> None:
        n = len(self.minions)
        if n == 0:
            self._ptr = 0
        else:
            self._ptr = (self._ptr + 1) % n

    # ── Board modification ───────────────────────────────────────────────────

    def position_of(self, uid: int) -> int:
        for i, m in enumerate(self.minions):
            if m.uid == uid:
                return i
        return -1

    def remove(self, minion: CombatMinion) -> int:
        """Remove minion from board; adjust ptr; return position it occupied."""
        pos = self.position_of(minion.uid)
        if pos == -1:
            return -1
        self.minions.pop(pos)
        # Update tribe tracking and Titus cache
        for t in minion.tribes:
            self.dead_tribe_types.add(t)
        if self._titus and minion.is_titus():
            self._titus = any(m.is_titus() for m in self.minions)
        # Adjust ptr
        n = len(self.minions)
        if n == 0:
            self._ptr = 0
        elif pos < self._ptr:
            self._ptr = max(0, self._ptr - 1)
        if n > 0 and self._ptr >= n:
            self._ptr = 0
        return pos

    def insert_at(self, pos: int, minion: CombatMinion) -> bool:
        """Insert minion at pos (clamped). Returns False if board full."""
        if len(self.minions) >= MAX_BOARD:
            return False
        pos = max(0, min(pos, len(self.minions)))
        self.minions.insert(pos, minion)
        if pos < self._ptr and len(self.minions) > 1:
            self._ptr += 1
        return True

    def adjacent_indices(self, pos: int) -> List[int]:
        result = []
        if pos > 0:
            result.append(pos - 1)
        if pos < len(self.minions) - 1:
            result.append(pos + 1)
        return result

    # ── Titus detection ──────────────────────────────────────────────────────

    def has_titus(self) -> bool:
        return self._titus

    # ── Outcome helpers ──────────────────────────────────────────────────────

    def alive(self) -> bool:
        return bool(self.minions)

    def win_damage(self) -> int:
        """Damage dealt when this side wins: tavern_tier + sum of minion tiers."""
        return self.tavern_tier + sum(m.tier for m in self.minions)


# ─────────────────────────────────────────────────────────────────────────────
# Board conversion
# ─────────────────────────────────────────────────────────────────────────────

_CLEAVE_CARDS = {"blade collector", "blade_collector"}

_KEY_TABLE = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ ",
    "abcdefghijklmnopqrstuvwxyz_",
)

def _normalize_key(s: str) -> str:
    return s.translate(_KEY_TABLE).replace("-", "_").replace("'", "").replace(",", "")


def _minion_from_dict(d: dict, uid: int) -> CombatMinion:
    """Convert a minion dict (parse_bg / TavernPool / MinionState format) to CombatMinion."""
    name = d.get("name", "")
    card_key = _normalize_key(d.get("card_id", name))

    raw_tribes = d.get("tribes") or []
    if not raw_tribes and d.get("tribe"):
        raw_tribes = [d["tribe"]]
    tribes = [t.upper() for t in raw_tribes if t]

    name_lower = name.lower()
    cleave = any(c in name_lower for c in _CLEAVE_CARDS)

    atk = (d.get("attack",        0)
         + d.get("perm_atk_bonus", 0)
         + d.get("game_atk_bonus", 0))
    hp  = (d.get("health",        0)
         + d.get("perm_hp_bonus",  0)
         + d.get("game_hp_bonus",  0))

    return CombatMinion(
        uid=uid,
        name=name,
        card_key=card_key,
        name_key=_normalize_key(name),
        attack=max(0, atk),
        health=max(1, hp),
        tier=max(1, d.get("tier", 1)),
        tribes=tribes,
        divine_shield=bool(d.get("divine_shield", False)),
        venomous=bool(d.get("venomous", False) or d.get("poisonous", False)),
        reborn=bool(d.get("reborn", False)),
        taunt=bool(d.get("taunt", False)),
        windfury=bool(d.get("windfury", False)),
        cleave=cleave,
        golden=bool(d.get("golden", False)),
    )


def _clone_side(board: List[dict], tavern_tier: int, uid_offset: int = 0) -> CombatSide:
    minions = [_minion_from_dict(d, uid_offset + i) for i, d in enumerate(board)]
    return CombatSide(minions, tavern_tier)


def _fast_clone_side(templates: List[CombatMinion], tavern_tier: int,
                     uid_offset: int = 0) -> CombatSide:
    """Clone pre-built CombatMinion templates for a trial (avoids dict parsing)."""
    minions = []
    for i, t in enumerate(templates):
        m = object.__new__(CombatMinion)
        m.__dict__.update(t.__dict__)
        m.uid = uid_offset + i
        m.dead = False
        m.attacks_this_turn = 0
        m.reborn_used = False
        m.killed_by_uid = -1
        m._stored_clone = None
        minions.append(m)
    return CombatSide(minions, tavern_tier)


# ─────────────────────────────────────────────────────────────────────────────
# Token factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_token(
    side: CombatSide,
    *,
    name: str,
    attack: int,
    health: int,
    tier: int = 1,
    tribes: Optional[List[str]] = None,
    taunt: bool = False,
    divine_shield: bool = False,
    reborn: bool = False,
    venomous: bool = False,
    windfury: bool = False,
) -> CombatMinion:
    nk = _normalize_key(name)
    return CombatMinion(
        uid=side.next_uid(),
        name=name,
        card_key=nk,
        name_key=nk,
        attack=attack,
        health=health,
        tier=tier,
        tribes=tribes or [],
        divine_shield=divine_shield,
        venomous=venomous,
        reborn=reborn,
        taunt=taunt,
        windfury=windfury,
        cleave=False,
        golden=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Deathrattle dispatch
# ─────────────────────────────────────────────────────────────────────────────

# Handler signature: (dead, pos, friendly, enemy, reborn_queue, trigger_count)
# trigger_count = 1 normally, 2 if Titus, 2 if golden, 4 if both.
# reborn_queue is List[(dead_minion, death_pos)] to process after all DRs.

_DRType = Callable[
    ["CombatMinion", int, "CombatSide", "CombatSide", list, int],
    None,
]

# DR dispatch: two-level lookup for speed.
# _DR_EXACT  maps full name_key → handler  (O(1) hit for registered cards)
# _DR_FRAGS  is a fallback list of (fragment, handler) for partial-name matches
_DR_EXACT: Dict[str, _DRType] = {}
_DR_FRAGS: List[Tuple[str, _DRType]] = []


def _dr_register(*keys: str) -> Callable:
    """Register a DR handler.  Single-word keys become exact lookups;
    keys containing a space are treated as fragment (substring) matchers."""
    def decorator(fn: _DRType) -> _DRType:
        for k in keys:
            kl = k.lower()
            if " " not in kl:
                _DR_EXACT[kl] = fn
            else:
                _DR_FRAGS.append((kl, fn))
        return fn
    return decorator


def _dr_dispatch(
    dead: CombatMinion,
    pos: int,
    friendly: CombatSide,
    enemy: CombatSide,
    reborn_queue: list,
    trigger_count: int,
) -> None:
    # Fast path: exact name_key match
    handler = _DR_EXACT.get(dead.name_key)
    if handler is not None:
        handler(dead, pos, friendly, enemy, reborn_queue, trigger_count)
        return
    # Slow path: substring search (rare — only for multi-word fragments like "three_lil")
    key = dead.name_key
    for fragment, h in _DR_FRAGS:
        if fragment in key:
            h(dead, pos, friendly, enemy, reborn_queue, trigger_count)
            return
    # No DR registered → no-op


# ── Token-summoning DRs ──────────────────────────────────────────────────────

@_dr_register("harmless_bonehead")
def _dr_bonehead(dead, pos, friendly, enemy, rq, tc):
    for i in range(2 * tc):
        friendly.insert_at(pos + i, _make_token(friendly, name="Skeleton", attack=1, health=1,
                                                 tribes=["UNDEAD"]))

@_dr_register("cord_puller")
def _dr_cord_puller(dead, pos, friendly, enemy, rq, tc):
    for i in range(tc):
        friendly.insert_at(pos + i, _make_token(friendly, name="Microbot", attack=1, health=1,
                                                 divine_shield=True, tribes=["MECH"]))

@_dr_register("buzzing_vermin")
def _dr_buzzing_vermin(dead, pos, friendly, enemy, rq, tc):
    for i in range(tc):
        friendly.insert_at(pos + i, _make_token(friendly, name="Beetle", attack=2, health=2,
                                                 tribes=["BEAST"]))

@_dr_register("forest_rover")
def _dr_forest_rover(dead, pos, friendly, enemy, rq, tc):
    for i in range(tc):
        friendly.insert_at(pos + i, _make_token(friendly, name="Beetle", attack=2, health=2,
                                                 tribes=["BEAST"]))

@_dr_register("runed_progenitor")
def _dr_runed_progenitor(dead, pos, friendly, enemy, rq, tc):
    for i in range(tc):
        friendly.insert_at(pos + i, _make_token(friendly, name="Beetle", attack=2, health=2,
                                                 tribes=["BEAST"]))

@_dr_register("turquoise_skitterer")
def _dr_turquoise_skitterer(dead, pos, friendly, enemy, rq, tc):
    bonus = 4 * tc
    for m in friendly.minions:
        if m.has_tribe("BEAST") and "beetle" in m.name.lower():
            m.attack += bonus
            m.health += bonus
    for i in range(tc):
        friendly.insert_at(pos + i, _make_token(friendly, name="Beetle", attack=2, health=2,
                                                 tribes=["BEAST"]))

@_dr_register("cadaver_caretaker")
def _dr_cadaver(dead, pos, friendly, enemy, rq, tc):
    for i in range(3 * tc):
        friendly.insert_at(pos + i, _make_token(friendly, name="Skeleton", attack=1, health=1,
                                                 tribes=["UNDEAD"]))

@_dr_register("handless_forsaken")
def _dr_handless(dead, pos, friendly, enemy, rq, tc):
    for i in range(tc):
        friendly.insert_at(pos + i, _make_token(friendly, name="Hand", attack=2, health=1,
                                                 reborn=True, tribes=["UNDEAD"]))

@_dr_register("twilight_hatchling")
def _dr_hatchling(dead, pos, friendly, enemy, rq, tc):
    for i in range(tc):
        friendly.insert_at(pos + i, _make_token(friendly, name="Whelp", attack=3, health=3,
                                                 tribes=["DRAGON"]))

@_dr_register("twilight_broodmother")
def _dr_broodmother(dead, pos, friendly, enemy, rq, tc):
    for i in range(2 * tc):
        friendly.insert_at(pos + i, _make_token(friendly, name="Twilight Hatchling",
                                                 attack=1, health=1, taunt=True,
                                                 tribes=["DRAGON"]))

@_dr_register("silky_shimmermoth")
def _dr_shimmermoth(dead, pos, friendly, enemy, rq, tc):
    for i in range(tc):
        friendly.insert_at(pos + i, _make_token(friendly, name="Beetle", attack=2, health=2,
                                                 tribes=["BEAST"]))

@_dr_register("eternal_summoner")
def _dr_eternal_summoner(dead, pos, friendly, enemy, rq, tc):
    for i in range(tc):
        friendly.insert_at(pos + i, _make_token(friendly, name="Eternal Knight",
                                                 attack=4, health=1, tribes=["UNDEAD"]))

@_dr_register("deathly_striker")
def _dr_deathly_striker(dead, pos, friendly, enemy, rq, tc):
    # Avenge + DR: summon the stored Undead from hand — we approximate with a 4/4
    for i in range(tc):
        friendly.insert_at(pos + i, _make_token(friendly, name="Undead Token",
                                                 attack=4, health=4, tribes=["UNDEAD"]))

@_dr_register("arid_atrocity")
def _dr_arid_atrocity(dead, pos, friendly, enemy, rq, tc):
    # 6/6 Golem + 1/1 per friendly tribe type that died this combat
    bonus = len(friendly.dead_tribe_types)
    for i in range(tc):
        friendly.insert_at(pos + i, _make_token(friendly, name="Golem",
                                                 attack=6 + bonus, health=6 + bonus))

@_dr_register("stitched_salvager")
def _dr_stitched_salvager(dead, pos, friendly, enemy, rq, tc):
    if dead._stored_clone is not None:
        for i in range(tc):
            clone = copy.copy(dead._stored_clone)
            clone.uid = friendly.next_uid()
            clone.dead = False
            clone.attacks_this_turn = 0
            friendly.insert_at(pos + i, clone)


# ── AoE damage DRs ───────────────────────────────────────────────────────────

@_dr_register("tunnel_blaster")
def _dr_tunnel_blaster(dead, pos, friendly, enemy, rq, tc):
    dmg = 3 * tc
    for m in list(friendly.minions) + list(enemy.minions):
        m.take_damage(dmg)

@_dr_register("silent_enforcer")
def _dr_silent_enforcer(dead, pos, friendly, enemy, rq, tc):
    dmg = 2 * tc
    for m in list(enemy.minions):
        m.take_damage(dmg)
    for m in list(friendly.minions):
        if not m.has_tribe("DEMON"):
            m.take_damage(dmg)

@_dr_register("spiked_savior")
def _dr_spiked_savior(dead, pos, friendly, enemy, rq, tc):
    # +1 HP and 1 damage to each friendly — net 0 HP but pops divine shields
    for m in list(friendly.minions):
        m.health += 1 * tc
        m.take_damage(1 * tc)

@_dr_register("photobomber")
def _dr_photobomber(dead, pos, friendly, enemy, rq, tc):
    if not enemy.minions:
        return
    target = max(enemy.minions, key=lambda m: m.health)
    target.take_damage(2 * tc)


# ── Stat-buff DRs ────────────────────────────────────────────────────────────

@_dr_register("silithid_burrower")
def _dr_silithid(dead, pos, friendly, enemy, rq, tc):
    for m in friendly.minions:
        if m.has_tribe("BEAST"):
            m.attack += 1 * tc
            m.health += 1 * tc

@_dr_register("showy_cyclist")
def _dr_showy(dead, pos, friendly, enemy, rq, tc):
    for m in friendly.minions:
        if m.has_tribe("NAGA"):
            m.attack += 2 * tc
            m.health += 2 * tc

@_dr_register("stellar_freebooter")
def _dr_stellar(dead, pos, friendly, enemy, rq, tc):
    if friendly.minions:
        import random as _r
        target = _r.choice(friendly.minions)
        target.health += dead.attack * tc

@_dr_register("three_lil")  # matches "three_lil_quilboar"
def _dr_three_lil(dead, pos, friendly, enemy, rq, tc):
    for m in friendly.minions:
        if m.has_tribe("QUILBOAR"):
            m.attack += 3 * tc
            m.health += 3 * tc

@_dr_register("leeroy_the_reckless")
def _dr_leeroy(dead, pos, friendly, enemy, rq, tc):
    """Destroy the minion that killed Leeroy."""
    killer_uid = dead.killed_by_uid
    if killer_uid < 0:
        return
    for m in enemy.minions:
        if m.uid == killer_uid:
            m.dead = True
            m.killed_by_uid = -1
            break


# ─────────────────────────────────────────────────────────────────────────────
# Start-of-combat triggers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_soc(
    side: CombatSide,
    opp: CombatSide,
    rng: random.Random,
    tavern_tier: int,
) -> None:
    """Fire start-of-combat effects for all minions on *side*, left to right."""
    for m in list(side.minions):  # snapshot — board may change during SOC
        key = m.name_key
        if "amber_guardian" in key:
            _soc_amber_guardian(m, side, rng)
        elif "humming_bird" in key or "hummingbird" in key:
            _soc_humming_bird(m, side)
        elif "prized_promo" in key:            # "prized_promo_drake" or "prized_promo-drake"
            _soc_promo_drake(m, side)
        elif "misfit_dragonling" in key:
            _soc_misfit_dragonling(m, tavern_tier)
        elif "fire_forged_evoker" in key or "fire-forged_evoker" in key:
            _soc_fire_evoker(m, side)
        elif "irate_rooster" in key:
            _soc_irate_rooster(m, side)
        elif "soulsplitter" in key:
            _soc_soulsplitter(m, side, rng)
        elif "stitched_salvager" in key:
            _soc_stitched_salvager(m, side)


def _soc_amber_guardian(m: CombatMinion, side: CombatSide, rng: random.Random):
    mult = 2 if m.golden else 1
    candidates = [x for x in side.minions if x is not m and x.has_tribe("DRAGON")]
    if candidates:
        t = rng.choice(candidates)
        t.attack += 2 * mult
        t.health += 2 * mult
        t.divine_shield = True


def _soc_humming_bird(m: CombatMinion, side: CombatSide):
    mult = 2 if m.golden else 1
    for x in side.minions:
        if x.has_tribe("BEAST"):
            x.attack += 1 * mult


def _soc_promo_drake(m: CombatMinion, side: CombatSide):
    mult = 2 if m.golden else 1
    for x in side.minions:
        if x.has_tribe("DRAGON"):
            x.attack += 4 * mult
            x.health += 4 * mult


def _soc_misfit_dragonling(m: CombatMinion, tavern_tier: int):
    mult = 2 if m.golden else 1
    m.attack += tavern_tier * mult
    m.health += tavern_tier * mult


def _soc_fire_evoker(m: CombatMinion, side: CombatSide):
    mult = 2 if m.golden else 1
    for x in side.minions:
        if x.has_tribe("DRAGON"):
            x.attack += 2 * mult
            x.health += 1 * mult


def _soc_irate_rooster(m: CombatMinion, side: CombatSide):
    mult = 2 if m.golden else 1
    pos = side.position_of(m.uid)
    if pos == -1:
        return
    for adj in side.adjacent_indices(pos):
        x = side.minions[adj]
        x.take_damage(1)          # 1 damage to adjacent
        x.attack += 4 * mult      # +4 attack to adjacent


def _soc_soulsplitter(m: CombatMinion, side: CombatSide, rng: random.Random):
    candidates = [x for x in side.minions
                  if x is not m and x.has_tribe("UNDEAD") and not x.reborn]
    if candidates:
        rng.choice(candidates).reborn = True


def _soc_stitched_salvager(m: CombatMinion, side: CombatSide):
    """Destroy the left neighbour; store a deep copy to summon on death."""
    pos = side.position_of(m.uid)
    if pos <= 0:
        return
    neighbour = side.minions[pos - 1]
    m._stored_clone = copy.copy(neighbour)   # snapshot before removal
    neighbour.dead = True                    # will be collected in next death wave


# ─────────────────────────────────────────────────────────────────────────────
# Attack and damage
# ─────────────────────────────────────────────────────────────────────────────

def _pick_target(attacker: CombatMinion, defender_side: CombatSide,
                 rng: random.Random) -> Optional[CombatMinion]:
    pool = [m for m in defender_side.minions if m.taunt] or defender_side.minions
    return rng.choice(pool) if pool else None


def _do_attack(
    attacker: CombatMinion,
    attacker_side: CombatSide,
    defender_side: CombatSide,
    rng: random.Random,
) -> None:
    """Execute one attack from *attacker* against a chosen target."""
    target = _pick_target(attacker, defender_side, rng)
    if target is None:
        return

    t_pos = defender_side.position_of(target.uid)

    # Target takes damage from attacker
    target.take_damage(attacker.attack, venomous_src=attacker.venomous,
                       killer_uid=attacker.uid)

    # Cleave: attacker also damages minions adjacent to the primary target
    if attacker.cleave and t_pos >= 0:
        for adj in defender_side.adjacent_indices(t_pos):
            defender_side.minions[adj].take_damage(attacker.attack,
                                                   venomous_src=attacker.venomous,
                                                   killer_uid=attacker.uid)

    # Counter-attack: target retaliates against attacker (primary target only)
    attacker.take_damage(target.attack, venomous_src=target.venomous,
                         killer_uid=target.uid)

    attacker.attacks_this_turn += 1


# ─────────────────────────────────────────────────────────────────────────────
# Death resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_deaths(side_a: CombatSide, side_b: CombatSide,
                    rng: random.Random, attacker_is_a: bool = True) -> None:
    """
    Iterative death resolution: process dead minions in waves until none remain.

    Processing order per wave: attacker's side first (BG convention), then
    defender's side.  Within a side: left-to-right by board position.
    """
    for _ in range(MAX_DR_WAVES):
        dead_a = [m for m in list(side_a.minions) if m.dead]
        dead_b = [m for m in list(side_b.minions) if m.dead]

        if not dead_a and not dead_b:
            break

        sides_ordered = (
            [(side_a, side_b, dead_a), (side_b, side_a, dead_b)] if attacker_is_a
            else [(side_b, side_a, dead_b), (side_a, side_b, dead_a)]
        )

        for friendly, enemy, dead_list in sides_ordered:
            reborn_queue: List[Tuple[CombatMinion, int]] = []
            has_titus = friendly.has_titus()

            for dead_m in dead_list:
                pos = friendly.remove(dead_m)
                if pos == -1:
                    pos = 0

                # Trigger count: 2 if golden, 2 if titus, 4 if both
                tc = (2 if dead_m.golden else 1) * (2 if has_titus else 1)
                _dr_dispatch(dead_m, pos, friendly, enemy, reborn_queue, tc)

                # Queue reborn if eligible
                if dead_m.reborn and not dead_m.reborn_used:
                    reborn_queue.append((dead_m, pos))

            # Apply reborn resurrections (after all DRs on this side have fired)
            for dead_m, death_pos in reborn_queue:
                insert_pos = min(death_pos, len(friendly.minions))
                copy_m = dead_m.make_reborn_copy(friendly.next_uid())
                friendly.insert_at(insert_pos, copy_m)


# ─────────────────────────────────────────────────────────────────────────────
# Single-trial combat
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_trial_fast(
    p_templates: List[CombatMinion],
    o_templates: List[CombatMinion],
    player_tier: int,
    opp_tier:    int,
    rng:         random.Random,
) -> Tuple[str, int, int]:
    """Like _run_one_trial but accepts pre-built CombatMinion templates."""
    side_p = _fast_clone_side(p_templates, player_tier, uid_offset=0)
    side_o = _fast_clone_side(o_templates, opp_tier,    uid_offset=1000)
    return _combat(side_p, side_o, rng, player_tier, opp_tier)


def _combat(
    side_p: CombatSide,
    side_o: CombatSide,
    rng:    random.Random,
    player_tier: int,
    opp_tier:    int,
) -> Tuple[str, int, int]:
    """Run one combat given two already-cloned CombatSide objects."""
    # ── Start-of-combat triggers ─────────────────────────────────────────────
    if rng.random() < 0.5:
        _apply_soc(side_p, side_o, rng, player_tier)
        _apply_soc(side_o, side_p, rng, opp_tier)
    else:
        _apply_soc(side_o, side_p, rng, opp_tier)
        _apply_soc(side_p, side_o, rng, player_tier)

    _resolve_deaths(side_p, side_o, rng, attacker_is_a=True)

    # ── Combat loop ──────────────────────────────────────────────────────────
    player_first = rng.random() < 0.5

    for turn in range(MAX_TURNS):
        if not side_p.alive() or not side_o.alive():
            break

        player_turn = (turn % 2 == 0) == player_first
        current = side_p if player_turn else side_o
        other   = side_o if player_turn else side_p

        attacker = current.pick_attacker()
        if attacker is None or not other.alive():
            break

        _do_attack(attacker, current, other, rng)
        _resolve_deaths(side_p, side_o, rng, attacker_is_a=player_turn)

        if not side_p.alive() or not side_o.alive():
            break

        # Windfury: second attack (same attacker, pointer does not advance yet)
        if (attacker.windfury
                and not attacker.dead
                and attacker.attacks_this_turn < 2
                and other.alive()):
            _do_attack(attacker, current, other, rng)
            _resolve_deaths(side_p, side_o, rng, attacker_is_a=player_turn)

        attacker.attacks_this_turn = 0
        current.advance_ptr()

    # ── Outcome ──────────────────────────────────────────────────────────────
    p_alive = side_p.alive()
    o_alive = side_o.alive()
    if p_alive and not o_alive:
        return "win",  side_p.win_damage(), 0
    elif o_alive and not p_alive:
        return "loss", 0, side_o.win_damage()
    else:
        return "tie",  0, 0


def _run_one_trial(
    player_board: List[dict],
    opp_board:    List[dict],
    player_tier:  int,
    opp_tier:     int,
    rng:          random.Random,
) -> Tuple[str, int, int]:
    """Simulate one combat from raw minion dicts."""
    side_p = _clone_side(player_board, player_tier, uid_offset=0)
    side_o = _clone_side(opp_board,    opp_tier,    uid_offset=1000)
    return _combat(side_p, side_o, rng, player_tier, opp_tier)


# ─────────────────────────────────────────────────────────────────────────────
# BGCombatSim — public API
# ─────────────────────────────────────────────────────────────────────────────

class BGCombatSim:
    """
    Monte Carlo BG combat simulator.

    Parameters
    ----------
    n_trials:
        Number of independent trials per ``simulate()`` call (default 200).
        200 trials takes ~1–3 ms for typical mid-game boards.

    Usage
    -----
    ::

        sim = BGCombatSim()
        result = sim.simulate(player_board, opp_board, player_tier=3, opp_tier=4)
        print(result.win_prob, result.expected_damage_taken)
    """

    def __init__(self, n_trials: int = DEFAULT_TRIALS, seed: Optional[int] = None):
        self.n_trials = n_trials
        self._rng = random.Random(seed)

    def simulate(
        self,
        player_board:  List[dict],
        opponent_board: List[dict],
        player_tier:   int = 1,
        opp_tier:      int = 1,
    ) -> SimResult:
        """Run Monte Carlo simulation and return aggregated SimResult.

        Parameters
        ----------
        player_board, opponent_board:
            Lists of minion dicts (keys: name, card_id, attack, health, tier,
            divine_shield, venomous/poisonous, reborn, taunt, windfury, golden,
            tribes/tribe, perm_atk_bonus, game_atk_bonus, perm_hp_bonus,
            game_hp_bonus).
        player_tier, opp_tier:
            Tavern tiers — used for win-damage calculation.
        """
        # Build minion templates once; each trial clones them (faster than re-parsing dicts)
        p_templates = [_minion_from_dict(d, i)        for i, d in enumerate(player_board)]
        o_templates = [_minion_from_dict(d, 1000 + i) for i, d in enumerate(opponent_board)]

        wins = losses = ties = 0
        total_dealt  = 0.0
        total_taken  = 0.0

        for _ in range(self.n_trials):
            try:
                result, dealt, taken = _run_one_trial_fast(
                    p_templates, o_templates,
                    player_tier, opp_tier,
                    self._rng,
                )
            except Exception:
                # Malformed input or unhandled edge case → conservative tie
                ties += 1
                continue

            if result == "win":
                wins  += 1
                total_dealt += dealt
            elif result == "loss":
                losses += 1
                total_taken += taken
            else:
                ties  += 1

        n = self.n_trials
        return SimResult(
            win_prob=wins / n,
            # Expected damage conditional on winning/losing
            expected_damage_dealt=total_dealt / max(wins,   1),
            expected_damage_taken=total_taken / max(losses, 1),
            trials=n,
        )
