"""Play one game with the current checkpoint and emit a readable trace."""
import json, logging, torch
logging.basicConfig(level=logging.ERROR)
for ln in ('env.game_loop','env.tavern_pool','agent.ppo','symbolic.firestone_client',
           'symbolic.effect_handler','env.trinket_handler','train'):
    logging.getLogger(ln).setLevel(logging.ERROR)
from agent.policy import BGPolicyNetwork, PTR_SHOP_OFF, PTR_BOARD_OFF, PTR_HAND_OFF, ACTION_TYPE_NAMES
from agent.ppo import PPOConfig, PPOTrainer
from train import load_card_defs, CARD_DEFS_PATH, EvalAgent, GreedyPlayAgent, HeuristicAgent
from env.game_loop import BattlegroundsGame
from env.matchmaker import Matchmaker
from env.tavern_pool import TavernPool
from symbolic.board_computer import SymbolicBoardComputer
from symbolic.firestone_client import FirestoneClient

HERO = 0

def mstr(m):
    if m is None: return "-"
    a = getattr(m,'attack',None); h = getattr(m,'health',None)
    if a is None: a = getattr(m,'base_atk',0)
    if h is None: h = getattr(m,'base_hp',0)
    a += getattr(m,'perm_atk_bonus',0)+getattr(m,'game_atk_bonus',0)
    h += getattr(m,'perm_hp_bonus',0)+getattr(m,'game_hp_bonus',0)
    kw=[]
    for k,s in (('taunt','T'),('divine_shield','D'),('reborn','R'),('windfury','W'),('venomous','V')):
        if getattr(m,k,False): kw.append(s)
    t=getattr(m,'tier','?')
    return f"{getattr(m,'name','?')[:18]}({a}/{h}|t{t}{'|'+''.join(kw) if kw else ''})"

def ptr_desc(t, p):
    if p is None or p < 0: return ""
    if t in (0,):  return f" shop[{p-PTR_SHOP_OFF}]"
    if t in (1,8,9): return f" board[{p-PTR_BOARD_OFF}]"
    if t in (2,):  return f" hand[{p-PTR_HAND_OFF}]"
    return f" ptr{p}"

if __name__ == "__main__":
    cd = load_card_defs(CARD_DEFS_PATH)
    pol = BGPolicyNetwork(card_dim=44,d_model=256,nhead=8,num_layers=4,scalar_dim=100,dropout=0.1)
    ck = torch.load('bg_agent_ppo.pt', map_location='cpu', weights_only=False)
    pol.load_state_dict(ck['model_state_dict']); pol.eval()
    trace = {"checkpoint": {"updates": ck['update_count'], "steps": ck['total_steps']}, "rounds": []}

    SEED = 4242
    agents = [EvalAgent(pol, player_id=0, device='cpu')]
    for pid in range(1,8):
        agents.append(GreedyPlayAgent(player_id=pid) if pid<=4 else HeuristicAgent(player_id=pid))
    g = BattlegroundsGame(card_defs=cd, agents=agents, board_computer=SymbolicBoardComputer(cd),
        firestone_client=FirestoneClient(firestone_path=None, mock_mode=True),
        matchmaker=Matchmaker(n_players=8, seed=SEED), tavern_pool=TavernPool(cd, seed=SEED),
        n_players=8, seed=SEED, shape_stats_weight=1.0)
    g.reset()

    cur = {"actions": [], "rnd": 0}
    orig_shop = g.step_shopping
    def wrapped(pid, t, p=-1, _o=orig_shop):
        ps = g.players[pid]
        if pid == HERO:
            before = [mstr(m) for m in ps.board]
            gold_before = ps.gold
        out = _o(pid, t, p)
        if pid == HERO:
            after = [mstr(m) for m in ps.board]
            cur["actions"].append({
                "act": ACTION_TYPE_NAMES[t].upper() + ptr_desc(t,p),
                "gold": f"{gold_before}->{ps.gold}",
                "reward": round(out[1], 4),
                "board_changed": before != after,
                "board_after": after,
            })
        return out
    g.step_shopping = wrapped

    orig_combat = g.step_combat
    def wcombat(pid, opp, _o=orig_combat):
        r = _o(pid, opp)
        if pid == HERO:
            cur["combat"] = {"vs": opp, "result": r["result"],
                             "dmg_taken": round(r["damage_taken"],1),
                             "dmg_dealt": round(r["damage_dealt"],1),
                             "hp_after": g.players[HERO].health}
        return r
    g.step_combat = wcombat

    orig_round = g._run_round if hasattr(g,'_run_round') else None
    res = None
    # capture per-round snapshots by hooking the shop draw
    orig_draw = g._draw_shop
    def wdraw(ps, _o=orig_draw):
        out = _o(ps)
        if ps.player_id == HERO:
            if cur["actions"] or cur.get("combat"):
                trace["rounds"].append(dict(cur))
            cur.clear(); cur.update({"actions": [], "rnd": g.round_num,
                                     "tier": ps.tavern_tier, "hp": ps.health,
                                     "gold": g._gold_for_round(g.round_num),
                                     "shop": [mstr(m) for m in out] if out else [],
                                     "board_start": [mstr(m) for m in ps.board]})
        return out
    g._draw_shop = wdraw

    res = g.run_game()
    if cur.get("actions") or cur.get("combat"):
        trace["rounds"].append(dict(cur))
    trace["result"] = {"placement": res.placements[HERO], "n_rounds": res.n_rounds,
                       "all_placements": {f"P{p}": res.placements[p] for p in range(8)},
                       "final_hp": g.players[HERO].health}
    trace["seats"] = {"P0": "TRAINED AGENT", **{f"P{p}": ("greedy" if p<=4 else "heuristic") for p in range(1,8)}}
    json.dump(trace, open('game_trace.json','w'), indent=1)
    print(f"placement={res.placements[HERO]} rounds={res.n_rounds} captured_rounds={len(trace['rounds'])}")
