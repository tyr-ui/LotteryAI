"""Evaluation Dashboard generation for LotteryAI v2.1."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
from storage import load_json, save_json

SCHEMA_VERSION = "1.0"
GAME_KEYS = ("loto6", "loto7", "miniloto", "numbers3", "numbers4")
GAME_NAMES = {"loto6":"LOTO6","loto7":"LOTO7","miniloto":"ミニロト","numbers3":"Numbers3","numbers4":"Numbers4"}
COMBINATION = {"loto6", "loto7", "miniloto"}

def _map(v: Any) -> Mapping[str, Any]:
    return v if isinstance(v, Mapping) else {}

def _list(v: Any) -> list[Any]:
    return list(v) if isinstance(v, Sequence) and not isinstance(v,(str,bytes,bytearray)) else []

def _num(v: Any):
    return None if isinstance(v,bool) or not isinstance(v,(int,float)) else v

def _round(v: Any):
    n=_num(v)
    return None if n is None else (n if isinstance(n,int) else round(float(n),6))

def _level(n:int)->str:
    return "未評価" if n<=0 else "データ不足" if n<5 else "参考値" if n<20 else "評価可能"

def _source(name: Any, ranked: Sequence[Any]) -> str|None:
    target=str(name or "")
    for item in ranked:
        row=_map(item)
        if str(row.get("config",""))==target and row.get("search_origin"):
            return str(row["search_origin"])
    low=target.lower()
    return next((s for s in ("experience","evolution","random","local","base") if low.startswith(s)),None)

def _history_rows(history: Sequence[Any], game:str)->list[Mapping[str,Any]]:
    rows=[_map(x) for x in history if _map(x).get("draw_type")==game and _map(x).get("status")=="evaluated"]
    return sorted(rows,key=lambda r:int(r.get("draw_no",0) or 0))

def _window(rows: Sequence[Mapping[str,Any]])->dict[str,Any]:
    if not rows:
        return {"evaluated_draws":0,"avg_best_match_count":None,"avg_all_pattern_matches":None,"max_best_match_count":None,"hit_rate_1match":None,"hit_rate_2match":None,"hit_rate_3match":None,"hit_rate_4match":None}
    def avg(key:str):
        vals=[float(v) for r in rows if (v:=_num(r.get(key))) is not None]
        return round(sum(vals)/len(vals),6) if vals else None
    best=[int(v) for r in rows if (v:=_num(r.get("best_match_count"))) is not None]
    return {"evaluated_draws":len(rows),"avg_best_match_count":avg("best_match_count"),"avg_all_pattern_matches":avg("avg_match_count"),"max_best_match_count":max(best) if best else None,"hit_rate_1match":avg("hit_rate_1match"),"hit_rate_2match":avg("hit_rate_2match"),"hit_rate_3match":avg("hit_rate_3match"),"hit_rate_4match":avg("hit_rate_4match")}

def _selected_rank(opt:Mapping[str,Any], selected:Any)->Mapping[str,Any]:
    ranked=_list(opt.get("ranked_configs"))
    for item in ranked:
        row=_map(item)
        if row.get("config")==selected:return row
    return _map(ranked[0]) if ranked else {}

def _experience(store:Mapping[str,Any])->dict[str,Any]:
    stats={}
    for name,value in _map(store.get("search_source_statistics")).items():
        row=_map(value)
        stats[str(name)]={"count":int(row.get("count",0) or 0),"share":_round(row.get("share")),"average_selection_score":_round(row.get("average_selection_score")),"best_selection_score":_round(row.get("best_selection_score"))}
    return {"history_count":int(store.get("history_count",0) or 0),"unique_config_count":int(store.get("unique_config_count",0) or 0),"average_selection_score":_round(store.get("average_selection_score")),"best_config_name":store.get("best_config_name"),"best_selection_score":_round(store.get("best_selection_score")),"source_statistics":stats}

def _game(game:str, run:Mapping[str,Any], hist:Sequence[Any], summary:Mapping[str,Any], opt_all:Mapping[str,Any], exp_all:Mapping[str,Any])->dict[str,Any]:
    rg=_map(_map(run.get("games")).get(game)); og=_map(opt_all.get(game)); eg=_map(_map(exp_all.get("games")).get(game))
    selected=rg.get("selected_config") or og.get("selected_config"); ranked=_list(og.get("ranked_configs")); sr=_selected_rank(og,selected)
    rows=_history_rows(hist,game); sm=_map(summary.get(game)); count=int(sm.get("evaluated_draws",len(rows)) or 0); meta=_map(og.get("search_metadata"))
    observed={"status":_level(count),"evaluated_draws":count,"all_time":{"avg_best_match_count":_round(sm.get("avg_best_match_count")),"avg_all_pattern_matches":_round(sm.get("avg_all_pattern_matches")),"max_best_match_count":_round(sm.get("max_best_match_count")),"best_draw_no":sm.get("best_draw_no"),"latest_evaluated_draw_no":sm.get("latest_evaluated_draw_no")},"recent_5":_window(rows[-5:]),"recent_20":_window(rows[-20:])}
    back={"selection_score":_round(sr.get("selection_score")),"avg_matches":_round(sr.get("avg_matches")),"average_matches_per_ticket":_round(sr.get("average_matches_per_ticket")),"random_uplift":_round(sr.get("random_uplift")),"tested_periods":sr.get("tested_periods"),"algorithm":meta.get("algorithm"),"requested_allocation":dict(_map(meta.get("requested_allocation"))),"effective_allocation":dict(_map(meta.get("effective_allocation")))}
    if game not in COMBINATION:
        nb=_map(og.get("numbers_backtest")) or sr
        back["numbers"]={"average_best_position_matches":_round(nb.get("average_best_position_matches") or nb.get("avg_matches")),"average_position_matches_per_ticket":_round(nb.get("average_position_matches_per_ticket") or nb.get("average_matches_per_ticket")),"average_best_unordered_matches":_round(nb.get("average_best_unordered_matches")),"average_unordered_matches_per_ticket":_round(nb.get("average_unordered_matches_per_ticket")),"straight_hits":nb.get("straight_hits"),"box_hits":nb.get("box_hits"),"straight_rate":_round(nb.get("straight_rate")),"box_rate":_round(nb.get("box_rate"))}
    warning=f"事後評価が{count}回のため、"+("長期成績は判断できません。" if count<5 else "成績は参考値です。") if count<20 else None
    return {"display_name":GAME_NAMES[game],"current":{"latest_draw_no":rg.get("latest_draw_no"),"next_draw_no":rg.get("next_draw_no"),"selected_config":selected,"selected_search_source":_source(selected,ranked),"prediction":[dict(_map(x)) for x in _list(rg.get("prediction")) if _map(x)],"previous_evaluation":dict(_map(rg.get("previous_evaluation")))},"observed_evaluation":observed,"optimizer_backtest":back,"experience":_experience(eg),"warnings":[warning] if warning else []}

def build_evaluation_dashboard(output_dir:Path)->dict[str,object]:
    run=_map(load_json(output_dir/"run_summary.json",default={})); hist=_list(load_json(output_dir/"evaluation_history.json",default=[])); summary=_map(load_json(output_dir/"evaluation_summary.json",default={})); opt=_map(load_json(output_dir/"optimizer_result.json",default={})); exp=_map(load_json(output_dir/"optimizer_experience.json",default={}))
    games={g:_game(g,run,hist,summary,opt,exp) for g in GAME_KEYS}; counts={g:int(_map(v.get("observed_evaluation")).get("evaluated_draws",0) or 0) for g,v in games.items()}; candidates=[]
    for g,v in games.items():
        metric=_num(_map(_map(v.get("observed_evaluation")).get("all_time")).get("avg_best_match_count"))
        if metric is not None:candidates.append((g,metric))
    minimum=min(counts.values(),default=0)
    return {"schema_version":SCHEMA_VERSION,"generated_at":run.get("generated_at") or opt.get("generated_at") or exp.get("updated_at"),"status":run.get("status","unknown"),"overall":{"full_run_status":run.get("status","unknown"),"best_observed_game":max(candidates,key=lambda x:x[1])[0] if candidates else None,"least_evaluated_games":[g for g,n in counts.items() if n==minimum],"experience_schema_version":exp.get("schema_version"),"warnings":[f"{GAME_NAMES[g]}の事後評価は{n}回です。" for g,n in counts.items() if n<5],"previous_dashboard_comparison":{"status":"unavailable","reason":"初回実装では過去Dashboardとの差分を保存していません。"}},"games":games}

def _prediction(game:str, rows:Sequence[Any])->str:
    out=[]
    for i,item in enumerate(rows,1):
        row=_map(item)
        if game in COMBINATION: label="・".join(str(v) for v in _list(row.get("numbers")))
        else:
            label=str(row.get("number") or row.get("number_text") or row.get("digits_text") or "")
            if not label:label="".join(str(v) for v in _list(row.get("digits") or row.get("numbers")))
        out.append(f"{i}. {label}")
    return "<br>".join(out) if out else "予想なし"

def render_evaluation_dashboard_markdown(dashboard:Mapping[str,object])->str:
    overall=_map(dashboard.get("overall")); games=_map(dashboard.get("games")); best=overall.get("best_observed_game"); least=[GAME_NAMES.get(str(x),str(x)) for x in _list(overall.get("least_evaluated_games"))]
    lines=["# LotteryAI Evaluation Dashboard","",f"- 生成日時: `{dashboard.get('generated_at')}`",f"- Full Run: **{overall.get('full_run_status')}**",f"- Dashboard schema: `{dashboard.get('schema_version')}`","","## 全体","",f"- 事後評価の平均最高一致が最も高いゲーム: **{GAME_NAMES.get(str(best),str(best)) if best else '判定不能'}**",f"- 事後評価が最も少ないゲーム: {', '.join(least) if least else '不明'}",""]
    for g in GAME_KEYS:
        game=_map(games.get(g)); cur=_map(game.get("current")); obs=_map(game.get("observed_evaluation")); alltime=_map(obs.get("all_time")); bt=_map(game.get("optimizer_backtest")); ex=_map(game.get("experience"))
        lines += [f"## {GAME_NAMES[g]}","",f"- 次回抽せん回: **{cur.get('next_draw_no')}**",f"- 採用Config: `{cur.get('selected_config')}`",f"- 採用元: `{cur.get('selected_search_source')}`",f"- 事後評価: **{obs.get('status')}** ({obs.get('evaluated_draws')}回)",f"- 平均最高一致: {alltime.get('avg_best_match_count')}",f"- 平均1口一致: {alltime.get('avg_all_pattern_matches')}",f"- 最大一致: {alltime.get('max_best_match_count')}",f"- Optimizer selection_score: {bt.get('selection_score')}",f"- Optimizer Random uplift: {bt.get('random_uplift')}",f"- Experience履歴: {ex.get('history_count')}件","","### 次回予想","",_prediction(g,_list(cur.get("prediction"))),""]
        nums=_map(bt.get("numbers"))
        if nums:lines += ["### Numbersバックテスト","",f"- 平均最高位置一致: {nums.get('average_best_position_matches')}",f"- 1口平均位置一致: {nums.get('average_position_matches_per_ticket')}",f"- 平均最高順不同一致: {nums.get('average_best_unordered_matches')}",f"- Straight: {nums.get('straight_hits')} ({nums.get('straight_rate')})",f"- Box: {nums.get('box_hits')} ({nums.get('box_rate')})",""]
        warnings=_list(game.get("warnings"))
        if warnings:lines += ["### 注意","",*[f"- {w}" for w in warnings],""]
    return "\n".join(lines).rstrip()+"\n"

def write_evaluation_dashboard(output_dir:Path)->dict[str,object]:
    dashboard=build_evaluation_dashboard(output_dir); save_json(output_dir/"evaluation_dashboard.json",dashboard); (output_dir/"evaluation_dashboard.md").write_text(render_evaluation_dashboard_markdown(dashboard),encoding="utf-8"); return dashboard

__all__=["SCHEMA_VERSION","build_evaluation_dashboard","render_evaluation_dashboard_markdown","write_evaluation_dashboard"]
