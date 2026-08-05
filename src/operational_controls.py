from __future__ import annotations

from collections import Counter
from hashlib import sha256
from itertools import combinations_with_replacement
from random import Random
from typing import Mapping, Sequence

from data_loader import dataframe_to_history
from features import build_model_context
from numbers_backtester import box_composition_signature
from predictor import _passes_repeat_filters, _passes_shape_filters


def _seed(game_key: str, draw_no: int) -> int:
    return int.from_bytes(sha256(f"{game_key}:{draw_no}".encode()).digest()[:8], "big")

def _lotto_uniform(config: Mapping[str, object], rng: Random, count: int=5):
    pool=list(range(int(config["min_num"]), int(config["max_num"])+1)); pick=int(config["pick_count"]); out=[]; seen=set()
    while len(out)<count:
        row=tuple(sorted(rng.sample(pool,pick)))
        if row not in seen: seen.add(row); out.append(list(row))
    return out

def _lotto_filtered(df, config: Mapping[str, object], filters: Mapping[str, object], rng: Random, count:int=5):
    try:
        history=dataframe_to_history(df, config)
        context=build_model_context(history, {**dict(config), **dict(filters)})
    except (TypeError, ValueError, KeyError):
        return _lotto_uniform(config, rng, count)
    pool=list(context.number_pool); out=[]; seen=set(); attempts=0; merged={**dict(config), **dict(filters)}
    while len(out)<count and attempts<10000:
        attempts+=1; row=tuple(sorted(rng.sample(pool, context.pick_count)))
        if row in seen: continue
        if not _passes_shape_filters(row, context, merged): continue
        if not _passes_repeat_filters(row, context): continue
        seen.add(row); out.append(list(row))
    for row in _lotto_uniform(config,rng,count*2):
        t=tuple(row)
        if len(out)>=count: break
        if t not in seen: seen.add(t); out.append(row)
    return out

def _numbers_uniform(digit_count:int, rng:Random, count:int=10):
    return [[rng.randrange(10) for _ in range(digit_count)] for _ in range(count)]

def _composition_matched_boxes(box_prediction: Sequence[Mapping[str, object]], digit_count:int, rng:Random):
    requested=Counter(box_composition_signature(row.get("numbers", row.get("digits", []))) for row in box_prediction)
    groups={}
    for candidate in combinations_with_replacement(range(10),digit_count): groups.setdefault(box_composition_signature(candidate),[]).append(candidate)
    result=[]
    for sig,count in sorted(requested.items()):
        for candidate in rng.sample(groups[sig],count): result.append(list(candidate))
    return result

def build_operational_controls(game_key:str, config:Mapping[str,object], dataframe, result:Mapping[str,object], target_draw_no:int, generated_at:str):
    rng=Random(_seed(game_key,target_draw_no)); family=str(config.get("family","lotto")).lower()
    base={"target_draw_no":target_draw_no,"generated_at":generated_at,"generated_before_draw":True,"control_seed":_seed(game_key,target_draw_no)}
    if family=="numbers":
        digit_count=int(config.get("digit_count", config.get("pick_count",0))); box=result.get("box_prediction",[])
        return {**base,"uniform_random_control":_numbers_uniform(digit_count,rng,10),"composition_matched_random_box_control":_composition_matched_boxes(box,digit_count,rng),"model_prediction":result.get("prediction",[]),"model_box_prediction":box}
    return {**base,"uniform_random_control":_lotto_uniform(config,rng,5),"filtered_random_control":_lotto_filtered(dataframe,config,result.get("selected_filters",{}),rng,5),"model_prediction":result.get("prediction",[])}
