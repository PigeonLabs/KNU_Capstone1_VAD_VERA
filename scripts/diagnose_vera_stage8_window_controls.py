"""Numerical all-one controls matched to new window features; no extra model calls."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from dataclasses import replace
from scripts.vera_stage8_common import *
from scripts.vera_stage8_eval import refine,summarize
from scripts.analyze_vera_stage8 import load

def main():
 assert json.loads((OUT/'step4/status.json').read_text())['status']=='complete';rr=load(OUT/'step4/frame_scores.csv');lookup={(r['scene'],r['original_split'],r['video'],r['frame']):r['label'] for r in rr if r['condition']=='PROB_10s'};output=[]
 for sec in [2,4]:
  config=replace(CONFIG,window_seconds=sec)
  for r in [r for r in rows() if r['split']=='evaluation']:
   fs=readlines(ROOT/'cache/stage8/window_features'/f'{sec}s'/r['scene']/r['original_split']/(r['video']+'.jsonl'));score=refine(np.ones(len(fs)),[f['feature'] for f in fs],r['length'],config)
   output.extend({'condition':f'ONE_{sec}s','scene':r['scene'],'original_split':r['original_split'],'video':r['video'],'frame':i,'label':lookup[(r['scene'],r['original_split'],r['video'],i)],**{k:float(v[i]) for k,v in score.items()}} for i in range(r['length']))
 write(OUT/'step4/window_matched_ONE_metrics.json',{'purpose':'numeric diagnostic; primary shared ONE comparator remains preregistered Step0 10s control','metrics':summarize(output),'extra_rounding':False});print(json.dumps({c:m['macro'] for c,m in summarize(output).items()},indent=2))
if __name__=='__main__':main()
