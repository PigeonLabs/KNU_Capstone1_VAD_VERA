"""Parallel independent comparisons; frozen sklearn evaluator remains unchanged."""
import argparse,json,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage8_common import OUT,write,evaluator_guard
from scripts.vera_stage8_eval import paired_bootstrap
from scripts.analyze_vera_stage8 import load

def worker(task):
 step,a,b=task;evaluator_guard();path=OUT/step/'bootstrap'/f'{a}_vs_{b}.json'
 if path.exists():return str(path)
 rr=load(OUT/step/'frame_scores.csv')+[r for r in load(OUT/'step0/frame_scores.csv') if r['condition']=='ONE']
 value=paired_bootstrap(rr,a,b);write(path,value);return f'{a} - {b}: '+str(value['ci95']['initial'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--step',choices=['step4','step5'],required=True);step=p.parse_args().step
 pairs=[('DINO','ONE'),('HYBRID','ONE'),('HYBRID','DINO'),('HYBRID','PROB')] if step=='step5' else [('PROB_2s','ONE'),('PROB_4s','ONE'),('PROB_2s','PROB_10s'),('PROB_4s','PROB_10s')]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for result in pool.map(worker,[(step,a,b) for a,b in pairs]):print(result,flush=True)
