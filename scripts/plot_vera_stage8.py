import sys,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.vera_stage8_common import OUT,REPORT

def main():
 rows=list(csv.DictReader((OUT/'step1/window_scores.csv').open()));fig,ax=plt.subplots(1,2,figsize=(10,3.5))
 ax[0].hist([float(r['dec']) for r in rows],bins=[-.25,.25,.75,1.25],color='#446b92');ax[0].set_title('DEC: 2 distinct scores');ax[0].set_xlabel('Decoded anomaly score');ax[0].set_ylabel('Segment count')
 ax[1].hist([float(r['prob']) for r in rows],bins=20,color='#397762');ax[1].set_title('PROB: 28 distinct scores');ax[1].set_xlabel('P(Yes | Yes, No)');ax[1].set_ylabel('Segment count');fig.tight_layout();fig.savefig(REPORT/'step1_score_histogram.svg');preview=ROOT/'runs/stage8';preview.mkdir(parents=True,exist_ok=True);fig.savefig(preview/'score_histogram.png',dpi=120);plt.close(fig)
 # Actual historical binary histograms, frame-weighted, with complete counts in JSON.
 old=json.loads((OUT/'step0/historical_score_distributions.json').read_text());fig,axes=plt.subplots(1,1,figsize=(11,4))
 zero=[h['histogram'].get('0.0',0) for h in old];one=[h['histogram'].get('1.0',0) for h in old];names=[h['source'].replace('experiments/','').replace('/frame_scores.csv','')+'/'+h['condition'] for h in old]
 x=np.arange(len(old));axes.bar(x,zero,label='score 0',color='#446b92');axes.bar(x,one,bottom=zero,label='score 1',color='#b66448');axes.set_xticks(x,names,rotation=90,fontsize=5);axes.set_ylabel('Frame count');axes.legend();fig.tight_layout();fig.savefig(REPORT/'historical_binary_histogram.svg');plt.close(fig)
 for filename,image in [('step0_sanity.md','historical_binary_histogram.svg'),('step1_prob_vs_dec.md','step1_score_histogram.svg')]:
  p=REPORT/filename;s=p.read_text();tag=f'![실제 점수 히스토그램]({image})'
  if tag not in s:p.write_text(s+'\n'+tag+'\n')
if __name__=='__main__':main()
