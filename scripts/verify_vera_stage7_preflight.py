"""Independent preflight identity and decoder telemetry check, without labels."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage7_common import OUT,sha,digest,protected,segments,input_identity
from scripts.vera_stage7_engine import parse_response,quality
from scripts.vera_stage4_common import write
import numpy as np

def main():
 frozen=json.loads((OUT/'frozen.json').read_text());assert digest({k:v for k,v in frozen.items() if k!='fingerprint'})==frozen['fingerprint']
 for p,h in frozen['source_sha256'].items():assert sha(ROOT/p)==h,p
 assert frozen['generation']=={'do_sample':False,'num_beams':1,'max_new_tokens':128} and frozen['model_config']['max_new_tokens']==128
 rows=json.loads((OUT/'inference_manifest.json').read_text())['preflight_normal_train'];refs=json.loads((OUT/'references.json').read_text());prompts=json.loads((OUT/'prompts.json').read_text());seen=[]
 for row in rows:
  available=segments(row['length'])
  for c in ['C0','N','X']:
   for i in np.linspace(0,len(available)-1,5,dtype=int):
    seg=available[i];p=OUT/'preflight'/c/row['scene']/f"{seg['center']:06d}.json";r=json.loads(p.read_text());q=r['request'];seen.append(r)
    assert q['images']==input_identity(c,row,seg,refs) and q['prompt']==prompts[row['scene']][c]
    assert q['fingerprint']==frozen['fingerprint'] and r['cache_key']==digest(q)
    assert r['prediction']==parse_response(r['response']) and r['parse_status']=='valid'
    assert r['actual_generation_config']==frozen['generation']
    assert len(r['generated_token_ids'])==r['generated_token_count']<=128
    assert r['ended_with_eos']==(r['generated_token_ids'][-1] in r['eos_token_ids'])
    assert r['explanation_quality']==quality(r['response'],r)
 assert len(seen)==60==len(list((OUT/'preflight').rglob('*.json')))
 report={'status':'passed','calls':60,'all_binary_valid':True,'generation_config_verified':True,'generated_token_mean':float(np.mean([r['generated_token_count'] for r in seen])),'max_generated_tokens':max(r['generated_token_count'] for r in seen),'EOS_ended':sum(r['ended_with_eos'] for r in seen),'same_value_duplicates':sum(r['explanation_quality']['duplicate_output'] for r in seen),'protected_previous_files':protected(),'fingerprint':frozen['fingerprint']}
 write(OUT/'preflight_verification.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
