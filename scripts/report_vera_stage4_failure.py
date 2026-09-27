"""Audit and publish the actual R04 extraction failure without fabricating a profile."""
import json,re,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import *
from scripts.build_vera_normal_context_v2 import evidence_errors

def main():
 out=ROOT/'experiments/stage4/R04/normal';status=json.loads((out/'status.json').read_text());assert status['status']=='failed'
 frozen=json.loads((out/'frozen.json').read_text())
 for p,h in frozen['source_sha256'].items():assert sha(ROOT/p)==h
 assert frozen['split_sha256']==sha(ROOT/'experiments/stage2_1/split.json')
 _,gen,audit=split_scene('R04');summaries=[];observed=0;fallback=[]
 for row in gen:
  for seg in segments(row['length']):
   r=json.loads((out/'observations'/row['original_split']/row['video']/f"{seg['center']:06d}.json").read_text())
   assert r['segment']==seg and r['video_id']==row['id']
   assert r['request_hash']==digest({k:r[k] for k in ['prompt','video_id','segment']});observed+=1
  p=out/'video_summaries'/row['original_split']/(row['video']+'.json');r=json.loads(p.read_text());summaries.append({'video_id':row['id'],**r['parsed']})
  if r.get('derived_not_model_json'):
   observations=[{'center':seg['center'],'description':json.loads((out/'observations'/row['original_split']/row['video']/f"{seg['center']:06d}.json").read_text())['response']} for seg in segments(row['length'])]
   idx=np.linspace(0,len(observations)-1,min(6,len(observations)),dtype=int).tolist()
   assert r['summary_fallback']['observations_sha256']==digest(observations)
   assert r['parsed']['facts']==[{'claim':observations[i]['description'],'center':observations[i]['center']} for i in idx];fallback.append(row['id'])
 rules=json.loads((out/'rules.json').read_text());proposed=json.loads((out/'candidates.json').read_text())['parsed']['rules'];assert len(rules)==len(proposed)==5
 for r,p in zip(rules,proposed):
  assert all(r[k]==v for k,v in p.items());assert evidence_errors(r,summaries)==r['rejection_reason'];assert not r['accepted'] and not r['support_eligible']
 assert not list((out/'support_checks').rglob('*.json')) and not list((out/'audit').rglob('*.json'))
 assert not (out/'normal_profile.json').exists() and not (out.parent/'evaluation').exists()
 allowed={r['id'] for r in gen};calls=readlines(out/'calls.jsonl');assert all(r.get('video_id') is None or r['video_id'] in allowed for r in calls)
 result={'status':'failure_verified','scene':'R04','experiment_status':'failed','observed_segments':observed,'generation_videos':len(gen),'reserved_audit_videos':len(audit),'candidate_rules':len(rules),'accepted_rules':0,'visual_support_calls':0,'normal_audit_windows_executed':0,'B_C_evaluation_executed':False,'source_hashes_verified':True,'normal_train_only':True,'all_candidate_rejections_recomputed':True,'summary_fallback_videos':fallback,'reason':'All five candidates cite R04/training/07 center 0, absent from that video summary. Available centers are 32,64,128,96,160. No automatic evidence reanchoring; no valid normal profile.'}
 write(out/'failure_verification.json',result)
 write(out/'runtime_summary.json',{'calls':len(calls),'sum_model_seconds':sum(r['seconds'] for r in calls),'peak_allocated_gib':max(r['peak_allocated_gib'] for r in calls),'peak_reserved_gib':max(r['peak_reserved_gib'] for r in calls),'timing_scope':'model.chat only; excludes preprocessing/loading/hashing/logging','experiment_status':'failed'})
 ledger=ROOT/'experiments/stage4/execution/controller_failure';ledger.mkdir(parents=True,exist_ok=True)
 for name in ['vera_stage4_continuation.json','vera_stage4_continuation.log']:shutil.copyfile(ROOT/'runs'/name,ledger/name)
 registry_path=ROOT/'experiments/stages.json';registry=json.loads(registry_path.read_text());key='stage4_R04_normal';assert key not in registry
 registry[key]={'title':'4단계 R04 정상 기준 생성 실패','status':'failed','execution_approved':True,'status_file':'experiments/stage4/R04/normal/status.json','artifact_directory':'experiments/stage4/R04/normal'};write(registry_path,registry)
 text=(ROOT/'README.md').read_text();text=text.replace('\n진행 상태는','\n| **4-R04-N** | **R04 정상 기준·질문 생성** | **실패** | 338구간 관찰 완료; 후보 5개 모두 잘못된 근거 구간 인용으로 탈락 |\n\n진행 상태는',1)
 text=re.sub(r'\*\*최신 완료:.*?\n','**4단계 현재 결과: R01–R03 A/B/C 평가 완료, R04 정상 기준 생성 실패.** R04 B/C 평가는 수행되지 않았다.\n',text,count=1)
 text+='\n## 4단계 R04 — 정상 기준 생성 실패\n\n정상 학습 영상 14개에서 338구간을 관찰하고 영상별 요약 14개를 만들었다. 후보 규칙 5개 모두 `R04/training/07`의 중심 프레임 `0`을 인용했지만, 입력 요약에 존재하는 중심은 `32, 64, 128, 96, 160`이었다. 기존 인용 검증 기준에 따라 모두 탈락했다. 이는 JSON 구문 오류가 아니라 근거 구간 선택 실패다. 근거 구간을 임의 교체하거나 정상 기준을 만들어 넣지 않았다.\n\n별도 점검용 정상 영상 4개는 예약되어 있었지만, 유효 후보가 없어 시각 근거 확인·정상 점검·B/C 평가는 실행되지 않았다. 이 실패를 정상 판정이나 AUROC 50으로 대체하지 않으며 R01–R04 전체 비교 완료로 표시하지 않는다. 모델 학습과 질문 optimizer 반복은 수행하지 않았다.\n\n'
 text+=f'요약 형식 실패 후 관찰 원문을 인용한 영상은 {len(fallback)}개다. 원래 응답·형식 복구·원문 인용 내역과 사용 구간을 보존했다.\n\n'
 text+='[실패 상태](experiments/stage4/R04/normal/status.json), [원래 후보](experiments/stage4/R04/normal/candidates.json), [모든 탈락 사유](experiments/stage4/R04/normal/invalid_candidates.json), [실패 검산](experiments/stage4/R04/normal/failure_verification.json), [실행 명령과 로그](experiments/stage4/execution/R04_normal), [모델 시간·VRAM](experiments/stage4/R04/normal/runtime_summary.json)에 실제 수행 내용을 기록했다.\n'
 (ROOT/'README.md').write_text(text);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
