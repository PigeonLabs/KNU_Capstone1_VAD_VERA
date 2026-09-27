"""Post-evaluation examples only; never used to construct or select prompts."""
import csv,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import SCENES,segments,write,sha,base

def main():
 out=ROOT/'experiments/stage4';result={'purpose':'Post-hoc interpretation after all four completed evaluations; labels choose illustrative examples only, never prompts, settings, or a best condition. No additional model calls.','selection':'First lexicographic A/B/C disagreement; then highest number of anomalous sampled frames among all-three-normal windows (lexicographic tie break), excluding duplicate example.','scenes':{},'source_sha256':{}}
 assert json.loads((out/'R04/normal/independent_verification.json').read_text())['status']=='passed'
 for scene in ['R01','R02','R03','R04']:
  folder=out/scene/'evaluation';assert json.loads((folder/'status.json').read_text())['status']=='complete'
  manifest=json.loads((folder/'inference_manifest.json').read_text());rows=list(csv.DictReader((folder/'frame_scores.csv').open()));labels={(r['original_split'],r['video'],int(r['frame'])):int(r['label']) for r in rows if r['condition']=='A'}
  result['source_sha256'][str((folder/'frame_scores.csv').relative_to(ROOT))]=sha(folder/'frame_scores.csv')
  candidates=[];literal={c:{'segments':0,'positive_segments':0,'positive_responses_naming_rule_id':0,'responses_with_uncertainty_phrase':0} for c in ['B','C']}
  for video in sorted(manifest,key=lambda r:(r['original_split'],r['video'])):
   prior_path=ROOT/'experiments/stage3/inference'/scene/video['original_split']/(video['video']+'.jsonl');prior={r['center']:r for r in base.readlines(prior_path)}
   for seg in segments(video['length']):
    answers={'A':{'prediction':prior[seg['center']]['prediction'],'response':prior[seg['center']]['response'],'source_path':str(prior_path.relative_to(ROOT)),'source_sha256':sha(prior_path)}}
    for c in ['B','C']:
     p=folder/'inference'/c/video['original_split']/video['video']/f"{seg['center']:06d}.json";record=json.loads(p.read_text());prediction=base.parse_response(record['response'])
     answers[c]={'prediction':prediction,'response':record['response'],'source_path':str(p.relative_to(ROOT)),'source_sha256':sha(p)}
     stat=literal[c];stat['segments']+=1;stat['positive_segments']+=prediction
     stat['positive_responses_naming_rule_id']+=int(prediction==1 and bool(re.search(r'\bN[1-9]\d*\b',record['response'])))
     stat['responses_with_uncertainty_phrase']+=int(bool(re.search(r'\b(?:unobservable|unclear|cannot determine|not visible|cannot see)\b',record['response'],re.I)))
    sampled=[labels[(video['original_split'],video['video'],i)] for i in seg['frame_ids']]
    scored=[labels[(video['original_split'],video['video'],i)] for i in range(seg['center'],seg['score_end'])]
    candidates.append({'video_id':video['id'],'original_split':video['original_split'],'video':video['video'],**seg,'sampled_frame_labels':sampled,'scored_frames':len(scored),'scored_anomalous_frames':sum(scored),'answers':answers})
  examples=[]
  disagreements=[r for r in candidates if len({a['prediction'] for a in r['answers'].values()})>1]
  if disagreements:examples.append({'kind':'first_prediction_disagreement',**disagreements[0]})
  missed=[r for r in candidates if all(a['prediction']==0 for a in r['answers'].values()) and r['scored_anomalous_frames']>0]
  missed.sort(key=lambda r:(-sum(r['sampled_frame_labels']),r['original_split'],r['video'],r['center']))
  if missed:examples.append({'kind':'all_conditions_missed_anomaly_with_most_sampled_anomalous_frames',**missed[0]})
  for stat in literal.values():
   stat['uncertainty_phrase_fraction']=stat['responses_with_uncertainty_phrase']/stat['segments']
   stat['rule_id_mention_fraction_among_positive_responses']=stat['positive_responses_naming_rule_id']/stat['positive_segments'] if stat['positive_segments'] else None
  result['scenes'][scene]={'examples':examples,'literal_response_indicators':literal,'indicator_caveat':'Literal mentions only, not validated semantic observability, evidence fidelity, or explanation correctness. Missing a rule ID alone does not prove the normal description was ignored.'}
 write(out/'explanation_examples.json',result)
 body='# 4단계 실제 판정과 미탐 예시\n\nR01–R04 평가 완료 후 고른 사후 해석 자료다. 정답은 사례 선택에만 사용했고 프롬프트 변경이나 조건 선택에 사용하지 않았다. 원문 응답은 모델의 설명이며, 물체 이름·상태의 사실성에 대한 사람 검증은 아니다. 전체 선택 규칙, 프레임 정답, 원본 응답 경로와 SHA256은 [JSON](explanation_examples.json)에 보존했다.\n'
 for scene,detail in result['scenes'].items():
  body+=f'\n## {scene}\n'
  for example in detail['examples']:
   kind='조건별 판정이 다른 첫 구간' if example['kind']=='first_prediction_disagreement' else '세 조건 모두 놓친 이상 구간'
   body+=f"\n### {kind}\n\n영상 `{example['video_id']}`, 중심 프레임 {example['center']}, 점수 부여 구간 [{example['center']}, {example['score_end']}), 이상 프레임 {example['scored_anomalous_frames']}/{example['scored_frames']}. 입력 프레임: {example['frame_ids']}. 입력 프레임 정답: {example['sampled_frame_labels']}. 프레임 ID는 0부터 시작한다.\n"
   for condition,answer in example['answers'].items():
    body+=f"\n**{condition}: {'이상' if answer['prediction'] else '정상'} 판정**\n\n"+'\n'.join('> '+line for line in answer['response'].splitlines())+'\n'
 (out/'explanation_examples.md').write_text(body)

 print(json.dumps({s:{'examples':len(r['examples']),'literal_response_indicators':r['literal_response_indicators']} for s,r in result['scenes'].items()},indent=2))
if __name__=='__main__':main()
