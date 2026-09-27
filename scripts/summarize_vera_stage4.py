"""Report completed canonical Stage4 results; protected scenes are read-only."""
import csv,json,re,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from scripts.vera_stage4_common import write,sha,SCENES

def main():
 out=ROOT/'experiments/stage4';normal=out/'R04/normal';audit=json.loads((normal/'independent_verification.json').read_text());assert audit['status']=='passed' and audit['experiment_status']=='complete'
 scenes={};rows=[];sources={}
 for scene in SCENES:
  p=out/scene/'evaluation';assert json.loads((p/'status.json').read_text())['status']=='complete';assert json.loads((p/'independent_verification.json').read_text())['status']=='passed'
  scenes[scene]=json.loads((p/'metrics.json').read_text());rows.extend(csv.DictReader((p/'frame_scores.csv').open()))
  for file in [p/'frame_scores.csv',p/'metrics.json',p/'frozen.json',out/scene/'normal/normal_profile.json']:sources[str(file.relative_to(ROOT))]=sha(file)
 for file in [normal/'independent_verification.json',normal/'input_cache_manifest.json',normal/'fact_table.json',normal/'candidate_rules_frozen.json',normal/'grounding_summary.json',out/'r04_protected_artifacts.json',out/'explanation_examples.json']:sources[str(file.relative_to(ROOT))]=sha(file)
 summary={'status':'complete','completed_evaluation_scenes':SCENES,'R04':audit,'scenes':scenes,'conditions':{},'sources_sha256':sources,'scope':'All four scenes, identical frame support for A/B/C; R01-R03 frozen scores read only, no rerun; R04 separated rule/grounding pipeline; no test-based selection.'}
 for c in ['A','B','C']:
  part=[r for r in rows if r['condition']==c];y=[int(r['label']) for r in part];pred=[float(r['final']) for r in part];counts={k:sum(scenes[s][c]['initial_binary'][k] for s in SCENES) for k in ['tp','fp','tn','fn']};counts['recall']=counts['tp']/(counts['tp']+counts['fn']);counts['fpr']=counts['fp']/(counts['fp']+counts['tn'])
  summary['conditions'][c]={'frames':len(part),'macro':{k:float(np.mean([scenes[s][c]['final'][k] for s in SCENES])) for k in ['auroc','ap']},'pooled':{'auroc':100*roc_auc_score(y,pred),'ap':100*average_precision_score(y,pred)},'initial_binary_pooled':counts}
 write(out/'summary.json',summary)
 tests=sorted((out/'execution/R04_separated_regression_final').glob('*/command.json'))[-1];assert json.loads(tests.read_text())['status']=='complete';passed=int(re.search(r'(\d+) passed',(tests.parent/'output.log').read_text()).group(1))
 candidates=json.loads((normal/'candidates.json').read_text())['parsed']['rules'];duplicates=json.loads((normal/'duplicate_candidates.json').read_text());grounds=json.loads((normal/'grounding_summary.json').read_text());profile=json.loads((normal/'normal_profile.json').read_text());rules=json.loads((normal/'rules.json').read_text())
 body='## 4단계 R04 — 규칙 생성·영상별 근거 선택 분리 (완료)\n\n**현재 공식 R04 실행은 정상 기준 생성과 A/B/C 평가까지 완료했다.** 정상 영상 14개의 관찰 338개·검증된 요약 14개를 입력 캐시로 사용했다. 모델 파일 22개·정상 원본 프레임 5,300개·관찰 및 요약 SHA256을 확인했고 모델/config·분할·샘플링·프롬프트·관찰 코드 블록도 동일함을 검증했다. 관찰 추론은 재실행하지 않고 규칙과 근거 선택은 새로 생성했다.\n\n'
 body+='실행 순서는 **영상별로 묶은 정상 summary → 규칙 텍스트만 생성 → 문자열 중복 제거 → 규칙 문구/해시 고정 → 규칙별·영상별 독립 grounding → Python의 distinct-video 계산 → 선택된 모든 근거의 시각 검증 → held-out 정상 audit → 정상 설명·질문 고정 → R04 평가**였다. grounding에는 한 영상의 facts만 주고 정확한 ID 하나 또는 NONE만 허용했다. 잘못된 출력은 NONE과 별개의 failed이며 의미 보정이나 재선택을 하지 않았다.\n\n'
 body+='### 실제 후보 규칙\n\n| ID | 조건 / 정상 기대 | 질문 | 처리 |\n|---|---|---|---|\n'
 dupids={r['candidate']['id'] for r in duplicates}
 for r in candidates:body+=f"| {r['id']} | 조건: {r['condition']}<br>정상: {r['normal_expectation']} | {r['question']} | "+('동일 질문 중복 제거; grounding 미실행' if r['id'] in dupids else 'grounding 및 시각 검증 수행')+' |\n'
 body+=f"\n후보 {len(candidates)}개, schema invalid {audit['schema_invalid']}개, 중복 제거 {len(duplicates)}개, 최종 채택 {audit['accepted_rules']}개다. normalization은 lowercase·strip·연속 공백 단일화이며 expectation 또는 question이 같으면 처음 것을 유지했다. N2–N5는 질문이 N1과 같아 grounding 전에 제외했다. 해당 규칙의 영상별 결과를 NONE으로 만들어 넣지 않았다.\n\n"
 body+='### 영상별 grounding과 시각 검증\n\n| 규칙 | 정상 영상 | 선택 fact | 중심 프레임 | grounding | visual support |\n|---|---|---|---:|---|---|\n'
 for g in grounds:
  rule=next(r for r in rules if r['id']==g['rule_id']);checks={(c['video_id'],c['center']):c['state'] for c in rule['support_checks']}
  for v in g['videos']:
   e=v['evidence'];body+=f"| {g['rule_id']} | {v['video_id']} | {v['fact_id'] or 'NONE' if v['state']!='failed' else 'FAILED'} | {e['center'] if e else '—'} | {v['state']} | {checks.get((v['video_id'],e['center']),'미실행') if e else '미실행'} |\n"
 body+='\n| 규칙 | grounded distinct videos | NONE | grounding failed | visual supported / contradicted / unobservable / failed | audit supported / contradicted / unobservable / failed | 채택 |\n|---|---:|---:|---:|---|---|---|\n'
 for g in grounds:
  c=audit['per_rule_checks'][g['rule_id']];states=['supported','contradicted','unobservable','failed'];body+=f"| {g['rule_id']} | {g['grounded_generation_videos']} | {g['none_videos']} | {g['grounding_failures']} | "+' / '.join(str(c['visual'][s]) for s in states)+' | '+' / '.join(str(c['audit'][s]) for s in states)+' | '+('채택' if g['rule_id'] in audit['accepted_rule_ids'] else '제외')+' |\n'
 body+='\n선택된 모든 generation-video 근거를 실제 8프레임 창에서 검사했다. distinct supported 영상 ≥3, 명확한 contradiction/failed 없음 조건을 유지했다. 별도 정상 영상 4개의 100개 창을 모두 audit했고, unobservable을 contradiction으로 바꾸지 않았다. 같은 VLM의 판단이므로 독립 사람 검증은 아니다.\n\n### 고정 정상 설명과 질문\n\n```text\n'+profile['normal_description']+'\n'+profile['questions']+'```\n\n'
 body+='정상 설명은 정지한 칼날의 배치에 집중하고 조건과 기대 상태가 유사하며 질문도 포괄적이다. 상세한 공정 순서·필수 동작이 검증됐다고 해석하지 않는다. 이런 한계에도 평가 결과를 보고 문구를 수정하지 않았다.\n\n'
 body+='### R04 실제 평가\n\nA=기존 Q0와 기존 ImageBind 특징을 검증 후 재사용, B=고정 정상 설명+Q0, C=동일 정상 설명+채택 질문. InternVL2-8B BF16 동결, 30 FPS 가정, stride 16·10초 창·8프레임, 기존 retrieval/smoothing/위치 가중치를 유지했다. 평가 프레임 정답은 B/C 추론이 모두 끝난 뒤 점수 계산에 사용했다.\n\n| 조건 | AUROC (%) | AP (%) | 초기 TP | FP | FN | 초기 recall (%) |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for c in ['A','B','C']:
  m=scenes['R04'][c];b=m['initial_binary'];body+=f"| {c} | {m['final']['auroc']:.2f} | {m['final']['ap']:.2f} | {b['tp']} | {b['fp']} | {b['fn']} | {100*b['recall']:.2f} |\n"
 body+='\nB의 단계별 점수는 다음과 같다.\n\n| 점수 단계 | AUROC (%) | AP (%) |\n|---|---:|---:|\n'
 for stage in ['initial','retrieved','smoothed','final']:
  m=scenes['R04']['B'][stage];body+=f"| {stage} | {m['auroc']:.2f} | {m['ap']:.2f} |\n"
 body+='\nA/C는 모든 구간을 정상으로 판정했다. B의 유일한 양성 구간은 R04/testing/15의 중심 32, 점수 구간 [32,48)로 프레임 정답 기준 FP 16개·TP 0개다. 다만 8개 입력 프레임 중 5개에는 이상 라벨이 있어 이를 정상 영상만 보고 낸 오탐으로 단정하지 않는다. B의 최종 순위 지표 상승은 smoothing/위치 가중치 이후 나타났고, 초기 프레임 단위 이상 recall은 0%였다. 이 결과만으로 직접적인 이상 검출 능력이 개선되었다고 결론 내리지 않는다. 상세 입력·점수 정렬과 원문은 [실제 판정 사례](explanation_examples.md)에 보존했다.\n'
 body+=f"\n전체 테스트 **{passed}개 통과**. R01–R03 산출물 **{audit['protected_R01_R03_files']:,}개 SHA256 불변**, 추론·평가 재실행 없음. [테스트 로그]({str((tests.parent/'output.log').relative_to(out))}), [보호 목록](r04_protected_artifacts.json), [normal 독립 검산](R04/normal/independent_verification.json), [평가 독립 검산](R04/evaluation/independent_verification.json).\n\n"
 body+='[후보 원문](R04/normal/candidates.json), [고정 규칙과 해시](R04/normal/candidate_rules_frozen.json), [14개 grounding 원문/결과](R04/normal/grounding_summary.json), [fact table](R04/normal/fact_table.json), [정상 설명](R04/normal/normal_description.txt), [질문](R04/normal/questions.txt), [평가 프롬프트](R04/evaluation/prompts.json), [정렬된 단계별 프레임 점수](R04/evaluation/frame_scores.csv), [실행 명령](execution/R04_separated_normal).\n\n'
 body+='## 4단계 — 현재 결과 종합\n\nR01–R03는 기존 고정 결과를 그대로 읽었으며 R04만 이번에 평가했다. 네 장면의 동일 16,862프레임에서 A/B/C를 비교한다.\n\n| 장면 | A AUROC / AP (%) | B AUROC / AP (%) | C AUROC / AP (%) |\n|---|---:|---:|---:|\n'
 for s in SCENES:body+='| '+s+' | '+' | '.join(f"{scenes[s][c]['final']['auroc']:.2f} / {scenes[s][c]['final']['ap']:.2f}" for c in ['A','B','C'])+' |\n'
 for unit in ['macro','pooled']:body+='| '+unit+' | '+' | '.join(f"{summary['conditions'][c][unit]['auroc']:.2f} / {summary['conditions'][c][unit]['ap']:.2f}" for c in ['A','B','C'])+' |\n'
 body+='\n이미 관찰한 IPAD 재분할의 탐색적 오프라인 평가다. 장면별 실행 프로토콜에는 차이가 있어 동일한 정상 기준 생성 방식의 네 장면 반복으로 해석하지 않는다. [종합 수치·해시](summary.json), [실제 판정 사례](explanation_examples.md). 공개 점수 재검산: `python scripts/verify_vera_stage4_summary.py`.\n'
 (out/'results.md').write_text(body)
 path=ROOT/'README.md';text=path.read_text();backup=ROOT/'runs/README_before_R04_separated.md';backup.parent.mkdir(exist_ok=True);shutil.copyfile(path,backup)
 heads=list(re.finditer(r'^## .*$',text,re.M));ranges=[]
 allowed={'## 4단계 R04 — fact-ID 근거 기반 정상 기준·질문 생성','## 4단계 R04 — 규칙 생성·영상별 근거 선택 분리 (완료)','## 4단계 — 현재 결과 종합'}
 for i,h in enumerate(heads):
  if h.group() in allowed:ranges.append((h.start(),heads[i+1].start() if i+1<len(heads) else len(text)))
 original=text
 for a,b in reversed(ranges):text=text[:a]+text[b:]
 for s in ['R01','R02','R03']:
  for section in re.findall(r'(?ms)^## 4단계 '+s+r' .*?(?=^## |\Z)',original):assert section.rstrip() in text
 text='\n'.join(line for line in text.splitlines() if not line.startswith('| **4-R04-'))+'\n'
 text=re.sub(r'^\*\*4단계 현재 상태:.*$', '**4단계 완료: R01–R03 기존 결과 보존, R04 규칙 생성·영상별 grounding과 A/B/C 평가 완료.** [현재 공식 결과](experiments/stage4/results.md).',text,count=1,flags=re.M)
 m=scenes['R04'];table=f"| **4-R04-N** | **R04 규칙 생성·영상별 근거 검증** | **완료** | 후보 {len(candidates)}개 / 중복 {len(duplicates)}개 / 채택 {audit['accepted_rules']}개, grounding 12/14영상 |\n| **4-R04-E** | **R04 정상 설명·질문 A/B/C 평가** | **완료** | AUROC A {m['A']['final']['auroc']:.2f} / B {m['B']['final']['auroc']:.2f} / C {m['C']['final']['auroc']:.2f}% |\n"
 text=text.replace('\n진행 상태는','\n'+table+'\n진행 상태는',1)
 linked=re.sub(r'\]\(([^)]+)\)',lambda m:']('+('experiments/stage4/'+m.group(1) if '://' not in m.group(1) else m.group(1))+')',body);path.write_text(text.rstrip()+'\n\n'+linked)
 write(out/'status.json',{'status':'complete','completed_evaluation_scenes':SCENES,'R04_protocol':'separated text rules and independent per-video grounding','all_four_scenes_complete':True,'frames_per_condition':16862,'finished_at':time.time()})
 p=ROOT/'experiments/stages.json';registry=json.loads(p.read_text())
 for kind in ['normal','evaluation']:registry['stage4_R04_'+kind]={'title':'R04 '+('규칙 생성·영상별 근거 검증' if kind=='normal' else 'A/B/C 평가'),'status':'complete','execution_approved':True,'status_file':f'experiments/stage4/R04/{kind}/status.json','artifact_directory':f'experiments/stage4/R04/{kind}'}
 registry['stage4']={'title':'R01–R04 정상 기준·질문 비교 종합','status':'complete','execution_approved':True,'status_file':'experiments/stage4/status.json','artifact_directory':'experiments/stage4'};write(p,registry)
 print(json.dumps(summary['conditions'],indent=2))
if __name__=='__main__':main()
