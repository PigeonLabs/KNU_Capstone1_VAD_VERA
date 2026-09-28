"""Deterministic post-hoc examples; never feeds back into the frozen prompts."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import sha,write,SCENES
OUT=ROOT/'experiments/stage5'

def main():
 assert json.loads((OUT/'independent_verification.json').read_text())['status']=='passed'
 windows=json.loads((OUT/'window_diagnostics.json').read_text());index={(r['condition'],r['video_id'],r['center']):r for r in windows}
 result={};text='# 5단계 실제 판정 사례\n\n모든 추론·지표가 고정된 뒤 결정적으로 선택한 사례다. 프롬프트 수정이나 선택에 사용하지 않았다. 점수 구간 라벨과 실제 입력 프레임 라벨은 다를 수 있다. 출력 설명은 모델의 주장이지 사람이 검증한 사실이 아니다.\n\n'
 for scene in SCENES:
  rows=sorted([r for r in windows if r['condition']=='P3' and r['scene']==scene],key=lambda r:(r['video_id'],r['center']));result[scene]=[];text+='## '+scene+'\n\n'
  selectors=[('기준 A는 음성이고 P3는 양성인 이상 포함 점수 구간',lambda r:r['scored_positive_frames']>0 and r['prediction']==1 and index[('A',r['video_id'],r['center'])]['prediction']==0),('P3가 양성인 정상 점수 구간',lambda r:r['scored_positive_frames']==0 and r['prediction']==1),('새 세 조건이 모두 음성인 이상 포함 점수 구간',lambda r:r['scored_positive_frames']>0 and all(index[(c,r['video_id'],r['center'])]['prediction']==0 for c in ['P1','P2','P3']))]
  for title,select in selectors:
   eligible=[r for r in rows if select(r)]
   if not eligible:text+='- '+title+': 해당 사례 없음.\n\n';continue
   w=eligible[0];_,split,video=w['video_id'].split('/');responses={}
   for c in ['P1','P2','P3']:
    p=OUT/'inference'/c/scene/split/video/f"{w['center']:06d}.json";raw=json.loads(p.read_text());responses[c]={'source':str(p.relative_to(ROOT)),'sha256':sha(p),'prediction':raw['prediction'],'response':raw['response']}
   case={'selection':title,'window':w,'baseline_A_prediction':index[('A',w['video_id'],w['center'])]['prediction'],'responses':responses};result[scene].append(case)
   text+='### '+title+'\n\n'+f"`{w['video_id']}`, 중심 {w['center']}, 점수 [{w['center']},{w['score_end']}), 점수 구간 이상 {w['scored_positive_frames']}/{w['scored_frames']}프레임, 입력 이상 {w['sampled_positive_frames']}/8프레임. 입력 ID `{w['frame_ids']}`.\n\n"
   for c,r in responses.items():text+=f"**{c} / Output {r['prediction']}** ([원문]({str(Path(r['source']).relative_to(OUT.relative_to(ROOT)))}))\n\n> "+r['response'].replace('\n','\n> ')+'\n\n'
 write(OUT/'explanation_examples.json',result);(OUT/'explanation_examples.md').write_text(text)
 print(json.dumps({'examples':{s:len(v) for s,v in result.items()},'posthoc_only':True}))
if __name__=='__main__':main()
