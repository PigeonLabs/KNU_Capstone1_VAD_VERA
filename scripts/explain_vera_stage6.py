"""Fixed, post-hoc response examples; no feedback to inference or reference selection."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import sha,write,SCENES
OUT=ROOT/'experiments/stage6'

def main():
 assert json.loads((OUT/'independent_verification.json').read_text())['status']=='passed'
 windows=json.loads((OUT/'window_diagnostics.json').read_text());index={(r['condition'],r['video_id'],r['center']):r for r in windows};result={}
 text='# 6단계 실제 응답 비교\n\n전체 추론과 지표가 고정된 뒤 각 조건에 해당하는 첫 (video ID, center) 사례를 골랐다. 원문 설명은 모델의 주장이며 사람이 검증한 시각적 사실이 아니다. 사례는 프롬프트·참조 선택에 피드백하지 않았다.\n\n'
 for scene in SCENES:
  rows=sorted([r for r in windows if r['condition']=='N' and r['scene']==scene],key=lambda r:(r['video_id'],r['center']));result[scene]=[];text+='## '+scene+'\n\n'
  selectors=[('C0는 음성, N은 양성인 이상 포함 점수 구간',lambda r:r['scored_positive_frames']>0 and r['prediction']==1 and index[('C0',r['video_id'],r['center'])]['prediction']==0),('입력 8장과 점수 구간 모두 정상인데 N이 양성인 구간',lambda r:r['scored_positive_frames']==0 and r['sampled_positive_frames']==0 and r['prediction']==1),('N과 X의 판정이 다른 구간',lambda r:r['prediction']!=index[('X',r['video_id'],r['center'])]['prediction']),('새 세 조건 모두 음성인 이상 포함 점수 구간',lambda r:r['scored_positive_frames']>0 and all(index[(c,r['video_id'],r['center'])]['prediction']==0 for c in ['C0','N','X']))]
  for title,select in selectors:
   candidates=[r for r in rows if select(r)]
   if not candidates:text+='- '+title+': 해당 사례 없음.\n\n';continue
   w=candidates[0];_,split,vid=w['video_id'].split('/');responses={}
   for c in ['P1','C0','N','X']:
    folder=ROOT/'experiments'/('stage5' if c=='P1' else 'stage6')/'inference'/c/scene/split/vid;p=folder/f"{w['center']:06d}.json";raw=json.loads(p.read_text())
    responses[c]={'source':str(p.relative_to(ROOT)),'sha256':sha(p),'prediction':raw['prediction'],'response':raw['response'],'reference_images':[] if c=='P1' else [r for r in raw['request']['images'] if r['role']=='reference']}
   result[scene].append({'selection':title,'window':w,'responses':responses});text+='### '+title+'\n\n'+f"`{w['video_id']}`, center={w['center']}, 점수 [{w['center']},{w['score_end']}), 이상 점수 프레임 {w['scored_positive_frames']}/{w['scored_frames']}, 이상 입력 {w['sampled_positive_frames']}/8.\n\n"
   for c,r in responses.items():
    link='../stage5/'+str(Path(r['source']).relative_to('experiments/stage5')) if c=='P1' else str(Path(r['source']).relative_to('experiments/stage6'))
    text+=f"**{c} / Output {r['prediction']}** ([원문]({link}))\n\n"
    if r['reference_images']:text+='참조: '+', '.join(f"`{x['video_id']}:{x['frame_id']}`" for x in r['reference_images'])+'\n\n'
    text+='> '+r['response'].replace('\n','\n> ')+'\n\n'
 write(OUT/'explanation_examples.json',result);(OUT/'explanation_examples.md').write_text(text);print(json.dumps({'examples':{s:len(v) for s,v in result.items()},'posthoc_only':True}))
if __name__=='__main__':main()
