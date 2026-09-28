import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage8_common import *
def main():
 out=OUT/'step5/explanations';frozen=json.loads((out/'frozen.json').read_text());assert sha(OUT/'step5/window_scores.csv')==frozen['score_source_sha256'];selection={r['video_id']:r for r in frozen['chosen']};normal={r['id'] for r in rows() if r['split']=='train' and r['video_label']==0};samples=[]
 for path in out.glob('R*/*/*.json'):
  r=json.loads(path.read_text());assert r['segment']['center']==int(selection[r['video_id']]['center']);assert len(r['images'])==12;assert {x['video_id'] for x in r['images'][:4]}<=normal;assert all(x['video_id']==r['video_id'] for x in r['images'][4:]);resolve_images(r['images']);assert r['generated_tokens']==len(r['generated_token_ids'])<=256;samples.append(r)
 assert len(samples)==len(selection)==37
 write(out/'verification.json',{'status':'passed','calls':37,'token_limit_hits':sum(r['at_token_limit'] for r in samples),'score_source_unchanged':True,'selection_no_labels':True,'references_normal_train_only':True,'semantic_accuracy':'not measured; generated descriptions are not verified annotations'})
 lines=['# 8단계 점수와 분리된 설명','','영상별 최고 평균 점수 구간 1개씩, 정상 학습 참조 4프레임과 query 8프레임으로 생성했습니다. 256토큰·repetition_penalty 1.1, 총 37회. **아래 문장은 VLM 생성문이며 참조와의 차이가 실제 이상이라는 검증은 아닙니다.** 정상 영상의 위상 차이나 가시성 차이도 설명에 포함될 수 있습니다.','', '| 영상 | 중심 | 생성 원문 |','|---|---:|---|']
 for r in sorted(samples,key=lambda r:r['video_id']):lines.append(f"| {r['video_id']} | {r['segment']['center']} | {r['response'].replace('|','/').replace(chr(10),' ')} |")
 (REPORT/'step5_explanations.md').write_text('\n'.join(lines)+'\n');print('37 explanations verified; scores unchanged; semantics unverified')
if __name__=='__main__':main()
