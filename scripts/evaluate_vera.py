"""Labels are opened only here, after frozen inference is complete."""
import csv
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from ipad.common import SCENES, write_json
from ipad.metrics import binary_metrics
from ipad.vera import VeraConfig, refine_scores, segments
from scripts.run_vera import CACHE, OUT, load_records, sha


def summarize(rows, column):
    by_scene={s: binary_metrics([r['label'] for r in rows if r['scene']==s],
                               [r[column] for r in rows if r['scene']==s]) for s in SCENES}
    result=dict(scenes=by_scene, pooled=binary_metrics([r['label'] for r in rows],[r[column] for r in rows]))
    result['macro']={m:float(np.mean([v[m] for v in by_scene.values() if v[m] is not None]))
                    if any(v[m] is not None for v in by_scene.values()) else None for m in ['auroc','auprc']}
    return result


def compare(rows):
    table={(r['scene'],int(r['video']),int(r['frame'])):r for r in rows}
    comparisons={}
    sources=[('IPAD', 'stage1_reproduction', 'evaluation/scores.csv', ['negative_psnr_with_phase']),
             ('DINOv2_reconstruction', 'stage2_dinov2','reconstruction/scores.csv',
              ['dino_patch6_with_phase','dino_patch12_with_phase','dino_cls_with_phase','dino_multilevel_with_phase'])]
    for name,folder,suffix,columns in sources:
        pairs=[]; source_hashes={}
        for scene in SCENES:
            path=ROOT/'experiments'/folder/scene/suffix
            if not path.exists(): raise RuntimeError(f'Missing baseline: {path}')
            source_hashes[str(path.relative_to(ROOT))]=sha(path)
            with path.open() as f:
                for old in csv.DictReader(f):
                    key=(scene,int(old['video']),int(old['frame']))
                    if key not in table: continue
                    new=table[key]
                    if int(float(old['label']))!=new['label']: raise ValueError('Baseline label disagreement')
                    pairs.append(dict(new,**{c:float(old[c]) for c in columns}))
        if len({(r['scene'],r['video'],r['frame']) for r in pairs})!=len(pairs): raise ValueError('Duplicate comparison rows')
        comparisons[name]=dict(frames=len(pairs), source_sha256=source_hashes, vera=summarize(pairs,'final'),
                               baselines={c:summarize(pairs,c) for c in columns})
    return comparisons


def main():
    config=VeraConfig(); frozen=json.loads((OUT/'frozen.json').read_text())
    manifest=json.loads((OUT/'test_manifest.json').read_text())
    if frozen['config']!=vars(config): raise RuntimeError('Configuration changed')
    for path,expected in frozen['source_sha256'].items():
        if sha(ROOT/path)!=expected: raise RuntimeError(f'Source changed: {path}')
    all_rows=[]; excluded=[]; inventory=[]; raw_counts={}; labels_inventory=[]
    for record in manifest:
        scene,video,length=record['scene'],record['video'],record['length']
        lp=ROOT/'IPAD_dataset'/scene/'test_label'/f'{int(video):03d}.npy'
        labels=np.load(lp,allow_pickle=False).reshape(-1)
        labels_inventory.append(dict(path=str(lp.relative_to(ROOT)),sha256=sha(lp)))
        if scene=='R02' and int(video) in (12,13,14):
            excluded.append(dict(scene=scene,video=video,frames=length,labels=len(labels),reason='pre-existing strict alignment exclusion'))
            continue
        if len(labels)!=length or not np.isin(labels,[0,1]).all(): raise ValueError('Unexpected label alignment')
        vp=OUT/'inference'/scene/f'{video}.vlm.jsonl'
        fp=CACHE/'feature_journals/inference'/scene/f'{video}.features.jsonl'
        vr,fr=load_records(vp),load_records(fp)
        ss=segments(length,config);centers=[s['center'] for s in ss]
        if set(vr)!=set(centers) or set(fr)!=set(centers): raise ValueError('Incomplete inference')
        for s in ss:
            for source in (vr,fr):
                row=source[s['center']]
                if row['fingerprint']!=frozen['fingerprint']: raise ValueError('Stale cache fingerprint')
                if any(row[k]!=s[k] for k in s): raise ValueError('Segment manifest mismatch')
        scores,details=refine_scores([vr[c]['score'] for c in centers],[fr[c]['feature'] for c in centers],length,config)
        raw_counts[f'{scene}/{video}']=dict(segments=len(centers),positive=sum(vr[c]['score'] for c in centers))
        write_json(OUT/'postprocessing'/scene/f'{video}.json',details)
        inventory.extend(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in (vp,fp))
        for i in range(length):
            all_rows.append(dict(scene=scene,video=video,frame=i,label=int(labels[i]),**{k:float(v[i]) for k,v in scores.items()}))
    with (OUT/'frame_scores.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    metrics={c:summarize(all_rows,c) for c in ['initial','retrieved','smoothed','final']}
    write_json(OUT/'metrics.json',metrics);write_json(OUT/'comparisons.json',compare(all_rows))
    write_json(OUT/'excluded.json',excluded);write_json(OUT/'raw_prediction_counts.json',raw_counts)
    write_json(OUT/'inference_inventory.json',inventory);write_json(OUT/'label_inventory.json',labels_inventory)
    lines=['# VERA의 IPAD 전이 평가','','고정 InternVL2-8B·UCF 공개 질문·ImageBind, 질문 재학습 없음. 원 논문 수치 재현이 아닌 오프라인 전이 평가.',
           '30 FPS는 가정이며 미래 프레임과 영상 전체 문맥을 사용합니다. AP는 sklearn average_precision_score입니다.','',
           '| 단계 | Macro AUROC (%) | Macro AP (%) | Pooled AUROC (%) | Pooled AP (%) |','|---|---:|---:|---:|---:|']
    for name,m in metrics.items():
        lines.append(f"| {name} | {m['macro']['auroc']:.3f} | {m['macro']['auprc']:.3f} | {m['pooled']['auroc']:.3f} | {m['pooled']['auprc']:.3f} |")
    lines+=['','| 장면 | 최종 AUROC (%) | 최종 AP (%) | 프레임 수 |','|---|---:|---:|---:|']
    for scene,m in metrics['final']['scenes'].items(): lines.append(f"| {scene} | {m['auroc']:.3f} | {m['auprc']:.3f} | {m['frames']} |")
    lines+=['',f'평가 영상 {len(manifest)-len(excluded)}개, 제외 {len(excluded)}개, 평가 프레임 {len(all_rows)}개.',
            'R02 12·13·14는 기존 라벨 불일치 규약에 따라 제외했습니다.',
            '동일 프레임 교집합 비교는 comparisons.json, 원시 응답은 inference/, 단계별 점수는 frame_scores.csv에 보존했습니다.',
            'R01의 짧은 영상에서는 10초 문맥 창이 많은 구간에 중복될 수 있습니다. 테스트 결과로 설정을 수정하지 않았습니다.',
            '설명은 VLM 생성문이며 검증된 주석이 아닙니다. 오디오는 입력하지 않으며 공개 질문은 그대로 유지했습니다.',
            '후속 실험은 결과 검토 후 별도로 결정합니다.']
    (OUT/'results.md').write_text('\n'.join(lines)+'\n')
    write_json(OUT/'status.json',dict(status='complete',completed_at=time.time(),fingerprint=frozen['fingerprint'],videos=len(manifest)-len(excluded),frames=len(all_rows)))
    print(json.dumps(metrics['final'],indent=2))


if __name__=='__main__':main()
