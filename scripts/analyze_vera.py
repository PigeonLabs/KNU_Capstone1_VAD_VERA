"""Descriptive diagnostics of the completed fixed experiment; no setting selection."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_vera import summarize
from scripts.run_vera import OUT, CACHE, sha
from ipad.common import SCENES, write_json


def main():
    if json.loads((OUT/'status.json').read_text())['status']!='complete':
        raise RuntimeError('Full frozen evaluation must finish first')
    with (OUT/'frame_scores.csv').open() as f:
        rows=[dict(r,frame=int(r['frame']),label=int(r['label']),**{k:float(r[k]) for k in ['initial','retrieved','smoothed','final']}) for r in csv.DictReader(f)]
    lookup={(r['scene'],int(r['video']),r['frame']):r for r in rows}
    pairs=[];hashes={}
    columns=['conditional_nn_with_phase','conditional_soft_with_phase','unconditional_nn_with_phase','unconditional_soft_with_phase']
    for scene in SCENES:
        p=ROOT/'experiments/stage2_dinov2'/scene/'prototype/scores.csv';hashes[str(p.relative_to(ROOT))]=sha(p)
        with p.open() as f:
            for old in csv.DictReader(f):
                key=(scene,int(old['video']),int(old['frame']))
                if key not in lookup:continue
                new=lookup[key]
                if int(old['label'])!=new['label']:raise ValueError('Label mismatch')
                pairs.append(dict(new,**{c:float(old[c]) for c in columns}))
    if len({(r['scene'],r['video'],r['frame']) for r in pairs})!=len(pairs):raise ValueError('Duplicate baseline frame')
    comparison=dict(frames=len(pairs),source_sha256=hashes,vera=summarize(pairs,'final'),
                    baselines={c:summarize(pairs,c) for c in columns})
    write_json(OUT/'prototype_comparison.json',comparison)
    runtimes={}
    for name,base in [('vlm',OUT),('features',CACHE/'feature_journals')]:
        samples=[json.loads(l) for p in (base/'inference').rglob('*.jsonl') for l in p.read_text().splitlines()]
        seconds=np.array([r['seconds'] for r in samples])
        runtimes[name]=dict(segments=len(samples),sum_seconds=float(seconds.sum()),mean_seconds=float(seconds.mean()),
                            p50_seconds=float(np.median(seconds)),p95_seconds=float(np.percentile(seconds,95)),
                            peak_allocated_gib=max(r['peak_allocated_bytes'] for r in samples)/2**30,
                            peak_reserved_gib=max(r['peak_reserved_bytes'] for r in samples)/2**30)
    write_json(OUT/'runtime_summary.json',runtimes)
    counts=json.loads((OUT/'raw_prediction_counts.json').read_text())
    diagnostics={}
    for scene in SCENES:
        group=[r for k,r in counts.items() if k.startswith(scene+'/')]
        diagnostics[scene]=dict(videos=len(group),segments=sum(r['segments'] for r in group),
                               positive_segments=sum(r['positive'] for r in group),
                               all_zero_videos=sum(r['positive']==0 for r in group),
                               all_one_videos=sum(r['positive']==r['segments'] for r in group))
    write_json(OUT/'prediction_diagnostics.json',diagnostics)
    metrics=json.loads((OUT/'metrics.json').read_text())
    comparisons=json.loads((OUT/'comparisons.json').read_text())
    final=metrics['final']
    lines=['# VERA IPAD 결과 해석', '',
           f"공개 질문 기반 추론 구현과 전체 평가는 완료했지만, 최종 macro AUROC는 {final['macro']['auroc']:.2f}%, pooled AUROC는 {final['pooled']['auroc']:.2f}%로 낮았습니다.",
           '정상 학습 영상 4개에서 실행 가능성을 점검한 뒤, 고정 설정으로 63개 테스트 영상·31,550프레임을 평가했습니다. R02 12·13·14는 기존 정렬 규약에 따라 제외했습니다.', '',
           '## 초기 판정의 문제', '',
           '| 장면 | 구간 수 | 이상 판정 구간 | 모든 구간이 정상 판정인 영상 |',
           '|---|---:|---:|---:|']
    for scene,d in diagnostics.items():
        lines.append(f"| {scene} | {d['segments']} | {d['positive_segments']} | {d['all_zero_videos']}/{d['videos']} |")
    positives=sum(d['positive_segments'] for d in diagnostics.values())
    lines += ['',f'전체 2,001개 구간 중 {positives}개만 이상으로 판정했습니다. R01·R02는 초기 점수가 모두 0이어서 검색·평활화·위치 가중치를 적용해도 AUROC가 50%입니다.',
              '초기 macro AUROC 50.21%에서 후처리 후 52.02%가 되었습니다. 현재 결과에서 우선 확인할 문제는 후처리보다 초기 시각 판정의 거의 전부 정상이라는 응답입니다.',
              'UCF에서 학습한 질문이 산업 공정에 맞지 않을 가능성, 8프레임·10초 문맥의 시간 정보 손실, VLM의 미세한 공정 동작 인식 한계는 가능한 설명입니다. 이번 실험만으로 원인을 확정하지는 않습니다.', '',
              '## 동일 프레임 교집합 비교', '',
              '아래 모든 방법은 공통 30,353프레임에서 재계산했습니다. 전체 프레임 VERA 지표와 구분합니다. 기존 seed 0 결과를 사용했으며 가장 높은 결과를 골라 설정을 바꾸지 않았습니다.', '',
              '| 방법 | Macro AUROC (%) | Macro AP (%) |', '|---|---:|---:|']
    cm=comparisons['IPAD']['vera']['macro']
    lines.append(f"| VERA | {cm['auroc']:.3f} | {cm['auprc']:.3f} |")
    for group,c in [*comparisons.items(),('DINOv2_prototype',comparison)]:
        for name,r in c['baselines'].items():
            m=r['macro'];lines.append(f"| {group}: {name} | {m['auroc']:.3f} | {m['auprc']:.3f} |")
    lines += ['', '## 실행 및 재현 범위', '',
              f"VLM 평균 {runtimes['vlm']['mean_seconds']:.3f}초/구간, ImageBind 평균 {runtimes['features']['mean_seconds']:.3f}초/구간입니다. 최대 할당 GPU 메모리는 각각 {runtimes['vlm']['peak_allocated_gib']:.2f}GiB, {runtimes['features']['peak_allocated_gib']:.2f}GiB입니다.",
              '시간은 전처리와 모델 계산을 포함한 구간 처리 시간이며 모델 로딩·manifest 검사·로그 쓰기 등 전체 실행 시간과 다릅니다. 오프라인 미래 문맥을 사용하므로 실시간 인과 추론 성능으로 해석하지 않습니다.',
              '정합성·디스크 보호 테스트 23개와 실제 정상 영상 GPU 진단을 통과했습니다. tokenizer 초기 호환성 실패 및 수정 이력은 preflight 로그에 보존했습니다.',
              '원 논문 학습 전체나 원 벤치마크 점수 재현은 아닙니다. BF16 eager attention, 다중 이미지 토큰 연결, 논문/공식 코드 차이, ImageBind 전처리 가정은 ../../docs/vera_ipad_protocol.md에 명시했습니다.',
              '공개 질문을 임의로 수정하거나 테스트 지표로 설정을 선택하지 않았습니다. 후속 실험은 이번 결과를 검토한 뒤 범위를 정합니다.', '',
              '근거: metrics.json, comparisons.json, prototype_comparison.json, prediction_diagnostics.json, runtime_summary.json, positive_explanations.json, frame_scores.csv.']
    (OUT/'interpretation.md').write_text('\n'.join(lines)+'\n')
    inventory=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name not in ['artifact_inventory.json','disk_guard.log']:
            inventory.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    write_json(OUT/'artifact_inventory.json',inventory)
    print(json.dumps(dict(runtime=runtimes,predictions=diagnostics),indent=2))

if __name__=='__main__':main()
