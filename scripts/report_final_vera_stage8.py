"""Generate final Korean report from completed artifacts only."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage8_common import OUT,REPORT,write,protected,evaluator_guard,sha

def main():
 evaluator_guard();prior=protected()
 for step in ['step1','step4','step5']:assert json.loads((OUT/step/'status.json').read_text())['status']=='complete'
 m1=json.loads((OUT/'step1/metrics.json').read_text());m4=json.loads((OUT/'step4/metrics.json').read_text());m5=json.loads((OUT/'step5/metrics.json').read_text());one=json.loads((OUT/'step0/metrics.json').read_text())['ONE'];seed=json.loads((OUT/'seed0_verification/status.json').read_text());assert seed['status']=='complete'
 lines=['# VERA × IPAD 8단계 실험 결과','','기존 R01–R04 재분할의 평가 37영상, 16,862프레임. 이전에 관찰한 자료이므로 모두 탐색적 결과입니다. 사용자가 별도 미관측 hold-out 없이 진행하고 확증 평가를 보류하도록 승인했습니다.','', '| 조건 | 초기 macro AUROC/AP (%) | smoothing·위치 OFF | 최종 위치 ON | 최종 Δ vs ONE AUROC/AP (pp) |','|---|---:|---:|---:|---:|']
 allm={**m1,**{k:v for k,v in m4.items() if k!='PROB_10s'},'DINO':m5['DINO'],'HYBRID':m5['HYBRID']}
 for c,d in allm.items():
  v=d['macro'];fmt=lambda s:f"{v[s]['auroc']:.4f} / {v[s]['ap']:.4f}";delta={k:v['final'][k]-one['macro']['final'][k] for k in ['auroc','ap']};lines.append(f"| {c} | {fmt('initial')} | {fmt('smoothed')} | {fmt('final')} | {delta['auroc']:.4f} / {delta['ap']:.4f} |")
 lines+=['','확정된 실행 분기와 해석:','', '- 10초 PROB 초기 macro AUROC 52.7044%로 사전 50±3% 분기에 해당했습니다. Step 2 정상 참조 비교와 Step 3 질문 최적화는 수행하지 않았습니다.','- 10초 창의 확률 점수는 2종의 이진 출력보다 28종으로 세분화됐지만 DEC 대비 초기·최종 AUROC 차이 CI가 0을 포함했습니다. 원인을 시각 인식 한계 하나로 확정하지 않습니다.','- 사전 분기에 따라 10초 PROB를 사용한 검증 17영상에서 λ=0이 선택돼 HYBRID와 DINO가 동일합니다. 높은 하이브리드 수치를 VLM 결합 이득으로 주장하지 않습니다.','- 창 길이 2/4/10초는 사전 고정 진단입니다. 높은 테스트 수치를 보고 창 길이를 새 기본값으로 선택하거나 질문/규칙을 다시 만들지 않았습니다.','- 모든 AUROC/AP는 sklearn, seed 0, 장면별 macro와 pooled를 함께 보존합니다. 비교 CI는 장면 내 영상 단위 2,000회 bootstrap이며 441회(22.05%)의 단일 클래스 macro 무효 표본을 숨기지 않았습니다.','- ONE은 위치 prior 대조군입니다. ORACLE은 라벨 주입 진단이며 모델 성능이나 수학적 상한이 아닙니다.','- 실제 FPS는 원본 영상 컨테이너 부재로 확인할 수 없어 30 FPS 가정을 유지했습니다. 외부 논문의 25 FPS를 이 로컬 자료의 실측으로 대신하지 않았습니다.','- VLM/영상 문맥은 미래 프레임을 포함한 오프라인 방식입니다. 온라인 성능이나 일반화 성능을 확증하지 않습니다.','', '| 장면 | PROB 초기 AUROC/AP | DINO 초기 AUROC/AP | DINO 최종 AUROC/AP |','|---|---:|---:|---:|']
 for scene in ['R01','R02','R03','R04']:
  values=[m1['PROB'][scene]['initial'],m5['DINO'][scene]['initial'],m5['DINO'][scene]['final']];lines.append('| '+scene+' | '+' | '.join(f"{v['auroc']:.4f} / {v['ap']:.4f}" for v in values)+' |')
 lines+=['','수행·재현 근거:','', '- [Step 0 대조군·분포](step0_sanity.md)', '- [Step 1 동일 forward DEC/PROB](step1_prob_vs_dec.md)', '- [Step 4 시간 창·후처리](step4_temporal.md)', '- [Step 5 검증 선택 하이브리드](step5_hybrid.md)', '- [모든 AUROC/AP 신뢰구간](confidence_intervals.md)', '- [처리 단계별 ONE 대비 차이](delta_vs_ONE.md)', '- [37개 별도 설명](step5_explanations.md)', '- [재현 명령](../../experiments/stage8/reproduction_commands.json)', '- [소스 SHA256](../../experiments/stage8/source_inventory.json)', '- [상세 실행 명령·stdout/stderr](../../experiments/stage8/execution/)', '',f'이전 1–7단계의 {prior:,}개 산출물/보호 소스 SHA가 불변입니다. R01–R03의 기존 Stage 4 결과도 변경하거나 재실행하지 않았습니다.','', '시드 초기화 보정: 초기 시간 창 실행의 956개 응답은 정답 평가 전에 중단해 진단 폴더에 보존했습니다. 정식 시간 창 실행은 명시적 seed 0으로 다시 수행했습니다. 검증 추론 522건과 설명 37건은 seed 0에서 전수 재현해 기존 토큰·응답·검증 logits/확률의 정확한 일치를 확인했습니다. 설정을 성능에 맞춰 바꾸지 않았습니다.','', '설명 원문의 의미적 정확도는 측정하지 않았습니다. 정상 참조와의 단순 차이가 결함이라는 보장은 없으며, 설명은 점수에 반영하지 않았습니다. 모델 가중치·원본 데이터·특징 캐시는 로컬에만 보존합니다.']
 (REPORT/'results.md').write_text('\n'.join(lines)+'\n')
 p=ROOT/'README.md';text=p.read_text();text=text.replace('# VERA의 IPAD 전이 평가\n', '# VERA의 IPAD 전이 평가\n\n**8단계 완료: 2초 창의 VLM 초기 AUROC는 61.91%, 10초 PROB는 52.70%였습니다. 검증 선택 하이브리드(10초 PROB)는 λ=0(DINO 단독)을 선택했습니다.** [전체 결과·시드 보정·한계](reports/stage8/results.md).\n',1);row='| **8** | **확률 점수·시간 창·검증 선택 하이브리드** | **완료·탐색적** | PROB 초기 AUROC 52.70%, 선택 HYBRID 초기 87.40%(λ=0); [결과](reports/stage8/results.md) |\n';pos=text.index('\n진행 상태는');text=text[:pos].rstrip()+'\n'+row+text[pos:];marker='\n### 8단계 Step 4 — 시간 창 비교 및 최종 완료\n';assert marker not in text
 section=[marker,'2/4/10초(30 FPS 가정) 창 비교와 후처리 분해를 완료했다. 평가 결과로 창 길이를 재선택하지 않았다.','', '| 창 | 초기 macro AUROC/AP (%) | 최종 macro AUROC/AP (%) |','|---|---:|---:|']
 for c,d in sorted(m4.items()):
  a=d['macro']['initial'];b=d['macro']['final'];section.append(f"| {c} | {a['auroc']:.2f} / {a['ap']:.2f} | {b['auroc']:.2f} / {b['ap']:.2f} |")
 section += ['','[8단계 전체 결과](reports/stage8/results.md) · [창별 4단계 점수·CI](reports/stage8/step4_temporal.md) · [전체 AUROC/AP 신뢰구간](reports/stage8/confidence_intervals.md).','', 'Step 2·3은 사전 분기에 따라 생략했다. 8단계의 성능 주장은 탐색적이며 미관측 hold-out 확증은 보류했다. 실패·재시작·시드 보정 로그도 보존한다.']
 p.write_text(text+'\n'.join(section)+'\n')
 write(OUT/'status.json',{'status':'complete','stage':8,'scope':'exploratory 37-video evaluation only','completed_steps':[0,1,4,5],'skipped_steps':[2,3],'reason_for_skip':'prespecified Step1 initial macro PROB 50±3 gate','confirmatory_holdout':'deferred by explicit user choice','protected_files':prior,'seed0_correction_verified':True,'finished_at':time.time()})
 p=ROOT/'experiments/stages.json';reg=json.loads(p.read_text());reg['stage8_step4']={'title':'시간 창·후처리 분해 및 8단계 완료 검증','status':'complete','execution_approved':True,'status_file':'experiments/stage8/step4/status.json','artifact_directory':'experiments/stage8/step4'};reg['stage8']={'title':'확률 점수·시간 창·검증 선택 하이브리드','status':'complete','execution_approved':True,'status_file':'experiments/stage8/status.json','artifact_directory':'experiments/stage8'};p.write_text(json.dumps(reg,ensure_ascii=False,indent=2)+'\n')
 print('Final report generated from completed results')
if __name__=='__main__':main()
