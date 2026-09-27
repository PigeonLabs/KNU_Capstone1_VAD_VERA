"""Finish already-authorized stages sequentially; publish only completed evidence."""
import json,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/jeong/Desktop/IPAD/runs/vera_publication/KNU_Capstone1_VAD_VERA')
WORK=Path('/tmp/vera_stage3_work');PY='/home/jeong/Desktop/IPAD/cache/vera/venv/bin/python'
LOCAL=ROOT/'runs/stage_continuation';LOCAL.mkdir(parents=True,exist_ok=True)
def state(step,**kw):
 (LOCAL/'status.json').write_text(json.dumps({'step':step,'updated_at':time.time(),**kw},indent=2)+'\n');print(step,flush=True)
def run(args):subprocess.run(args,cwd=ROOT,check=True)
def logged(stage,step,args):run([PY,'scripts/logged_command.py','--stage',stage,'--step',step,'--',*args])
def copy(name):shutil.copyfile(WORK/name,ROOT/'scripts'/name)
def publish(stage,message):run([PY,'scripts/publish_vera_stage.py','--stage',stage,'--message',message])
try:
 state('waiting_for_training')
 while True:
  status=json.loads((ROOT/'experiments/stage2_2/status.json').read_text())
  if status['status'] in ['failed','paused_low_disk']:raise RuntimeError('Training stopped: '+str(status))
  if status['status']=='complete':
   commands=sorted((ROOT/'experiments/stage2_2/execution/training').glob('*/command.json'))
   if commands and json.loads(commands[-1].read_text())['status']=='complete':break
  time.sleep(5)
 state('auditing_training');copy('audit_vera_training.py')
 logged('stage2_2','independent_audit',[PY,'scripts/audit_vera_training.py'])
 logged('stage2_2','report',[PY,str(WORK/'report_vera_stage.py'),'stage2_2'])
 publish('stage2_2','2-2단계 완료: learner–optimizer 600회 질문 최적화와 전체 이력')
 state('selecting_questions');copy('select_vera_questions.py')
 logged('stage2_3','selection',[PY,'scripts/select_vera_questions.py'])
 logged('stage2_3','report',[PY,str(WORK/'report_vera_stage.py'),'stage2_3'])
 publish('stage2_3','2-3단계 완료: 검증 정확도로 질문 선택·동결')
 state('evaluating');copy('evaluate_vera_learned.py')
 logged('stage3','evaluation',[PY,'scripts/evaluate_vera_learned.py'])
 state('auditing_evaluation');copy('audit_vera_evaluation.py')
 logged('stage3','independent_audit',[PY,'scripts/audit_vera_evaluation.py'])
 logged('stage3','report',[PY,str(WORK/'report_vera_stage.py'),'stage3'])
 # These orchestration/report sources now describe only stages actually executed.
 copy('report_vera_stage.py');copy('continue_vera_stages.py')
 publish('stage3','3단계 완료: 선택 질문의 IPAD 평가·동일 프레임 비교·독립 검산')
 state('complete')
except BaseException as exc:
 state('failed',error=repr(exc));raise
