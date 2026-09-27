"""Sequential execution of approved scene-specific stage4; publish terminal evidence only."""
import json,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/jeong/Desktop/IPAD/runs/vera_publication/KNU_Capstone1_VAD_VERA');WORK=Path('/tmp/vera_stage4_work');PY='/home/jeong/Desktop/IPAD/cache/vera/venv/bin/python'
STATE=ROOT/'runs/vera_stage4_continuation.json'
def state(**kw):STATE.write_text(json.dumps({'time':time.time(),**kw},ensure_ascii=False,indent=2)+'\n');print(kw,flush=True)
def run(args):subprocess.run(args,cwd=ROOT,check=True)
def logged(step,args):run([PY,'scripts/logged_command.py','--stage','stage4','--step',step,'--',*args])
def copy(name):shutil.copyfile(WORK/name,ROOT/'scripts'/name)
def publish(scene,kind):run([PY,'scripts/publish_vera_stage.py','--stage',f'stage4_{scene}_{kind}','--message',f'4단계 {scene}: '+('정상 기준 생성·근거 확인 완료' if kind=='normal' else '독립 A/B/C 평가·검산 완료')])
def complete(p):return p.exists() and json.loads(p.read_text())['status']=='complete'
try:
 # R01 normal extraction was started interactively; never start a duplicate worker.
 first=ROOT/'experiments/stage4/R01/normal/status.json'
 while not complete(first):
  s=json.loads(first.read_text()) if first.exists() else {}
  if s.get('status') in ['failed','paused_low_disk']:raise RuntimeError('R01 normal extraction stopped: '+str(s))
  state(step='waiting_for_existing_R01_normal');time.sleep(5)
 while True:
  logs=sorted([p for p in (ROOT/'experiments/stage4/execution').glob('*/*/command.json') if 'scripts/build_vera_normal_context.py' in json.loads(p.read_text())['command'] and json.loads(p.read_text())['command'][-1]=='R01'],key=lambda p:json.loads(p.read_text())['started_unix'])
  if logs and json.loads(logs[-1].read_text())['status']=='complete':break
  time.sleep(1)
 for scene in ['R01','R02','R03','R04']:
  for kind in ['normal','evaluation']:
   registry=json.loads((ROOT/'experiments/stages.json').read_text())
   if f'stage4_{scene}_{kind}' in registry:continue
   state(scene=scene,step=kind)
   terminal=ROOT/'experiments/stage4'/scene/kind/'status.json'
   if terminal.exists() and json.loads(terminal.read_text()).get('status')=='running':
    while not complete(terminal):
     current=json.loads(terminal.read_text())
     if current['status'] in ['failed','paused_low_disk']:raise RuntimeError('Existing stage stopped: '+str(current))
     time.sleep(5)
    logfiles=sorted([p for p in (ROOT/'experiments/stage4/execution').glob('*/*/command.json') if json.loads(p.read_text())['command'][-1]==scene and any('build_vera_normal_context' in arg for arg in json.loads(p.read_text())['command'])],key=lambda p:json.loads(p.read_text())['started_unix'])
    while logfiles and json.loads(logfiles[-1].read_text())['status']=='running':time.sleep(1)
   if not complete(terminal):
    if terminal.exists() and json.loads(terminal.read_text()).get('status')=='paused_low_disk':raise RuntimeError('Explicit disk-pause resume required')
    script=('build_vera_normal_context.py' if scene=='R01' else 'build_vera_normal_context_v2.py') if kind=='normal' else 'evaluate_vera_normal_context.py'
    if kind=='evaluation':copy(script)
    logged(scene+'_'+kind,[PY,'scripts/'+script,'--scene',scene])
   copy('audit_vera_stage4.py')
   logged(scene+'_'+kind+'_audit',[PY,'scripts/audit_vera_stage4.py','--scene',scene,'--kind',kind])
   logged(scene+'_'+kind+'_report',[PY,str(WORK/'report_vera_stage4.py'),'--scene',scene,'--kind',kind])
   publish(scene,kind)
 state(step='summarizing')
 copy('summarize_vera_stage4.py')
 logged('summary',[PY,'scripts/summarize_vera_stage4.py'])
 copy('report_vera_stage4.py');copy('continue_vera_stage4.py')
 run([PY,'scripts/publish_vera_stage.py','--stage','stage4','--message','4단계 완료: R01–R04 독립 정상 기준·질문 A/B/C 종합'])
 state(step='complete')
except BaseException as exc:
 state(step='failed',error=repr(exc),traceback=traceback.format_exc());raise
