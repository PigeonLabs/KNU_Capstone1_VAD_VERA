"""Append-only command capture for approved experiment/publication steps."""
import argparse
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import hashlib,json,os,re,subprocess,sys,time,uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--step',required=True)
    p.add_argument('--config',type=Path);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    for name in (a.stage,a.step):
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',name):p.error('Invalid stage/step name')
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:p.error('Missing command')
    joined=' '.join(command)
    if re.search(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-proj-[A-Za-z0-9_-]{20,})',joined):
        p.error('Credentials must not be placed in command arguments')
    now=datetime.now(timezone.utc);run_id=now.strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=ROOT/'experiments'/a.stage/'execution'/a.step/run_id;out.mkdir(parents=True)
    record=dict(stage=a.stage,step=a.step,attempt=run_id,command=command,cwd=str(ROOT),
                started_utc=now.isoformat(),started_kst=now.astimezone(ZoneInfo('Asia/Seoul')).isoformat(),
                started_unix=time.time(),status='running',
                environment_values_logged=False,stdout_stderr='combined in output.log')
    if a.config:
        path=a.config.resolve();record['config']=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    target=out/'command.json';target.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    code=1;child=None
    try:
        with (out/'output.log').open('wb') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            while data:=child.stdout.readline():
                log.write(data);log.flush();sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
            code=child.wait()
    except BaseException as exc:
        record['exception']=repr(exc)
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        code=130 if isinstance(exc,KeyboardInterrupt) else 1
    finally:
        finished=datetime.now(timezone.utc)
        record.update(finished_utc=finished.isoformat(),finished_kst=finished.astimezone(ZoneInfo('Asia/Seoul')).isoformat(),
                      elapsed_seconds=time.time()-record['started_unix'],
                      exit_code=code,status='complete' if code==0 else 'failed')
        target.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    print('Execution evidence:',out.relative_to(ROOT))
    return code

if __name__=='__main__':raise SystemExit(main())
