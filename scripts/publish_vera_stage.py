"""Commit approved terminal stage evidence and push main without force."""
import argparse
from datetime import datetime,timezone
import hashlib,json,re,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REMOTE='https://github.com/PigeonLabs/KNU_Capstone1_VAD_VERA.git'
EXTENSIONS={'.md','.py','.json','.jsonl','.csv','.tsv','.txt','.toml','.yaml','.yml','.sh','.log'}
ALLOWED_ROOTS={'docs','experiments','ipad','scripts','tests','configs','reports'}
ALLOWED_FILES={'AGENTS.md','README.md','.gitignore','requirements.txt','requirements.lock.txt','THIRD_PARTY_NOTICES.md'}


def git(*args,check=True):
    return subprocess.run(['git',*args],cwd=ROOT,check=check,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)


def publication_files(exclude_prefixes=()):
    files=sorted(set(git('ls-files','--cached','--others','--exclude-standard').stdout.splitlines()))
    records=[]
    for name in files:
        if any(name.startswith(prefix) for prefix in exclude_prefixes):continue
        path=Path(name)
        if not(name in ALLOWED_FILES or path.parts[0] in ALLOWED_ROOTS):raise RuntimeError('Unapproved publication path: '+name)
        if name!='.gitignore' and path.suffix.lower() not in EXTENSIONS and not (name.startswith('reports/stage8/') and path.suffix.lower()=='.svg'):raise RuntimeError('Excluded file type: '+name)
        if any(part in {'cache','IPAD_dataset','.venv','__pycache__','runs'} for part in path.parts):raise RuntimeError('Local-only path: '+name)
        actual=ROOT/path
        if not actual.exists() and not actual.is_symlink():continue  # Tracked deletion; staged explicitly below.
        if actual.is_symlink() or not actual.is_file():raise RuntimeError('Nonregular publication file: '+name)
        if actual.stat().st_size>50*1024**2:raise RuntimeError('Oversized publication file: '+name)
        data=actual.read_bytes()
        if b'\x00' in data:raise RuntimeError('Binary content: '+name)
        data.decode('utf-8')
        if path.suffix.lower()=='.svg' and re.search(rb'<(?:script|image|foreignObject)\b|(?:href|src)=[\"\'](?:https?:|data:|file:)',data,re.I):raise RuntimeError('Non-static analysis SVG: '+name)
        if re.search(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-proj-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)',data):
            raise RuntimeError('Potential secret in '+name)
        records.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--message',required=True)
    p.add_argument('--check-only',action='store_true');p.add_argument('--exclude-prefix',action='append',default=[]);a=p.parse_args()
    registry=json.loads((ROOT/'experiments/stages.json').read_text())
    stage=registry[a.stage]
    if stage['status'] not in {'complete','failed','diagnostic'}:raise RuntimeError('Only a terminal approved stage can be published')
    if not stage['execution_approved']:raise RuntimeError('Stage execution was not approved')
    if json.loads((ROOT/stage['status_file']).read_text())['status']!=stage['status']:raise RuntimeError('Registry and experiment state differ')
    if git('remote','get-url','origin').stdout.strip().rstrip('/')!=REMOTE:raise RuntimeError('Unexpected publication remote')
    if git('symbolic-ref','--short','HEAD').stdout.strip()!='main':raise RuntimeError('Publication must occur on main')
    files=publication_files(a.exclude_prefix)
    print('Publication audit:',len(files),'files,',sum(r['bytes'] for r in files),'bytes')
    if a.check_only:return
    git('fetch','origin')
    remote_head=git('rev-parse','--verify','refs/remotes/origin/main',check=False)
    if remote_head.returncode==0:
        if git('merge-base','--is-ancestor',remote_head.stdout.strip(),'HEAD',check=False).returncode!=0:
            raise RuntimeError('Remote main contains unseen work; reconcile without force before publication')
    target=ROOT/'docs/publication_manifest.json'
    target.write_text(json.dumps(dict(stage=a.stage,generated_utc=datetime.now(timezone.utc).isoformat(),
                                     files=[r for r in files if r['path']!='docs/publication_manifest.json'],
                                     excludes_self=True),indent=2)+'\n')
    names=[r['path'] for r in publication_files(a.exclude_prefix)]
    # Only audited paths are staged; never stage model/data/cache or unrelated files.
    git('add','--',*names)
    # Also record tracked deletions, such as user-requested removal of proposals.
    deleted=git('ls-files','--deleted').stdout.splitlines()
    if deleted:git('add','--',*deleted)
    if git('diff','--cached','--quiet',check=False).returncode==0:
        print('No new changes to commit')
    else:
        print(git('commit','-m',a.message).stdout.strip())
    receipt=dict(stage=a.stage,commit=git('rev-parse','HEAD').stdout.strip(),remote=REMOTE,branch='main')
    result=git('push','origin','HEAD:main',check=False)
    receipt.update(push_exit_code=result.returncode,push_stdout=result.stdout,push_stderr=result.stderr,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    if result.returncode==0:
        receipt['remote_commit']=git('ls-remote','origin','refs/heads/main').stdout.split()[0]
        receipt['verified']=receipt['remote_commit']==receipt['commit']
    directory=ROOT/'runs/publication';directory.mkdir(parents=True,exist_ok=True)
    path=directory/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    path.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(receipt,indent=2,ensure_ascii=False))
    if result.returncode or not receipt.get('verified'):raise RuntimeError('Push not verified; receipt retained, never force-push')

if __name__=='__main__':main()
