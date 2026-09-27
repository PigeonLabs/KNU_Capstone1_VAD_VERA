"""Recreate VERA's isolated overlay environment and pinned upstream source files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'cache/vera'
REF=ROOT/'experiments/vera_ipad/source_reference'
SOURCES={
 'vera-framework/VERA':'15b8bcb8574a977c229c577f50bfe6f06d07106e',
 'facebookresearch/ImageBind':'53680b02d7e37b19b124fa37bae4b6c98c38f5be',
 'lucazanella/lavad':'1ad46c666d1b3cfb262f3dd84769acf873285056',
}

def fetch(repo,path,target):
    url=f'https://raw.githubusercontent.com/{repo}/{SOURCES[repo]}/{path}'
    content=urllib.request.urlopen(url,timeout=60).read()
    if target.exists() and target.read_bytes()!=content:
        raise RuntimeError(f'Preserving conflicting source: {target}')
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.write_bytes(content)
    return dict(url=url,path=str(target.relative_to(ROOT)),sha256=hashlib.sha256(content).hexdigest())

def main():
    p=argparse.ArgumentParser();p.add_argument('--install',action='store_true');a=p.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True)
    python=CACHE/'venv/bin/python'
    if a.install:
        if not python.exists():subprocess.run(['uv','venv','--python',str(ROOT/'.venv/bin/python'),str(CACHE/'venv')],check=True)
        sites=json.loads(subprocess.check_output([str(python),'-c','import site,json;print(json.dumps(site.getsitepackages()))'],text=True))
        base=json.loads(subprocess.check_output([str(ROOT/'.venv/bin/python'),'-c','import site,json;print(json.dumps(site.getsitepackages()))'],text=True))
        Path(sites[0],'ipad_base.pth').write_text('\n'.join(base)+'\n')
        lock=ROOT/'experiments/vera_ipad/requirements-vera.lock.txt'
        ordinary=[x for x in lock.read_text().splitlines() if not x.startswith('torchaudio==')]
        subprocess.run(['uv','pip','install','--python',str(python),'--no-deps',*ordinary],check=True)
        subprocess.run(['uv','pip','install','--python',str(python),'--no-deps','torchaudio==2.11.0','--index-url','https://download.pytorch.org/whl/cu128'],check=True)
    inventory=[]
    for repo,revision in SOURCES.items():
        if repo.endswith('/VERA'):
            paths=['VERA_learner_instruct.txt','VERA_optimizer_instruct.txt','generate_initial_score.py','compute_auc.py','training.py','README.md']
            for path in paths:inventory.append(fetch(repo,path,REF/'VERA'/path))
        elif repo.endswith('/ImageBind'):
            req=urllib.request.Request(f'https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1',headers={'User-Agent':'VERA-reproduction'})
            tree=json.load(urllib.request.urlopen(req,timeout=60))
            for entry in tree['tree']:
                path=entry['path']
                if entry['type']=='blob' and (path.startswith('imagebind/') or path=='LICENSE'):
                    inventory.append(fetch(repo,path,CACHE/'ImageBind'/path))
        else:
            for path in ['libs/ImageBind/imagebind/data.py','src/models/create_index.py']:
                inventory.append(fetch(repo,path,REF/('lavad_'+Path(path).name)))
    target=ROOT/'experiments/vera_ipad/source_inventory.json'
    target.write_text(json.dumps(dict(revisions=SOURCES,files=inventory),indent=2)+'\n')
    print(f'Pinned source inventory: {target}')

if __name__=='__main__':main()
