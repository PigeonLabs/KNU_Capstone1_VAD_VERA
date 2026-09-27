"""Verify failure evidence and the boundary between public analysis and local assets."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('publisher',ROOT/'scripts/publish_vera_stage.py')
publisher=importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)

@pytest.mark.parametrize('exit_code',[0,7])
def test_command_attempts_preserve_outputs_and_failures(tmp_path,exit_code):
    scripts=tmp_path/'scripts';scripts.mkdir()
    shutil.copyfile(ROOT/'scripts/logged_command.py',scripts/'logged_command.py')
    command=[sys.executable,str(scripts/'logged_command.py'),'--stage','test_stage','--step','check','--',
             sys.executable,'-c',f'import sys; print("result"); print("detail",file=sys.stderr); sys.exit({exit_code})']
    for _ in range(2):
        completed=subprocess.run(command,capture_output=True,text=True)
        assert completed.returncode==exit_code
    attempts=list((tmp_path/'experiments/test_stage/execution/check').iterdir())
    assert len(attempts)==2
    for attempt in attempts:
        metadata=json.loads((attempt/'command.json').read_text())
        assert metadata['exit_code']==exit_code
        assert metadata['status']==('complete' if exit_code==0 else 'failed')
        assert metadata['started_kst'].endswith('+09:00')
        assert metadata['finished_kst'].endswith('+09:00')
        assert metadata['command']==command[7:]
        log=(attempt/'output.log').read_text()
        assert 'result' in log and 'detail' in log

@pytest.mark.parametrize('name,data,error',[
    ('experiments/weights.pth',b'tensor','Excluded file type'),
    ('experiments/cache/features.json',b'{}','Local-only path'),
    ('experiments/notes.json',b'abc\x00def','Binary content'),
    ('experiments/log.txt',('ghp_'+'a'*35).encode(),'Potential secret'),
])
def test_publication_rejects_local_or_secret_content(tmp_path,monkeypatch,name,data,error):
    path=tmp_path/name;path.parent.mkdir(parents=True);path.write_bytes(data)
    monkeypatch.setattr(publisher,'ROOT',tmp_path)
    monkeypatch.setattr(publisher,'git',lambda *a,**kw:SimpleNamespace(stdout=name+'\n'))
    with pytest.raises(RuntimeError,match=error):publisher.publication_files()

def test_unapproved_stage_never_reaches_git(tmp_path,monkeypatch):
    (tmp_path/'experiments').mkdir()
    (tmp_path/'experiments/stages.json').write_text(json.dumps({'stage2':{'status':'proposed','execution_approved':False}}))
    monkeypatch.setattr(publisher,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['publish','--stage','stage2','--message','test'])
    def forbid_git(*a,**kw):raise AssertionError('Git must not be invoked for unapproved execution')
    monkeypatch.setattr(publisher,'git',forbid_git)
    with pytest.raises(RuntimeError,match='terminal approved'):publisher.main()
