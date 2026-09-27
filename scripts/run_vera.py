"""Run the approved frozen VERA transfer evaluation, with resumable per-segment logs."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ipad.common import SCENES, seed_everything, write_json
from ipad.data import frames, videos
from ipad.vera import VeraConfig, QUESTIONS, parse_response, refine_scores, segments

OUT = ROOT/'experiments/vera_ipad'
CACHE = ROOT/'cache/vera'
DATA = ROOT/'IPAD_dataset'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8*1024**2):
            h.update(block)
    return h.hexdigest()


def event(kind, **values):
    with (OUT/'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=time.time(), kind=kind, **values), ensure_ascii=False)+'\n')


def disk_guard():
    if (OUT/'disk_pause.json').exists() or (ROOT/'runs/disk_pause.json').exists():
        raise RuntimeError('Latched disk pause requires explicit user-directed resume')
    free = shutil.disk_usage(ROOT).free
    if free <= 10*1024**3:
        write_json(OUT/'disk_pause.json', dict(status='paused_low_disk', free_bytes=free, automatic_resume=False))
        raise RuntimeError('Disk reserve reached; no automatic resume')


def manifest(split):
    records = []
    for scene in SCENES:
        for video in videos(DATA, scene, split):
            fs = frames(video)
            ids = [int(f.stem) for f in fs]
            if ids != list(range(len(fs))):
                raise ValueError(f'Noncontiguous frame IDs: {video}')
            records.append(dict(scene=scene, video=video.name, length=len(fs),
                                relative_path=str(video.relative_to(ROOT)),
                                frames_sha256=[sha(f) for f in fs]))
    return records


def freeze(config):
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [ROOT/'ipad/vera.py', ROOT/'ipad/vera_models.py', ROOT/'scripts/run_vera.py',
               ROOT/'scripts/evaluate_vera.py', *sorted((OUT/'source_reference').rglob('*.py')),
               OUT/'source_reference/VERA/VERA_learner_instruct.txt']
    definition = dict(config=asdict(config), questions=QUESTIONS,
                      source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources})
    fingerprint = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    path = OUT/'frozen.json'
    if path.exists():
        if json.loads(path.read_text())['fingerprint'] != fingerprint:
            raise RuntimeError('Frozen run differs from current source/config; preserve run and create a new explicit experiment')
    else:
        write_json(path, dict(fingerprint=fingerprint, **definition))
        write_json(OUT/'test_manifest.json', manifest('testing'))
        write_json(OUT/'normal_manifest.json', manifest('training'))
        (OUT/'questions.txt').write_text(QUESTIONS)
    for name, command in [('environment.txt',[sys.executable,'-c',
         'import importlib.metadata as m; print("\\n".join(sorted({d.metadata["Name"]+"=="+d.version for d in m.distributions()})))']),
         ('gpu.txt',['nvidia-smi'])]:
        p=OUT/name
        if not p.exists(): p.write_text(subprocess.check_output(command, text=True))
    event('invocation', command=sys.argv, executable=sys.executable, fingerprint=fingerprint,
          python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda)
    return fingerprint


def load_records(path):
    result = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['center'] in result: raise ValueError('Duplicate segment in journal')
            result[row['center']] = row
    return result


def journal(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())


def run_stage(stage, diagnostic, config, fingerprint):
    from ipad.vera_models import InternVL, ImageBind
    disk_guard()
    records = json.loads((OUT/('normal_manifest.json' if diagnostic else 'test_manifest.json')).read_text())
    if diagnostic: records = [next(r for r in records if r['scene']==s) for s in SCENES]
    selected = []
    for r in records:
        # Exclusion is fixed by pre-existing dataset alignment audit, not predictions.
        if not diagnostic and r['scene']=='R02' and int(r['video']) in (12,13,14): continue
        selected.append(r)
    prefix = 'diagnostic' if diagnostic else 'inference'
    suffix = 'vlm' if stage=='vlm' else 'features'
    pending = []
    for r in selected:
        base = OUT if stage=='vlm' else CACHE/'feature_journals'
        path = base/prefix/r['scene']/f"{r['video']}.{suffix}.jsonl"
        existing = load_records(path)
        ss = segments(r['length'],config)
        if diagnostic: ss = ss[:1]
        if any(s['center'] not in existing for s in ss): pending.append((r,path,existing,ss))
    if not pending: return
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = (InternVL(CACHE/'InternVL2-8B', (OUT/'source_reference/VERA/VERA_learner_instruct.txt').read_text(),config)
             if stage=='vlm' else ImageBind(CACHE, OUT/'source_reference'))
    event('model_loaded', stage=stage, diagnostic=diagnostic, seconds=time.perf_counter()-started)
    for r, path, existing, ss in pending:
        fs=frames(ROOT/r['relative_path'])
        if len(fs)!=r['length'] or [sha(p) for p in fs]!=r['frames_sha256']:
            raise RuntimeError('Input frames changed after configuration freeze')
        for segment in ss:
            if segment['center'] in existing: continue
            disk_guard()
            start=time.perf_counter()
            row=dict(**segment, fingerprint=fingerprint, scene=r['scene'],video=r['video'])
            try:
                if stage=='vlm':
                    response=model.predict([fs[i] for i in segment['frame_ids']])
                    row['response']=response
                    row['score']=parse_response(response)
                else:
                    feature, ids=model.encode(fs,segment,config.imagebind_samples)
                    if not np.isfinite(feature).all() or np.linalg.norm(feature)==0: raise ValueError('Invalid feature')
                    row.update(feature=feature.tolist(), imagebind_frame_ids=ids)
                torch.cuda.synchronize()
                row.update(seconds=time.perf_counter()-start,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
                journal(path,row)
            except Exception:
                event('segment_failed', stage=stage, diagnostic=diagnostic, row=row, traceback=traceback.format_exc())
                raise
        print(f'{prefix} {stage} {r["scene"]}/{r["video"]} complete',flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['diagnostic','full','all'],default='all')
    a=p.parse_args();os.chdir(ROOT)
    config=VeraConfig();seed_everything(config.seed);torch.set_num_threads(8)
    disk_guard();fingerprint=freeze(config)
    try:
        write_json(OUT/'status.json',dict(status='running', phase=a.phase,started_at=time.time()))
        if a.phase in ('diagnostic','all'):
            for stage in ('vlm','features'): run_stage(stage,True,config,fingerprint)
            write_json(OUT/'diagnostic_complete.json',dict(status='diagnostic',fingerprint=fingerprint))
        if a.phase in ('full','all'):
            if not (OUT/'diagnostic_complete.json').exists(): raise RuntimeError('Normal-only diagnostic required first')
            if not (OUT/'model_inventory.json').exists(): raise RuntimeError('Model SHA256 inventory required')
            for stage in ('vlm','features'): run_stage(stage,False,config,fingerprint)
            subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_vera.py')],check=True)
        else:
            write_json(OUT/'status.json',dict(status='diagnostic',fingerprint=fingerprint))
    except BaseException:
        status='paused_low_disk' if (OUT/'disk_pause.json').exists() else 'failed'
        write_json(OUT/'status.json',dict(status=status,traceback=traceback.format_exc()))
        event('run_failed',traceback=traceback.format_exc())
        raise


if __name__=='__main__': main()
