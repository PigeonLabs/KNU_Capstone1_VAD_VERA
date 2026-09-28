"""Explicit seed wrapper for unchanged frozen deterministic runners."""
import sys,runpy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ipad.common import seed_everything
seed_everything(0)
script=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(script,run_name='__main__')
