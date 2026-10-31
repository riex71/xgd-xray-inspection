"""Installer-only Windows temporary-directory workaround; no host site-packages imports.

Some managed Windows ACLs make tempfile.mkdir(mode=0o700) unreadable immediately.
Use ordinary mkdir in the verified workspace temp root, retaining unique names.
This shim runs only during pip installation and is absent from inference/tests.
"""
import os,sys,tempfile,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
TMP=ROOT/'outputs/followup/reproduction/install_tmp'
TMP.mkdir(parents=True,exist_ok=True)
def mkdtemp(suffix=None,prefix=None,dir=None):
    p=TMP/((prefix or 'pip-')+uuid.uuid4().hex+(suffix or ''))
    if not p.resolve().is_relative_to(TMP.resolve()):raise RuntimeError('installer tmp escaped workspace')
    p.mkdir();return str(p)
tempfile.mkdtemp=mkdtemp
from pip._internal.cli.main import main
raise SystemExit(main(sys.argv[1:]))
