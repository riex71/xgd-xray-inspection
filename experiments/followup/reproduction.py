"""Fresh wheel installation, never reuse host site-packages/PYTHONPATH as proof of isolation."""
import argparse,datetime,email,hashlib,json,os,shutil,subprocess,sys,venv,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/followup/reproduction'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def wheel_inventory():
    out=OUT/'wheelhouse';out.mkdir(parents=True,exist_ok=True);rows=[]
    cache=Path(os.environ.get('LOCALAPPDATA',''))/'pip/cache/http-v2'
    for p in cache.rglob('*.body'):
        try:
            with p.open('rb') as f:
                if f.read(4)!=b'PK\x03\x04':continue
            with zipfile.ZipFile(p) as z:
                names=z.namelist();wheel=next(n for n in names if n.endswith('.dist-info/WHEEL'))
                md=email.message_from_bytes(z.read(wheel.rsplit('/',1)[0]+'/METADATA'))
                wm=email.message_from_bytes(z.read(wheel));tags=wm.get_all('Tag')
                pieces=[t.split('-') for t in tags]
                if len({p[1] for p in pieces})==1 and len({p[2] for p in pieces})==1:
                    tag='.'.join(dict.fromkeys(p[0] for p in pieces))+'-'+pieces[0][1]+'-'+pieces[0][2]
                else:tag=tags[0]
                name=md['Name'].replace('-','_');version=md['Version']
                dest=out/f'{name}-{version}-{tag}.whl'
                h=sha(p)
                if dest.exists() and sha(dest)!=h:continue
                if not dest.exists():shutil.copyfile(p,dest)
                rows.append({'wheel':dest.name,'sha256':h,'bytes':dest.stat().st_size})
        except (OSError,StopIteration,zipfile.BadZipFile,KeyError,TypeError):continue
    (OUT/'WHEEL_INVENTORY.json').write_text(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),
        'read_only_source':'local pip HTTP cache; wheel bodies validated as ZIP/WHEEL/METADATA','wheels':rows},indent=1),encoding='utf8')
    return out

def install():
    OUT.mkdir(parents=True,exist_ok=True);wheelhouse=wheel_inventory();v=OUT/'fresh_venv'
    py=v/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    # The managed Windows TEMP has denied child ensurepip writes in this session.
    # A fresh workspace temporary directory fixes the installation harness, not package imports.
    tmp=OUT/'install_tmp';tmp.mkdir(parents=True,exist_ok=True)
    os.environ['TEMP']=str(tmp);os.environ['TMP']=str(tmp)
    pip_ok=py.exists() and subprocess.run([str(py),'-m','pip','--version'],capture_output=True).returncode==0
    if not pip_ok:
        venv.EnvBuilder(with_pip=False,system_site_packages=False).create(v)
        bundled=next((Path(sys.base_prefix)/'Lib/ensurepip/_bundled').glob('pip-*.whl'))
        site=v/'Lib/site-packages'
        # Bootstrap pip from Python's official bundled wheel, never from existing host site-packages.
        with zipfile.ZipFile(bundled) as z:
            for name in z.namelist():
                if not (site/name).resolve().is_relative_to(site.resolve()):raise RuntimeError('unsafe bundled wheel path')
            z.extractall(site)
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
    # Offline wheel installation first. A failure is preserved, not silently substituted with the host environment.
    with (OUT/'INSTALL.log').open('a',encoding='utf8') as log:
        r=subprocess.run([str(py),str(ROOT/'experiments/followup/install_isolated.py'),'install','--no-index','--find-links',str(wheelhouse),'-r',str(ROOT/'requirements.txt')],
                         cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    report={'time':datetime.datetime.now().astimezone().isoformat(),'exit_code':r.returncode,
        'requirements_sha256':sha(ROOT/'requirements.txt'),'system_site_packages':False,'PYTHONPATH_removed':True,
        'status':'INSTALLED_NOT_YET_REPRODUCED' if r.returncode==0 else 'INSTALLATION_BLOCKED_MISSING_LOCAL_WHEEL',
        'environment':'outputs/followup/reproduction/fresh_venv','host_site_packages_used':False,
        'pip_bootstrap':'official bundled wheel; installer-only ordinary-mkdir temporary directory workaround'}
    (OUT/'INSTALL_STATUS.json').write_text(json.dumps(report,indent=1),encoding='utf8')
    print(json.dumps(report),flush=True)
    if r.returncode:raise SystemExit(r.returncode)

if __name__=='__main__':install()
