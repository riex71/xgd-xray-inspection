"""Build a relative-path source ZIP allowlist; never delete research artifacts or expose secrets."""
import collections
import importlib.metadata as md
import json
import re
from pathlib import Path

from experiments import lib as L


def main():
    suspicious=[]; includes=[]; excluded=[]; personal=[]
    # absolute home-directory paths (personal account names) must not ship in text files
    # (pattern assembled from pieces so that this file does not match itself)
    home=re.compile(r"(?i)(?:\b[a-z]:[\\/]+" + "us" + r"ers[\\/]+[^\\/\s\"]+|/" + "ho" + r"me/[^/\s\"]+|/" + "Us" + r"ers/[^/\s\"]+)")
    secret=re.compile(r'(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|sk-[A-Za-z0-9]{30,})')
    for p in L.ROOT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(L.ROOT).as_posix()
        # A manifest cannot contain a stable hash of itself. Keep it alongside the allowlist.
        if rel=='outputs/experiments/resume/SUBMISSION_FILE_MANIFEST.json':continue
        omit=any(x in p.parts for x in ('__pycache__','.pytest_cache','isolated_package','isolated_env','.git','versions'))
        omit=omit or rel.startswith('outputs/controls/') or '/cache/' in rel or rel.startswith('outputs/retrain/') or p.suffix=='.log'
        omit=omit or rel.startswith(('data/clean_v3/','data/v3tr_orig/')) or p.suffix=='.cache'
        omit=omit or p.name.startswith('.env') or p.suffix.lower() in ('.pem','.key','.pfx')
        # runtime process-state files change after this manifest is written (runner STATUS / locks / progress); they are not evidence
        omit=omit or (rel.startswith('outputs/experiments/resume/') and (p.name in ('STATUS.json','PROGRESS.json') or p.name.startswith('PROCESS_LOCK')))
        if omit:
            excluded.append(rel);continue
        if p.suffix.lower() in ('.py','.md','.yaml','.yml','.json','.txt','.csv') and p.stat().st_size<8_000_000:
            content=p.read_text(encoding='utf8',errors='ignore')
            if secret.search(content):suspicious.append(rel)
            if home.search(content):personal.append(rel)
        includes.append({'path':rel,'size':p.stat().st_size,'sha256':L.X.sha256_file(p)})
    dist=md.distribution('ultralytics')
    license_files=[dist.locate_file(f) for f in dist.files if str(f).endswith('licenses/LICENSE')]
    if license_files:
        p=L.ROOT/'licenses/ULTRALYTICS_AGPL-3.0.txt';p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.write_bytes(Path(license_files[0]).read_bytes())
        if not any(x['path']==p.relative_to(L.ROOT).as_posix() for x in includes):
            includes.append({'path':p.relative_to(L.ROOT).as_posix(),'size':p.stat().st_size,'sha256':L.X.sha256_file(p)})
    report={'source_zip_preflight_pass':not suspicious and not personal,'credential_suspected_paths_only':suspicious,
            'personal_home_path_files':personal,
            'evaluation_GT_absent':all(not L.XC.label_path(L.CFG,s).exists() for s in L.XC.test_stems(L.CFG)),
            'included':includes,'excluded_paths':excluded,
            'note':'allowlist only, no zip generated or artifact deleted; data is for competition submission, unrestricted redistribution not certified',
            'requirements_versions':{k:md.version(k) for k in ('torch','torchvision','ultralytics','numpy','scipy','opencv-python','Pillow','PyYAML')},
            'normal_products':0,'physical_specimen_disjoint_certified':False}
    L.jdump(L.OUT/'resume/SUBMISSION_FILE_MANIFEST.json',report)
    if suspicious:raise RuntimeError('Credential-suspected files (paths only): '+', '.join(suspicious))
    if personal:raise RuntimeError('Absolute home-directory paths in submission text files: '+', '.join(personal))
    if not report['evaluation_GT_absent']:raise RuntimeError('Evaluation labels present in operational label folder')
    print('submission allowlist',len(includes),'files; no suspected credentials or home-directory paths; original research files preserved',flush=True)


if __name__=='__main__':main()
