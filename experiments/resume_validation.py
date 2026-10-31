"""Resume the already-authorized, preregistered validation after the existing controls job.

Never restarts candidate/model search, never reads evaluation79 GT, and stops with a
nonzero status on any mandatory failure. State/logs remain inside this model folder.
"""
import csv
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/experiments/resume'
OUT.mkdir(parents=True,exist_ok=True)


def state(stage,status='RUNNING',**kw):
    p=OUT/'STATUS.json'
    old=json.loads(p.read_text(encoding='utf8')) if p.exists() else {}
    old.update({'stage':stage,'status':status,'updated':datetime.datetime.now().astimezone().isoformat(),'evaluation79_label_reads':0,**kw})
    p.write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf8')
    print(stage,status,kw,flush=True)


def call(name,args):
    state(name)
    with open(OUT/f'{name}.log','a',encoding='utf8') as log:
        p=subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                         env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    if p.returncode:
        raise RuntimeError(f'{name} exited {p.returncode}; see {name}.log')
    state(name,'STEP_COMPLETE')


def completed(folder):
    p=folder/'results.csv'
    if not p.exists() or not (folder/'weights/last.pt').exists(): return False
    with p.open(encoding='utf8') as f: rows=list(csv.DictReader(f))
    return len(rows)>=40 and int(float(rows[-1]['epoch']))==40


def collect():
    import yaml
    from src import xgd_config as XC
    from src import train_dfl as T
    cfg=XC.load(); records=[]
    archive=ROOT.parent/'00_과거버전_보관'/'4번 분석 모델'/'outputs/validation/runs/t2'
    for prof,yp in (('dfl_src12','yolo_src12'),('dfl_src3_2','yolo_src3_2')):
        fold=cfg['profiles'][prof]['training_split']['held_fold']
        for sd in (17,29,43):
            folder=archive/f'b1_sc4_orig_f{fold}' if sd==17 else ROOT/f'outputs/controls/baseline/retrain/{prof}_s{sd}'
            src=folder/'weights/last.pt'
            if sd!=17 and not completed(folder): raise RuntimeError(f'incomplete {folder}')
            args=yaml.safe_load((folder/'args.yaml').read_text(encoding='utf8'))
            expected=T.train_args(cfg,sd)
            diff={k:[v,args.get(k)] for k,v in expected.items() if k!='verbose' and v!=args.get(k)}
            if diff: raise RuntimeError(f'Not a matched baseline: {prof} {sd}: {diff}')
            dest=XC.checkpoint(cfg,yp,sd); dest.parent.mkdir(parents=True,exist_ok=True)
            sh=hashlib.sha256(src.read_bytes()).hexdigest()
            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=sh:
                raise RuntimeError(f'Refuse to overwrite different weight {dest}')
            if not dest.exists(): shutil.copy2(src,dest)
            records.append({'profile':yp,'seed':sd,'file':dest.relative_to(ROOT).as_posix(),'sha256':sh,
                            'source_run':str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else f'archived historical b1_sc4_orig_f{fold}',
                            'recipe_matched_keys':len(expected)-1,'checkpoint_policy':'last, not best',
                            'training_args':{k:args.get(k) for k in expected},
                            'historical_seed17_environment_equivalence':'not independently byte-proven'})
    (ROOT/'weights/baseline_yolo26s/MANIFEST.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
    lineage=[]
    for name,seeds in (('baseline',(29,43)),('v3only',(17,29,43))):
        for prof in ('dfl_src12','dfl_src3_2'):
            expected_train,expected_held=T.stems(cfg,prof)
            for sd in seeds:
                base=ROOT/f'outputs/controls/{name}/retrain';folder=base/f'{prof}_s{sd}'
                if not completed(folder):
                    continue  # active job is not called completed; supervisor waits before final collect.
                for kind,expected_stems in (('train',expected_train),('held',expected_held)):
                    paths=(base/f'lists/{prof}_s{sd}_{kind}.txt').read_text(encoding='utf8').splitlines()
                    if [Path(p).stem for p in paths]!=expected_stems:raise RuntimeError(f'Wrong loader lineage {name}/{prof}/{sd}/{kind}')
                    if any(not Path(p).exists() for p in paths):raise RuntimeError('Loader path does not exist')
                args=yaml.safe_load((folder/'args.yaml').read_text(encoding='utf8'))
                diff={k:[v,args.get(k)] for k,v in T.train_args(cfg,sd).items() if k!='verbose' and v!=args.get(k)}
                if diff:raise RuntimeError(f'Wrong control recipe {name}/{prof}/{sd}: {diff}')
                lineage.append({'control':name,'profile':prof,'seed':sd,'train_images':len(expected_train),'held_images':len(expected_held),
                                'input':'v3tr_orig' if name=='baseline' else 'clean_v3','loader_order_exact':True,
                                'training_list_sha256':hashlib.sha256((base/f'lists/{prof}_s{sd}_train.txt').read_bytes()).hexdigest(),
                                'config_sha256':hashlib.sha256((ROOT/f'outputs/controls/{name}/config.yaml').read_bytes()).hexdigest()})
    (OUT/'CONTROL_LINEAGE.json').write_text(json.dumps(lineage,ensure_ascii=False,indent=2),encoding='utf8')


def main():
    lock=OUT/'PROCESS_LOCK.json'
    # exclusive creation prevents two resume supervisors from consuming the GPU queue simultaneously
    try:
        with lock.open('x',encoding='utf8') as f: json.dump({'pid':os.getpid(),'start':datetime.datetime.now().isoformat()},f)
    except FileExistsError:
        raise SystemExit('Resume process lock exists. Check STATUS/log and live PID; do not start a duplicate.')
    try:
        call('tests',['-m','pytest','-q','tests','-p','no:cacheprovider'])
        call('E1_rule_rescore',['-m','experiments.e1_rules_on_x14'])
        folders=[ROOT/f'outputs/controls/{name}/retrain/{p}_s{s}' for name,seeds in [('baseline',(29,43)),('v3only',(17,29,43))]
                 for p in ('dfl_src12','dfl_src3_2') for s in seeds]
        start=time.monotonic()
        while not all(completed(p) for p in folders):
            remaining=[p.relative_to(ROOT).as_posix() for p in folders if not completed(p)]
            state('WAIT_EXISTING_CONTROL_TRAINING',remaining=remaining)
            if time.monotonic()-start>3*3600: raise RuntimeError('Controls still incomplete after 3h; no restart attempted')
            time.sleep(30)
        collect()
        # Seals bind recipes, source, runtime, initial/final weights and all actual input/label bytes.
        call('execution_seal',['-m','experiments.execution_seal'])
        call('predictions',['-m','experiments.e2_predict','--groups','clean','r1','r2','r2b','r3','grid','c2','native','baseline','v3only'])
        call('score',['-m','experiments.e3_score'])
        call('data_error_audit',['-m','experiments.e4_audit'])
        call('deployment_calibrate_12',['-m','src.xgd_analyzer','calibrate','--profile','dfl_src12'])
        call('deployment_calibrate_3_2',['-m','src.xgd_analyzer','calibrate','--profile','dfl_src3_2'])
        call('verify',['-m','src.verify'])
        call('predict79',['-m','src.predict_test'])
        call('isolated_reproduction',['-m','experiments.isolated_reproduction'])
        call('final_report',['-m','experiments.final_report'])
        state('FINISHED','COMPLETE')
    except BaseException as e:
        state('STOPPED','FAILED',error=str(e))
        raise
    finally:
        # Keep provenance of the original process lock, even after successful/failed completion.
        record=json.loads(lock.read_text(encoding='utf8'));record['finished']=datetime.datetime.now().isoformat()
        lock.write_text(json.dumps(record,indent=2),encoding='utf8')


if __name__=='__main__': main()
