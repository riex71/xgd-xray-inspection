"""Standalone reproduction in a separate workspace copy and new venv (shared pinned site packages).
No eval79 GT, no model adoption, no data/weight overwrite of the parent package.
"""
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from experiments import lib as L


def copy_tree(src,dst):
    if not src.exists():return
    shutil.copytree(src,dst,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.cache','*.tmp'))


def run(py,cwd,name,args):
    log=L.OUT/'resume'/f'reproduction_{name}.log'
    with log.open('a',encoding='utf8') as f:
        r=subprocess.run([str(py),'-u',*args],cwd=cwd,stdout=f,stderr=subprocess.STDOUT,
                         env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONPATH':'','PYTHONNOUSERSITE':'1'})
    if r.returncode:raise RuntimeError(f'{name} failed {r.returncode}; see {log.name}')


def main():
    dst=L.ROOT/'outputs/isolated_package'
    if dst.exists():raise RuntimeError('Isolated reproduction directory already exists: inspect, do not overwrite previous evidence')
    dst.mkdir(parents=True)
    for d in ('src','tools','weights'):copy_tree(L.ROOT/d,dst/d)
    for d in ('raw','labels_orig','models','initial'):copy_tree(L.ROOT/'data'/d,dst/'data'/d)
    for p in (L.ROOT/'data').glob('*.json'): (dst/'data').mkdir(exist_ok=True);shutil.copy2(p,dst/'data'/p.name)
    shutil.copy2(L.ROOT/'data/split.csv',dst/'data/split.csv')
    for p in ('config.yaml','requirements.txt','run_all.py'):shutil.copy2(L.ROOT/p,dst/p)
    copy_tree(L.ROOT/'outputs/validation/reference',dst/'outputs/validation/reference')
    env=L.ROOT/'outputs/isolated_env'
    subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(env)],check=True)
    py=env/'Scripts/python.exe' if os.name=='nt' else env/'bin/python'
    run(py,dst,'fixed_weights',['run_all.py'])
    verify=json.loads((dst/'outputs/validation/VERIFY_REPORT.json').read_text(encoding='utf8'))
    if not verify['VERIFY_PASS']:raise RuntimeError('isolated verification failed')
    before=json.loads((L.ROOT/'outputs/test_predictions.json').read_text(encoding='utf8'))
    after=json.loads((dst/'outputs/test_predictions.json').read_text(encoding='utf8'))
    if before!=after:raise RuntimeError('isolated default predictions/decisions differ from parent current execution')
    # Separate full re-training is user-authorized reproduction, not a new detector research arm.
    run(py,dst,'full_retrain',['run_all.py','full'])
    cfg=L.XC.load(dst/'outputs/full_run/config_full.yaml');cfg['_root']=dst;cfg['_artifacts']=dst/'outputs/full_run'
    cfg['paths']['artifacts']='outputs/full_run'
    per={}
    for prof in ('dfl_src12','dfl_src3_2'):
        per[prof]={}
        for sd in L.SEEDS:
            model=L.Model(prof,sd,cfg=cfg,key=f'isolated_retrain_{prof}_s{sd}')
            boxes={}
            for s in L.targets(prof):
                w,h=L.size_of(s);sc=model.scale(w,h);boxes[s]=model.predict(L.clean_img(s),sc['imgsz'])
            gt={s:L.gt(s) for s in boxes};rec={s:L.image_eval(b,gt[s],*L.size_of(s)) for s,b in boxes.items()}
            per[prof][sd]=L.detection_metrics(rec,boxes,gt)
            del model
    result={'time':datetime.datetime.now().astimezone().isoformat(),'isolated_default_VERIFY_PASS':True,
            'isolated_default_predictions_decisions_equal':True,'venv':'new venv, shared pinned system site-packages',
            'fresh_dependency_wheel_install_verified':False,'new_OS_gpu_environment_verified':False,
            'full_retrain_executed':True,'retrain_profiles_seeds':per,'checkpoint_policy':'final last.pt, 40 epochs',
            'submission_weights_replaced':False,'evaluation79_GT_used':False,
            'claim':'standalone same-hardware reproduction; cross-environment numerical guarantees not established'}
    L.jdump(L.OUT/'resume/REPRODUCTION_REPORT.json',result)
    print('isolated fixed-weight and full retraining reproduction completed',flush=True)


if __name__=='__main__':main()
