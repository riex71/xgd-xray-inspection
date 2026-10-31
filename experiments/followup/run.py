"""Single GPU queue for the new follow-up. No duplicate runner, no legacy research reruns."""
import datetime
import json
import os
import subprocess
import sys
import time

from experiments.followup import common as F,synth as S


def call(name,args):
    F.status(stage=name,status='RUNNING',evaluation79_GT_reads=0)
    with (F.OUT/f'{name}.log').open('a',encoding='utf8') as log:
        result=subprocess.run([sys.executable,'-u',*args],cwd=F.ROOT,stdout=log,stderr=subprocess.STDOUT,
                              env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    if result.returncode:raise RuntimeError(f'{name} exit {result.returncode}; inspect preserved log')


def main():
    F.OUT.mkdir(parents=True,exist_ok=True);p=F.OUT/'PROCESS_LOCK.json'
    with p.open('x',encoding='utf8') as f:json.dump({'pid':os.getpid(),'start':datetime.datetime.now().astimezone().isoformat()},f)
    try:
        call('tests',['-m','pytest','-q','tests','-p','no:cacheprovider'])
        call('preflight',['-m','experiments.followup.preflight'])
        call('synth',['-m','experiments.followup.synth'])
        call('dfl_clean',['-m','experiments.followup.predict','--family','dfl','--views',*F.VIEWS,'--conds','dev_clean'])
        objects=[c for c in S.CONDS if not c.startswith('STRESS_')]
        call('dfl_objects',['-m','experiments.followup.predict','--family','dfl','--views',*F.VIEWS,'--conds',*objects])
        stress=[c for c in S.CONDS if c.startswith('STRESS_')]
        call('dfl_stress',['-m','experiments.followup.predict','--family','dfl','--views','base','--conds',*stress])
        call('v3only_raw',['-m','experiments.followup.predict','--family','v3only','--views','base','--conds','dev_clean','synth'])
        # Subsequent CPU analysis code must be present and its prospective seal published before training.
        # This guard prevents a partially implemented follow-up from silently choosing a model.
        F.status(stage='AWAIT_EXECUTION_IMPLEMENTATION',status='RUNNING',evaluation79_GT_reads=0)
        while not (F.OUT/'ANALYSIS_IMPLEMENTATION_READY.json').exists():time.sleep(30)
        call('cheap_comparison',['-m','experiments.followup.analyze','--stage','screen'])
        call('trace05_training',['-m','experiments.followup.train_trace05'])
        call('trace05_raw',['-m','experiments.followup.predict','--family','trace05','--views','base','--conds','dev_clean','synth'])
        call('final_comparison',['-m','experiments.followup.analyze','--stage','final'])
        F.status(stage='RESULTS_READY_FOR_SELECTION',status='COMPLETE',evaluation79_GT_reads=0,
                 final_deployment_generated=False,reason='Selection/recalibration qualification is a separate required stage')
    except BaseException as e:
        F.status(stage='STOPPED',status='FAILED',error=str(e),evaluation79_GT_reads=0);raise


if __name__=='__main__':main()
