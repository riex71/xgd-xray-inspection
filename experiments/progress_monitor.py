"""Read-only progress monitor for the existing supervisor; does not start training or scoring."""
import datetime
import json
import time
from pathlib import Path

from experiments import lib as L
from experiments.e2_predict import SHIFTS,R3_CONDS,GRID,Synth,r2_variants


def expected():
    todo=[]
    for prof in ('dfl_src12','dfl_src3_2'):
        cond={'clean','native',*(f'R1_{x}_{y}' for x,y in SHIFTS),*(f'R3_{x}' for x in R3_CONDS),
              *r2_variants(Synth(prof),True).keys(),'GRID_1.1_clean','R2h_removed','R2i_sham','R2j_object_only'}
        if prof=='dfl_src12':cond.update(f'GRID_{f:g}_{q}' for f,q in GRID)
        for sd in L.SEEDS:
            todo.extend((f'{prof}_s{sd}',c,len(L.targets(prof))) for c in sorted(cond))
    for typ in ('baseline','v3only'):
        for prof in ('dfl_src12','dfl_src3_2'):
            for sd in L.SEEDS:
                key=f'yolo_{prof[4:]}_s{sd}' if typ=='baseline' else f'v3only_{prof}_s{sd}'
                cond=('clean','native') if typ=='baseline' else ('clean','R2e_trace','R2f_free_M1','R2f_free_M2','R2g_mark_M1','R2g_mark_M2')
                todo.extend((key,c,len(L.targets(prof))) for c in cond)
    return todo


def main():
    todo=expected();out=L.OUT/'resume/PROGRESS.json';total=sum(n for _,_,n in todo)
    start=time.monotonic();first=None
    for _ in range(12*60*2):
        try:
            status=json.loads((L.OUT/'resume/STATUS.json').read_text(encoding='utf8'))
            done=complete=skipped=0;latest=None
            for key,c,n in todo:
                p=L.CACHE/key/f'{c}.json'
                if not p.exists():continue
                r=json.loads(p.read_text(encoding='utf8'));done+=min(n,len(r));skipped+=sum(x is None for x in r.values())
                complete+=int(len(r)==n)
                mt=p.stat().st_mtime
                if latest is None or mt>latest[0]:latest=(mt,key,c,len(r),n)
            if first is None:first=done
            elapsed=time.monotonic()-start;rate=(done-first)/elapsed if elapsed>90 else 0
            partial={'updated':datetime.datetime.now().astimezone().isoformat(),'supervisor':status,
                     'prediction_attempts':done,'prediction_attempts_planned':total,'condition_caches_complete':complete,
                     'condition_caches_planned':len(todo),'skipped_synthesis_attempts':skipped,
                     'recent_prediction_attempts_per_second':rate,'prediction_remaining_seconds_estimate':(total-done)/rate if rate>0 else None,
                     'estimate_scope':'predictions only; excludes CPU scoring, calibration, verification, full reproduction and reporting',
                     'latest_cache':latest,'evaluation79_label_reads':0}
            L.jdump(out,partial)
            if status.get('status') in ('COMPLETE','FAILED'):break
        except (OSError,json.JSONDecodeError):pass
        time.sleep(30)


if __name__=='__main__':main()
