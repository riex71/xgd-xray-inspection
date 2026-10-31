"""Status-only ledger; never keep prediction files open or inspect eval79 GT."""
import datetime,json,time
from experiments.followup import common as F

def main():
    previous=None
    while True:
        p=F.OUT/'STATUS.json'
        try:state=json.loads(p.read_text(encoding='utf8'))
        except (OSError,json.JSONDecodeError):time.sleep(5);continue
        key=tuple(state.get(k) for k in ('stage','candidate','family','profile','condition','model','view','status'))
        if key!=previous:
            previous=key
            with (F.OUT/'EXECUTION_EXPOSURE_LEDGER.jsonl').open('a',encoding='utf8') as f:
                record={'observer_time':datetime.datetime.now().astimezone().isoformat(),'state':state,
                    'source_lock_sha256':F.sha_file(F.OUT/'ANALYSIS_EXECUTION_LOCK.json'),
                    'observation':'status event; scores not read; missed intermediate transitions possible',
                    'evaluation79_GT_reads':0,'evidence':'adaptive internal development execution'}
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
        if state.get('status')=='COMPLETE' and (F.OUT/'FOLLOWUP_COMPLETED.json').exists():
            # The resumed parent script has an old completion label. Preserve it, then reconcile factual status.
            F.atomic_json(F.OUT/'RESUMED_PARENT_LAST_STATUS.json',state)
            final=json.loads((F.OUT/'FOLLOWUP_COMPLETED.json').read_text(encoding='utf8'))
            F.status(stage='FOLLOWUP_COMPLETED',status='COMPLETE',final_deployment_generated=True,
                     evaluation79_GT_reads=0,pipeline_signature=final['pipeline_signature'],
                     submission_status=json.loads((F.OUT/'PACKAGE_READINESS.json').read_text(encoding='utf8'))['status'])
            return
        if state.get('status')=='FAILED':return
        time.sleep(5)

if __name__=='__main__':main()
