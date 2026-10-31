"""Seal follow-up raw collection inputs; preserve all legacy snapshots and labels."""
import datetime
import json
from pathlib import Path

from experiments import lib as L
from experiments.followup import common as F


def main():
    F.OUT.mkdir(parents=True,exist_ok=True)
    files=['config.yaml','docs/PREREG_FOLLOWUP.md','data/metadata.json','data/validation_split.json','data/v3_pixel_sha256.json',
           'experiments/followup/common.py','experiments/followup/synth.py','experiments/followup/predict.py',
           'experiments/e2_predict.py','experiments/lib.py','src/xgd_analyzer.py','src/preprocess.py']
    inputs={p:F.sha_file(F.ROOT/p) for p in files}
    metadata={r['stem']:r for r in json.loads((F.ROOT/'data/metadata.json').read_text(encoding='utf8'))['records']}
    images={s:F.sha_file(L.XC.raw_path(L.CFG,s)) for s in sorted(L.META)}
    labels={s:F.sha_file(L.XC.label_path(L.CFG,s)) for s in sorted(L.DEV)}
    for s in L.META:
        if images[s]!=metadata[s]['raw_sha256']:raise RuntimeError('Raw image changed')
        if s in labels and labels[s]!=metadata[s]['label_sha256']:raise RuntimeError('Original TXT changed')
    weights={F.model_key(f,p,sd):F.sha_file(F.checkpoint(f,p,sd)) for f in ('dfl','v3only')
             for p in ('dfl_src12','dfl_src3_2') for sd in L.SEEDS}
    record={'inputs':inputs,'images':images,'development_original_labels':labels,'weights':weights,
            'evaluation79_GT_reads':0,'baseline_version':'XGD-1.1.0','formal_qualification':'UNQUALIFIED',
            'prior_exposure':'all development E1-E9 and evaluation metadata/distribution already observed'}
    p=F.OUT/'RAW_COLLECTION_LOCK.json'
    if p.exists():
        old=json.loads(p.read_text(encoding='utf8'))
        if old['sealed']!=record:raise RuntimeError('Raw collection lock changed; prior preserved')
    else:F.atomic_json(p,{'time':datetime.datetime.now().astimezone().isoformat(),'sealed':record})
    print('Follow-up input/source seal PASS',len(images),'raw images',len(labels),'development labels',flush=True)


if __name__=='__main__':main()
