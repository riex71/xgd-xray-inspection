"""Seal CPU decision procedures and trace05 provenance before follow-up comparisons are opened."""
import argparse,datetime,importlib.metadata as md,json
from experiments import lib as L
from experiments.followup import common as F,policy as P,analyze as A
from src import train_dfl as T

LOCK=F.OUT/'ANALYSIS_EXECUTION_LOCK.json'
SOURCES=('experiments/followup/analyze.py','experiments/followup/policy.py','experiments/followup/selection.py',
 'experiments/followup/complete.py','experiments/followup/diagnostics.py','experiments/followup/execution_lock.py',
 'experiments/followup/train_trace05.py','src/followup_analyzer.py','src/followup_safety.py',
 'experiments/followup/report.py','experiments/followup/fresh_check.py','experiments/followup/reproduction.py',
 'experiments/followup/install_isolated.py',
 'experiments/followup/distribution_probe.py',
 'docs/FOLLOWUP_SELECTION_CONTRACT.md')

def sources():return {p:F.sha_file(F.ROOT/p) for p in SOURCES}

def verify():
    lock=json.loads(LOCK.read_text(encoding='utf8'))
    if lock['sources']!=sources():raise RuntimeError('CPU analysis/source lock mismatch; no decision execution allowed')
    if lock.get('high_group_audit_sha256')!=F.sha_file(F.ROOT/'outputs/experiments/E4_audit/RESULTS.json'):
        raise RuntimeError('Frozen high-confidence group audit changed')
    from experiments.followup.preflight import main as input_preflight
    input_preflight()
    return lock

def seal():
    if LOCK.exists():return verify()
    if any((F.OUT/p).exists() for p in ('SCREEN_COMPARISONS.json','FINAL_COMPARISONS.json','SELECTION_LOCK.json')):
        raise RuntimeError('Cannot claim prospective lock after a new comparison result exists')
    from experiments.followup.train_trace05 import prepare
    prepare()
    train={'initial_checkpoint_sha256':F.sha_file(F.ROOT/L.CFG['train']['init_weights']),
        'model_yaml_sha256':F.sha_file(F.ROOT/L.CFG['train']['model']),
        'training_data_manifest_sha256':F.sha_file(F.OUT/'train/trace05/DATA_MANIFEST.json'),
        'source_sha256':F.sha_file(F.ROOT/'src/train_dfl.py'),'runner_sha256':F.sha_file(F.ROOT/'experiments/followup/train_trace05.py'),
        'configuration':{k:v for k,v in L.CFG.items() if not k.startswith('_')},
        'seed_training_arguments':{str(s):T.train_args(L.CFG,s) for s in L.SEEDS},
        'train_stems':{p:L.XC.profile_training_stems(L.CFG,p) for p in A.PROFILES},
        'held_final_Ultralytics_validation':'library runs final validation despite val=False; last checkpoint only, no best-epoch or recipe selection',
        'new_training_count':6,'repeated_images_not_independent':True}
    F.atomic_json(F.OUT/'train/trace05/TRAIN_LOCK.json',train)
    lock={'time':datetime.datetime.now().astimezone().isoformat(),'sources':sources(),
          'high_group_audit_sha256':F.sha_file(F.ROOT/'outputs/experiments/E4_audit/RESULTS.json'),
          'candidates':P.candidates(),'raw_collection_lock_sha256':F.sha_file(F.OUT/'RAW_COLLECTION_LOCK.json'),
          'trace05_train_lock_sha256':F.sha_file(F.OUT/'train/trace05/TRAIN_LOCK.json'),
          'group_date_splits':{p:{'parts':A.splits(p)[0],'scope':A.splits(p)[1]} for p in A.PROFILES},
          'packages':{k:md.version(k) for k in ('torch','ultralytics','numpy','scipy','opencv-python')},
          'prior_exposure':'XGD-1.1 E1-E9 and eval79 metadata/decisions; no eval79 GT, no follow-up comparison yet',
          'evidence':'prospective implementation/selection contract inside adaptive internal research'}
    F.atomic_json(LOCK,lock)
    return lock

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seal',action='store_true');a=ap.parse_args()
    print('execution seal',seal()['time'] if a.seal else verify()['time'])
