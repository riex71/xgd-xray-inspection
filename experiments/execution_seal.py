"""Seal reproducibility and candidate exposure facts, without secrets or absolute personal paths."""
import datetime
import importlib.metadata as md
import json
import platform
from pathlib import Path

from experiments import lib as L


def main():
    import torch
    from experiments.resume_validation import collect
    collect()  # Validate actual completed controls and loader lists in this fresh child process.
    files={}
    for pattern in ('src/*.py','experiments/*.py','tests/*.py','docs/*.md','data/models/*.yaml','weights/*.pt',
                    'weights/baseline_yolo26s/*','data/initial/*.pt','data/raw/*.bmp','data/labels_orig/*.txt',
                    'outputs/controls/*/retrain/*/weights/last.pt','outputs/controls/*/retrain/*/args.yaml'):
        for p in L.ROOT.glob(pattern):
            if p.is_file(): files[p.relative_to(L.ROOT).as_posix()]=L.X.sha256_file(p)
    for f in ('config.yaml','requirements.txt','data/metadata.json','data/split.csv','data/validation_split.json','data/v3_pixel_sha256.json'):
        files[f]=L.X.sha256_file(L.ROOT/f)
    e1=json.loads((L.OUT/'E1_rules_x14/RESULTS.json').read_text(encoding='utf8'))
    err=max(v for f in e1['families'].values() for c in f['conditions'].values() for v in c.get('x14_check',{}).values())
    if err>1e-12: raise RuntimeError(f'Legacy independent re-score disagrees: {err}')
    record={'time':datetime.datetime.now().astimezone().isoformat(),'sha256':files,
            'environment':{'python':platform.python_version(),'os':platform.system(),'torch':torch.__version__,
                           'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                           'packages':{k:md.version(k) for k in ('ultralytics','numpy','scipy','opencv-python','PyYAML','pytest')}},
            'legacy_E1_metric_max_abs_error':err,
            'exposure':{'E1_X14_repeated_results_already_seen':True,'candidate_signal_gates_defined_before_E2_results':True,
                        'development356_repeated_adaptive_use':True,'independent_final_test':False,
                        'evaluation79_label_access_forbidden':True,'new_detector_search':False,
                        'ultralytics_final_epoch_validation_may_run_even_with_val_false':True,
                        'control_checkpoint_policy':'fixed final last.pt; no best-epoch or recipe selection from final validation',
                        'controls_are_user_authorized_comparators_not_replacement_search':True},
            'primary_calibration':'P-IND-strict: grade/reference/lambda use opposite date half only',
            'diagnostic_calibration':'P-IND-deployment-grade: grade uses whole development target data; not independent',
            'prediction_cache_policy':'reject missing/mismatched source/model/config/input seals; preserve prior results'}
    p=L.OUT/'resume/EXECUTION_SEAL.json'
    if p.exists():
        raise RuntimeError('Execution seal already exists; inspect the sealed run rather than overwrite it')
    L.jdump(p,record)
    print('sealed',len(files),'files',flush=True)


if __name__=='__main__': main()
