"""One registered trace-strength intervention, same clean COCO initialization/recipe/t2/3seeds."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from experiments import lib as L
from experiments.followup import common as F
from src import preprocess as P,train_dfl as T


def prepare():
    root=L.ROOT/'data/followup_trace05';records={}
    metadata={r['stem']:r for r in json.loads((F.ROOT/'data/metadata.json').read_text(encoding='utf8'))['records']}
    cfg=L.CFG['preprocess'];lo,hi=cfg['train_marks_per_image']
    for s in sorted(L.DEV):
        r=metadata[s];g=L.gray(L.clean_img(s));rng=np.random.default_rng(int(r['raw_sha256'][:8],16))
        spots=P.free_spots(g,P._avoid(s,g),int(rng.integers(lo,hi+1)),rng)
        choose=L.rng_for(r['raw_sha256'],'trace05');kept=[]
        for x,y in spots:
            marked=P.draw_mark(g,x,y,rng) # consume identical mark colour RNG even when this mark is dropped
            retain=bool(choose.random()<.5);kept.append(retain)
            if retain:g=L.gray(P.clean_v3(marked))
        p=root/'images'/f'{s}.png';lab=root/'labels'/f'{s}.txt'
        if p.exists():
            if P.pixel_sha256(L.gray(P.read_color(p)))!=P.pixel_sha256(g):raise RuntimeError('trace05 existing input changed')
        else:P.write_image(p,L.bgr(g))
        lab.parent.mkdir(parents=True,exist_ok=True);source=L.XC.label_path(L.CFG,s).read_bytes()
        if lab.exists() and lab.read_bytes()!=source:raise RuntimeError('derived TXT changed')
        if not lab.exists():lab.write_bytes(source)
        records[s]={'positions':spots,'retained':kept,'image_sha256':F.sha_file(p),'pixel_sha256':P.pixel_sha256(g),
                    'official_TXT_sha256':F.sha_file(lab),'raw_sha256':r['raw_sha256']}
    F.publish(F.OUT/'train/trace05/DATA_MANIFEST.json',records,{'preprocess':F.sha_file(F.ROOT/'src/preprocess.py'),
              'runner':F.sha_file(Path(__file__)),'original_pixel_seal':F.sha_file(F.ROOT/'data/v3_pixel_sha256.json')})
    return records


def main():
    prepare()
    lock=json.loads((F.OUT/'train/trace05/TRAIN_LOCK.json').read_text(encoding='utf8'))
    if lock['runner_sha256']!=F.sha_file(Path(__file__)) or lock['source_sha256']!=F.sha_file(F.ROOT/'src/train_dfl.py'):
        raise RuntimeError('Trace05 source changed after registration')
    if lock['training_data_manifest_sha256']!=F.sha_file(F.OUT/'train/trace05/DATA_MANIFEST.json'):
        raise RuntimeError('Trace05 training input lineage mismatch')
    if lock['initial_checkpoint_sha256']!=F.sha_file(F.ROOT/L.CFG['train']['init_weights']):raise RuntimeError('Trace05 initialization changed')
    cfg=copy.deepcopy({k:v for k,v in L.CFG.items() if not k.startswith('_')})
    cfg['train']['train_input']='followup_trace05';cfg['paths']['artifacts']='outputs/followup/train/trace05'
    p=F.OUT/'train/trace05/config.yaml';payload=yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False)
    if p.exists() and p.read_text(encoding='utf8')!=payload:raise RuntimeError('trace05 config changed')
    if not p.exists():p.write_text(payload,encoding='utf8')
    cfg=L.XC.load(p)
    for prof in ('dfl_src12','dfl_src3_2'):
        for sd in L.SEEDS:
            dest=cfg['_artifacts']/f'retrain/{prof}_s{sd}'
            if (dest/'weights/last.pt').exists():
                rows=list(csv.DictReader((dest/'results.csv').open(encoding='utf8')))
                if not rows or int(float(rows[-1]['epoch']))!=40:raise RuntimeError('Incomplete training preserved; not called complete')
                continue
            F.status(stage='train_trace05',status='RUNNING',profile=prof,seed=sd,evaluation79_GT_reads=0)
            T.train(cfg,prof,sd)
    print('trace05 registered six training runs complete',flush=True)


if __name__=='__main__':main()
