"""Complete user-required negative controls omitted by the interrupted runner; no model/recipe changes."""
import datetime
import hashlib
import importlib.metadata as md
import json
from pathlib import Path

import cv2
import numpy as np

from experiments import lib as L
from experiments.e2_predict import Synth


def make(syn,s,kind):
    B=syn.base(s);g=B['g8'].copy()
    if kind=='R2h_removed':
        for b in B['bx']:g=L.remove_object(g,b)
        return L.bgr(g),[],{'removed_windows':len(B['bx']),'original_centres':B['centres']}
    if kind=='R2i_sham':
        pts=[]
        for j in range(2):
            p=L.free_spot(g,B['centres']+B['marks']+pts,L.rng_for('sham-background',s,j),min_dist=25,edge=14)
            if p is None:continue
            pts.append(p);g=L.remove_object(g,[p[0]-5,p[1]-5,p[0]+5,p[1]+5])
        return L.bgr(g),B['gt'],{'sham_windows':len(pts),'sham_centres':pts,'no_feasible_window':not pts}
    out=np.full_like(g,int(np.median(g)))
    for b in B['bx']:
        x0,y0,x1,y1=map(int,[max(0,b[0]-8),max(0,b[1]-8),min(B['w'],b[2]+9),min(B['h'],b[3]+9)])
        out[y0:y1,x0:x1]=g[y0:y1,x0:x1]
    return L.bgr(out),B['gt'],{'preserved_windows':len(B['bx']),'margin_px':8}


def main():
    out=L.OUT/'resume/CONTROL_COMPLETION_LOCK.json'
    source={f:L.X.sha256_file(L.ROOT/f) for f in ('experiments/extra_controls.py','experiments/e3_score.py','experiments/final_report.py',
                                                                  'experiments/e2_predict.py','experiments/lib.py','experiments/operational_timing.py')}
    if not out.exists():
        L.jdump(out,{'time':datetime.datetime.now().astimezone().isoformat(),'source':source,
                     'reason':'complete user-required removed/sham/context controls; no model/threshold/gate change',
                     'prior_E1_and_some_R1_R3_predictions_exist':True})
    elif json.loads(out.read_text(encoding='utf8'))['source']!=source:
        raise RuntimeError('Control completion source mismatch; old lock preserved')
    marker=L.OUT/'resume/CONTROL_COMPLETION_DONE.json'
    if marker.exists():return
    for prof in ('dfl_src12','dfl_src3_2'):
        syn=Synth(prof);stems=L.targets(prof)
        for sd in L.SEEDS:
            model=L.Model(prof,sd)
            for cond in ('R2h_removed','R2i_sham','R2j_object_only'):
                p=L.CACHE/model.key/f'{cond}.json';seal=p.with_suffix('.manifest.json')
                signature={'checkpoint':model.an.ckpt_sha256,'source':{f:sha for f,sha in source.items() if f not in ('experiments/e3_score.py','experiments/final_report.py','experiments/operational_timing.py')},
                           'inputs':{f:L.X.sha256_file(L.ROOT/f) for f in ('data/metadata.json','data/split.csv','data/v3_pixel_sha256.json')},
                           'packages':{k:md.version(k) for k in ('torch','ultralytics','numpy','opencv-python')},'condition':cond}
                if seal.exists() and json.loads(seal.read_text(encoding='utf8'))!=signature:raise RuntimeError('extra control cache mismatch')
                if not seal.exists():L.jdump(seal,signature)
                data=json.loads(p.read_text(encoding='utf8')) if p.exists() else {}
                for i,s in enumerate(stems):
                    if s in data:continue
                    img,gt,details=make(syn,s,cond);h,w=img.shape[:2];sc=model.scale(w,h)
                    data[s]={'boxes':model.predict(img,sc['imgsz']),'gt':gt,'resolution':[w,h],'signals':L.signals_of(img),'imgsz':sc['imgsz'],'control':details}
                    if (i+1)%20==0:L.jdump(p,data)
                L.jdump(p,data);print('extra control',prof,sd,cond,flush=True)
            del model
    L.jdump(marker,{'complete':True,'evaluation79_label_reads':0,'original_labels_modified':False})


if __name__=='__main__':main()
