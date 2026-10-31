"""Synthetic test images for the follow-up, generated once and reused by every model / view.

The generators and RNG streams are the E2 ones (experiments/e2_predict.Synth, imported unchanged), so the images are the same synthetic
cases E2/E3 scored. Images are stored as PNG with a completion manifest (meta.json + meta.done.json: GT, special points, quality
signals, PNG hashes).

python -m experiments.followup.synth [--conds ...]
"""
import argparse
from functools import lru_cache

import cv2

from experiments import e2_predict as E2
from experiments import lib as L
from experiments.followup import common as F

CONDS = ["R2f_free_M1", "R2f_free_M2", "R2a_move_M1_0", "R2a_move_M2_0", "R2a_same_M1_0", "R2a_same_M2_0",
         "R2c_context_M1", "R2c_context_M2", "R2g_mark_M1", "R2g_mark_M2", "R2e_trace"]
STRESS=['blur04','blur08','resample085','resample07','noise2','noise4','composition',*E2.R3_CONDS]
CONDS+=['STRESS_'+c for c in STRESS]+['R2d_surround','R2h_removed','R2i_sham','R2j_object_only']
CONDS+=[c.replace('M2','M2v2') for c in tuple(CONDS) if 'M2' in c]


def generator_signature(profile, cond, stems):
    return {"profile": profile, "condition": cond, "stems_sha256": F.sha_bytes("\n".join(stems).encode()),
            "generator_sha256": {p: F.sha_file(L.ROOT / p) for p in ("experiments/followup/synth.py","experiments/extra_controls.py","experiments/e2_predict.py", "experiments/lib.py", "src/preprocess.py")},
            'data':{p:F.sha_file(L.ROOT/p) for p in ('data/metadata.json','data/v3_pixel_sha256.json','data/validation_split.json')}}


def build(profile, cond, makers=None):
    stems = L.targets(profile)
    d = F.SYNTH / profile / cond
    meta_path = d / "meta.json"
    sig = generator_signature(profile, cond, stems)
    done = F.read_done(meta_path, sig)
    if done is not None:
        return done
    if makers is None:
        makers = generators(profile)
    meta = {}
    for s in stems:
        img, extra = makers[cond](s)
        if img is None:
            meta[s] = None
            continue
        ok, buf = cv2.imencode(".png", img)
        assert ok
        p = d / f"{s}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(buf.tobytes())
        F._replace(tmp, p)
        h, w = img.shape[:2]
        meta[s] = {**extra, "resolution": [w, h], "png_sha256": F.sha_bytes(buf.tobytes()), "signals": L.signals_of(img)}
    F.publish(meta_path, meta, sig)
    return meta


def load_image(profile, cond, stem):
    path=F.SYNTH/profile/cond/f'{stem}.png'
    meta=verified_meta(profile,cond)
    if not meta or F.sha_file(path)!=meta[stem]['png_sha256']:raise RuntimeError('Synthetic PNG provenance mismatch')
    return L.X.read_image(path)


@lru_cache(maxsize=128)
def verified_meta(profile,cond):
    return F.read_done(F.SYNTH/profile/cond/'meta.json')


def insert_m2v2(g8,patch,cx,cy):
    """Align destination ROI exactly to source-mask ROI; old M2 remains untouched."""
    import numpy as np
    h,w=g8.shape;ph,pw=patch['win'].shape;bw,bh=patch['size'];ox,oy=patch['off']
    x=int(round(cx-bw/2-ox));y=int(round(cy-bh/2-oy))
    if x<1 or y<1 or x+pw>=w-1 or y+ph>=h-1:return None,None
    mask=cv2.dilate(patch['mask']*255,np.ones((3,3),np.uint8))
    if not mask.any():return None,None
    src=np.zeros_like(g8);src[y:y+ph,x:x+pw]=patch['win']
    full=np.zeros_like(g8);full[y:y+ph,x:x+pw]=mask
    xx,yy,ww,hh=cv2.boundingRect(full)
    center=(xx+ww//2,yy+hh//2)
    out=L.gray(cv2.seamlessClone(L.bgr(src),L.bgr(g8),full,center,cv2.NORMAL_CLONE))
    bx=[(x+ox)/w,(y+oy)/h,(x+ox+bw)/w,(y+oy+bh)/h]
    return out,bx


def generators(profile):
    syn=E2.Synth(profile);v=E2.r2_variants(syn,include_sweep=False)
    from experiments.extra_controls import make
    for c in STRESS:
        v['STRESS_'+c]=lambda s,c=c:(L.degrade_any(L.clean_img(s),c,s),{'gt':L.gt(s),'points':{}})
    for c in ('R2h_removed','R2i_sham','R2j_object_only'):
        def control(s,c=c):
            im,gt,detail=make(syn,s,c);return im,{'gt':gt,'control':detail,'points':{}}
        v[c]=control
    legacy=dict(v)
    for c in CONDS:
        if 'M2v2' not in c:continue
        def corrected(s,c=c):
            original=c.replace('M2v2','M2');old,ex=legacy[original](s)
            if old is None:return None,ex
            B=syn.base(s);pt=ex['points']
            if 'move' in c or 'same' in c:
                j=pt['object_index'];patch=L.object_patch(B['g8'],B['bx'][j]);base=L.remove_object(B['g8'],B['bx'][j]);spot=pt['requested']
            elif 'context' in c:
                C=syn.base(pt['recipient']);patch=L.object_patch(B['g8'],B['bx'][0]);base=C['g8'];spot=pt['new']
            else:
                t,j=pt['donor'];D=syn.base(t);patch=L.object_patch(D['g8'],D['bx'][j]);base=B['g8'];spot=pt['new']
            g,bx=insert_m2v2(base,patch,*spot)
            if g is None:return None,ex
            if 'mark' in c:
                cx=(bx[0]+bx[2])*g.shape[1]/2;cy=(bx[1]+bx[3])*g.shape[0]/2
                g=L.gray(E2.P.clean_v3(E2.P.draw_mark(g,int(round(cx)),int(round(cy)),L.rng_for('R2g',s))))
            ex={**ex,'synthesis_version':'M2v2_ROI_ALIGNMENT_ONLY','legacy_geometry_preserved':True}
            if max(abs(a-b) for a,b in zip(bx,ex['gt'][pt['object_index']] if 'object_index' in pt else ex['gt'][-1]))>1e-9:
                raise RuntimeError('Corrected source ROI changed intended derived GT')
            return L.bgr(g),ex
        v[c]=corrected
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="+", default=CONDS)
    a = ap.parse_args()
    for prof in ("dfl_src12", "dfl_src3_2"):
        makers = generators(prof)
        for c in a.conds:
            m = build(prof, c, makers)
            F.status(stage="synth", profile=prof, condition=c, images=sum(v is not None for v in m.values()))
            print("synth", prof, c, sum(v is not None for v in m.values()), flush=True)


if __name__ == "__main__":
    main()
