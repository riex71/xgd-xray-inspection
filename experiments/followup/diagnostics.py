"""Locked diagnostic accounting. No search, no evaluation79 labels, no pipeline alteration."""
import collections,math
import cv2
import numpy as np
from scipy.stats import binom
from experiments import lib as L,e2_predict as E2
from experiments.followup import common as F,synth as S,analyze as A,policy as P
from src.followup_safety import fixed_sequence

def patch_stats(gray,box):
    h,w=gray.shape;cx=(box[0]+box[2])*w/2;cy=(box[1]+box[3])*h/2
    x,y=int(round(cx)),int(round(cy));r=8
    win=gray[max(0,y-r):min(h,y+r+1),max(0,x-r):min(w,x+r+1)].astype(float)
    if win.size<25:return None
    yy,xx=np.indices(win.shape);cx2=x-max(0,x-r);cy2=y-max(0,y-r)
    d=np.hypot(xx-cx2,yy-cy2);core=win[d<=2];ring=win[(d>=5)&(d<=8)]
    contrast=float(np.median(ring)-np.quantile(core,.2)) if core.size and ring.size else None
    edge=np.hypot(cv2.Sobel(win,cv2.CV_64F,1,0),cv2.Sobel(win,cv2.CV_64F,0,1))
    return {'contrast':contrast,'ring_brightness':float(np.median(ring)) if ring.size else None,
            'ring_noise_MAD':float(1.4826*np.median(np.abs(ring-np.median(ring)))) if ring.size else None,
            'boundary_gradient':float(np.mean(edge[(d>=4)&(d<=6)])),
            'darkest_location_minus_nominal':list(map(int,(np.unravel_index(np.argmin(win),win.shape)[1]-cx2,
                                                            np.unravel_index(np.argmin(win),win.shape)[0]-cy2)))}

def synthesis_audit():
    rows=[]
    for prof in A.PROFILES:
        syn=E2.Synth(prof)
        for cond in ('R2f_free_M2','R2a_move_M2_0','R2a_same_M2_0','R2c_context_M2'):
            m=S.verified_meta(prof,cond);m2=S.verified_meta(prof,cond.replace('M2','M2v2'))
            for s,e in m.items():
                if e is None or m2.get(s) is None:continue
                pts=e['points'];j=pts.get('object_index',len(e['gt'])-1);bx=e['gt'][j]
                donor,idx=pts.get('donor',(s,pts.get('object_index',0)));D=syn.base(donor)
                patch=L.object_patch(D['g8'],D['bx'][idx]);mask=cv2.dilate(patch['mask']*255,np.ones((3,3),np.uint8))
                _,_,mw,mh=cv2.boundingRect(mask)
                r={'profile':prof,'stem':s,'condition':cond,'donor':donor,'donor_object_index':idx,
                   'nominal_GT_unchanged':e['gt']==m2[s]['gt'],'mask_ROI':[mw,mh],
                   'legacy_ROI_shift_px':[-int(mw%2==0),-int(mh%2==0)],
                   'M2v2_ROI_shift_px':[0,0],
                   'source':patch_stats(D['g8'],D['gt'][idx]),
                   'M2':patch_stats(L.gray(S.load_image(prof,cond,s)),bx),
                   'M2v2':patch_stats(L.gray(S.load_image(prof,cond.replace('M2','M2v2'),s)),bx)}
                for kind in ('M2','M2v2'):
                    r[kind]['contrast_ratio_vs_source']=r[kind]['contrast']/r['source']['contrast'] if r['source']['contrast'] else None
                rows.append(r)
    report={'evidence':'implementation / train-development synthesis diagnostic','rows':rows,
        'pairs':len(rows),'legacy_shifted_pairs':sum(any(r['legacy_ROI_shift_px']) for r in rows),
        'M2v2_definition':'ROI alignment only; no contrast search or post-result adaptation',
        'contrast_ratios':{kind:{'P10_P50_P90':np.percentile([r[kind]['contrast_ratio_vs_source'] for r in rows
                 if r[kind]['contrast_ratio_vs_source'] is not None],[10,50,90]).tolist()} for kind in ('M2','M2v2')},
        'limits':'GT/core offsets and Poisson brightness/contrast/PSF changes remain; M2v2 is not certified photometric realism',
        'source_URL':'https://raw.githubusercontent.com/opencv/opencv/4.x/modules/photo/src/seamless_cloning.cpp'}
    F.atomic_json(F.OUT/'SYNTHESIS_GEOMETRY_PHOTOMETRY_AUDIT.json',report)
    return report

def nominal_ltt(trust,fail,lambdas,alpha,delta=.10):
    rows=[]
    for lam in lambdas:
        acc=np.asarray(trust)>=lam;n=int(acc.sum());k=int(np.asarray(fail,dtype=int)[acc].sum())
        p=float(binom.cdf(k,n,alpha)) if n else 1.
        rows.append({'lambda':float(lam),'n_accept':n,'accepted_failures':k,'pvalue':p,
                     'conditional_risk':k/n if n else None,'coverage':n/len(trust) if len(trust) else None})
    return {'rows':rows,'bonferroni_passed':[i for i,r in enumerate(rows) if r['pvalue']<=delta/len(rows)],
            'fixed_sequence':fixed_sequence([r['pvalue'] for r in rows],delta),
            'delta':delta,'alpha':alpha,'iid_exchangeability_verified':False,'qualified':False}

def ltt_report(selection):
    out={'evidence':'internal calibration diagnostics; no formal guarantee','seed':17,'no_seed_replication_as_sample':True,
         'alpha_primary':.05,'alpha_sensitivity':.10,'fresh_family_error_budget_sensitivity':'.10/(3 devices * 4 procedures * 2 alphas)',
         'multiplicity_limitation':'This illustrative budget does not cover prior E1-E9 / 29 policies / trace-family adaptive selection; no simultaneous formal guarantee',
         'source_threshold_policy':'10 source-training trust quantiles, 3 fixed coverages .5/.7/.9, one fixed .5, sequence .1 to 1 first failure stop',
         'minimum_zero_failure_accepted':{str(a):{str(k):math.ceil(math.log(.1/k)/math.log(1-a)) for k in (10,3,1)} for a in (.05,.10)},'profiles':{}}
    for prof,item in selection['profiles'].items():
        cfg=item['policy'];family=item['family'];R=A.processed(family,prof,17,'dev_clean',cfg)[1]
        src=A.raw(family,prof,17,'base','dev_clean');trust=[]
        for s in L.XC.profile_training_stems(L.CFG,prof):
            w,h=L.size_of(s);b=F.e0(src[s]['boxes'],w,h,cfg['e0']);trust.append(L.X.trust_of(b,cfg['score']))
        lam={cov:float(np.quantile(trust,1-cov)) for cov in np.arange(.1,1.01,.1)}
        profiles=[]
        for dev,cal,ev in A.splits(prof)[0]:
            t=[R[s]['trust'] for s in cal];f=[R[s]['fail'] for s in cal]
            row={'device':dev,'cal_stems':cal,'unique_images':len(cal),'date_groups':len({L.META[s]['date'] for s in cal}),'alphas':{}}
            for alpha in (.05,.10):
                sets={'10_Bonferroni':[lam[x] for x in sorted(lam)],
                      '3_Bonferroni':[float(np.quantile(trust,1-x)) for x in (.5,.7,.9)],
                      '1_source_fixed':[float(np.quantile(trust,.5))],
                      'fixed_sequence':[lam[x] for x in sorted(lam)]}
                row['alphas'][str(alpha)]={name:{'procedure_only':nominal_ltt(t,f,ls,alpha),
                    'simultaneous_budget_sensitivity':nominal_ltt(t,f,ls,alpha,.1/24)} for name,ls in sets.items()}
            profiles.append(row)
        out['profiles'][prof]=profiles
    F.atomic_json(F.OUT/'LTT_PROCEDURE_DIAGNOSTICS.json',out)
    return out
