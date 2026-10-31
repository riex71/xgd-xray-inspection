"""One-factor follow-up comparisons on sealed raw predictions; every changed pipeline is recalibrated.
Results are internal development evidence. No evaluation79 GT, independent qualification, or hidden threshold search.
"""
import argparse
import collections
import copy
import datetime
from functools import lru_cache
import json
import math
from types import SimpleNamespace

import numpy as np

from experiments import lib as L
from experiments.followup import common as F,policy as P,synth as S

PROFILES=('dfl_src12','dfl_src3_2')
SCREEN_CONDS=('dev_clean','R2f_free_M1','R2f_free_M2','R2f_free_M2v2','R2a_move_M1_0','R2a_move_M2_0',
              'R2a_move_M2v2_0','R2a_same_M1_0','R2a_same_M2_0','R2a_same_M2v2_0','R2c_context_M1','R2c_context_M2','R2c_context_M2v2','R2e_trace')
CLEAN_SIGNALS={}


def initialize():
    global CLEAN_SIGNALS
    if not CLEAN_SIGNALS:CLEAN_SIGNALS={s:L.signals_of(L.clean_img(s)) for s in sorted(L.DEV)}


@lru_cache(maxsize=1024)
def raw(family,prof,sd,view,cond):
    p=F.PRED/F.model_key(family,prof,sd)/view/f'{cond}.json'
    stems,digest=prediction_inputs(prof,cond)
    stub=SimpleNamespace(an=SimpleNamespace(ckpt_sha256=F.sha_file(F.checkpoint(family,prof,sd))),
                         rng=L.XC.profile_scale_range(L.CFG,prof))
    record=F.read_done(p,F.signature(stub,view,cond,stems,digest))
    if record is None:raise RuntimeError(f'Required prediction not completed: {p}')
    return record


@lru_cache(maxsize=128)
def prediction_inputs(prof,cond):
    from experiments.followup.predict import inputs_for
    stems,loader,digest=inputs_for(prof,cond)
    return stems,digest


@lru_cache(maxsize=128)
def meta(prof,cond):return S.verified_meta(prof,cond) if cond!='dev_clean' else None


def data(prof,s,cond):
    if cond=='dev_clean':return L.gt(s),L.size_of(s),CLEAN_SIGNALS[s]
    m=meta(prof,cond)[s]
    return m.get('gt',L.gt(s)),m['resolution'],m['signals']


@lru_cache(maxsize=16)
def source_bias(family,prof,sd):
    errors=[];cache=raw(family,prof,sd,'base','dev_clean')
    for s in L.XC.profile_training_stems(L.CFG,prof):
        w,h=L.size_of(s);boxes=F.e0(cache[s]['boxes'],w,h,4.)
        errors.extend(L.pair_errors_8px(boxes,L.gt(s),w,h,signed=True))
    return np.median([e[1:] for e in errors],axis=0).tolist() if errors else [0.,0.]


@lru_cache(maxsize=1)
def high_groups():
    audit=json.loads((L.OUT/'E4_audit/RESULTS.json').read_text(encoding='utf8'))
    parent={s:s for s in L.META}
    def root(s):
        while parent[s]!=s:parent[s]=parent[parent[s]];s=parent[s]
        return s
    for row in audit['near_pairs']:
        if row['max_NCC']>=.998:
            a,b=root(row['a']),root(row['b']);parent[max(a,b)]=min(a,b)
    groups=collections.defaultdict(list)
    for s in sorted(parent):groups[root(s)].append(s)
    return {s:gid for gid,ss in groups.items() for s in ss},dict(groups)


@lru_cache(maxsize=2)
def splits(prof):
    gid,groups=high_groups();train=set(L.XC.profile_training_stems(L.CFG,prof));held=L.targets(prof)
    contaminated={s for s in held if set(groups[gid[s]])&train}
    usable=[s for s in held if s not in contaminated];parts=[];moves=[]
    for dev in sorted({L.META[s]['device'] for s in usable}):
        ss=[s for s in usable if L.META[s]['device']==dev];a,b=map(set,L.date_halves(ss))
        for g in sorted({gid[s] for s in ss}):
            member=set(groups[g])&set(ss);ca,cb=len(member&a),len(member&b)
            if ca and cb:
                early=min(member,key=lambda s:(L.META[s]['date'],s));side=0 if ca>cb or (ca==cb and early in a) else 1
                moved=member&(b if side==0 else a);moves.extend(sorted(moved))
                (a if side==0 else b).update(member);(b if side==0 else a).difference_update(member)
        if a and b:
            parts.extend([(dev,sorted(a),sorted(b)),(dev,sorted(b),sorted(a))])
    return parts,{'high_group_threshold':.998,'group_members':groups,'source_connected_held_excluded':sorted(contaminated),
                  'moved_between_halves':moves,'usable_held':usable,'physical_specimen_disjoint_certified':False}


def processed(family,prof,sd,cond,cfg):
    return processed_cached(family,prof,sd,cond,json.dumps(cfg,sort_keys=True))


@lru_cache(maxsize=64)
def processed_cached(family,prof,sd,cond,config_json):
    cfg=json.loads(config_json)
    cache=raw(family,prof,sd,'base',cond);boxes={};records={};details={}
    for s,e in cache.items():
        if s not in L.targets(prof):continue
        gt,(w,h),sig=data(prof,s,cond)
        views={'base':e['boxes']}
        for v in F.VIEWS[1:]:
            requested=v in P.required_views(cfg)
            if cond.startswith('STRESS_') and not requested:continue
            # Never silently treat an unmeasured view signal as safe.
            if requested or (family=='dfl' and not cond.startswith('STRESS_')):
                views[v]=raw(family,prof,sd,v,cond)[s]['boxes']
        others=[raw(family,prof,t,'base',cond)[s]['boxes'] for t in L.SEEDS if t!=sd]
        control=raw('v3only',prof,sd,'base',cond).get(s,{}).get('boxes')
        bx,flag,detail=P.process(views,others,control,w,h,cfg,source_bias(family,prof,sd) if cfg['localization']=='source_bias' else (0.,0.))
        boxes[s]=bx;records[s]=P.image_eval(bx,gt,w,h,cfg['score']);details[s]={**detail,'flag':flag,'signals':sig}
    return boxes,records,details


def grouped_metrics(rr,bx,gt,sz,key):
    groups=collections.defaultdict(list)
    for s in rr:groups[key(s)].append(s)
    return {str(k):L.detection_metrics({s:rr[s] for s in ss},{s:bx[s] for s in ss},gt,sz) for k,ss in groups.items()}


@lru_cache(maxsize=512)
def calibration(prof,cal_tuple,config_json,family):
    cfg=json.loads(config_json);cal=list(cal_tuple)
    R={sd:processed(family,prof,sd,'dev_clean',cfg)[1] for sd in L.SEEDS}
    sig={s:CLEAN_SIGNALS[s] for s in cal};session={s:L.META[s]['date'] for s in cal}
    ref=L.X.build_quality_reference(cal,sig,session);seeds={};succ=[];fps=[];coverage=[];statuses=[]
    for sd in L.SEEDS:
        rr=R[sd];trust=[rr[s]['trust'] for s in cal];fail=[rr[s]['fail'] for s in cal]
        lam,bound=L.X.crc_lambda(trust,fail,.05)
        seeds[sd]={'N':len(cal),'lambda':lam if math.isfinite(lam) else None,'bound':bound}
        succ.append(sum(rr[s]['tp3'] for s in cal)/max(1,sum(rr[s]['n'] for s in cal)))
        fps.append(np.mean([rr[s]['fp3'] for s in cal]))
        if cfg['grade']=='G2':
            coverage.append(np.mean([t>=lam for t in trust]));statuses.append('QUALIFIED_DEVICE_CONDITIONAL' if math.isfinite(lam) else 'FULL_REINSPECTION')
        elif cfg['grade']=='G1':
            folds=[]
            for date in sorted(set(session.values())):
                tr=[s for s in cal if session[s]!=date];va=[s for s in cal if session[s]==date]
                ll,_=L.X.crc_lambda([rr[s]['trust'] for s in tr],[rr[s]['fail'] for s in tr],.05)
                folds.append((sum(rr[s]['trust']>=ll and rr[s]['fail'] for s in va)/len(va),sum(rr[s]['trust']>=ll for s in va)/len(va)))
            cov=np.mean([f[1] for f in folds]) if folds else 0.
            coverage.append(cov);statuses.append('QUALIFIED_DEVICE_CONDITIONAL' if folds and max(f[0] for f in folds)<=.08 and np.mean([f[0] for f in folds])<=.05 and cov>=.5 else 'FULL_REINSPECTION')
        else:
            gated={s:rr[s]['trust'] for s in cal}
            for date in sorted(set(session.values())):
                qref=L.X.build_quality_reference(cal,sig,session,exclude_session=date)
                for s in cal:
                    if session[s]!=date:continue
                    z={q:(sig[s][q]-qref[q]['loc'])/qref[q]['scale'] for q in cfg['signals']}
                    if max(abs(x) for x in z.values())>cfg['z_review']:gated[s]-=1.
            cluster={s:L.META[s]['date']+L.META[s]['device'] for s in cal}
            if len(set(cluster.values()))>=2:
                assessment=L.X.crc_split_assessment(gated,{s:rr[s]['fail'] for s in cal},cal,cluster,alphas=(.05,),splits=200,delta=.10)['0.05']['CRC']
                coverage.append(assessment['median_coverage']);statuses.append(assessment['status'])
            else:coverage.append(0.);statuses.append('FULL_REINSPECTION')
    status=collections.Counter(statuses).most_common(1)[0][0]
    grade=L.X.qualification_grade(min(succ),float(np.mean(fps)),float(np.median(coverage)),status).value
    if cfg['grade']=='G2':grade='QUALIFIED' if min(succ)>=.95 and np.mean(fps)<=.05 and min(coverage)>=.5 and all(v['lambda'] is not None for v in seeds.values()) else 'UNQUALIFIED'
    for v in seeds.values():v['grade']=grade
    return {'profile':prof,'reference':ref,'seeds':seeds,'grade':grade,'cal_stems':cal,'dates':sorted(set(session.values())),
            'statistical_preconditions_established':False,'formal_qualification':'UNQUALIFIED'}


def calibration_key(cfg):
    # Review-only signals do not change detector/trust/GT losses; equivalent calibration inputs share the computation.
    c=copy.deepcopy(cfg);c['id']='CAL';c['safety']='none';c['radius']=3.;c['low']=.1
    return json.dumps(c,sort_keys=True)


def evaluate(family,prof,cond,cfg):
    parts,group_info=splits(prof);per={}
    by_seed={sd:processed(family,prof,sd,cond,cfg) for sd in L.SEEDS}
    packages={tuple(cal):calibration(prof,tuple(cal),calibration_key(cfg),family) for _,cal,ev in parts}
    for sd,(bx,rr,detail) in by_seed.items():
        gt={s:data(prof,s,cond)[0] for s in rr};sz={s:data(prof,s,cond)[1] for s in rr}
        detector=L.detection_metrics(rr,bx,gt,sz)
        detector['R4']=sum(r['tp4'] for r in rr.values())/max(1,sum(r['n'] for r in rr.values())) if sum(r['n'] for r in rr.values()) else None
        gid,_=high_groups()
        detector['by_device']=grouped_metrics(rr,bx,gt,sz,lambda s:L.META[s]['device'])
        detector['by_date']=grouped_metrics(rr,bx,gt,sz,lambda s:L.META[s]['date'])
        detector['by_high_group']=grouped_metrics(rr,bx,gt,sz,lambda s:gid[s])
        detector['actual_missing8']=sum(r['actual_missing8'] for r in rr.values())
        detector['localization_only_objects']=sum(r['localization_only_objects'] for r in rr.values())
        detector['far_FP']=sum(r['far_FP'] for r in rr.values())
        detector['raw_top30_frequency']=sum(len(raw(family,prof,sd,'base',cond)[s]['boxes'])>=30 for s in rr)/len(rr) if rr else None
        design={};rows=[]
        for label in ('group_date_calibration','QUALIFIED_FIXED_DIAGNOSTIC'):
            decisions={};eligible={};calls=[]
            for device,cal,ev in parts:
                pkg=copy.deepcopy(packages[tuple(cal)])
                if label=='QUALIFIED_FIXED_DIAGNOSTIC':
                    for v in pkg['seeds'].values():v['grade']='QUALIFIED'
                for s in ev:
                    if s not in rr:continue
                    _,(w,h),sig=data(prof,s,cond);r=rr[s];flag=detail[s]['flag']
                    d,reasons,z=P.decision(pkg,sd,sig,r,flag,cfg,w,h)
                    decisions[s]=d;eligible[s]=r;calls.append(detail[s]['model_calls'])
                    if label=='group_date_calibration':rows.append({'stem':s,'seed':sd,'condition':cond,'device':device,'date':L.META[s]['date'],
                        'GT_count':r['n'],'prediction_count':r['n_det'],'FN3':r['n']-r['tp3'],'FP3':r['fp3'],
                        'actual_missing8':r['actual_missing8'],'localization_only_objects':r['localization_only_objects'],
                        'far_FP':r['far_FP'],'decision':d,'flag':flag,'trust':r['trust'],'reasons':reasons,
                        'tp3':r['tp3'],'fail':r['fail'],'n_det':r['n_det'],'n':r['n'],
                        'calibration_dates':pkg['dates'],'quality_z':z,'localization_spread':detail[s]['localization_spread']})
            m=L.decision_metrics(decisions,eligible)
            m['FN8_joint_risk']=sum(decisions[s]=='ACCEPT' and eligible[s]['actual_missing8']>0 for s in eligible)/len(eligible) if eligible else None
            m['localization_joint_risk']=sum(decisions[s]=='ACCEPT' and eligible[s]['localization_only_objects']>0 for s in eligible)/len(eligible) if eligible else None
            m['far_FP_joint_risk']=sum(decisions[s]=='ACCEPT' and eligible[s]['far_FP']>0 for s in eligible)/len(eligible) if eligible else None
            m['model_calls_mean']=float(np.mean(calls)) if calls else None
            for cost in (10,50,100):
                loss=sum((cost if eligible[s]['actual_missing8']>0 else 10) if decisions[s]=='ACCEPT' and eligible[s]['fail'] else
                         0 if decisions[s]=='ACCEPT' else 1 if decisions[s]=='REVIEW' else 3 for s in eligible)
                m[f'assumed_cost_{cost}']=loss/len(eligible) if eligible else None
            design[label]=m
        detector['decisions']=design;detector['rows']=rows
        per[sd]=detector
    summary=L.summarize_seeds(per,('R3','F13','pFROC3'),('FN','FP_per_image3'))
    summary['group_protocol']={k:v for k,v in group_info.items() if k!='group_members'}
    summary['calibrations']=[{'device':L.META[p['cal_stems'][0]]['device'],'N':len(p['cal_stems']),'dates':p['dates'],'grade':p['grade']} for p in packages.values()]
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['screen','final'],required=True);stage=ap.parse_args().stage
    from experiments.followup.execution_lock import verify
    verify()
    initialize()
    families=['dfl'] if stage=='screen' else ['dfl','v3only','trace05']
    result={'time':datetime.datetime.now().astimezone().isoformat(),'stage':stage,'evidence':'adaptive internal development comparison',
            'evaluation79_GT_reads':0,'independent_qualification':'UNQUALIFIED','candidates':{},'candidate_values':P.candidates()}
    for family in families:
        candidates=P.candidates() if family=='dfl' else [P.default()]
        for cfg in candidates:
            key=family+'/'+cfg['id'];result['candidates'][key]={}
            for prof in PROFILES:
                conds=SCREEN_CONDS if family=='dfl' else ('dev_clean','R2f_free_M1','R2f_free_M2','R2f_free_M2v2','R2a_move_M1_0','R2a_move_M2_0','R2a_same_M2_0','R2e_trace')
                if cfg['id']=='P0' or cfg['id'].startswith(('Q_','GRADE_')):
                    conds=tuple(conds)+tuple(c for c in S.CONDS if c.startswith('STRESS_'))
                C={}
                for cond in conds:
                    if cond!='dev_clean' and family!='dfl' and cfg['localization']!='base':continue
                    F.status(stage='analyze_'+stage,status='RUNNING',candidate=key,profile=prof,condition=cond,evaluation79_GT_reads=0)
                    C[cond]=evaluate(family,prof,cond,cfg)
                result['candidates'][key][prof]=C
                F.atomic_json(F.OUT/f'comparisons/{stage}_{family}_{cfg["id"]}_{prof}.json',C)
                print('compared',stage,key,prof,flush=True)
    gid,groups=high_groups()
    result['high_groups']=groups
    F.atomic_json(F.OUT/f'{stage.upper()}_COMPARISONS.json',result)
    print('comparisons complete',stage,flush=True)
    if stage=='final':
        from experiments.followup.complete import main as complete
        complete()


if __name__=='__main__':main()
