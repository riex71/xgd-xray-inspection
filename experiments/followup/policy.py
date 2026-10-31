"""Pure matched inference and safety candidates; no GT input to inference/completeness checks."""
import math
import numpy as np

from experiments import lib as L
from experiments.followup import common as F


def default():
    return dict(id='P0',score=.25,e0=4.,localization='base',safety='none',radius=3.,low=.1,
                z_review=3.,z_full=6.,signals=['Q2','Q4'],grade='G0')


def required_views(cfg):
    needed={'base'};mode=cfg['localization']
    if mode=='flip_median':needed.update(('hflip','vflip','hvflip'))
    if mode=='scale_median':needed.update(('sm32','sp32'))
    if mode in ('union','triggered_union'):needed.update(('hflip','vflip','sp32'))
    if cfg['safety'] in ('view','spread','bundle'):needed.update(('hflip','vflip','sm32','sp32'))
    return tuple(v for v in F.VIEWS if v in needed)


def candidates():
    result=[default()]
    def add(ID,**kw):result.append({**default(),'id':ID,**kw})
    for v in (.05,.10,.15):add(f'P1_low{v}',safety='residual',low=v)
    for v in (.15,.35):add(f'P1b_score{v}',score=v)
    for v in (2.,6.):add(f'E0_{v:g}',e0=v)
    for kind in ('seed','view','v3only','bundle'):
        for radius in (3.,6.):add(f'P2_{kind}_{radius:g}',safety=kind,radius=radius)
    add('P3_always',localization='union')
    add('P3_triggered',localization='triggered_union')
    add('L1_flip',localization='flip_median')
    add('L2_scale',localization='scale_median')
    add('L3_source_bias',localization='source_bias')
    add('L4_view_spread',safety='spread')
    add('Q_3_8',z_full=8.)
    for v in (4.,5.):add(f'Q_{v:g}_8',z_review=v,z_full=8.)
    for sig in ('Q2','Q4'):add('Q_only_'+sig,signals=[sig])
    add('GRADE_G1',grade='G1');add('GRADE_G2',grade='G2')
    return result


def match(a,b,w,h,radius):
    used=set();pairs=[]
    for i,x in enumerate(a):
        choices=[(math.hypot(((x[0]+x[2])-(y[0]+y[2]))*w/2,((x[1]+x[3])-(y[1]+y[3]))*h/2),j)
                 for j,y in enumerate(b) if j not in used]
        choices=[(d,j) for d,j in choices if d<=radius]
        if choices:
            d,j=min(choices);used.add(j);pairs.append((i,j,d))
    return pairs,len(a)-len(pairs),len(b)-len(used)


def confirmed(boxes,w,h,cfg):
    return [b for b in F.e0(boxes,w,h,cfg['e0']) if b[4]>=cfg['score']]


def residual_flag(raw,boxes,w,h,low,score):
    # Missed objects are not represented by mean/min score; unmatched low-score proposals are one *imperfect* signal.
    return any(low<=b[4]<score and not any(math.hypot(((b[0]+b[2])-(q[0]+q[2]))*w/2,
                         ((b[1]+b[3])-(q[1]+q[3]))*h/2)<=6 for q in boxes) for b in raw)


def fuse_median(views,w,h):
    base=sorted(views[0],key=lambda b:-b[4]);matched=[[b] for b in base]
    for other in views[1:]:
        for i,j,_ in match(base,other,w,h,3.)[0]:matched[i].append(other[j])
    return [median_box(g) for g in matched]


def median_box(group):
    a=np.array(group);cx,cy=np.median((a[:,0]+a[:,2])/2),np.median((a[:,1]+a[:,3])/2)
    bw,bh=np.median(a[:,2]-a[:,0]),np.median(a[:,3]-a[:,1])
    return [cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2,float(np.median(a[:,4]))]


def fuse_union(views,w,h):
    clusters=[]
    for b in sorted([q for v in views for q in v],key=lambda b:-b[4]):
        target=next((g for g in clusters if math.hypot(((g[0][0]+g[0][2])-(b[0]+b[2]))*w/2,
                            ((g[0][1]+g[0][3])-(b[1]+b[3]))*h/2)<=4),None)
        if target is None:clusters.append([b])
        else:target.append(b)
    return [median_box(g)[:4]+[max(q[4] for q in g)] for g in clusters][:30]


def process(raw_views,other_seeds,v3only,w,h,cfg,bias=(0.,0.)):
    base=F.e0(raw_views['base'],w,h,cfg['e0']);high=confirmed(base,w,h,cfg)
    def differs(b):return match(high,confirmed(b,w,h,cfg),w,h,cfg['radius'])[1:]!=(0,0)
    seed_flag=any(differs(b) for b in other_seeds)
    view_flag=any(differs(raw_views[v]) for v in ('hflip','vflip','sm32','sp32') if v in raw_views)
    control_flag=differs(v3only) if v3only is not None else False
    low_flag=residual_flag(raw_views['base'],high,w,h,cfg['low'],cfg['score'])
    uncertainty=[]
    for b in high:
        xy=[[(b[0]+b[2])*w/2,(b[1]+b[3])*h/2]]
        for name in ('hflip','vflip','sm32','sp32'):
            if name not in raw_views:continue
            alt=confirmed(raw_views[name],w,h,cfg)
            pair,_,_=match([b],alt,w,h,3)
            if pair:
                q=alt[pair[0][1]];xy.append([(q[0]+q[2])*w/2,(q[1]+q[3])*h/2])
        uncertainty.append(float(np.sqrt(np.var(xy,axis=0).sum())))
    flags={'residual':low_flag,'seed':seed_flag,'view':view_flag,'v3only':control_flag,
           'bundle':seed_flag or view_flag or control_flag,'spread':any(x>1.0 for x in uncertainty),'none':False}
    mode=cfg['localization'];needed_views={'base'}
    if mode=='flip_median':
        base=fuse_median([F.e0(raw_views[v],w,h,cfg['e0']) for v in ('base','hflip','vflip','hvflip')],w,h);needed_views.update(('hflip','vflip','hvflip'))
    if mode=='scale_median':
        base=fuse_median([F.e0(raw_views[v],w,h,cfg['e0']) for v in ('base','sm32','sp32')],w,h);needed_views.update(('sm32','sp32'))
    if mode in ('union','triggered_union') and (mode=='union' or low_flag or seed_flag):
        base=fuse_union([confirmed(raw_views[v],w,h,cfg) for v in ('base','hflip','vflip','sp32')],w,h);needed_views.update(('hflip','vflip','sp32'))
    if mode=='source_bias':base=[[b[0]-bias[0]/w,b[1]-bias[1]/h,b[2]-bias[0]/w,b[3]-bias[1]/h,b[4]] for b in base]
    if cfg['safety'] in ('view','spread','bundle'):needed_views.update(('hflip','vflip','sm32','sp32'))
    calls=len(needed_views)+2*int(cfg['safety'] in ('seed','bundle') or mode=='triggered_union')+int(cfg['safety'] in ('v3only','bundle'))
    return F.e0(base,w,h,cfg['e0']),flags[cfg['safety']],{'flags':flags,'localization_spread':uncertainty,'model_calls':calls}


def image_eval(boxes,gt,w,h,threshold):
    selected=[b for b in boxes if b[4]>=threshold];n_det=len(selected);r={'n':len(gt),'n_det':n_det,
                      'trust':float(np.mean([b[4] for b in selected])) if selected else 0.}
    for radius in (1,2,3,4,5,8):
        rows=L.centre_matches({'x':selected},{'x':gt},{'x':(w,h)},radius)
        tp=sum(x[1] for x in rows);r[f'tp{radius}']=tp;r[f'fp{radius}']=n_det-tp
    r['tp_iou'],r['fp_iou']=L.iou_counts(selected,gt,w,h,thr=threshold)
    r.update(fail=int(r['tp3']<r['n'] or r['fp3']>0),fail_fn=int(r['tp3']<r['n']),fail_fp=int(r['fp3']>0),
             actual_missing8=len(gt)-r['tp8'],localization_only_objects=r['tp8']-r['tp3'],far_FP=n_det-r['tp8'],
             err8=L.pair_errors_8px(selected,gt,w,h,thr=threshold),signed8=L.pair_errors_8px(selected,gt,w,h,thr=threshold,signed=True))
    return r


def decision(pkg,sd,sig,rec,flag,cfg,w,h):
    ref=pkg['reference']
    try:
        z={q:(sig[q]-ref[q]['loc'])/ref[q]['scale'] for q in ('Q2','Q4')}
        valid=all(math.isfinite(v) and math.isfinite(ref[q]['scale']) and ref[q]['scale']>0 for q,v in z.items())
    except (KeyError,TypeError,ValueError,ZeroDivisionError):valid=False;z={}
    if not valid:return 'FULL_REINSPECTION',['촬영 품질 신호/기준에 비정상 값: 자동 판정 중단'],z
    m=max(abs(z[q]) for q in cfg['signals'])
    state='FULL_REINSPECTION' if m>cfg['z_full'] else 'REVIEW' if m>cfg['z_review'] else 'NORMAL'
    sc=L.X.scale_policy(w,h,L.XC.profile_scale_range(L.CFG,pkg['profile']),L.CFG['scale_policy'])
    ok,why=L.X.scale_envelope(sc,L.CFG['scale_policy'])
    cal=dict(pkg['seeds'][sd]);cal['lambda']=float('inf') if cal['lambda'] is None else cal['lambda']
    d,reasons,_=L.X.decide(state,cal,rec['trust'],rec['n_det'],L.CFG['crc'],ok,why,(),True)
    if d=='ACCEPT' and flag:d='REVIEW';reasons.append('잔여 후보/예측 불일치: 영상 검출 완결성 미확인')
    return d,reasons,z
