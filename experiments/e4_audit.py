"""Read-only data/lineage and object-condition audit; eval79 image hashes only, never its GT."""
import collections
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np

from experiments import lib as L
from experiments.e3_score import load_cache


def dhash(img):
    z=cv2.resize(img,(9,8),interpolation=cv2.INTER_AREA)
    return np.packbits(z[:,1:]>z[:,:-1]).tobytes()


def ncc(a,b):
    best=-1.0
    for angle in (-3,0,3):
        for scale in (.9,1.0,1.1):
            m=cv2.getRotationMatrix2D((31.5,31.5),angle,scale)
            v=cv2.warpAffine(b,m,(64,64),borderMode=cv2.BORDER_REFLECT_101)
            v=cv2.copyMakeBorder(v,4,4,4,4,cv2.BORDER_REFLECT_101)
            score=float(cv2.matchTemplate(v,a,cv2.TM_CCOEFF_NORMED).max())
            best=max(best,score)
    return best


def main():
    records=json.loads((L.ROOT/'data/metadata.json').read_text(encoding='utf8'))['records']
    meta={r['stem']:r for r in records}
    stems=sorted(meta)
    small={s:cv2.resize(L.gray(L.clean_img(s)),(64,64),interpolation=cv2.INTER_AREA) for s in stems}
    dh={s:int.from_bytes(dhash(z),'big') for s,z in small.items()}
    exact=collections.defaultdict(list); bad=[]
    for s in stems:
        sh=L.X.sha256_file(L.XC.raw_path(L.CFG,s)); exact[sh].append(s)
        if sh!=meta[s]['raw_sha256']: bad.append(s)
        if s in L.DEV and L.X.sha256_file(L.XC.label_path(L.CFG,s))!=meta[s]['label_sha256']: bad.append(s+' label')
    if bad: raise RuntimeError(f'Data provenance mismatch: {bad[:5]}')
    # Candidate restriction is disclosed: low dHash OR untransformed NCC >= .95.
    # Re-shoots with materially changed product/view may evade this image-only screen.
    candidates=[]; near=[]
    for i,a in enumerate(stems):
        for b in stems[i+1:]:
            ham=(dh[a]^dh[b]).bit_count()
            direct=float(cv2.matchTemplate(small[a],small[b],cv2.TM_CCOEFF_NORMED)[0,0])
            if ham<=6 or direct>=.95:
                candidates.append((a,b,ham,direct))
    for k,(a,b,ham,direct) in enumerate(candidates):
        sim=ncc(small[a],small[b])
        if sim>=.98:
            near.append({'a':a,'b':b,'dhash_distance':ham,'max_NCC':sim,'same_device':meta[a]['device']==meta[b]['device'],
                         'same_date':meta[a]['date']==meta[b]['date'],'a_role':meta[a]['role'],'b_role':meta[b]['role']})
        if (k+1)%500==0: print('near-duplicate candidates',k+1,'/',len(candidates),flush=True)
    profile={}
    for prof in ('dfl_src12','dfl_src3_2'):
        train=set(L.XC.profile_training_stems(L.CFG,prof)); held=set(L.targets(prof))
        cross=[r for r in near if (r['a'] in train and r['b'] in held) or (r['b'] in train and r['a'] in held)]
        exclude={r['a'] if r['a'] in held else r['b'] for r in cross}
        ds={}
        for sd in L.SEEDS:
            raw=load_cache(f'{prof}_s{sd}','clean'); bx={s:r['boxes'] for s,r in raw.items() if s not in exclude}
            gs={s:L.gt(s) for s in bx}; rec={s:L.image_eval(bx[s],gs[s],*L.size_of(s)) for s in bx}
            ds[sd]=L.detection_metrics(rec,bx,gs)
        # Source-only, non-image position prior: count mode, then median normalized boxes
        # sorted along x. Diagnostic control, never used as an operating model.
        count=collections.Counter(len(L.gt(s)) for s in train).most_common(1)[0][0]
        group=[sorted(L.gt(s),key=lambda b:(b[0]+b[2])/2) for s in train if len(L.gt(s))==count]
        prior=np.median(np.asarray(group),axis=0).tolist()
        pb={s:[b+[1.0] for b in prior] for s in held};gg={s:L.gt(s) for s in held}
        pr={s:L.image_eval(pb[s],gg[s],*L.size_of(s)) for s in held}
        prior_metrics=L.detection_metrics(pr,pb,gg)
        # Swap GT with the first deterministic other-date image of the same device.
        # Similar fixed placements can make this control weak; publish pairing, never hide it.
        permutations={}
        for sd in L.SEEDS:
            raw=load_cache(f'{prof}_s{sd}','clean');bx={s:r['boxes'] for s,r in raw.items()}
            swapped={}
            for s in sorted(held):
                pool=[t for t in sorted(held) if meta[t]['device']==meta[s]['device'] and meta[t]['date']!=meta[s]['date']]
                t=pool[int(L.rng_for('bbox-permutation',s).integers(len(pool)))]
                swapped[s]=L.gt(t)
            rec={s:L.image_eval(bx[s],swapped[s],*L.size_of(s)) for s in held}
            permutations[sd]=L.detection_metrics(rec,bx,swapped)
        profile[prof]={'train':len(train),'held':len(held),'train_device_counts':dict(collections.Counter(meta[s]['device'] for s in train)),
                       'shared_device_dates':sorted({(meta[s]['device'],meta[s]['date']) for s in train}&{(meta[s]['device'],meta[s]['date']) for s in held}),
                       'near_pairs_cross_split':cross,'excluded_held_images':sorted(exclude),'screen_disjoint_subset_metrics':ds,
                       'source_only_position_prior':prior_metrics,'bbox_permutation_diagnostic':permutations,
                       'permutation_limit':'GT position/count may repeat across dates; this control does not establish absence of shortcuts'}
    # GT-layout similarity is a specimen *proxy*; fixed placement itself can create spurious links.
    geometry={}
    for s in L.DEV:
        mask=L.product_mask()(L.gray(L.clean_img(s)),empty=0)
        ys,xs=np.nonzero(mask)
        if not len(xs): continue
        w,h=L.size_of(s)
        centers=np.array([((b[0]+b[2])*w/2,(b[1]+b[3])*h/2) for b in L.gt(s)])
        geometry[s]=(centers-np.array([xs.min(),ys.min()]))/np.maximum([xs.max()-xs.min(),ys.max()-ys.min()],1)
    layout=[]
    from scipy.optimize import linear_sum_assignment
    for a,b in itertools.combinations(sorted(geometry),2):
        if meta[a]['device']==meta[b]['device'] or meta[a]['date']!=meta[b]['date']: continue
        A,B=geometry[a],geometry[b]
        if len(A)!=len(B) or not len(A): continue
        best=float('inf')
        for flipx,flipy in itertools.product((False,True),repeat=2):
            C=B.copy()
            if flipx:C[:,0]=1-C[:,0]
            if flipy:C[:,1]=1-C[:,1]
            delta=np.linalg.norm(A[:,None,:]-C[None,:,:],axis=2)
            ii,jj=linear_sum_assignment(delta);best=min(best,float(delta[ii,jj].max()))
        if best<=.03: layout.append({'a':a,'b':b,'max_normalized_center_delta':best,'status':'UNPROVEN_SPECIMEN_PROXY'})
    rows=[]
    for prof in ('dfl_src12','dfl_src3_2'):
        raw={sd:load_cache(f'{prof}_s{sd}','clean') for sd in L.SEEDS}
        for s in L.targets(prof):
            g=L.gray(L.clean_img(s)); h,w=g.shape
            mask=L.product_mask()(g,empty=0)
            dist=cv2.distanceTransform(mask,cv2.DIST_L2,5)
            qs=L.gt(s)
            per={sd:L.centre_matches({s:raw[sd][s]['boxes']},{s:qs},{s:(w,h)},3) for sd in L.SEEDS}
            for j,q in enumerate(qs):
                x0,y0,x1,y1=L.box_px(q,w,h);cx,cy=(x0+x1)/2,(y0+y1)/2
                a,b,c,d=max(0,int(x0)),max(0,int(y0)),min(w,int(math.ceil(x1))),min(h,int(math.ceil(y1)))
                inn=g[b:d,a:c].astype(float); ring=g[max(0,b-5):min(h,d+5),max(0,a-5):min(w,c+5)].astype(float)
                rows.append({'stem':s,'device':L.META[s]['device'],'date':L.META[s]['date'],'gt':j,
                             'area_px':(x1-x0)*(y1-y0),'min_side_px':min(x1-x0,y1-y0),'bbox_aspect':(x1-x0)/max(1e-6,y1-y0),
                             'edge_distance_px':float(dist[min(h-1,int(cy)),min(w-1,int(cx))]),
                             'contrast_proxy':float((np.median(ring)-np.mean(inn))/(np.std(ring)+1e-6)) if inn.size and ring.size else None,
                             'miss_by_seed':{sd:not any(z[1] and z[3]==j and z[0]>=L.THR for z in per[sd]) for sd in L.SEEDS}})
    bins={'min_side_px':(0,5,10,20,float('inf')),'edge_distance_px':(0,5,15,30,float('inf')),
          'contrast_proxy':(-float('inf'),0,.5,1,float('inf')),'bbox_aspect':(0,.5,1,2,float('inf'))}
    breakdown={}
    for feature,edges in bins.items():
        groups={}
        for lo,hi in zip(edges,edges[1:]):
            for dev in sorted({r['device'] for r in rows}):
                rr=[r for r in rows if r['device']==dev and r[feature] is not None and lo<=r[feature]<hi]
                groups[f'{dev}: [{lo},{hi})']={'objects':len(rr),'miss_by_seed':{sd:sum(r['miss_by_seed'][sd] for r in rr) for sd in L.SEEDS},
                                                'miss_rate_by_seed':{sd:sum(r['miss_by_seed'][sd] for r in rr)/len(rr) if rr else None for sd in L.SEEDS}}
        breakdown[feature]=groups
    interaction={}
    for dev in sorted({r['device'] for r in rows}):
        for small_flag in (False,True):
            for edge_flag in (False,True):
                for low_contrast in (False,True):
                    rr=[r for r in rows if r['device']==dev and (r['min_side_px']<10)==small_flag and (r['edge_distance_px']<15)==edge_flag
                        and r['contrast_proxy'] is not None and (r['contrast_proxy']<.5)==low_contrast]
                    interaction[f'{dev}:small={small_flag},edge={edge_flag},contrast_low={low_contrast}']={'objects':len(rr),
                        'miss_rate_by_seed':{sd:sum(r['miss_by_seed'][sd] for r in rr)/len(rr) if rr else None for sd in L.SEEDS}}
    figures=[]; fp_locations=[]
    for prof in ('dfl_src12','dfl_src3_2'):
        raw=load_cache(f'{prof}_s17','clean')
        chosen=0
        for s,e in raw.items():
            w,h=L.size_of(s);g=L.gt(s);bx=e['boxes']
            mt=L.centre_matches({s:bx},{s:g},{s:(w,h)},3)
            # Predictions are already sorted by confidence after E0.
            for i,r in enumerate(mt):
                if not r[1] and r[0]>=L.THR:
                    b=sorted(bx,key=lambda b:-b[4])[i]
                    fp_locations.append({'stem':s,'device':L.META[s]['device'],'date':L.META[s]['date'],
                                         'center_normalized':[(b[0]+b[2])/2,(b[1]+b[3])/2],'confidence':b[4]})
            ev=L.image_eval(bx,g,w,h)
            if ev['fail'] and chosen<6:
                img=L.clean_img(s).copy()
                for b in g:
                    x0,y0,x1,y1=L.box_px(b,w,h);cv2.rectangle(img,(round(x0),round(y0)),(round(x1),round(y1)),(0,220,0),1)
                for b in bx:
                    if b[4]<L.THR:continue
                    x0,y0,x1,y1=L.box_px(b,w,h);cv2.rectangle(img,(round(x0),round(y0)),(round(x1),round(y1)),(0,0,255),1)
                img=cv2.resize(img,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST)
                p=L.OUT/'E4_audit/examples'/f'{prof}_{s}.png';p.parent.mkdir(parents=True,exist_ok=True)
                cv2.imencode('.png',img)[1].tofile(str(p))
                figures.append({'file':p.relative_to(L.ROOT).as_posix(),'stem':s,'seed':17,'GT_green':True,'prediction_red':True,
                                'FN':ev['n']-ev['tp3'],'FP':ev['fp3'],'trust':ev['trust']})
                chosen+=1
    res={'raw_images_verified':len(stems),'development_labels_verified':len(L.DEV),'evaluation79_label_reads':0,
         'physical_specimen_disjoint_certified':False,'normal_product_images':0,
         'exact_duplicate_groups':[v for v in exact.values() if len(v)>1],
         'near_search':{'candidate_count':len(candidates),'all_image_pairs':len(stems)*(len(stems)-1)//2,
                       'candidate_rule':'dHash <=6 OR direct 64px NCC >=.95',
                       'NCC_search':'64px grid shifts +/-4px, rotations {-3,0,3}, scales {.9,1,1.1}',
                       'limitation':'coarse image-grid diagnostic; not exhaustive physical re-shoot identification'},
         'near_pairs':near,'unproven_same_date_cross_device_layout_pairs':layout,'profile_audit':profile,'objects':rows,
         'examples':figures,'false_positive_locations_seed17':fp_locations,'fixed_feature_bins':bins,'condition_breakdown':breakdown,
         'size_edge_contrast_device_interactions':interaction,
         'material_density_available':False,'bbox_aspect_is_true_object_shape':False,
         'training_source_bytes_note':'current raw/label hashes verified; historical manifest/path does not prove consumed bytes for every old run'}
    L.jdump(L.OUT/'E4_audit'/'RESULTS.json',res)
    print('E4 audit finished; candidate pairs',len(candidates),'near pairs',len(near),flush=True)


if __name__=='__main__': main()
