"""CPU scoring of provenance-sealed development prediction caches. No evaluation79 labels.

P-IND strict is primary. Full-data operating grades are a separately labelled diagnostic,
not independent calibration. No detector, threshold or signal rule is fitted here.
"""
import collections
import json
import importlib.metadata as package_metadata
import math
from pathlib import Path
from functools import lru_cache

import numpy as np

from experiments import lib as L
from experiments.e1_rules_on_x14 import arm_scale
from src import xgd_analyzer as X


@lru_cache(maxsize=128)
def checked_weight_hash(path, size, mtime):
    return L.X.sha256_file(path)


def load_cache(key, condition):
    p = L.CACHE / key / f"{condition}.json"
    if not p.with_suffix(".manifest.json").exists():
        raise RuntimeError(f"Unsealed cache: {p}")
    manifest=json.loads(p.with_suffix('.manifest.json').read_text(encoding='utf8'))
    for section in ('source','inputs'):
        if any(L.X.sha256_file(L.ROOT/f)!=sha for f,sha in manifest[section].items()):
            raise RuntimeError(f'Changed {section} for cached prediction {p}; preserved, not scored')
    if any(package_metadata.version(k)!=v for k,v in manifest['packages'].items()):
        raise RuntimeError(f'Changed runtime dependencies for {p}')
    prof,sd=key.rsplit('_s',1)
    cp=L.ROOT/f'outputs/controls/v3only/retrain/{prof[7:]}_s{sd}/weights/last.pt' if prof.startswith('v3only_') else L.XC.checkpoint(L.CFG,prof,int(sd))
    st=cp.stat()
    if checked_weight_hash(str(cp),st.st_size,st.st_mtime_ns)!=manifest['checkpoint']:
        raise RuntimeError(f'Changed checkpoint for {p}')
    return json.loads(p.read_text(encoding="utf8"))


def pairs(a, b, w, h, radius):
    """One-to-one, score-ordered correspondence; missing matches are retained explicitly."""
    aa, bb = L.centers_px(a, w, h), L.centers_px(b, w, h)
    used, result = set(), []
    for i, (x, y, _) in enumerate(aa):
        choices = [(math.hypot(x-u, y-v), j) for j, (u,v,_) in enumerate(bb) if j not in used and math.hypot(x-u,y-v) <= radius]
        if choices:
            e, j = min(choices)
            used.add(j)
            result.append((i, j, e))
    return result, len(aa)-len(result), len(bb)-len(used)


def boxshift(boxes, dx, dy, w, h):
    return [[x1-dx/w, y1-dy/h, x2-dx/w, y2-dy/h, c] for x1,y1,x2,y2,c in boxes]


def condition_records(raw, condition):
    boxes, gts, resolutions, extra = {}, {}, {}, {}
    for s, r in raw.items():
        if r is None:
            continue
        w, h = r.get("resolution", L.size_of(s))
        b, g = r["boxes"], r.get("gt", L.gt(s))
        if condition.startswith("R1_"):
            _, dx, dy = condition.split("_")
            dx, dy = int(dx), int(dy)
            # Inverse prediction coordinates; exclude only GT moved within 4px of image edge.
            retained = [q for q in g if 4 <= (q[0]+q[2])*w/2+dx <= w-4 and 4 <= (q[1]+q[3])*h/2+dy <= h-4]
            extra[s] = {"boundary_excluded": len(g)-len(retained)}
            g, b = retained, boxshift(b, dx, dy, w, h)
        boxes[s], gts[s], resolutions[s] = b, g, (w,h)
    recs = {s: L.image_eval(boxes[s], gts[s], *resolutions[s]) for s in boxes}
    return recs, boxes, gts, resolutions, extra


def analyze_profile(profile):
    tg = L.targets(profile)
    keys = {sd: f"{profile}_s{sd}" for sd in L.SEEDS}
    clean = {sd: load_cache(k, "clean") for sd,k in keys.items()}
    cr = {sd: condition_records(d, "clean")[0] for sd,d in clean.items()}
    signals = {s: clean[17][s]["signals"] for s in tg}
    primary = L.design(profile, cr, signals, "IND", "strict")
    diagnostics = L.design(profile, cr, signals, "IND", "deployment")
    time_primary = L.design(profile, cr, signals, "TIME", "strict")
    result = {"primary": "P-IND-strict", "grade_diagnostic": "P-IND-deployment-grade (grade uses evaluation data)",
              "calibration_roles": {s: {"cal_dates": p["cal_dates"], "n": p["n_cal"], "grade":p["grade"]} for s,p in primary.items()},
              "conditions": {}, "examples": [], "translation": {}, "synthetic": {}}
    conds = sorted(set.intersection(*(set(p.stem for p in (L.CACHE/k).glob('*.json') if not p.name.endswith('.manifest.json')) for k in keys.values())))
    for condition in conds:
        per = {}
        for sd,key in keys.items():
            raw = load_cache(key, condition)
            recs, bx, gt, sz, ex = condition_records(raw, condition)
            met = L.detection_metrics(recs, bx, gt, sz)
            met["skipped_synthesis"] = len(tg)-len(recs)
            met["boundary_excluded"] = sum(v.get("boundary_excluded",0) for v in ex.values())
            met["device"] = {d: L.detection_metrics({s:r for s,r in recs.items() if L.META[s]["device"]==d},bx,gt,sz)
                             for d in sorted({L.META[s]["device"] for s in recs})}
            met["session_groups"] = {d: L.detection_metrics({s:r for s,r in recs.items() if L.META[s]["device"]+"/"+L.META[s]["date"]==d},bx,gt,sz)
                                    for d in sorted({L.META[s]["device"]+"/"+L.META[s]["date"] for s in recs})}
            for label,pkg in (("P-IND-strict",primary),("P-IND-deployment-grade",diagnostics),("P-TIME-strict",time_primary)):
                ds, eligible = {}, {}
                for s,r in recs.items():
                    if s not in pkg:
                        continue
                    w,h = sz[s]
                    sc = X.scale_policy(w,h,L.XC.profile_scale_range(L.CFG,profile),L.CFG["scale_policy"],"native" if condition=="native" else "x8")
                    sc["imgsz"], sc["effective_scale"] = raw[s]["imgsz"], raw[s]["imgsz"]/max(w,h)
                    ds[s], gate, z = L.decide_image(pkg[s],sd,raw[s]["signals"],r,sc)
                    eligible[s]=r
                    if label=="P-IND-strict" and r["fail"] and len(result["examples"]) < 200:
                        result["examples"].append({"stem":s,"seed":sd,"condition":condition,"decision":ds[s],"trust":r["trust"],"quality":gate,
                                                   "lambda":pkg[s]["seeds"][sd]["lambda"],"FN":r["n"]-r["tp3"],"FP":r["fp3"]})
                met[label] = L.decision_metrics(ds,eligible)
            if condition.startswith("R1_"):
                errors, miss_a, miss_b, clean_lost, recovered = [],0,0,0,0
                clean_boxes = {s:clean[sd][s]["boxes"] for s in recs}
                for s,r in recs.items():
                    pm,ma,mb = pairs(clean_boxes[s],bx[s],*sz[s],3)
                    errors += [p[2] for p in pm]
                    miss_a+=ma; miss_b+=mb
                    # Same retained GT, before and after shift, one-to-one matching.
                    rc = L.image_eval(clean_boxes[s],gt[s],*sz[s])
                    cl = L.centre_matches({s:clean_boxes[s]},{s:gt[s]},{s:sz[s]},3)
                    sh = L.centre_matches({s:bx[s]},{s:gt[s]},{s:sz[s]},3)
                    a={z[3] for z in cl if z[1] and z[0]>=L.THR}; b={z[3] for z in sh if z[1] and z[0]>=L.THR}
                    clean_lost+=len(a-b); recovered+=len(b-a)
                met["translation"]={"paired_P50_P90_P95_max":np.percentile(errors,[50,90,95,100]).tolist() if errors else None,
                                    "unmatched_clean":miss_a,"unmatched_shift":miss_b,"clean_GT_lost":clean_lost,"GT_recovered":recovered}
            if condition.startswith("R2"):
                new_hit=old_hit=trace_hit=traces=nspots=0
                removed_hits=removed_total=sham_missing=0
                for s,r in raw.items():
                    if r is None: continue
                    w,h = r.get("resolution",L.size_of(s)); pt=r.get("points",{})
                    if "new" in pt:
                        nspots+=1; new_hit+=L.detected_at(r["boxes"],w,h,pt["new"])
                    if "old" in pt:
                        old_hit+=L.detected_at(r["boxes"],w,h,pt["old"],4)
                    for p in pt.get("spots",[]):
                        traces+=1; trace_hit+=L.detected_at(r["boxes"],w,h,p,6)
                    for p in r.get('control',{}).get('original_centres',[]):
                        removed_total+=1;removed_hits+=L.detected_at(r['boxes'],w,h,p,3)
                    sham_missing+=int(r.get('control',{}).get('no_feasible_window',False))
                met["synthetic"]={"n_new":nspots,"new_hits":new_hit,"new_hit_rate":new_hit/nspots if nspots else None,
                                  "old_hit_count":old_hit,"n_fake_trace":traces,"trace_hits":trace_hit,"trace_hit_rate":trace_hit/traces if traces else None,
                                  'removed_sites':removed_total,'removed_site_hits':removed_hits,'sham_no_feasible_window_images':sham_missing}
            per[sd]=met
        result["conditions"][condition] = L.summarize_seeds(per,keys_min=("R3","pFROC3","F13"),keys_max=("FN","FP_per_image3"))
    metrics=[(cond,sd,m) for cond,C in result["conditions"].items() for sd,m in C["per_seed"].items()]
    result["true_worst_condition_seed"] = {"3px":min(((m["R3"],c,s) for c,s,m in metrics if m["R3"] is not None),default=None),
                                            "pFROC":min(((m["pFROC3"],c,s) for c,s,m in metrics if m.get("pFROC3") is not None),default=None),
                                            "FP_image":max(((m["FP_per_image3"],c,s) for c,s,m in metrics if m["FP_per_image3"] is not None),default=None)}
    return result


def controls():
    out={}
    for typ,profs in (("baseline",("yolo_src12","yolo_src3_2")),("v3only",("dfl_src12","dfl_src3_2"))):
        out[typ]={}
        for prof in profs:
            per={}
            for sd in L.SEEDS:
                key=f"{prof}_s{sd}" if typ=="baseline" else f"v3only_{prof}_s{sd}"
                per[sd]={}
                for p in sorted((L.CACHE/key).glob('*.json')):
                    if p.name.endswith('.manifest.json'): continue
                    r,b,g,z,_=condition_records(load_cache(key,p.stem),p.stem)
                    m=L.detection_metrics(r,b,g,z)
                    raw=load_cache(key,p.stem)
                    if p.stem.startswith("R2"):
                        n=hit=0
                        for s,e in raw.items():
                            if e is None: continue
                            pt=e.get('points',{}); w,h=e.get('resolution',L.size_of(s))
                            points=pt.get('spots',[]) if 'trace' in p.stem else ([pt['new']] if 'new' in pt else [])
                            for q in points:
                                n+=1;hit+=L.detected_at(e['boxes'],w,h,q,6 if 'trace' in p.stem else 3)
                        m['target_hits'],m['target_count']=hit,n
                    per[sd][p.stem]=m
            out[typ][prof]=per
    return out


def signal_analysis():
    """Each predeclared signal independently. No weight/threshold search and no automatic deployment changes."""
    out={"adoption_policy":"predeclared gates; internal evidence only; no automatic adoption or threshold search", "profiles":{}}
    for prof in ("dfl_src12","dfl_src3_2"):
        keys={sd:f'{prof}_s{sd}' for sd in L.SEEDS}
        cs={sd:load_cache(k,'clean') for sd,k in keys.items()}
        cr={sd:condition_records(c,'clean')[0] for sd,c in cs.items()}
        sig={s:r['signals'] for s,r in cs[17].items()}
        pkg=L.design(prof,cr,sig,'IND','strict')
        rows=[]
        for sd,key in keys.items():
            tr=[load_cache(key,f'R1_{dx}_{dy}') for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))]
            sc=load_cache(key,'GRID_1.1_clean')
            for s,c in cs[sd].items():
                w,h=L.size_of(s); b=c['boxes']; T=[]
                for sh,(dx,dy) in zip(tr,((1,0),(-1,0),(0,1),(0,-1))):
                    T.append(boxshift(sh[s]['boxes'],dx,dy,w,h))
                def flagged(alt):
                    return pairs(b,alt,w,h,1.5)[1]>0 or pairs(b,alt,w,h,3)[2]>0
                c1=any(flagged(t) for t in T)
                c2=flagged(sc[s]['boxes'])
                c3=any(pairs(b,cs[t][s]['boxes'],w,h,3)[1:]!=(0,0) for t in L.SEEDS if t!=sd)
                r=cr[sd][s]; st=arm_scale(prof,s,'rule')
                decision=L.decide_image(pkg[s],sd,c['signals'],r,st)[0]
                # Associations are not assertions of detector independence or statistical qualification.
                distances=[e for t in T for _,_,e in pairs(b,t,w,h,3)[0]]
                rows.append({'profile':prof,'seed':sd,'stem':s,'device':L.META[s]['device'],'session':L.META[s]['date'],
                             'fail':r['fail'],'FN':r['n']-r['tp3'],'FP':r['fp3'],'base_decision':decision,'confidence':r['trust'],
                             'C1_translation':c1,'C2_scale':c2,'C3_seed':c3,
                             'paired_translation_variance':float(np.var(distances)) if distances else None})
        checks={}
        for signal in ('C1_translation','C2_scale','C3_seed'):
            af=[r for r in rows if r['fail'] and r['base_decision']=='ACCEPT']
            correct=[r for r in rows if not r['fail']]
            captured=sum(r[signal] for r in af)
            additional=sum(r[signal] and r['base_decision']=='ACCEPT' for r in correct)/max(1,len(correct))
            checks[signal]={'accepted_failures_before':len(af),'captured':captured,'capture_fraction':captured/len(af) if af else None,
                            'additional_correct_hold_rate':additional,
                            'status':'INCONCLUSIVE_NO_ACCEPTED_FAILURES' if not af else
                                     ('INTERNAL_GATE_MET_NOT_INDEPENDENTLY_QUALIFIED' if captured/len(af)>=.5 and additional<=.1 else 'NOT_ADOPTED'),
                            'flagged_failures':sum(r[signal] and r['fail'] for r in rows),
                            'all_failures':sum(r['fail'] for r in rows),'flagged_correct':sum(r[signal] and not r['fail'] for r in rows),
                            'all_correct':len(correct)}
        out['profiles'][prof]={'rows':rows,'candidates':checks,'seed_disagreement_is_independent_detector':False}
    L.jdump(L.OUT/'E3_scoring'/'SAFETY_SIGNALS.json',out)
    return out


def main():
    from experiments.extra_controls import main as complete_required_controls
    complete_required_controls()
    res={"evidence":"development robustness / safety evidence; repeated development pool, not independent qualification",
         "evaluation79_label_reads":0,"profiles":{},"controls":controls()}
    for prof in ("dfl_src12","dfl_src3_2"):
        res["profiles"][prof]=analyze_profile(prof)
        print('E3 scored',prof,flush=True)
    L.jdump(L.OUT/'E3_scoring'/'RESULTS.json',res)
    signal_analysis()


if __name__=='__main__':
    main()
