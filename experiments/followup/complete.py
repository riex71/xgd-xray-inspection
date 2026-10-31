"""Selection -> fixed validation -> recalibration -> operational reproduction -> final predictions.

This is loaded only after final comparisons. No new values are generated from results.
Every failure has a non-zero exit; existing source/results/checkpoints remain intact.
"""
import collections,copy,csv,datetime,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from experiments import lib as L
from experiments.followup import common as F,policy as P,analyze as A,synth as S,selection as SEL
from src import followup_analyzer as RT
from src.followup_safety import registry_entry

def task(stage,args,python=sys.executable):
    F.status(stage=stage,status='RUNNING',evaluation79_GT_reads=0)
    with (F.OUT/f'{stage}.log').open('a',encoding='utf8') as log:
        r=subprocess.run([str(python),'-u',*args],cwd=F.ROOT,stdout=log,stderr=subprocess.STDOUT,
            env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')})
    if r.returncode:raise RuntimeError(stage+': non-zero execution; preserved log')

def collect_selected(item,prof):
    needed=P.required_views(item['policy']);family=item['family']
    if family!='dfl' and len(needed)>1:
        task('selected_views_'+family+'_'+prof,['-m','experiments.followup.predict','--family',family,'--profiles',prof,'--views',*needed,'--conds','dev_clean','synth'])
    if family=='dfl' and len(needed)>1:
        task('selected_stress_views_'+prof,['-m','experiments.followup.predict','--family','dfl','--profiles',prof,'--views',*needed,'--conds',*[c for c in S.CONDS if c.startswith('STRESS_')]])

def validate_item(prof,item):
    family=item['family'];cfg=item['policy'];reports={};base={}
    for c in ('dev_clean',*S.CONDS):
        F.status(stage='selected_fixed_validation',status='RUNNING',profile=prof,condition=c,evaluation79_GT_reads=0)
        reports[c]=A.evaluate(family,prof,c,cfg);base[c]=A.evaluate('dfl',prof,c,P.default())
    reasons=SEL.clean_gate(base['dev_clean'],reports['dev_clean'])
    for c in ('dev_clean',*S.CONDS):
        # Fixed post-selection validation can reject, never retune a value.
        if SEL.decision(reports[c],'joint_risk')>SEL.decision(base[c],'joint_risk')+1e-12:reasons.append(c+': realized joint risk increased')
        if SEL.decision(reports[c],'FN8_joint_risk')>SEL.decision(base[c],'FN8_joint_risk')+1e-12:reasons.append(c+': actual-missing ACCEPT risk increased')
    if SEL.decision(reports['dev_clean'],'coverage')+.000000000001<.5*SEL.decision(base['dev_clean'],'coverage'):
        reasons.append('clean coverage below preregistered usability minimum')
    if family!='dfl':
        for c in ('R2f_free_M1','R2f_free_M2','R2f_free_M2v2','R2a_move_M1_0','R2a_move_M2_0','R2a_move_M2v2_0'):
            # The training intervention must not hide degraded sensitivity in a pooled average.
            if SEL.mean(reports[c],'actual_missing8')>SEL.mean(base[c],'actual_missing8')+1e-12:
                reasons.append(c+': detector actual missing count increased')
    F.atomic_json(F.OUT/f'SELECTED_VALIDATION_{prof}_{family}_{cfg["id"]}.json',{'candidate':item,'candidate_results':reports,'baseline_results':base,'rejection_reasons':reasons,
        'evidence':'post-selection internal development validation; not independent final validation'})
    return not reasons,reasons,reports,base

def calibrate(selection):
    config={'version':'XGD-1.2.0','profiles':copy.deepcopy(selection['profiles']),'weights':{},
        'calibration_file':'outputs/followup/deployment/CALIBRATION.json','formal_qualification':'UNQUALIFIED',
        'independent_qualification':False,'reporting_threshold_policy':'selected fixed policy; GT count never used at inference'}
    metadata=json.loads((F.ROOT/'data/metadata.json').read_text(encoding='utf8'))['records']
    for prof,item in config['profiles'].items():
        if item['policy']['localization']=='source_bias':
            item['source_bias']={str(sd):A.source_bias(item['family'],prof,sd) for sd in L.SEEDS}
        for sd in L.SEEDS:
            family=item['family'];config['weights'][f'{family}/{prof}/{sd}']=F.sha_file(F.checkpoint(family,prof,sd))
            if item['policy']['safety'] in ('v3only','bundle'):
                config['weights'][f'v3only/{prof}/{sd}']=F.sha_file(F.checkpoint('v3only',prof,sd))
    config['pipeline_signature']=RT.sign(config)
    cal={'pipeline_signature':config['pipeline_signature'],'time':datetime.datetime.now().astimezone().isoformat(),
        'profiles':{},'formal_qualification':'UNQUALIFIED','evaluation79_GT_reads':0,
        'claim':'within-provider empirical operating rules; no exchangeability/normal-product or arbitrary-device guarantee',
        'dataset_manifest_sha256':F.sha_file(F.OUT/'RAW_COLLECTION_LOCK.json'),
        'calibration_role':'all available held development labels, after internal policy selection; deployment calibration, not fresh validation'}
    for prof,item in config['profiles'].items():
        devs={};registry={}
        for d in sorted({L.META[s]['device'] for s in L.targets(prof)}):
            stems=sorted(s for s in L.targets(prof) if L.META[s]['device']==d)
            pkg=A.calibration(prof,tuple(stems),A.calibration_key(item['policy']),item['family'])
            pkg['pipeline_signature']=config['pipeline_signature'];pkg['official_TXT_sha256']={s:F.sha_file(L.XC.label_path(L.CFG,s)) for s in stems}
            devs[d]=pkg;registry[d]=registry_entry(d,[r for r in metadata if r['stem'] in stems],F.sha_file(F.ROOT/'data/metadata.json'),config['pipeline_signature'])
            registry[d].update({'preprocessing':'src.preprocess.clean_v3; original pixels retained','preprocessing_sha256':F.sha_file(F.ROOT/'src/preprocess.py'),
                               'effective_input_scale_range':L.XC.profile_scale_range(L.CFG,prof),'physical_magnification':'unknown'})
        cal['profiles'][prof]={'devices':devs,'registry':registry}
    F.atomic_json(F.OUT/'deployment/CALIBRATION.json',cal)
    config['calibration_sha256']=F.sha_file(F.OUT/'deployment/CALIBRATION.json')
    F.atomic_json(RT.CONFIG,config)
    return config

def completed_contract(config):
    import torch
    reports={};checks=[]
    for prof,item in config['profiles'].items():
        stems=sorted(L.targets(prof))[:8]
        for sd in L.SEEDS:
            an=RT.FollowupAnalyzer(prof,sd);m=an.model(item['family'],sd)
            s=stems[0];img=L.clean_img(s);h,w=img.shape[:2];z=m.scale(w,h)['imgsz']
            b1=F.predict_view(m,img,'base')[0];b2=F.predict_view(m,img,'base')[0]
            assert b1==b2,'repeat forward must be bit exact in current deterministic implementation'
            coords=np.asarray(b1,dtype=float)
            roundtrip=float(np.max(np.abs(((coords[:,:4]*[w,h,w,h])/[w,h,w,h]-coords[:,:4])*[w,h,w,h]))) if len(coords) else 0.
            assert roundtrip<=.05
            r=m.an.model.predict([img]*4,imgsz=z,rect=F.PC['rect'],conf=F.PC['conf'],iou=F.PC['iou'],max_det=30,
                verbose=False,device=m.an.device,half=F.PC['half'])
            dif=[]
            for q in r:
                arr=np.c_[q.boxes.xyxy.cpu().numpy(),q.boxes.conf.cpu().numpy()];arr[:,:4]/=[w,h,w,h]
                bb=arr.tolist();pairs,na,nb=P.match(b1,bb,w,h,.05)
                assert na==nb==0,'batch1/4 raw matching/coordinate contract failed'
                dif.extend(p[2] for p in pairs)
                assert len(P.confirmed(b1,w,h,item['policy']))==len(P.confirmed(bb,w,h,item['policy'])),'threshold/E0 decision correspondence changed'
                views={v:A.raw(item['family'],prof,sd,v,'dev_clean')[s]['boxes'] for v in P.required_views(item['policy'])}
                oth=[A.raw(item['family'],prof,t,'base','dev_clean')[s]['boxes'] for t in L.SEEDS if t!=sd]
                control=A.raw('v3only',prof,sd,'base','dev_clean')[s]['boxes'];pkg=an.cal['profiles'][prof]['devices'][L.META[s]['device']]
                cp=copy.deepcopy(pkg);cp['seeds']={int(k):v for k,v in cp['seeds'].items()};ds=[]
                for base_boxes in (b1,bb):
                    views['base']=base_boxes;out,flag,_=P.process(views,oth,control,w,h,item['policy'],item.get('source_bias',{}).get(str(sd),(0,0)))
                    rec=P.image_eval(out,L.gt(s),w,h,item['policy']['score'])
                    ds.append(P.decision(cp,sd,A.CLEAN_SIGNALS[s],rec,flag,item['policy'],w,h)[0])
                assert ds[0]==ds[1],'batch1/4 final threshold/E0/safety decision invariance failed'
            assert max(dif,default=0)<=.05
            assert all(torch.isfinite(p).all() for p in m.an.model.model.parameters()),'nonfinite completed parameter'
            records=A.processed(item['family'],prof,sd,'dev_clean',item['policy'])
            for s in stems:
                md={'calibration_id':L.META[s]['device'],'session':L.META[s]['date'],'preprocessed':False}
                actual=an.analyze(image_path=L.XC.raw_path(L.CFG,s),metadata=md)
                bx,rr,detail=records;w,h=L.size_of(s)
                pkg=an.cal['profiles'][prof]['devices'][md['calibration_id']]
                cp=copy.deepcopy(pkg);cp['seeds']={int(k):v for k,v in cp['seeds'].items()}
                expected,_,_=P.decision(cp,sd,A.CLEAN_SIGNALS[s],rr[s],detail[s]['flag'],item['policy'],w,h)
                assert actual['decision']==expected,'runtime/cache decision mismatch'
                assert actual['object_count']==rr[s]['n_det'],'runtime/cache detection mismatch'
                checks.append({'stem':s,'seed':sd,'profile':prof,'decision':expected,'detections':actual['object_count']})
            reports[f'{prof}/{sd}']={'batch1_4_max_center_delta_px':max(dif,default=0),'repeat_forward_bit_exact':True,
                'checkpoint_sha256':m.an.ckpt_sha256,'finite_parameters':True,'raw_matching_correspondence':True,
                'coordinate_roundtrip_px':roundtrip,'fixed_threshold_E0_decision_invariant':True}
            del an,m;r=None;torch.cuda.empty_cache()
    F.atomic_json(F.OUT/'COMPLETED_PIPELINE_CONTRACT.json',{'status':'PASS','models':reports,'runtime_cache_checks':checks,
        'coordinate_encode_decode_roundtrip_px':max(v['coordinate_roundtrip_px'] for v in reports.values()),'coordinate_roundtrip_note':'measured normalized xyxy multiply/divide with existing validated geometry; no new decode',
        'evidence':'implementation evidence; no generalization improvement inferred'})

def resolution_test(config):
    import cv2
    prof='dfl_src3_2';out={};registry_used=False
    stems=sorted(s for s in L.targets(prof) if L.META[s]['device']=='1호기' and L.size_of(s)==(352,332))
    excluded_objects=sum(sum(not(18<=(g[0]+g[2])*352/2<334) for g in L.gt(s)) for s in stems)
    for sd in L.SEEDS:
        an=RT.FollowupAnalyzer(prof,sd);records={'base':{},'crop':{},'canvas':{}};states={k:{} for k in records}
        for s in stems:
            image=L.clean_img(s);h,w=image.shape[:2];crop=np.ascontiguousarray(image[:,18:334]);canvas=np.full_like(image,int(np.median(image)));canvas[:,18:334]=crop
            GT=L.gt(s)
            for kind,im in (('base',image),('crop',crop),('canvas',canvas)):
                r=an.analyze(image=im,metadata={'calibration_id':'1호기','session':L.META[s]['date'],'preprocessed':True})
                ww=im.shape[1];bb=[[v/scale for v,scale in zip(d['bbox_xyxy'],(ww,h,ww,h))]+[d['confidence']] for d in r['detections']]
                gt=GT if kind!='crop' else [[(g[0]*w-18)/316,g[1],(g[2]*w-18)/316,g[3]] for g in GT if 18<=(g[0]+g[2])*w/2<334]
                records[kind][s]=P.image_eval(bb,gt,ww,h,config['profiles'][prof]['policy']['score']);states[kind][s]=r['quality']['state']
        stats={k:L.detection_metrics(v) for k,v in records.items()};base=stats['base']
        result={}
        for kind in ('crop','canvas'):
            agree=np.mean([states[kind][s]==states['base'][s] for s in stems]) if stems else 0.
            result[kind]={'metrics':stats[kind],'R3_delta':stats[kind]['R3']-base['R3'] if stems else None,
                'FP_per_image_delta':stats[kind]['FP_per_image3']-base['FP_per_image3'] if stems else None,'quality_state_agreement':float(agree),
                'compatibility_diagnostic_pass':bool(stems and not excluded_objects and stats[kind]['R3']>=base['R3']-.005-1e-12 and
                    abs(stats[kind]['FP_per_image3']-base['FP_per_image3'])<=.01+1e-12 and agree>=.95)}
        out[str(sd)]={'images':len(stems),'baseline':base,**result};del an
    report={'per_seed':out,'crop_pixels_retained':True,'crop_x':[18,334],'GT_objects_outside_crop':excluded_objects,'canvas_fill':'original image median',
        'actual_evaluation316_native_auto_approved':False,'reason':'native316 ROI/acquisition unknown; development crop success alone cannot establish equivalence',
        'independent_qualification':False}
    F.atomic_json(F.OUT/'RESOLUTION_COMPATIBILITY.json',report)

def predict_final(config):
    # This function is the only new evaluation79 prediction event. No GT path is referenced.
    path=F.OUT/'deployment/TEST_PREDICTIONS.json'
    if path.exists():raise RuntimeError('final evaluation prediction already exists; no silent regeneration')
    result={};timings=[];transitions=collections.Counter();base=json.loads((L.ROOT/'outputs/test_predictions.json').read_text(encoding='utf8'))['images']
    routes={'3호기':'dfl_src12','1호기':'dfl_src3_2','2호기':'dfl_src3_2'}
    for prof in A.PROFILES:
        an=RT.FollowupAnalyzer(prof,17)
        warm=sorted(L.targets(prof))[0];an.analyze(image_path=L.XC.raw_path(L.CFG,warm),metadata={'calibration_id':L.META[warm]['device'],'preprocessed':False})
        for s in L.XC.test_stems(L.CFG):
            if routes[L.META[s]['device']]!=prof:continue
            r=an.analyze(image_path=L.XC.raw_path(L.CFG,s),metadata={'calibration_id':L.META[s]['device'],'session':L.META[s]['date'],'preprocessed':False})
            result[s]={'device':L.META[s]['device'],'date':L.META[s]['date'],**r};timings.append(r.get('elapsed_seconds',0))
            transitions[base[s]['decision']+' -> '+r['decision']]+=1
        del an
    F.atomic_json(path,{'time':datetime.datetime.now().astimezone().isoformat(),'images':result,'n_images':len(result),'evaluation79_GT_reads':0,
        'decision_counts':dict(collections.Counter(r['decision'] for r in result.values())),
        'baseline_to_new_transition':dict(transitions),'latency_mean_P95_seconds':[float(np.mean(timings)),float(np.percentile(timings,95))],
        'includes':'read/clean/GPU/postprocess/quality/new policy; excludes model loading, acquisition and human review',
        'pipeline_signature':config['pipeline_signature'],'performance_inference':'none; evaluation labels not used'})
    rows=[];boxes=[];yo=F.OUT/'deployment/test_predictions_yolo';yo.mkdir(parents=True,exist_ok=True)
    for s,r in sorted(result.items()):
        rows.append({'image':s,'device':r['device'],'date':r['date'],'decision':r['decision'],'objects':r['object_count'],
                     'confirmation_scope':r['confirmation_scope'],'completeness_status':r['completeness_status'],'product_pass_certified':False,
                     'reason':' | '.join(r['reason']),'version':config['version']})
        w,h=L.size_of(s);lines=[]
        for b in r['detections']:
            x0,y0,x1,y1=b['bbox_xyxy'];sc=b['confidence'];boxes.append({'image':s,'decision':r['decision'],'x0':x0,'y0':y0,'x1':x1,'y1':y1,'score':sc})
            lines.append(f'0 {(x0+x1)/2/w:.6f} {(y0+y1)/2/h:.6f} {(x1-x0)/w:.6f} {(y1-y0)/h:.6f} {sc:.6f}')
        (yo/f'{s}.txt').write_text('\n'.join(lines)+'\n' if lines else '',encoding='utf8')
    for name,data in (('test_predictions_images.csv',rows),('test_predictions_boxes.csv',boxes)):
        with (F.OUT/'deployment'/name).open('w',encoding='utf-8-sig',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(data[0]) if data else ['image']);wr.writeheader();wr.writerows(data)

def fresh_reproduce():
    install=F.OUT/'reproduction/INSTALL_STATUS.json'
    if not install.exists() or json.loads(install.read_text(encoding='utf8'))['exit_code']!=0:
        raise RuntimeError('Fresh dependency installation incomplete; cannot reuse old reproduction PASS')
    py=F.OUT/'reproduction/fresh_venv/Scripts/python.exe'
    task('fresh_tests',['-m','pytest','-q','tests','-p','no:cacheprovider'],py)
    task('fresh_inference',['-m','experiments.followup.fresh_check'],py)

def main():
    from experiments.followup.execution_lock import verify
    verify();A.initialize()
    comparison=json.loads((F.OUT/'FINAL_COMPARISONS.json').read_text(encoding='utf8'));selection=SEL.choose(comparison)
    F.atomic_json(F.OUT/'SELECTION_LOCK.json',selection)
    decisions={}
    for prof,item in selection['profiles'].items():
        collect_selected(item,prof);passed,reasons,reports,base=validate_item(prof,item)
        history=[{'item':copy.deepcopy(item),'passed':passed,'reasons':reasons}]
        if not passed and item['family']!='dfl':
            item['family']='dfl';item['reverted_training_intervention']=True;collect_selected(item,prof)
            passed,reasons,reports,base=validate_item(prof,item);history.append({'item':copy.deepcopy(item),'passed':passed,'reasons':reasons})
        if not passed:
            item['family']='dfl';item['policy']=P.default();item['reverted_policy']=True
            passed,reasons,reports,base=validate_item(prof,item);history.append({'item':copy.deepcopy(item),'passed':passed,'reasons':reasons})
        if not passed:raise RuntimeError('Even baseline matched self-validation failed')
        item['final_validation_passed']=True;decisions[prof]=history
    F.atomic_json(F.OUT/'FINAL_SELECTION.json',{'selection':selection,'validation_history':decisions,
        'interpretation':'internal adaptive selection; no formal safety/generalization guarantee'})
    from experiments.followup.diagnostics import synthesis_audit,ltt_report
    synthesis_audit();ltt_report(selection)
    task('DFL_distribution_diagnostic',['-m','experiments.followup.distribution_probe'])
    config=calibrate(selection)
    task('post_selection_tests',['-m','pytest','-q','tests','-p','no:cacheprovider'])
    completed_contract(config);resolution_test(config);fresh_reproduce();predict_final(config)
    from experiments.followup.report import main as report
    report()
    F.atomic_json(F.OUT/'FOLLOWUP_COMPLETED.json',{'time':datetime.datetime.now().astimezone().isoformat(),
        'pipeline_signature':config['pipeline_signature'],'evaluation79_GT_reads':0,'formal_qualification':'UNQUALIFIED',
        'status':'COMPLETED_INTERNAL_RESEARCH_AND_REPRODUCTION','submission_zip_created':False})
    print('Follow-up final selection, recalibration, reproduction, and reporting completed',flush=True)
