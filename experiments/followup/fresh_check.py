"""Actual newly installed environment execution, never PYTHONPATH/site-packages substitution."""
import datetime,importlib.metadata as md,json,sys
from pathlib import Path
from experiments import lib as L
from experiments.followup import common as F,policy as P,analyze as A
from src.followup_analyzer import FollowupAnalyzer,CONFIG

def main():
    assert sys.prefix!=sys.base_prefix,'fresh venv required'
    assert not any('Python313/Lib/site-packages' in x.replace('\\','/') for x in sys.path),'host site-packages must not be used'
    A.initialize();config=json.loads(CONFIG.read_text(encoding='utf8'));rows=[];cpu=[]
    for prof,item in config['profiles'].items():
        for sd in L.SEEDS:
            s=sorted(L.targets(prof))[0];meta={'calibration_id':L.META[s]['device'],'session':L.META[s]['date'],'preprocessed':False}
            an=FollowupAnalyzer(prof,sd);result=an.analyze(image_path=L.XC.raw_path(L.CFG,s),metadata=meta)
            ref=A.processed(item['family'],prof,sd,'dev_clean',item['policy'])
            bx,rec,detail=ref;w,h=L.size_of(s);high=P.confirmed(bx[s],w,h,item['policy'])
            out=[[x/(w if i%2==0 else h) for i,x in enumerate(d['bbox_xyxy'])]+[d['confidence']] for d in result['detections']]
            pairs,na,nb=P.match(high,out,w,h,.05);assert na==nb==0,'fresh GPU prediction mismatches locked cache'
            pkg=an.cal['profiles'][prof]['devices'][meta['calibration_id']];pkg['seeds']={int(k):v for k,v in pkg['seeds'].items()}
            expected,_,_=P.decision(pkg,sd,A.CLEAN_SIGNALS[s],rec[s],detail[s]['flag'],item['policy'],w,h)
            assert result['decision']==expected,'fresh GPU decision mismatches locked rule'
            rows.append({'profile':prof,'seed':sd,'image':s,'max_center_delta_px':max((x[2] for x in pairs),default=0),'decision_same':True,
                         'checkpoint_sha256':result['model']['checkpoint_sha256']})
            del an
            if sd==17:
                c=FollowupAnalyzer(prof,sd,device='cpu');r=c.analyze(image_path=L.XC.raw_path(L.CFG,s),metadata=meta)
                cb=[[x/(w if i%2==0 else h) for i,x in enumerate(d['bbox_xyxy'])]+[d['confidence']] for d in r['detections']]
                cp,ca,cc=P.match(high,cb,w,h,.05)
                cpu.append({'profile':prof,'image':s,'unmatched':ca+cc,'max_matched_center_delta_px':max((x[2] for x in cp),default=0),
                            'decision_same':r['decision']==result['decision'],'CPU_qualified_for_general_use':False})
                del c
    report={'status':'PASS','time':datetime.datetime.now().astimezone().isoformat(),'fresh_dependency_install':True,'shared_site_packages':False,
        'packages':{k:md.version(k) for k in ('torch','torchvision','ultralytics','numpy','scipy','opencv-python','PyYAML','pytest')},
        'GPU_cache_checks':rows,'CPU_GPU_sample_comparison':cpu,'CPU_and_second_GPU_universal_decision_invariance_claimed':False,
        'different_GPU':'UNTESTED: only one physical GPU available','evaluation79_GT_reads':0,
        'pipeline_signature':config['pipeline_signature']}
    F.atomic_json(F.OUT/'reproduction/FRESH_REPRODUCTION.json',report)
    print('actual fresh-install inference PASS; CPU comparison reported separately',flush=True)

if __name__=='__main__':main()
