"""Additional safety contracts for a newly signed candidate; never modify XGD-1.1 artifacts."""
import hashlib,json,math


def registry_entry(device,records,source_sha,signature):
    rr=[r for r in records if r['device']==device]
    return {'device':device,'native_resolutions':sorted({tuple(r['resolution']) for r in rr}),
            'orientation':'unknown; exact width/height order is checked','ROI':'unknown',
            'acquisition_settings':'unknown','calibration_dates':sorted({r['date'] for r in rr}),
            'provider_source_sha256':source_sha,'pipeline_signature':signature,'automatic_new_resolution_approval':False}


def resolution_compatible(entry,resolution,signature):
    if not isinstance(entry,dict) or entry.get('pipeline_signature')!=signature:return False,'등록부/후속 보정 서명 불일치'
    if not isinstance(resolution,(list,tuple)) or len(resolution)!=2 or any(not isinstance(v,int) or isinstance(v,bool) or v<=0 for v in resolution):
        return False,'잘못된 native 해상도'
    if tuple(resolution) not in [tuple(x) for x in entry.get('native_resolutions',[])]:return False,'보정에 등록되지 않은 native 해상도/방향'
    return True,'등록된 native 해상도; ROI/실제 촬영 설정은 unknown'


def confirmation_schema(flag,decision):
    return {'confirmation_scope':'detected_object_positions_only','completeness_status':'INCONSISTENT' if flag else 'NOT_CERTIFIED',
            'product_pass_certified':False,'detections_confirmed':decision=='ACCEPT'}


def fixed_sequence(pvalues,delta=.10):
    passed=[]
    for i,p in enumerate(pvalues):
        if not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1:return {'passed':passed,'stopped_at':i,'invalid':True}
        if p>delta:return {'passed':passed,'stopped_at':i,'invalid':False}
        passed.append(i)
    return {'passed':passed,'stopped_at':None,'invalid':False}


def pipeline_signature(policy,weight_hashes,source_hashes):
    payload={'policy':policy,'weights':weight_hashes,'sources':source_hashes,'version':'XGD-1.2-candidate'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
