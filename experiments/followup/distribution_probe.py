"""Diagnostic DFL side distributions, not a calibrated center posterior / new policy.

Frozen original DFL only, first 8 sorted held and first 8 sorted source images/profile/seed.
No training, no threshold choice; missed objects without a proposal remain invisible.
"""
import datetime,math
import numpy as np
import torch
from scipy.stats import spearmanr
from ultralytics.utils import ops
from experiments import lib as L
from experiments.followup import common as F,policy as P,analyze as A

def main():
    rows=[];contracts=[]
    for prof in A.PROFILES:
        stems=sorted(set(sorted(L.targets(prof))[:8]+L.XC.profile_training_stems(L.CFG,prof)[:8]))
        for sd in L.SEEDS:
            model=F.load_model('dfl',prof,sd);head=model.an.model.model.model[-1];capture={}
            def hook(module,inputs,output):
                if not isinstance(output,tuple) or not isinstance(output[1],dict):return
                pred=output[1]
                if 'one2one' in pred:pred=pred['one2one' if module.end2end else 'one2many']
                if 'boxes' in pred:capture['pred']=pred
            handle=head.register_forward_hook(hook)
            try:
                for s in stems:
                    capture.clear();img=L.clean_img(s);h,w=img.shape[:2];z=model.scale(w,h)['imgsz']
                    b,unused=F.predict_view(model,img,'base');high=P.confirmed(b,w,h,P.default());gt=L.gt(s)
                    pred=capture.get('pred')
                    if pred is None:
                        contracts.append({'profile':prof,'seed':sd,'stem':s,'status':'INCONCLUSIVE','reason':'fused/export forward did not expose logits'})
                        continue
                    with torch.inference_mode():
                        decoded=head._get_decode_boxes(pred)[0].T.detach().clone()
                        if not head.end2end and not head.xyxy:decoded=ops.xywh2xyxy(decoded)
                        decoded=ops.scale_boxes((z,z),decoded,(h,w)).cpu().numpy()
                    logits=pred['boxes'][0].detach();stride=head.strides.reshape(-1).detach().cpu().numpy()
                    cls=pred['scores'][0,0].sigmoid().detach().cpu().numpy();used=set();used_gt=set();matched=0
                    for box in high:
                        px=np.array(box[:4])*[w,h,w,h]
                        delta=np.max(np.abs(decoded-px),axis=1);indices=np.flatnonzero((delta<=.05)&(np.abs(cls-box[4])<=1e-5))
                        indices=[int(i) for i in indices if i not in used]
                        if len(indices)!=1:continue # correspondence ambiguity is not silently hidden
                        i=indices[0];used.add(i);matched+=1
                        p=logits[:,i].reshape(4,head.reg_max).float().softmax(1).cpu().numpy();bins=np.arange(head.reg_max)
                        mu=(p*bins).sum(1);var=(p*(bins-mu[:,None])**2).sum(1)
                        entropy=-(p*np.log(np.clip(p,1e-12,1))).sum(1)
                        gain=min(z/h,z/w);vx=(var[0]+var[2])*.25*(stride[i]/gain)**2;vy=(var[1]+var[3])*.25*(stride[i]/gain)**2
                        cx,cy=(box[0]+box[2])*w/2,(box[1]+box[3])*h/2
                        near=[(math.hypot(cx-(g[0]+g[2])*w/2,cy-(g[1]+g[3])*h/2),j) for j,g in enumerate(gt) if j not in used_gt]
                        near=[(d,j) for d,j in near if d<=8]
                        errors=[]
                        if near:
                            d,j=min(near);used_gt.add(j);errors=[d]
                        rows.append({'profile':prof,'seed':sd,'stem':s,'source_training_image':s in L.XC.profile_training_stems(L.CFG,prof),
                            'query_index':i,'matching_max_corner_delta_px':float(delta[i]),'score':box[4],
                            'side_entropy_mean':float(entropy.mean()),'center_variance_independence_proxy_xy':[float(vx),float(vy)],
                            'center_spread_independence_proxy':float(np.sqrt(vx+vy)),
                            'nearest_unused_GT_error_within8px':errors[0] if errors else None,
                            'not_a_calibrated_center_posterior':True})
                    rec=P.image_eval(b,gt,w,h,.25)
                    contracts.append({'profile':prof,'seed':sd,'stem':s,'reported_detections':len(high),'unique_logit_query_correspondence':matched,
                        'GT_objects':len(gt),'TP3':rec['tp3'],'FN3':len(gt)-rec['tp3'],'FP3':rec['fp3'],
                        'reg_max':head.reg_max,'status':'PASS' if matched==len(high) else 'INCONCLUSIVE_QUERY_CORRESPONDENCE'})
            finally:handle.remove()
            del model;torch.cuda.empty_cache()
    usable=[r for r in rows if r['nearest_unused_GT_error_within8px'] is not None and not r['source_training_image']]
    def corr(key):
        if len(usable)<3:return None
        x=[r[key] for r in usable];y=[r['nearest_unused_GT_error_within8px'] for r in usable]
        value=spearmanr(x,y).statistic
        return float(value) if np.isfinite(value) else None
    F.atomic_json(F.OUT/'DFL_DISTRIBUTION_DIAGNOSTIC.json',{'time':datetime.datetime.now().astimezone().isoformat(),
        'source_sha256':F.sha_file(F.ROOT/'experiments/followup/distribution_probe.py'),'evidence':'frozen representation / localization diagnostic',
        'rows':rows,'correspondence_contracts':contracts,'conditional_held_error_Spearman':{k:corr(k) for k in ('side_entropy_mean','center_spread_independence_proxy')},
        'limits':'Separate l/t/r/b marginals do not identify calibrated center uncertainty; independence approximation ignores side covariance. '
                 'Same-image seeds are repetitions. Nearest 8px association is conditional, not FN-inclusive detection success. '
                 'An object with no proposal has no entropy measurement. No operating policy uses this diagnostic.',
        'evaluation79_GT_reads':0,'formal_qualification':'UNQUALIFIED'})

if __name__=='__main__':main()
