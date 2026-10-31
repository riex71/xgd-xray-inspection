"""Newly signed XGD follow-up runtime. Inference reads no TXT / GT / expected object count.

Run after selection/recalibration:
python -m src.followup_analyzer --image sample.bmp --metadata metadata.json --out result.json
The baseline src.xgd_analyzer and XGD-1.1 artifacts remain immutable.
"""
import argparse,copy,json,math,time,importlib.metadata as dependency_metadata,sys
from pathlib import Path
import numpy as np
from experiments import lib as L
from experiments.followup import common as F,policy as P
from src import xgd_analyzer as X
from src.followup_safety import confirmation_schema,resolution_compatible,pipeline_signature

CONFIG=F.OUT/'deployment/CONFIG.json'

def runtime_sources():
    result={p:F.sha_file(F.ROOT/p) for p in ('src/followup_analyzer.py','src/followup_safety.py',
        'experiments/followup/policy.py','experiments/followup/common.py','experiments/lib.py',
        'src/xgd_analyzer.py','src/xgd_config.py','src/preprocess.py','config.yaml','data/metadata.json',
        'data/validation_split.json','data/v3_pixel_sha256.json')}
    result['runtime_versions']={k:dependency_metadata.version(k) for k in ('torch','torchvision','ultralytics','numpy','scipy','opencv-python','PyYAML')}
    result['Python_version']=list(sys.version_info[:3])
    return result

def sign(config):return pipeline_signature(config['profiles'],config['weights'],runtime_sources())

class FollowupAnalyzer:
    def __init__(self,profile,seed=17,device=None,config_path=CONFIG):
        self.config=json.loads(Path(config_path).read_text(encoding='utf8'));self.profile=profile;self.seed=int(seed)
        self.item=self.config['profiles'][profile];self.cfg=self.item['policy'];self.family=self.item['family'];self.models={}
        self.actual_signature=sign(self.config);self.compatible=self.actual_signature==self.config['pipeline_signature']
        for key,expected in self.config['weights'].items():
            family,prof,sd=key.split('/')
            self.compatible &= F.sha_file(F.checkpoint(family,prof,int(sd)))==expected
        cal_path=F.ROOT/self.config['calibration_file']
        self.cal=json.loads(cal_path.read_text(encoding='utf8'))
        self.compatible &= F.sha_file(cal_path)==self.config['calibration_sha256'] and self.cal['pipeline_signature']==self.actual_signature
        self.device=device

    def model(self,family,seed):
        key=(family,seed)
        if key not in self.models:
            m=F.load_model(family,self.profile,seed)
            if self.device is not None:m.an.device=self.device
            expected=self.config['weights'].get(f'{family}/{self.profile}/{seed}')
            if expected!=m.an.ckpt_sha256:self.compatible=False
            self.models[key]=m
        return self.models[key]

    def invalid(self,reasons):
        return {'decision':'FULL_REINSPECTION','reason':reasons,'detections':[],'object_count':0,'trust':None,
                'rules_version':self.config['version'],'pipeline_signature':self.actual_signature,
                'model':{'profile':self.profile,'seed':self.seed,'family':self.family},
                'formal_qualification':'UNQUALIFIED',**confirmation_schema(False,'FULL_REINSPECTION')}

    def analyze(self,image_path=None,image=None,metadata=None):
        started=time.perf_counter()
        if not self.compatible:return self.invalid(['모델/정책/전처리/보정/의존성 서명 불일치: 재보정 필요'])
        if not isinstance(metadata,dict):return self.invalid(['장비 메타데이터 객체가 필요함'])
        md=dict(metadata);dev=md.get('calibration_id');pre=md.get('preprocessed',False)
        if not isinstance(pre,bool):return self.invalid(['preprocessed는 bool이어야 함'])
        img=X.read_image(image_path) if image is None else image
        problems=X.input_problems(img,dev)
        if problems:return self.invalid(problems)
        clean=img if pre else X.protocol(L.CFG)[0](img);h,w=clean.shape[:2]
        img_problems=X.input_problems(clean,dev)
        if img_problems:return self.invalid(img_problems)
        main=self.model(self.family,self.seed);views={};count=0
        def view(v):
            nonlocal count
            if v not in views:views[v]=F.predict_view(main,clean,v)[0];count+=1
            return views[v]
        view('base');others=[];control=None
        if self.cfg['safety'] in ('seed','bundle') or self.cfg['localization']=='triggered_union':
            for sd in L.SEEDS:
                if sd!=self.seed:others.append(F.predict_view(self.model(self.family,sd),clean,'base')[0]);count+=1
        if self.cfg['safety'] in ('v3only','bundle'):
            control=F.predict_view(self.model('v3only',self.seed),clean,'base')[0];count+=1
        needed=set(P.required_views(self.cfg))
        if self.cfg['localization']=='triggered_union':
            high=P.confirmed(views['base'],w,h,self.cfg)
            low=P.residual_flag(views['base'],high,w,h,self.cfg['low'],self.cfg['score'])
            disagreement=any(P.match(high,P.confirmed(b,w,h,self.cfg),w,h,self.cfg['radius'])[1:]!=(0,0) for b in others)
            if not (low or disagreement):needed={'base'}
        for v in F.VIEWS:
            if v in needed:view(v)
        bias=self.item.get('source_bias',{}).get(str(self.seed),(0.,0.))
        boxes,flag,detail=P.process(views,others,control,w,h,self.cfg,bias)
        trust=X.trust_of(boxes,self.cfg['score']);confirmed=[b for b in boxes if b[4]>=self.cfg['score']]
        signals=L.signals_of(clean);problems=X.input_problems(clean,dev,signals,boxes,trust)
        pkg=self.cal['profiles'][self.profile]['devices'].get(dev)
        compatible,reasons=resolution_compatible(self.cal['profiles'][self.profile]['registry'].get(dev),[int(w),int(h)],self.actual_signature)
        if pkg is None:
            d='FULL_REINSPECTION';why=['미등록 장비 보정'];z={};state='FULL_REINSPECTION'
        else:
            # JSON keys are strings; no calibration override bypass.
            cp=copy.deepcopy(pkg);cp['seeds']={int(k):v for k,v in cp['seeds'].items()}
            d,why,z=P.decision(cp,self.seed,signals,{'trust':trust,'n_det':len(confirmed)},flag,self.cfg,w,h)
            zm=(max(abs(z[q]) for q in self.cfg['signals']) if all(q in z and X._finite(z[q]) for q in ('Q2','Q4')) else math.inf)
            state='FULL_REINSPECTION' if zm>self.cfg['z_full'] else 'REVIEW' if zm>self.cfg['z_review'] else 'NORMAL'
        if not self.compatible:problems.append('모델/정책/전처리/보정 서명 불일치: 재보정 필요')
        if not compatible:problems.append(reasons)
        if problems:d='FULL_REINSPECTION';why=problems+why
        detections=[{'bbox_xyxy':[b[0]*w,b[1]*h,b[2]*w,b[3]*h],
                     'center':[(b[0]+b[2])*w/2,(b[1]+b[3])*h/2],'confidence':b[4]} for b in confirmed]
        sc=main.scale(w,h)
        return {'decision':d,'reason':why,'detections':detections,'object_count':len(detections),'trust':trust,
            'scale':sc,'quality':{'signals':signals,'z':z,'state':state},'model_calls':count,
            'localization_spread':detail['localization_spread'],'signals_of_disagreement':detail['flags'],
            'rules_version':self.config['version'],'pipeline_signature':self.actual_signature,
            'model':{'profile':self.profile,'seed':self.seed,'family':self.family,'checkpoint_sha256':main.an.ckpt_sha256},
            'native_resolution_registered':compatible,'elapsed_seconds':time.perf_counter()-started,
            'formal_qualification':'UNQUALIFIED','probability_guarantee':False,
            **confirmation_schema(flag,d)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--image',required=True);ap.add_argument('--metadata',required=True)
    ap.add_argument('--profile',default='dfl_src12');ap.add_argument('--seed',type=int,default=17);ap.add_argument('--out',required=True)
    a=ap.parse_args();md=json.loads(Path(a.metadata).read_text(encoding='utf8'))
    result=FollowupAnalyzer(a.profile,a.seed).analyze(image_path=a.image,metadata=md)
    F.atomic_json(a.out,result)

if __name__=='__main__':main()
