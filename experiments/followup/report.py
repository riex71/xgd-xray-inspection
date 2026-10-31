"""Generate source-grounded reports and a verified submission allowlist, never fabricate unfinished evidence."""
import collections,datetime,json,re
from pathlib import Path
import numpy as np
from experiments import lib as L
from experiments.followup import common as F,analyze as A,selection as S,policy as P

def load(name):return json.loads((F.OUT/name).read_text(encoding='utf8'))
def number(v):return 'N/A' if v is None or not isinstance(v,(int,float)) else f'{v:.4f}'
def write(name,body):
    p=F.ROOT/'docs'/name;p.write_text(body,encoding='utf8')

def risk_curves(selection):
    curves={}
    for prof,item in selection['profiles'].items():
        curves[prof]={}
        for label,family,cfg in (('baseline','dfl',P.default()),('selected',item['family'],item['policy'])):
            _,rr,details=A.processed(family,prof,17,'dev_clean',cfg);points=[]
            # Descriptive score-threshold curve, not a source of new operating thresholds or qualification.
            for coverage in np.arange(.1,1.01,.1):
                lam=float(np.quantile([r['trust'] for r in rr.values()],1-coverage))
                decisions={s:'ACCEPT' if r['trust']>=lam and r['n_det']>0 and not details[s]['flag'] else 'REVIEW' for s,r in rr.items()}
                m=L.decision_metrics(decisions,rr);points.append({'coverage_target':float(coverage),'lambda_diagnostic':lam,**m})
            curves[prof][label]=points
    out={'curves':curves,'classification':'descriptive eligibility-fixed diagnostic; empirical device grade/Q gates are omitted here',
         'not_operating_policy':True,'not_threshold_selection':True,'formal_qualification':'UNQUALIFIED','sample_seed':17}
    F.atomic_json(F.OUT/'RISK_COVERAGE_DIAGNOSTIC.json',out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,(prof,lines) in zip(axs,curves.items()):
        for label,pts in lines.items():ax.plot([p['coverage'] for p in pts],[p['joint_risk'] for p in pts],marker='o',label=label)
        ax.set(xlabel='Accepted image fraction (diagnostic)',ylabel='P(ACCEPT and failure)',title=prof);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Internal development risk-coverage; no formal qualification');fig.tight_layout()
    fig.savefig(F.OUT/'risk_coverage.png',dpi=180);plt.close(fig)

def submission_manifest(config):
    files=set();excluded=['AGENTS.md','CLAUDE.md','versions/','outputs/followup/cache/','outputs/followup/reproduction/fresh_venv/',
        '__pycache__/','.env','credentials','intermediate epoch weights','absolute-path training logs/args','evaluation GT']
    for pattern in ('src/*.py','experiments/*.py','experiments/followup/*.py','tests/*.py','tools/*.py','docs/*.md',
                    'requirements.txt','README.md','config.yaml','data/metadata.json','data/split.csv','data/validation_split.json',
                    'data/v3_pixel_sha256.json','data/models/*.yaml','data/initial/*.pt','data/raw/*.bmp','data/labels_orig/*.txt',
                    'data/clean_v3/images/*.png','data/clean_v3/labels/*.txt','data/v3tr_orig/images/*.png','data/v3tr_orig/labels/*.txt',
                    'weights/*.pt','weights/MANIFEST.json','outputs/calibration/*.json','outputs/followup/deployment/*.json',
                    'outputs/followup/deployment/*.csv','outputs/followup/deployment/test_predictions_yolo/*.txt'):
        files.update(p for p in F.ROOT.glob(pattern) if p.is_file())
    for name in ('FINAL_SELECTION.json','ANALYSIS_EXECUTION_LOCK.json','COMPLETED_PIPELINE_CONTRACT.json','RESOLUTION_COMPATIBILITY.json',
                 'LTT_PROCEDURE_DIAGNOSTICS.json','SYNTHESIS_GEOMETRY_PHOTOMETRY_AUDIT.json','DFL_DISTRIBUTION_DIAGNOSTIC.json','RISK_COVERAGE_DIAGNOSTIC.json','risk_coverage.png'):
        files.add(F.OUT/name)
    for key in config['weights']:
        fam,prof,sd=key.split('/');files.add(F.checkpoint(fam,prof,int(sd)))
    selected_trace=any(i['family']=='trace05' for i in config['profiles'].values())
    if selected_trace:
        files.update(p for p in (F.ROOT/'data/followup_trace05').rglob('*') if p.is_file())
        files.update(p for p in (F.OUT/'train/trace05').glob('*.json') if p.is_file())
    for prof,item in config['profiles'].items():
        files.add(F.OUT/f'SELECTED_VALIDATION_{prof}_{item["family"]}_{item["policy"]["id"]}.json')
    for old in ('E1_rules_x14','E3_scoring','E4_audit','E6_data_profile','E8_failure_geometry','E9_missed_object_blindspot'):
        path=L.OUT/old/'RESULTS.json'
        if path.exists():files.add(path)
    flags=[];rows=[]
    for p in sorted(files):
        rel=p.relative_to(F.ROOT).as_posix();reason=[]
        if not p.exists():reason.append('missing required file')
        elif p.suffix.lower() in ('.md','.py','.txt','.csv','.json','.yaml','.yml'):
            text=p.read_text(encoding='utf8-sig',errors='replace')
            if re.search(r'[A-Za-z]:[\\/]+Users[\\/]',text,re.I):reason.append('personal absolute path')
            if re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',text):reason.append('credential pattern')
        if reason:flags.append({'path':rel,'reasons':reason})
        if p.exists():rows.append({'path':rel,'bytes':p.stat().st_size,'sha256':F.sha_file(p)})
    result={'status':'READY_FOR_REVIEW' if not flags else 'BLOCKED_SANITIZATION_REQUIRED','zip_created':False,
            'files':rows,'excluded':excluded,'flags_paths_only':flags,
            'blind_scan_scope':'absolute personal paths and high-confidence credential patterns; human name/logo visual review still required',
            'data_rights':'KAMP licensed project data; do not infer redistribution rights from private/public repository status',
            'checkpoint_selection':'only registered final/representative checkpoints, no epoch sweep artifacts'}
    F.atomic_json(F.OUT/'SUBMISSION_ALLOWLIST.json',result)
    return result

def main():
    final=load('FINAL_SELECTION.json');selection=final['selection'];config=load('deployment/CONFIG.json')
    comparison=load('FINAL_COMPARISONS.json');contract=load('COMPLETED_PIPELINE_CONTRACT.json');fresh=load('reproduction/FRESH_REPRODUCTION.json')
    pred=load('deployment/TEST_PREDICTIONS.json');resolution=load('RESOLUTION_COMPATIBILITY.json');now=datetime.datetime.now().astimezone().isoformat()
    intro=f'# XGD 후속 실험 결과\n\n실행 시각: {now}. 최종 서명: `{config["pipeline_signature"]}`. 기존 1.1 산출물은 versions/XGD-1.1.0과 기존 outputs에 보존했다.\n\n'
    table='| profile | family / policy | R3 baseline→selected | F1 baseline→selected | FP/img baseline→selected | worst R3 | P90 baseline→selected |\n|---|---|---|---|---|---|---|\n'
    decision_table='| profile | coverage | joint risk | conditional risk | REVIEW | FULL | cost50 |\n|---|---|---|---|---|---|---|\n'
    failures=[]
    for prof,item in selection['profiles'].items():
        r=load(f'SELECTED_VALIDATION_{prof}_{item["family"]}_{item["policy"]["id"]}.json');base=r['baseline_results']['dev_clean'];new=r['candidate_results']['dev_clean']
        p90=lambda z:np.mean([x['err_P50_P90_P95'][1] for x in S.seeds(z)])
        table+=f'| {prof} | {item["family"]} / {item["policy"]["id"]} | {S.mean(base,"R3"):.4f}→{S.mean(new,"R3"):.4f} | {S.mean(base,"F13"):.4f}→{S.mean(new,"F13"):.4f} | {S.mean(base,"FP_per_image3"):.4f}→{S.mean(new,"FP_per_image3"):.4f} | {min(x["R3"] for x in S.seeds(new)):.4f} | {p90(base):.3f}→{p90(new):.3f} |\n'
        for label,z in (('baseline',base),('selected',new)):
            for sd,m in z['per_seed'].items():
                d=m['decisions'][S.DESIGN];decision_table+=f'| {prof}/{sd}/{label} | {number(d["coverage"])} | {number(d["joint_risk"])} | {number(d["conditional_risk"])} | {number(d["review_rate"])} | {number(d["full_rate"])} | {number(d["assumed_cost_50"])} |\n'
        for c,z in r['candidate_results'].items():
            for sd,m in z['per_seed'].items():failures.extend({'profile':prof,'condition':c,**x} for x in m['rows'] if x['fail'])
    F.atomic_json(F.OUT/'FINAL_FAILURE_ROWS.json',failures)
    risk_curves(selection)
    limits='\n원본 TXT는 primary GT다. 8px 진단 매칭은 미검출과 3px 위치 실패를 분해할 뿐 성공 기준을 바꾸지 않는다. 같은 seed/영상 반복은 독립 표본이 아니다. 합성 donor/recipient는 날짜·보정 역할을 넘을 수 있으며 추가 독립 표본으로 세지 않는다. 고신뢰 이미지 그룹은 실제 시험편 ID를 인증하지 않는다. 이 pipeline의 실제 정상 제품 검증 0장, qualified 외부 독립 장비/시험편 자료 부재, 반복 개발 노출로 formal qualification=UNQUALIFIED다. ACCEPT는 검출 위치 확인만 의미하며 product pass/completeness 보장이 아니다.\n'
    write('FOLLOWUP_EXPERIMENTS.md',intro+table+'\n29개 정책의 전체 비교는 outputs/followup/FINAL_COMPARISONS.json, 강도 0/0.5/1과 source 선택 사유는 FINAL_SELECTION.json에 있다. 결과 후 추가 값을 만들지 않았다. 후보 탈락과 기준선 복귀 이유도 보존했다.\n\nDFL side 분포 진단은 DFL_DISTRIBUTION_DIAGNOSTIC.json에 있다. l/t/r/b 주변 분포를 center posterior로 부르지 않고, covariance=0 근사로 구한 분산 proxy와 8px 조건부 오차의 관계만 보고했다. 검출이 없는 실제 객체의 분포는 관측하지 못하므로 공통 미탐 안전 신호로 채택하지 않았다.\n'+limits)
    write('RISK_COVERAGE_TRADEOFF.md',intro+decision_table+'\n가정 비용: 실제 미검출 ACCEPT 10/50/100(50 주), 위치/FP-only ACCEPT 10, REVIEW 1, FULL 3. 실제 사람 확인 정확도/시간은 미측정이다. 곡선 risk_coverage.png는 등급/Q gate를 고정하지 않은 기술적 진단이며 운영 정책을 대신하지 않는다. 실제 운영 결과는 표의 날짜/그룹 분리 결과다.\n'+limits)
    write('FINAL_SELECTION.md',intro+table+'\n```json\n'+json.dumps(selection['profiles'],ensure_ascii=False,indent=2)+'\n```\n\n선택→고정 공격 확인→복귀는 내부 adaptive 개발이며 independent final validation이 아니다. 평가79 정답을 사용하지 않고 고정 뒤 한 번 생성했다.\n'+limits)
    write('RESOLUTION_COMPATIBILITY.md',intro+'개발 등록부는 deployment/CALIBRATION.json의 profile/device/정확한 native 폭×높이/촬영일/서명으로 구성한다. ROI·실제 촬영 설정·물리 배율은 unknown이다.\n\n```json\n'+json.dumps(resolution,ensure_ascii=False,indent=2)+'\n```\n\n실제 평가316 영상은 개발 crop 시험 통과 여부와 무관하게 자동 등록하지 않았다.\n')
    write('CHANGELOG_XGD-1.2.md',intro+'기존 detector/source/GT/원 1.1 체크포인트를 보존했다. 실제 채택은 FINAL_SELECTION 표의 정책/강도뿐이다. 새 해상도 등록부, 보정 서명, 완결성 미인증 필드를 추가했다. 새 설치 추론 검증은 FRESH_REPRODUCTION.json, completed batch 계약은 COMPLETED_PIPELINE_CONTRACT.json에 있다. 과거 초기 cache/TEMP/WHEEL harness 실패는 별도 로그에 보존했으며 모델 성능 실패로 집계하지 않는다.\n'+table)
    for name in ('FINAL_SUMMARY.md','FINAL_VALIDATION_REPORT.md','ISSUE_RESPONSE_TABLE.md','RUBRIC_EVIDENCE.md','LIMITATIONS.md','FAILURE_EXAMPLES.md'):
        p=F.ROOT/'docs'/name;original=p.read_text(encoding='utf8');marker='\n<!-- FOLLOWUP_FINAL_UPDATE -->\n'
        if marker in original:raise RuntimeError('final report update already present; no silent overwrite')
        extra=marker+intro+table+decision_table+limits+'\n후속 근거: FOLLOWUP_EXPERIMENTS.md → FINAL_SELECTION.md → RISK_COVERAGE_TRADEOFF.md → RESOLUTION_COMPATIBILITY.md. 숫자는 기존 역사 결과와 합산하지 않는다.\n'
        if name=='RUBRIC_EVIDENCE.md':extra+='\n서면평가 연결: 데이터15(DEV356 원본 lineage·그룹·촬영일), 모델40(동일조건 DFL와 채택 후보 3seed), 오류15(FINAL_FAILURE_ROWS.json의 미검출/위치/FP), 활용10(coverage·REVIEW/FULL·실측 latency), 차별성10(완결성 미인증·리스크 trade-off·해상도 서명), 재현10(실제 fresh wheel 설치·batch/coordinate 계약). 타 팀의 유일성/우월성을 주장하지 않는다.\n'
        p.write_text(original+extra,encoding='utf8')
    readme=F.ROOT/'README.md';readme.write_text('# 현재 실행 버전: XGD-1.2.0\n\n최종 선택과 제한은 [docs/FINAL_SELECTION.md](docs/FINAL_SELECTION.md)에 있다. 기존 1.1은 versions/XGD-1.1.0에 보존했다.\n\n후속 재실행: `python -m experiments.followup.run`(완료 cache/lock 보존, 중복 실행 금지). 최종 단일 영상: `python -m src.followup_analyzer --image sample.bmp --metadata metadata.json --profile dfl_src12 --out result.json`. 1·2호기는 dfl_src3_2, 3호기는 dfl_src12이며 장비 보정을 다른 장비에 라우팅하지 않는다.\n\n'+readme.read_text(encoding='utf8'),encoding='utf8')
    allow=submission_manifest(config)
    F.atomic_json(F.OUT/'PACKAGE_READINESS.json',{'status':allow['status'],'zip_created':False,'fresh_install_inference':fresh['status'],
        'completed_contract':contract['status'],'formal_qualification':'UNQUALIFIED','evaluation79_GT_reads':0,
        'evaluation79_images':pred['n_images'],'blocking_paths':allow['flags_paths_only'],'different_GPU':'UNTESTED'})
    print('source-grounded follow-up docs and submission allowlist generated',flush=True)

if __name__=='__main__':main()
