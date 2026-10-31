"""Render factual reports from completed validation artifacts. Missing evidence stays BLOCKED."""
import collections
import csv
import json
import subprocess
import sys
from pathlib import Path

from experiments import lib as L


def read(p): return json.loads((L.ROOT/p).read_text(encoding='utf8'))


def text(p,s):
    p=L.ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s+'\n',encoding='utf8')


def f(v):
    return 'N/A' if v is None or v=='N/A' else f'{v:.4f}' if isinstance(v,(float,int)) else str(v)


def main():
    if not (L.OUT/'resume/OPERATIONAL_TIMING.json').exists():
        subprocess.run([sys.executable,'-u','-m','experiments.operational_timing'],cwd=L.ROOT,check=True)
    # The original supervisor may already be running from the interrupted-session resume.
    # Run this newly added mandatory stage at report time as well; never duplicate a completed run.
    if not (L.OUT/'resume/REPRODUCTION_REPORT.json').exists():
        subprocess.run([sys.executable,'-u','-m','experiments.isolated_reproduction'],cwd=L.ROOT,check=True)
    s=read('outputs/experiments/E3_scoring/RESULTS.json')
    a=read('outputs/experiments/E4_audit/RESULTS.json')
    q=read('outputs/experiments/E3_scoring/SAFETY_SIGNALS.json')
    v=read('outputs/validation/VERIFY_REPORT.json')
    e=read('outputs/experiments/E1_rules_x14/RESULTS.json')
    pred=read('outputs/test_predictions.json')['summary']
    seal=read('outputs/experiments/resume/EXECUTION_SEAL.json')
    reproduction=read('outputs/experiments/resume/REPRODUCTION_REPORT.json')
    timing=read('outputs/experiments/resume/OPERATIONAL_TIMING.json')
    legacy_max=max(x for F in e['families'].values() for C in F['conditions'].values() for x in C.get('x14_check',{}).values())
    tests=(L.OUT/'resume/tests.log').read_text(encoding='utf8')
    passed=v['VERIFY_PASS'] and legacy_max<=1e-12 and 'passed' in tests and 'failed' not in tests and reproduction['isolated_default_VERIFY_PASS']
    # No clean installed environment, human review accuracy, external qualification or full re-training
    # is inferred from a same-environment regression. These remain separate readiness requirements.
    lines=['# XGD-1.1 최종 검증 보고서','',f'GPU 추론/원본 전처리 재현: {v["VERIFY_PASS"]}; X14 독립 재채점 최대 차이 {legacy_max}.',
           '검출 가중치는 동결. 변경은 안전 판정·산출물 호환성·평가 코드이며 검출기 성능 향상으로 해석하지 않는다.',
           '주 평가 P-IND-strict는 반대 날짜 절반으로 품질 기준·CRC·운영 등급을 모두 만든다. 전체 자료 운영 등급 진단과 구분한다.',
           '개발 자료는 반복 열람된 development generalization benchmark. 독립 최종 검증 또는 임의 장비 보장으로 부르지 않는다.','',
           '| profile | seed | 3px recall | F1@3px | IoU.5 F1 | FN | FP/장 | P90(8px 조건부) | coverage | 전체 위험 | ACCEPT 조건부 위험 | 실패 포착 | 올바른 보류 |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for prof,F in s['profiles'].items():
        for sd,m in F['conditions']['clean']['per_seed'].items():
            d=m['P-IND-strict']
            lines.append(f'|{prof}|{sd}|{f(m["R3"])}|{f(m["F13"])}|{f(m["F1_iou"])}|{m["FN"]}|{f(m["FP_per_image3"])}|{f(m["err_P50_P90_P95"][1] if m["err_P50_P90_P95"] else None)}|{f(d["coverage"])}|{f(d["joint_risk"])}|{f(d["conditional_risk"])}|{f(d["failure_capture"])}|{f(d["correct_hold_rate"])}|')
        lines+=['',f'**{prof} 실제 최악 조건×seed:** {F["true_worst_condition_seed"]}',
                f'학습 장비·날짜/유사 중복 범위: {a["profile_audit"][prof]["train_device_counts"]}, 근접쌍 제외 보류 {len(a["profile_audit"][prof]["excluded_held_images"])}장.',
                f'안전 후보 개별 평가(자동 채택 아님): {q["profiles"][prof]["candidates"]}','']
    lines += ['## 동일 조건 모델 비교','',
              '같은 TXT/t2/recipe/seed와 같은 X8 입력 크기. 일반 YOLO seed17은 historical run이며 나머지는 이번 matched 비교 학습이다. 환경의 완전한 동일성까지 소급 증명하지 않는다.',
              '|개발 집합|모델|추론|3seed F1@3px 평균|3seed recall 평균|최악 recall|FP/장 평균|AP50(101point/E0/top30)|',
              '|---|---|---|---|---|---|---|---|']
    import numpy as np
    for prof,yp in (('dfl_src12','yolo_src12'),('dfl_src3_2','yolo_src3_2')):
        for mode in ('clean','native'):
            for name,rr in [('DFL',s['profiles'][prof]['conditions'][mode]['per_seed']),
                            ('YOLO26s',{sd:C[mode] for sd,C in s['controls']['baseline'][yp].items()})]:
                mm=list(rr.values())
                lines.append(f'|{prof}|{name}|{mode}|{f(np.mean([r["F13"] for r in mm]))}|{f(np.mean([r["R3"] for r in mm]))}|{f(min(r["R3"] for r in mm))}|{f(np.mean([r["FP_per_image3"] for r in mm]))}|{f(np.mean([r["AP50_101pt_E0_top30"] for r in mm]))}|')
    lines += ['', '## 최종 DFL 표시 흔적 대조','',
              '|집합|조건|DFL-v3tr 3seed 대상 hit 평균|v3only 3seed 대상 hit 평균|해석|','|---|---|---|---|---|']
    for prof in ('dfl_src12','dfl_src3_2'):
        for cond in ('R2e_trace','R2f_free_M1','R2f_free_M2','R2g_mark_M1','R2g_mark_M2'):
            final=[m['synthetic']['trace_hit_rate' if 'trace' in cond else 'new_hit_rate'] for m in s['profiles'][prof]['conditions'][cond]['per_seed'].values()]
            ctrl=[m[cond]['target_hits']/m[cond]['target_count'] if m[cond]['target_count'] else None for m in s['controls']['v3only'][prof].values()]
            lines.append(f'|{prof}|{cond}|{f(np.mean([x for x in final if x is not None]))}|{f(np.mean([x for x in ctrl if x is not None]))}|흔적 경보율은 낮을수록, 새 객체 hit는 높을수록 좋음. 실제 검출 및 3seed 결과를 함께 해석|')
    lines += ['', '## 고정 조건별 운영 검증','',
              '|profile|조건|3px 평균|실제 최악seed|FN 평균|FP/장 평균|운영seed17 coverage|운영seed17 ACCEPT 실패/ACCEPT|',
              '|---|---|---|---|---|---|---|---|']
    for prof,F in s['profiles'].items():
        for cond,C in F['conditions'].items():
            m=C['mean'];d=C['seed17']['P-IND-strict']
            count = f'{d["accepted_failures"]}/{d["accept"]}' if d['accept'] else 'N/A (ACCEPT 0)'
            lines.append(f'|{prof}|{cond}|{f(m.get("R3"))}|{f(C["worst"].get("R3"))}|{f(m["FN"])}|{f(m.get("FP_per_image3"))}|{f(d["coverage"])}|{count}|')
    lines+=['## 판정 변경과 한계','',
            'UNQUALIFIED/미등록/보정 불일치는 FULL, REVIEW 등급은 자동 ACCEPT 금지. 유효 입력 배율 범위 집행, CRC 동점 묶음, 검출 0개 REVIEW, 비정상 입력 차단을 회귀 검사했다.',
            'CRC의 α=.05는 교환가능성 가정하의 기대 전체 위험이며 ACCEPT 조건부 위험 5% 보장이 아니다. LTT 테스트가 수치상 성립하더라도 독립 교환가능성 전제는 미확인이다.',
            '정상 제품 0장: 정상 합격률/특이도·현장 인간 재검사 성능을 검증하지 않았다. ACCEPT는 검출 위치 자동 확정이며 제품 합격이 아니다.',
            'P90은 8px 안에 매칭된 예측의 조건부 값. 미탐 포함 2/3px 초과율·매칭률·FN·FP를 항상 같이 읽는다. ACCEPT 0이면 조건부 위험은 N/A다.',
            '물리적 시험편 ID가 없다. 유사 영상/배치 proxy를 찾아도 specimen-disjoint 인증이 되지 않는다. 장비2는 일부 학습에 포함돼 완전 unseen 장비가 아니다.',
            '현재 label 파일은 historical serialization이며 provider TXT와 수치 동등성 감사 결과가 있다. raw-byte provider original과 구분한다. historical training consumed bytes의 완전한 증명은 별도 한계다.',
            f'평가79 정답 사용0회; 고정 새 규칙 예측 분포: {pred["decisions"]}. 이 분포는 정확도가 아니며 규칙 재조정에 쓰지 않았다.',
            '전처리 손상/합성 generator/위치 sweep은 실제 새 이물·장비·제품 분포를 대체하지 않는다. bbox 종횡비·영상 대비는 재질·밀도 ground truth가 아니다.',
            '추가 signal의 내부 gate 통과는 독립 qualification이 아니므로 자동 배포 변경을 수행하지 않았다. 독립 정상/장비/시험편 calibration 이전 formal guarantee는 UNQUALIFIED.',
            '별도 작업 사본+새 가상환경에서 기본 실행을 재현했다. 가상환경은 설치된 고정 패키지를 공유하므로 fresh wheel 재설치·새 OS/GPU 검증과는 다르다.',
            '동일 원본/조리법/초기값에서 6개 DFL full 재학습 경로를 실행했다. 결과는 REPRODUCTION_REPORT.json에서 frozen DFL의 동일 seed/fold와 나란히 비교하며 재학습 가중치로 제출 모델을 교체하지 않았다.']
    lines+=['','## 실제 처리 시간과 재검사 시나리오','',
            '배치1, 모델 로드 후 1회 warmup, profile별 정렬된 개발 영상20장. 원본 읽기·전처리·GPU 추론·E0·판정을 포함한다. 공장 촬영/통신/사람 검수 시간은 제외한다.',
            '|profile|평균 초/장|P50/P90/P95 초|','|---|---|---|']
    for prof,t in timing['profiles'].items():
        lines.append(f'|{prof}|{f(t["mean_seconds"])}|{[round(x,4) for x in t["P50_P90_P95_seconds"]]}|')
    lines+=['','아래 작업자 시간은 측정 결과가 아닌 명시적 가정이다. REVIEW 10초, FULL 30초로 놓은 1000장 처리 시나리오다.',
            '|profile|clean 운영seed17 REVIEW율|FULL율|가정 검수시간 분/1000장|','|---|---|---|---|']
    for prof,F in s['profiles'].items():
        d=F['conditions']['clean']['seed17']['P-IND-strict']
        lines.append(f'|{prof}|{f(d["review_rate"])}|{f(d["full_rate"])}|{f(1000*(10*d["review_rate"]+30*d["full_rate"])/60)}|')
    text('docs/FINAL_VALIDATION_REPORT.md','\n'.join(lines))
    if a.get('examples'):
        text('docs/FAILURE_EXAMPLES.md','\n'.join(['# 운영 seed17 실제 FN/FP 사례','', '초록 = TXT GT, 빨강 = score≥.25 예측. 원본 좌표로 표시한 뒤 3배 확대했다.','']+
            [f'![{r["stem"]} FN={r["FN"]}, FP={r["FP"]}](../{r["file"]})' for r in a['examples']]))
    # Every major criticism is assigned a stable ID and an evidence boundary, not copied as a new claim.
    issues=[
        ('S01','§8','UNQUALIFIED도 ACCEPT','수정·검증 완료','tests/test_safety.py','enum과 운영 차단 집행'),
        ('S02','§8','배율 envelope 미집행','수정·검증 완료','tests/test_safety.py','0.95 허용오차·32px rounding 유지'),
        ('S03','§8','CRC 동점 경계 위험','수정·검증 완료','tests/test_safety.py','25장 반례에서 수용 없음'),
        ('S04','§8','검출0 자동 확정','수정·검증 완료','tests/test_safety.py','항상 REVIEW/FULL'),
        ('S05','§8','NaN/Inf/메타데이터/보정 불일치','수정·검증 완료','tests/test_safety.py','unknown 품질 상태·음수 λ도 차단'),
        ('S06','§5','CRC joint와 conditional 혼동','수정·검증 완료','docs/FINAL_VALIDATION_REPORT.md','각 분모·ACCEPT0 N/A·CI95 명시'),
        ('S07','§5','QUALIFIED를 확률 보장으로 해석','수정·검증 완료','src/xgd_analyzer.py','enum 운영 등급과 probability_guarantee 분리'),
        ('S08','§5','정상 화질 고신뢰도 실패','추가 실험으로 확인','outputs/experiments/E3_scoring/SAFETY_SIGNALS.json','3개 고정 신호 개별 평가; 자동 채택 아님'),
        ('S09','§5','LTT α=.05 미성립·표본 부족','가용 데이터로 확인 불가','docs/FINAL_VALIDATION_REPORT.md','독립 exchangeability/normal 자료 없음'),
        ('S10','§6','stress 전면 보류의 생산성 비용','추가 실험으로 확인','outputs/experiments/E3_scoring/RESULTS.json','coverage·correct hold·조건부 N/A'),
        ('S11','§6','under를 정상 운영으로 해석','근거상 지적 수정','outputs/experiments/E1_rules_x14/RESULTS.json','under/over/native 진단과 X8 운영 분리'),
        ('D01','§3','물리적 배율·pixel pitch 미제공','가용 데이터로 확인 불가','data/metadata.json','X8은 유효 network 입력 비율만'),
        ('D02','§7','specimen/re-shoot 인증 부재','가용 데이터로 확인 불가','outputs/experiments/E4_audit/RESULTS.json','image similarity/proxy만; 물리ID 없음'),
        ('D03','§7','exact hash만으로 근접 중복 누락','추가 실험으로 확인','outputs/experiments/E4_audit/RESULTS.json','435장 후보 search와 coarse NCC; 탐색 한계 공개'),
        ('D04','§7','실제 loader t2와 설명 split 다름','수정·검증 완료','weights/MANIFEST.json','validation_split t2를 canonical로 명시'),
        ('D05','§3','장비2를 완전 unseen으로 표현','수정·검증 완료','config.yaml','train 3호기100+2호기32, 새 날짜 평가'),
        ('D06','§7','품질 기준에 평가 절반 통계','추가 실험으로 확인','outputs/experiments/E3_scoring/RESULTS.json','strict P-IND 및 time grade/calibration 분리'),
        ('D07','§7','개발자료 반복 선택','가용 데이터로 확인 불가','outputs/experiments/resume/EXECUTION_SEAL.json','기존 adaptive exposure 제거 불가, 독립 claim 금지'),
        ('D08','§7','정상제품0·GT개수/위치 좁음','가용 데이터로 확인 불가','outputs/experiments/E4_audit/RESULTS.json','위치 합성은 독립 정상 자료 대체 안함'),
        ('D09','§7','hold65/test79 provenance와 독립성','수정·검증 완료','docs/RESUME_20261005.md','원본 forensic 회수와 historical 사용이력 구분; 65 미포함,79 GT 미사용'),
        ('D10','§7','historical label raw bytes','수정·검증 완료','data/metadata.json','현재 SHA/수치 provider 동등성, 과거 consumed bytes 한계'),
        ('M01','§2','일반 YOLO 3seed matched 비교 부족','추가 실험으로 확인','outputs/experiments/E3_scoring/RESULTS.json','같은 recipe/input size; historical17 환경 한계 공개'),
        ('M02','§2','F1/IoU 수치 누락','수정·검증 완료','outputs/experiments/E3_scoring/RESULTS.json','3/5px F1·IoU.5 F1, mAP 아님'),
        ('M03','§3','X8 성능과 판정 게이트 혼동','수정·검증 완료','docs/FINAL_VALIDATION_REPORT.md','동일 weights 좌표 개선 vs 위험/coverage 분리'),
        ('M04','§3','P90가 FN 제외','수정·검증 완료','experiments/lib.py','FN·매칭률·전체-object 초과율 병기'),
        ('M05','§6','최악조건 값이 평균인 문제','수정·검증 완료','outputs/experiments/E3_scoring/RESULTS.json','condition×seed 전역 최악 별도 산출'),
        ('R01','§7','정수 이동/stride phase 없음','추가 실험으로 확인','outputs/experiments/E3_scoring/RESULTS.json','14개 고정 이동·역좌표·unmatched·경계 제외'),
        ('R02','§7','local moved-object/position shortcut','추가 실험으로 확인','outputs/experiments/E3_scoring/RESULTS.json','M1/M2·old-site·updated GT'),
        ('R03','§7','위치 sweep·문맥 교체','추가 실험으로 확인','outputs/experiments/E3_scoring/RESULTS.json','10개구간·타 날짜 수신 제품·합성 한계'),
        ('R04','§7','주변만 열화·고신뢰 실패','추가 실험으로 확인','outputs/experiments/E3_scoring/RESULTS.json','객체+8px 보존, 주변σ1.2'),
        ('R05','§7','최종 DFL 흔적 무력화 인과 대조','추가 실험으로 확인','outputs/experiments/E3_scoring/RESULTS.json','v3only vs v3tr 3seed 고정; 사후 variant 없음'),
        ('R06','§7','표시제거가 tiny signal 훼손 가능','추가 실험으로 확인','outputs/experiments/E3_scoring/RESULTS.json','traceless vs fake-mark clean paired'),
        ('R07','§6','noise/PSF·배율 상호작용','추가 실험으로 확인','outputs/experiments/E3_scoring/RESULTS.json','고정 correlated noise/PSF/scale grid; 실제 scanner 아님'),
        ('R08','§2','크기×대비×edge FN분석 부족','추가 실험으로 확인','outputs/experiments/E4_audit/RESULTS.json','object별3seed miss, 수치proxy/빈범주 공개'),
        ('O01','§9','평가79 차단의 정확성 불명','가용 데이터로 확인 불가','outputs/test_predictions.json','정답미사용, 차단/coverage만'),
        ('O02','§9','사람·재검사 처리량 검증 없음','가용 데이터로 확인 불가','docs/RUBRIC_EVIDENCE.md','실제 현장 검증 불가, 가정 비용 시나리오만'),
        ('O03','§10','새 알고리즘/타팀 유일성 과장','수정·검증 완료','README.md','known DFL/CRC/LTT+문제 특성에 맞춘 검증 통합'),
        ('P01','§2','기존 PASS 복사/새출력을 기준으로 비교','수정·검증 완료','outputs/validation/VERIFY_REPORT.json','old reference 그대로, 새 GPU boxes와 비교'),
        ('P02','§2','오류인데 exit0','수정·검증 완료','src/verify.py','필수검사 FAIL면 nonzero, supervisor STOP'),
        ('P03','§2','새 설치 환경 재현 미검증','추가 실험으로 확인','outputs/experiments/resume/REPRODUCTION_REPORT.json','새 venv+standalone 사본; pinned 의존성 공유, clean install 미검증'),
        ('P04','§2','full 재학습 동일 성능 미검증','추가 실험으로 확인','outputs/experiments/resume/REPRODUCTION_REPORT.json','6개 동일 recipe 재학습의 실제 지표, 비트일치/개선 가정 없음'),
    ]
    table=['# 독립 평가 지적별 대응표','', '표의 추가 실험 확인은 실행 증거의 확보이며 성능 개선/shortcut 부재의 결론을 뜻하지 않는다.','',
           '|ID|검토 절/지적|재확인·수정/실험|관련 파일/결과 경로|완료 기준|현재 상태|남은 한계|','|---|---|---|---|---|---|---|']
    for ID,section,issue,state,path,note in issues:
        criterion='지적 경로 재현·회귀 PASS' if state=='수정·검증 완료' else '고정 실험 결과와 분모를 보존' if state=='추가 실험으로 확인' else '근거·해석 경계를 기록'
        limitation='독립 자료에서의 성능/안전 보장을 뜻하지 않음' if state!='가용 데이터로 확인 불가' else note
        table.append('|'+ '|'.join((ID,section+' '+issue,note,path,criterion,state,limitation))+'|')
    text('docs/ISSUE_RESPONSE_TABLE.md','\n'.join(table))
    rubric=['# 서면평가 근거 대응','', '|항목|배점|실제 근거|한계|','|---|---|---|---|',
            '|데이터 이해 및 진단|15|data/metadata.json, split.csv, E4_audit/RESULTS.json|물리 시험편ID·정상 제품·밀도 미제공|',
            '|AI 예측모델 개발|40|E3_scoring/RESULTS.json controls: YOLO26s/DFL 3seed, 3/5px F1·IoU.5 F1·pFROC|IoU F1≠mAP; 반복 개발benchmark|',
            '|영향요인 및 오류분석|15|E3 조건별·seed별·device/session, E4 객체 size/edge/contrast|shape/density는 영상proxy, 합성≠실제 scanner|',
            '|현장 활용방안|10|FINAL_VALIDATION_REPORT, test_predictions_images.csv|검출 위치ACCEPT≠제품합격; 사람검수효율 미측정|',
            '|창의성·차별성|10|표시 흔적 대조+유효입력배율+품질/위험/coverage 통합검증|DFL/CRC/LTT 기존연구, 타팀 제출물 미확인|',
            '|코드 및 재현성|10|run_all.py, requirements.txt, execution seal, old-reference GPU VERIFY|독립 clean-install/전면재학습 차이는 별도 검증|',
            '', '현장 절차: REVIEW는 표시 위치 작업자 재확인, FULL은 자동판정 중지와 장비/영상조건 점검·전수 재검사.',
            '새 정상 촬영일은 사람의 독립 확인→오프라인 reference 재작성→새 버전 검증을 거쳐 반영한다. 현재 실제 사람 승인을 받았다는 뜻은 아니다.',
            '비용 모델은 시나리오: 총 업무시간 = n_REVIEW×t_REVIEW + n_FULL×t_FULL. t를 실제 측정하지 않았으므로 생산성 개선으로 확정하지 않는다.']
    text('docs/RUBRIC_EVIDENCE.md','\n'.join(rubric))
    text('docs/CHANGELOG_XGD-1.1.md','\n'.join(['# XGD-1.1 변경 기록','',
        '원 XGD-1.0.0 및 중단 시점 XGD-1.1 코드는 versions에 보존. 원본 BMP/TXT·DFL weights·봉인 reference는 그대로다.',
        '판정: grade enum/envelope/tied CRC/검출0/비정상 입력/version-source compatibility 집행.',
        '검증: strict 보정설계와 전체등급 진단 구분, 조건부/전체 위험 분모, 실제condition×seed 최악.',
        '비교: 고정 YOLO baseline29/43와 v3only17/29/43 2profile을 새 학습; 최종 DFL 교체 없음.',
        'E1 규칙별 변경수: outputs/experiments/E1_rules_x14/RESULTS.json의 changes_v10_to_v11_P-X14_all_seeds.',
        f'평가79 기존39/8/32 → 새 분포 {pred["decisions"]}. 좌표 정확도 개선 주장 아님.']))
    # 2026-10-06 (resume_continue): result-linked issue table, rubric evidence, change log, limitations and the readiness decision are
    # generated from the completed artifacts by experiments/final_docs.py (overwrites the static tables above with measured findings).
    from experiments import final_docs
    readiness=final_docs.main(passed)
    L.jdump(L.OUT/'resume/PACKAGE_READINESS.json',readiness)
    subprocess.run([sys.executable,'-u','-m','experiments.package_preflight'],cwd=L.ROOT,check=True)
    print('reports written; ZIP','READY' if readiness['zip_ready'] else 'BLOCKED',readiness.get('blocking_items'),flush=True)
    if not passed: raise SystemExit(1)


if __name__=='__main__': main()
