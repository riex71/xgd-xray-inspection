"""XGD-1.2 sections placed at the top of the generated core documents (FINAL_SUMMARY, ISSUE_RESPONSE_TABLE, RUBRIC_EVIDENCE, LIMITATIONS).

Called by experiments/final_docs.py when outputs/followup/FOLLOWUP_COMPLETED.json exists; the XGD-1.1 content that follows is kept as the
baseline record. Numbers come only from completed follow-up artifacts (docs12.Art); nothing is selected or tuned here.
"""
import collections

from experiments.followup import docs12 as D
from experiments.followup import tables as T

FAM = T.FAM


def available():
    return (D.OUT / "FOLLOWUP_COMPLETED.json").exists()


def _final(a, prof):
    it = a.final_item(prof)
    return it, (a.r(it["family"], it["policy"]["id"], prof) if it else None), a.r("dfl", "P0", prof)


def _fmt_item(it):
    if not it:
        return "미완료"
    fam = {"dfl": "DFL(흔적 1.0)", "trace05": "DFL(흔적 0.5배)", "v3only": "DFL(흔적 0)"}.get(it["family"], it["family"])
    return f"{fam} + {it['policy']['id']} {D.NAMES.get(it['policy']['id'], '')}".strip()


def summary_section(a):
    f = D.f
    L = ["# XGD-1.2 결과 요약 (현재 버전)", "",
         "XGD-1.2는 1.1을 기준선으로 보존한 채 후속 지시문의 우선순위대로 후보를 사전 등록·비교·선택·재보정한 버전이다. 모든 수치는 같은 공급처 개발 자료의 내부 결과이고 독립 외부 검증이 아니다. "
         "**ACCEPT는 검출한 이물 위치만 확인한다는 뜻이며 제품 합격이나 영상 전체 완결성(놓친 이물 없음) 보장이 아니다.**", ""]
    L += ["## 채택한 변경", ""]
    for prof in T.PROFILES:
        it, x, b = _final(a, prof)
        L.append(f"- {FAM[prof]}: {_fmt_item(it)}")
    L += ["- 모든 장비: 보정 등록부에 없는 native 해상도는 FULL(사유 출력), 파이프라인 서명(정책·추가 모델·의존성 포함) 불일치는 FULL, 출력에 `confirmation_scope`·`completeness_status`·`product_pass_certified=false` 추가.", ""]
    L += ["## 기준선(1.1 정책, 같은 설계로 재보정) 대비", "",
          "| 장비군 | 지표 | 기준선 | XGD-1.2 |", "| --- | --- | --- | --- |"]
    for prof in T.PROFILES:
        it, x, b = _final(a, prof)
        if not (x and b):
            continue
        for name, key, p in (("정상 3px 재현율(3 seed 평균)", "R3", 4), ("정상 F1@3px", "F13", 4), ("정상 오탐/장", "FP", 3), ("최저 seed 3px 재현율", "R3_worst", 4),
                             ("정상 자동 확정 A(날짜·그룹 분리 보정)", "clean_A", 3), ("14조건 평균 전체 위험", "risk_mean", 4),
                             ("14조건 평균 놓침 자동 확정 위험(8px 진단)", "FN8_risk_mean", 4), ("합성 조건 놓침 자동 확정(등급 고정 진단)", "synth_FN8_risk_diag", 3),
                             ("가정 비용50(14조건)", "cost50", 3), ("모델 호출/장", "calls", 1)):
            L.append(f"| {FAM[prof]} | {name} | {f(b[key], p)} | {f(x[key], p)} |")
    L += ["", "## 우선순위별 결과", "",
          "| 문제 | 대응(후보) | 측정 결과 | 남은 한계 |", "| --- | --- | --- | --- |"]
    best = {}
    for prof in T.PROFILES:
        it, x, b = _final(a, prof)
        best[prof] = (it, x, b)
    def pick(cid, prof, key, p=3):
        r = a.r("dfl", cid, prof)
        return f(r[key], p) if r else "N/A"
    tabs = (a.sel or {}).get("selection", {}).get("selection_tables", {})

    def status(cid):
        """'3호기용(1·2호기 자료로 판정) 통과/탈락(사유)' for both directions; tables are keyed by target profile."""
        out = []
        for target, tab in tabs.items():
            src = next(p for p in T.PROFILES if p != target)
            v = tab["policies"].get(cid)
            if v is None:
                continue
            why = "; ".join(dict.fromkeys(r.split(":")[-1].strip() for r in v["reasons"]))
            out.append(f"{FAM[target]}용({FAM[src]} 자료) {'통과' if v['admissible'] else '탈락'}" + (f"({why})" if why else ""))
        return ", ".join(out)
    L.append("| 놓친 이물이 자동 확정됨 | 잔여 저점수 후보(P1), seed·view·대조 모델 불일치(P2), 의심 영상 합집합(P3) | " +
             "; ".join(f"{FAM[p]} 놓침 자동 확정 위험 {f(best[p][2]['FN8_risk_mean'], 4)} → {f(best[p][1]['FN8_risk_mean'], 4)}, 정상 A {f(best[p][2]['clean_A'])} → {f(best[p][1]['clean_A'])}"
                       for p in T.PROFILES if best[p][1] and best[p][2]) +
             ". **채택: P2_bundle_6**(비용50 최소). 추가 호출 없는 잔여 후보 확인 P1_low0.05(장당 호출 1회)도 두 방향 모두 통과했다. 선택 표 비용50은 " +
             ", ".join(f"{FAM[t]}용({FAM[next(p for p in T.PROFILES if p != t)]} 자료) P1 {f(tab['policies']['P1_low0.05']['cost50'])} 대 묶음 {f(tab['policies']['P2_bundle_6']['cost50'])}"
                       for t, tab in tabs.items() if 'P1_low0.05' in tab['policies'] and 'P2_bundle_6' in tab['policies']) +
             "로, 한쪽은 거의 같고 다른 쪽은 묶음 확인이 뚜렷이 낮아 사전 규칙(비용 최소)상 선택되지 않았다. 처리 시간이 중요하면 검토할 대안이다. | "
             "'전부 동의'한 공통 놓침은 남는다. 합성 이물 근거이며 실제 현장 놓침 빈도가 아니다. 장당 모델 호출이 1 → 8회로 늘어난다 |")
    L.append("| 3~4px 위치 오차 | 뒤집기·배율 TTA(L1·L2), 원천 편향 보정(L3), view 흩어짐(L4), 중복 억제 2/6px | " +
             "; ".join(f"{FAM[p]} L1 정상 R3 {pick('L1_flip', p, 'R3', 4)}(기준 {pick('P0', p, 'R3', 4)})" for p in T.PROFILES) +
             f". L1: {status('L1_flip')}. L2·L3·L4·E0도 관문 또는 비용 기준 미달. **채택 없음**(정책은 하나만 선택). | "
             "장비별 일정한 중심 치우침(E10)은 원천 장비 보정으로 설명되지 않는다. 라벨 중심 오차 여부는 미확인. TTA와 완결성 확인의 결합은 다음 버전 후보 |")
    fam_rows = [(p, a.r("trace05", "P0", p), a.r("v3only", "P0", p), a.r("dfl", "P0", p)) for p in T.PROFILES]
    rej = {p: [h for h in (a.sel or {}).get("validation_history", {}).get(p, []) if not h["passed"]] for p in T.PROFILES}
    L.append("| 흔적 증강의 검출 손실·합성 이물 둔감 | 강도 0 / 0.5배(신규 6개) / 1.0 | " +
             "; ".join(f"{FAM[p]} 합성 실제 놓침 비율 1.0 {f(d['synth_actual_missing_rate'])}, 0.5배 {f(t['synth_actual_missing_rate'])}, 0 {f(v['synth_actual_missing_rate'])}; "
                       f"정상 R3 {f(d['R3'], 4)}/{f(t['R3'], 4)}/{f(v['R3'], 4)}" for p, t, v, d in fam_rows if t and v and d) +
             ". 선택 단계에서 3호기용은 흔적 0, 1·2호기용은 0.5배가 뽑혔지만 고정 검증에서 탈락: " +
             "; ".join(f"{FAM[p]} {h['item']['family']}({', '.join(h['reasons'][:3])})" for p in T.PROFILES for h in rej[p]) + ". **채택 없음(흔적 1.0 유지)** | "
             "흔적 0.5배는 P0끼리 비교하면 합성 놓침이 줄었지만, 묶음 확인과 함께 쓰면 위험은 채택안보다 낮지 않고 M2 계열 자동 확정만 늘었다. "
             "한 조건의 영상 1장 증가로 탈락 — 비열등성 기준을 미리 등록해 다음 버전에서 재검증할 후보(FINAL_SELECTION) |")
    L.append("| 약한 열화의 과차단 | FULL 경계 8, REVIEW 경계 4·5, 신호 하나 | " +
             "; ".join(f"{FAM[p]} 정상 A Q_4_8 {pick('Q_4_8', p, 'clean_A')}·Q2만 {pick('Q_only_Q2', p, 'clean_A')}(기준 {pick('P0', p, 'clean_A')})" for p in T.PROFILES) +
             f". Q_4_8: {status('Q_4_8')}; Q2만: {status('Q_only_Q2')}. **채택 없음** | E11: 운영 신호 Q4가 실패가 적은 열화에도 경보. 품질 변화는 실패 확률이 아님 |")
    L.append("| 다른 해상도 입력 | 보정 등록부(정확한 폭×높이) | 등록부 밖 해상도 → FULL(사유 출력). **채택(안전 수정)**. 개발 1호기 crop/canvas 시험은 진단 | 316×332 1호기는 ROI·촬영 설정을 몰라 미등록 |")
    L.append("| 작은 보정 자료 | 등급 절차 G1/G2, LTT 절차 비교 | " +
             "; ".join(f"{FAM[p]} G2 정상 A {pick('GRADE_G2', p, 'clean_A')}·위험 {pick('GRADE_G2', p, 'risk_mean', 4)}(기준 {pick('P0', p, 'risk_mean', 4)})" for p in T.PROFILES) +
             f". G1: {status('GRADE_G1')}; G2: {status('GRADE_G2')}. **채택 없음** | LTT α 0.05는 실패 0건에서도 자동 확정 독립 영상 45~90장이 필요. 교환가능성 미확인 → 확률 보장 없음 |")
    L += [""]
    if a.pred:
        reasons = collections.Counter()
        for v in a.pred["images"].values():
            for r in v["reason"]:
                if "해상도" in r:
                    reasons[(v["device"], "미등록 해상도")] += 1
                elif "X12 정상 envelope" in r:
                    reasons[(v["device"], "품질 FULL")] += 1
                elif "경계 구간" in r:
                    reasons[(v["device"], "품질 REVIEW")] += 1
        L += ["## 평가 79장(정답 미사용, 판정 분포만)", ""] + D.transitions(a) + [
              "- 사유(영상 수, 한 영상에 여러 사유 가능): " + ", ".join(f"{d} {k} {n}" for (d, k), n in sorted(reasons.items())),
              "- 원인 1: 1호기 32장은 해상도 316×332가 1호기 보정 등록부(352×332)에 없어 FULL이다(1.2 해상도 규칙).",
              "- 원인 2: 2호기의 1.2 품질 기준은 CRC와 같은 **보류 개발 영상(6월 5개 촬영일, 100장)**으로만 만든다. "
              "1.1은 1·2호기용 검출기 학습에 쓴 8/31·9/1 촬영 32장까지 넣어(132장) Q4 기준이 중심 0.134·척도 0.039로 넓었고, 1.2 기준은 중심 0.156·척도 0.010이다. "
              "평가 2호기 45장(2020년 9월 촬영)의 Q4는 0.026~0.100으로, 1.1 기준 z는 −2.8~−0.9(정상 범위)였지만 1.2 기준 z는 −12.6~−5.5다. "
              "학습에만 쓰여 보류 라벨로 검출 성능이 검증된 적 없는 9월형 촬영 조건이므로 42장 FULL, 3장 REVIEW가 된다. "
              "이 설계(배포 보정 = 보류 개발 라벨)는 03:41 봉인 계약에서 정했고, 평가 79장 분포를 보고 바꾸지 않았다.",
              "- 그래서 1.2는 평가 79장에서 자동 확정이 0장이다. 이는 성능 저하나 개선의 증거가 아니라, 검증된 운영 범위 밖 입력을 사람 확인으로 넘긴 결과다."]
    if a.fresh or a.contract:
        L += ["", "## 재현", "",
              f"- 운영 코드와 캐시 판정 일치·배치1/4 좌표 계약: {a.contract.get('status') if a.contract else '미완료'}",
              f"- 새 의존성 설치 venv(기존 site-packages 미사용)에서 회귀 테스트·동결 추론 재현: {a.fresh.get('status') if a.fresh else '미완료'}; 다른 GPU 미검증."]
    L += ["", "상세: `docs/FOLLOWUP_EXPERIMENTS.md` → `docs/FINAL_SELECTION.md` → `docs/RISK_COVERAGE_TRADEOFF.md` → `docs/RESOLUTION_COMPATIBILITY.md` → `docs/CHANGELOG_XGD-1.2.md`.",
          "", "---", "", "아래는 XGD-1.1 기준선 결과(역사 기록)다.", ""]
    return "\n".join(L)


def issue_section(a):
    f = D.f
    rows = []
    def add(i, item, action, result, kind, path):
        rows.append(f"| F{i:02d} | {item} | {action} | {result} | {kind} | {path} |")
    sel = a.sel or {}
    fin = {p: a.final_item(p) for p in T.PROFILES}
    adopted_policy = {p: (fin[p]["policy"]["id"] if fin[p] else None) for p in T.PROFILES}
    adopted_family = {p: (fin[p]["family"] if fin[p] else None) for p in T.PROFILES}
    tabs = sel.get("selection", {}).get("selection_tables", {})
    adopted_ids = sorted({c for c in adopted_policy.values() if c})

    def kind_for(cids, bundle_component=False):
        if any(adopted_policy[p] in cids for p in T.PROFILES):
            return "1 채택"
        if bundle_component and any((adopted_policy[p] or "").startswith("P2_bundle") for p in T.PROFILES):
            return "1 채택(묶음 확인의 구성 요소)"
        passed = [f"{c}({sum(bool(t['policies'].get(c, {}).get('admissible')) for t in tabs.values())}/{len(tabs)}방향)"
                  for c in cids if any(t["policies"].get(c, {}).get("admissible") for t in tabs.values())]
        if passed:
            return f"미채택 — {', '.join(passed)} 관문 통과(괄호 = 통과 방향 수). 통과한 방향에서도 비용 규칙상 {', '.join(adopted_ids)} 선택"
        return "4 기준선 유지(관문·위험·비용 기준 미달)"
    add(1, "놓친 이물 자동 확정(E9)", "영상 검출 완결성 확인 후보 P1/P2/P3 비교 후 선택", "; ".join(f"{FAM[p]} {adopted_policy[p]}" for p in T.PROFILES),
        kind_for([c for c in D.NAMES if c.startswith(("P1", "P2", "P3"))]), "docs/FOLLOWUP_EXPERIMENTS.md §1, outputs/followup/FINAL_SELECTION.json")
    add(2, "출력 점수와 잔여 후보 분리", "출력 점수 0.15/0.35(P1b)와 잔여 확인(P1)을 따로 비교", "표 참조", kind_for(["P1_low0.05", "P1_low0.1", "P1_low0.15", "P1b_score0.15", "P1b_score0.35"]), "docs/FOLLOWUP_EXPERIMENTS.md §1")
    add(3, "예측 간 누락·개수·위치 불일치", "seed·view·v3only·묶음, 반경 3/6px", "표 참조", kind_for([c for c in D.NAMES if c.startswith("P2")]), "docs/FOLLOWUP_EXPERIMENTS.md §1")
    add(4, "의심 영상 재검사 합집합", "의심 영상만/항상 추가 view 합집합, 재보정", "표 참조", kind_for(["P3_always", "P3_triggered"]), "docs/FOLLOWUP_EXPERIMENTS.md §1")
    add(5, "C3(seed 불일치)의 조건부 재평가", "위험–자동 확정 비용 시나리오 10/50/100", "docs/RISK_COVERAGE_TRADEOFF.md", kind_for(["P2_seed_3", "P2_seed_6"], bundle_component=True), "docs/RISK_COVERAGE_TRADEOFF.md")
    add(6, "확인 범위·완결성 출력", "confirmation_scope·completeness_status·product_pass_certified 필드", "운영 출력에 추가", "1 채택", "src/followup_analyzer.py, src/followup_safety.py")
    add(7, "3~4px 위치 오차 분해", "8px 진단 매칭으로 '빗나감'과 '없음' 분리(3px 기준 유지)", "실패 행 FINAL_FAILURE_ROWS.json", "1 채택(보고 지표)", "outputs/followup/FINAL_FAILURE_ROWS.json")
    add(8, "x/y 부호 오차 분포", "장비·촬영일·seed·크기·위치별(E10)", "3호기 dx·dy ≈ −0.35px, 1·2호기 dx ≈ +0.9px의 일정한 치우침", "기술 통계", "outputs/experiments/E10_localization_bias/")
    add(9, "TTA 좌표 결합", "뒤집기(L1)·배율(L2) 중심 중앙값", "표 참조", kind_for(["L1_flip", "L2_scale"]), "docs/FOLLOWUP_EXPERIMENTS.md §2")
    add(10, "중심 편향 보정", "원천 장비 학습 영상으로만 추정(L3, zero-shot)", "표 참조", kind_for(["L3_source_bias"]), "docs/FOLLOWUP_EXPERIMENTS.md §2")
    add(11, "좌표 불확실성 신호", "view 흩어짐(L4), DFL 변 분포 진단", "DFL 분포는 보정된 중심 불확실성이 아님", kind_for(["L4_view_spread"]), "outputs/followup/DFL_DISTRIBUTION_DIAGNOSTIC.json")
    add(12, "합성 방법 충돌(M1/M2)", "M2 ROI 정합 버전 M2v2, 짝지은 시험, 합성 점검", f"ROI 어긋남 {a.syn['legacy_shifted_pairs']}/{a.syn['pairs']}쌍" if a.syn else "미완료", "1 채택(시험 방법)", "outputs/followup/SYNTHESIS_GEOMETRY_PHOTOMETRY_AUDIT.json")
    add(13, "흔적 증강 강도", "0 / 0.5배(신규 6개) / 1.0 비교", "; ".join(f"{FAM[p]} 최종 {adopted_family[p]}" for p in T.PROFILES),
        "1 채택" if any(adopted_family[p] in ("trace05", "v3only") for p in T.PROFILES) else "4 기준선 유지", "docs/FOLLOWUP_EXPERIMENTS.md §3")
    add(14, "해상도 호환성", "보정 등록부와 미등록 해상도 FULL", "등록부 밖 해상도 자동 확정 차단", "1 채택(안전 수정)", "docs/RESOLUTION_COMPATIBILITY.md")
    add(15, "FULL↔REVIEW 배분", "FULL 경계 8(자동 확정 경계 유지)", "표 참조", kind_for(["Q_3_8"]), "docs/FOLLOWUP_EXPERIMENTS.md §4")
    add(16, "자동 확정 경계 완화", "REVIEW 4·5 / FULL 8, 재보정 후 열화 위험", "표 참조", kind_for(["Q_4_8", "Q_5_8"]), "docs/FOLLOWUP_EXPERIMENTS.md §4")
    add(17, "품질 신호 필요성·추가 신호", "Q2만/Q4만, E11 추가 신호(진단)", "E11: Q4가 약한 열화에 과민", kind_for(["Q_only_Q2", "Q_only_Q4"]), "outputs/experiments/E11_quality_signals/")
    add(18, "작은 보정 자료 등급 절차", "G1(촬영일 LODO), G2(분할 평가 없음)", "표 참조", kind_for(["GRADE_G1", "GRADE_G2"]), "docs/FOLLOWUP_EXPERIMENTS.md §6")
    add(19, "LTT 절차·표본 요건", "10점/3점/1점 Bonferroni, 고정 순서", "확률 보장 불성립(교환가능성 미확인)", "4 기준선 유지(보고만)", "outputs/followup/LTT_PROCEDURE_DIAGNOSTICS.json")
    add(20, "중복 억제 거리", "E0 2/6px", "표 참조", kind_for(["E0_2", "E0_6"]), "docs/FOLLOWUP_EXPERIMENTS.md §2")
    add(21, "고신뢰/약한 그룹 분리", "NCC ≥ 0.998 연결 성분은 분리 단위, NCC ≥ 0.98 등은 민감도", "3호기 13장 이동으로 일부 촬영일이 두 절반에 걸침", "1 채택(평가 설계)", "outputs/followup/ANALYSIS_EXECUTION_LOCK.json")
    add(22, "선택·보정·평가 분리", "반대 장비군 선택, 대상 장비군 날짜·그룹 분리 보정", "선택 충돌 시 사전 규칙", "1 채택(평가 설계)", "docs/FOLLOWUP_SELECTION_CONTRACT.md")
    add(23, "불변 캐시·감시기", "서명·완료 manifest·원자적 게시, 상태 파일만 감시", "중단 3회(05:09 tuple/list 비교, 08:34 진단의 CPU/GPU head 불일치, 11:43 인코딩 이름)를 봉인 파일 수정 없이 래퍼로 재개", "1 채택", "experiments/followup/common.py, outputs/followup/RESUME_AMENDMENT_01~03.json")
    add(24, "새 위험 회귀 테스트", "잔여 후보·근접 객체·해상도·서명·고정 순서 LTT·미완료 캐시 등", "pytest 통과(로그)", "1 채택", "tests/test_followup_*.py")
    add(25, "새 의존성 설치 재현", "기존 site-packages 없이 고정 wheel 설치 후 테스트·동결 추론", a.fresh.get("status") if a.fresh else "미완료", "1 채택", "outputs/followup/reproduction/")
    add(26, "평가 79장 최종 예측", "최종 고정 뒤 한 번, 정답 미사용", f"{a.pred.get('decision_counts') if a.pred else '미완료'}", "1 채택", "outputs/followup/deployment/TEST_PREDICTIONS.json")
    add(27, "제출 목록", "AGENTS.md·CLAUDE.md 제외, 개인 경로·자격 증명 검사", (f"{a.allow.get('status')}, 파일 {len(a.allow.get('files', []))}개·{a.allow.get('total_bytes', 0) / 1e6:.1f}MB, ZIP 미생성. 60MB 넘는 텍스트 1개는 같은 규칙으로 따로 검사(ALLOWLIST_LARGE_FILE_SCAN.json)" if a.allow else "미완료"), "1 채택", "outputs/followup/SUBMISSION_ALLOWLIST.json, PACKAGE_READINESS.json")
    add(28, "보고서·발표 반영", "기존 원본은 옛 모델 기준이라 수정하지 않고 반영 자료·체크리스트 작성", "docs/REPORT_MATERIAL_XGD-1.2.md", "1 채택", "docs/REPORT_MATERIAL_XGD-1.2.md")
    add(29, "외부 자료", "적합성·라이선스·중복 확인 가능한 외부 정상/이물 X-ray 없음", "사용하지 않음", "4 자료 부족", "docs/LIMITATIONS.md")
    L = ["# 후속 지시문(2026-10-06) 항목별 처리", "",
         "처리 유형: 1 채택 / 2 trade-off 선택 / 3 완화 시나리오(채택 아님) / 4 기준선 유지·자료 부족. 표의 수치는 docs/FOLLOWUP_EXPERIMENTS.md에 있다.", "",
         "| ID | 항목 | 조치 | 결과 | 유형 | 근거 |", "| --- | --- | --- | --- | --- | --- |"] + rows + ["", "---", "", "아래는 XGD-1.1 독립 평가 지적 대응표(역사 기록)다.", ""]
    return "\n".join(L)


def rubric_section(a):
    L = ["# 서면평가 6개 항목 근거 — XGD-1.2 추가분", "",
         "- **데이터 15:** 고신뢰 근접 중복 그룹(NCC ≥ 0.998)과 약한 시험편 proxy를 구분했다. 날짜·그룹 분리 보정 설계, 평가 1호기 형식 변화(E7). "
         "근거: `outputs/followup/ANALYSIS_EXECUTION_LOCK.json`(group_date_splits), `outputs/experiments/E7_eval79_reasons/`.",
         "- **모델 40:** 같은 조건 기준선(1.1 정책 재보정)과 후보 29개·흔적 강도 3종을 3 seed로 비교하고, 사전 규칙으로 선택했다. "
         "근거: `docs/FOLLOWUP_EXPERIMENTS.md`, `docs/FINAL_SELECTION.md`, `outputs/followup/comparisons/final_*`.",
         "- **오류 15:** 3px 실패를 놓침(8px 안 예측 없음)·위치 오차·원거리 오탐으로 나누고(FINAL_FAILURE_ROWS), 부호 오차(E10)와 합성 방법별 짝 비교를 했다. "
         "근거: `outputs/followup/FINAL_FAILURE_ROWS.json`, `outputs/experiments/E10_localization_bias/`.",
         "- **활용 10:** 자동 확정(검출 위치만)·REVIEW(완결성 불일치 시 위치 후보 표시)·FULL(해상도·품질·서명)의 흐름, 위험–자동 확정–비용 trade-off, 실측 처리 시간. "
         "근거: `docs/RISK_COVERAGE_TRADEOFF.md`, `outputs/followup/deployment/TEST_PREDICTIONS.json`.",
         "- **차별성 10:** 검출 완결성 확인을 비용 기준으로 선택하고, 해상도 등록부·파이프라인 서명과 결합했다. 기존 방법 인용, 타 팀 우월성 주장 없음. 근거: `docs/FINAL_SELECTION.md`.",
         "- **재현 10:** 새 의존성 설치 venv 재현, 배치1/4·좌표 계약, 불변 캐시와 재개 기록. "
         "근거: `outputs/followup/reproduction/FRESH_REPRODUCTION.json`, `outputs/followup/COMPLETED_PIPELINE_CONTRACT.json`.", "", "---", "",
         "아래는 XGD-1.1 기준선 근거(역사 기록)다.", ""]
    return "\n".join(L)


def limitations_section(a):
    calls = []
    for p in T.PROFILES:
        it, x, b = _final(a, p)
        if x:
            calls.append(f"{FAM[p]} {D.f(x['calls'], 1)}회")
    L = ["# 남은 한계 — XGD-1.2 추가분", "",
         "1. 완결성 확인은 여러 예측이 **함께 놓친** 이물을 잡지 못한다. 합성 시험에서 줄어든 위험은 실제 현장 놓침 빈도의 증거가 아니다.",
         "2. 불일치 확인은 모델 호출을 늘린다" + (f"(장당 {', '.join(calls)})" if calls else "") + ". 처리 시간은 측정했지만 다른 GPU·현장 장비에서는 검증하지 않았다.",
         "3. 그룹(NCC ≥ 0.998)을 지키려고 3호기 13장을 다른 절반으로 옮겨 일부 촬영일이 보정·평가 양쪽에 걸친다. 약한 proxy(NCC ≥ 0.98)는 한 덩어리라 분리 단위로 쓰지 못했다.",
         "4. 정책 선택은 반대 장비군 자료로 했지만, 선택이 충돌하면 두 장비군 결과를 함께 써서 조정했다. 전체 분석은 같은 개발 자료를 반복해서 본 내부 검증이다.",
         "5. 장비별로 일정한 중심 치우침(3호기 −0.35px, 1·2호기 dx +0.9px)이 있다. 원천 장비 보정으로는 설명되지 않고, 라벨 중심 관례인지 모델 편향인지 구분하지 못했다.",
         "6. 흔적 증강 강도 비교는 합성 이물(M2/M2v2)에 크게 의존한다. M2v2는 ROI 정합만 고친 것이며 대비·PSF 사실성을 인증하지 않는다.",
         "7. 해상도 등록부는 정확한 폭×높이만 승인한다. 316×332 1호기처럼 새 형식을 쓰려면 그 조건의 라벨 보정 자료가 필요하다.",
         "8. 외부 정상·이물 X-ray 자료는 적합성·권리·중복을 확인할 수 있는 것이 없어 쓰지 않았다. 정상 제품 특이도와 '이물 없음' 자동 판정은 여전히 평가할 수 없다.",
         "9. 2호기 9월형 촬영 조건(8/31·9/1, Q4가 낮은 잡음 특성)은 1·2호기용 검출기 학습에만 쓰여 보류 라벨이 없다. 그래서 1.2는 9월 평가 2호기 영상을 품질 FULL/REVIEW로 보내고, "
         "평가 79장 자동 확정은 0장이다. 이 조건을 자동 확정하려면 학습에 쓰지 않은 같은 조건의 라벨 영상으로 보정해야 한다.", "", "---", "",
         "아래는 XGD-1.1 기준선의 한계(역사 기록)다.", ""]
    return "\n".join(L)
