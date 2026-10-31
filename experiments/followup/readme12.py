"""Model-folder README for the current version (XGD-1.2.0), built from completed artifacts (no new values).

python -m experiments.followup.readme12      -> README.md (the previous README is kept in versions/XGD-1.1.0/README.md)
"""
import json

from experiments import lib as L
from experiments.followup import docs12 as D
from experiments.followup import tables as T

FAM = T.FAM


def main():
    a = D.Art()
    f = D.f
    fin = {p: a.final_item(p) for p in T.PROFILES}
    famname = {"dfl": "DFL(흔적 증강 1.0)", "trace05": "DFL(흔적 증강 0.5배)", "v3only": "DFL(흔적 증강 0)"}
    rows = {p: (a.r(fin[p]["family"], fin[p]["policy"]["id"], p) if fin[p] else None, a.r("dfl", "P0", p)) for p in T.PROFILES}
    e6 = json.loads((L.OUT / "E6_data_profile/RESULTS.json").read_text(encoding="utf8"))
    lat = (a.pred or {}).get("latency_mean_P95_seconds")
    calls_txt = ", ".join(f"{FAM[p]} {f(rows[p][0]['calls'], 1)}" for p in T.PROFILES if rows[p][0])
    L_ = ["# 과제④ X-ray 이물 위치 탐지·재검사 판정 분석기 (XGD-1.2.0)", "",
          "완제품 X-ray 영상에서 작은 이물(대부분 폭 5~21px)의 위치를 찾아 표시하고, 결과마다 세 판정 중 하나를 붙인다.", "",
          "- **ACCEPT:** 검출한 이물 위치를 자동으로 확인한다.",
          "- **REVIEW:** 작업자가 확인한다. 예측 사이에 불일치가 있으면 위치 후보를 표시한 채 REVIEW로 보낸다.",
          "- **FULL_REINSPECTION:** 전수 재검사하고 장비·촬영 조건을 점검한다.", "",
          "> **ACCEPT는 '검출한 위치를 자동 확인'한다는 뜻이며 제품 합격이나 '놓친 이물 없음'(완결성)의 보장이 아니다.** 출력에 `confirmation_scope=detected_object_positions_only`, "
          "`completeness_status`(NOT_CERTIFIED 또는 INCONSISTENT), `product_pass_certified=false`를 함께 적는다. 개발 자료에 정상 제품 영상이 없어 검출 0개 영상은 항상 REVIEW다.", "",
          "모든 수치는 같은 공급처가 준 개발 영상 356장 안의 내부 검증 결과이며 독립 외부 검증이 아니다. 기준선 XGD-1.1.0은 `versions/XGD-1.1.0/`에 보존했다.", "",
          "## 1. XGD-1.2에서 바뀐 것", ""]
    for p in T.PROFILES:
        it = fin[p]
        if it:
            L_.append(f"- **{FAM[p]}:** 검출기 {famname.get(it['family'], it['family'])}, 판정 정책 `{it['policy']['id']}` — {D.NAMES.get(it['policy']['id'], '')}")
    L_ += ["- **해상도 등록부:** 장비 보정에 등록되지 않은 native 해상도는 FULL로 보내고 사유를 출력한다.",
           "- **파이프라인 서명:** 정책·추가 모델 가중치·의존성 버전까지 서명한다. 서명이 맞지 않으면 FULL로 보내 재보정을 요구한다.",
           "- 바뀐 파이프라인은 모두 다시 보정했다. 선택은 사전 등록(`docs/PREREG_FOLLOWUP.md`)과 결과 전 봉인 계약(`docs/FOLLOWUP_SELECTION_CONTRACT.md`)을 따랐다.", "",
           "## 2. 한눈에 보기 (개발 보류 폴드, 3 seed 평균, 날짜·그룹 분리 보정 설계)", "",
           "| 장비군 | 지표 | 기준선(1.1 정책, 같은 설계 재보정) | XGD-1.2 |", "| --- | --- | --- | --- |"]
    for p in T.PROFILES:
        x, b = rows[p]
        if not (x and b):
            continue
        for name, key, prec in (("정상 3px 재현율", "R3", 4), ("정상 F1@3px", "F13", 4), ("정상 오탐/장", "FP", 3), ("최저 seed 3px 재현율", "R3_worst", 4),
                                ("정상 자동 확정 비율 A", "clean_A", 3), ("14조건 평균 전체 위험 P(ACCEPT∧실패)", "risk_mean", 4),
                                ("14조건 평균 놓친 이물 자동 확정 위험", "FN8_risk_mean", 4), ("모델 호출/장", "calls", 1)):
            L_.append(f"| {FAM[p]} | {name} | {f(b[key], prec)} | {f(x[key], prec)} |")
    L_ += ["", "- 14조건 = 정상 1 + 합성 이물(M1·M2·M2v2: 추가·이동·원위치·다른 날짜 문맥) 12 + 흔적만 1. 합성 조건 결과는 실제 현장 놓침 빈도의 증거가 아니다.",
           "- 3px 일대일 중심 매칭이 주 기준이다. 4/5px 성공은 현장 허용 오차 시나리오일 뿐 모델 개선이 아니다.",
           "- 날짜·그룹 분리 보정에서 운영 등급(QUALIFIED)이 성립한 보정 절반은 3호기 2개 중 1개(61장), 1·2호기 4개 중 2개(1호기 79장, 2호기 80장)뿐이다. 나머지 절반의 영상은 정상이어도 FULL이 되므로 정상 자동 확정 비율 A가 낮다(정상 FULL 3호기 0.61, 1·2호기 0.71). 배포 보정(보류 전체)은 세 장비 모두 QUALIFIED다.", ""]
    if a.pred:
        L_ += ["## 3. 평가 79장 (정답 미사용, 판정 분포만)", "", f"- XGD-1.2: {a.pred.get('decision_counts')}",
               "- 1.1 → 1.2 전이: " + ", ".join(f"{k} {v}" for k, v in sorted(a.pred.get("baseline_to_new_transition", {}).items())),
               "- 1호기 32장은 해상도 316×332가 1호기 보정(352×332)에 등록되지 않아 FULL이다.",
               "- 2호기 45장(FULL 42, REVIEW 3)은 품질 기준이 달라진 결과다. 1.2는 품질 기준을 CRC와 같은 보류 개발 영상(6월 100장)으로만 만들고, 1.1은 검출기 학습에 쓴 8/31·9/1 영상 32장까지 넣었다. 9월 평가 영상의 Q4 z가 1.1 기준 −2.8~−0.9에서 1.2 기준 −12.6~−5.5로 바뀌었다(`docs/CHANGELOG_XGD-1.2.md`).",
               "- 정확도 비교가 아니며, 분포를 보고 규칙을 바꾸지 않았다. 1.2의 평가 79장 자동 확정 0장은 검증된 운영 범위 밖 입력을 사람 확인으로 넘긴 결과다.", ""]
    L_ += ["## 4. 데이터", "",
           "- 제공 BMP 435장 = 개발 356장 + 평가 79장. 영상 1장은 완제품 1개의 검사다. 정상 제품은 0장이다.",
           "- 장비: 1호기 352×332(6~7월), 2호기 316×332(6월, 8/31·9/1), 3호기 576×444(6월). 평가 79장은 모두 9월이다.",
           "- 정답은 제공 TXT다. 색 표시는 정답이나 특징으로 쓰지 않고 지운다(v3). 평가 79장 정답은 폴더에 없고 읽지 않았다.",
           "- 근접 중복: NCC ≥ 0.98 기준은 공통 제품 배경 때문에 1·2호기 327장이 한 덩어리가 된다. 그래서 분리 단위로는 NCC ≥ 0.998 고신뢰 그룹만 쓴다. 물리적 시험편 ID는 없다.", ""]
    L_ += ["## 5. 분석 흐름 (XGD-1.2, `src/followup_analyzer.py`)", "",
           "원본 BMP → ① 색 표시 제거(v3) → ② 입력 유효 배율 정규화(X8, 물리 배율 보정 아님) → ③ DFL 검출 + 4px 중복 제거 → "
           "④ 검출 완결성 확인(seed 17·29·43, 좌우·상하 뒤집기와 배율 ±32 view, 흔적 무력화 없는 대조 모델의 확정 검출이 6px 안에서 일대일로 맞는지) → "
           "⑤ 촬영 품질 검사(X12: Q2·Q4, |z| > 3 REVIEW, > 6 FULL) → ⑥ 장비별 CRC(기대 전체 위험 α 0.05, 다시 보정한 λ) → ⑦ 해상도 등록부·서명 확인 → 판정", "",
           "- FULL 조건: 입력·신호 오류, 보정 없음·불일치·서명 불일치, 등급 UNQUALIFIED, 유효 배율 범위 밖, 품질 FULL, 미등록 native 해상도.",
           "- REVIEW 조건: 품질 REVIEW, 등급 REVIEW, 검출 0개, 신뢰도 < λ, 예측 간 불일치(완결성 미확인).",
           "- ACCEPT: 위 조건이 하나도 없을 때만.",
           f"- 연산 비용: 장당 모델 호출 {calls_txt}회"
           + (f", 평가 79장 종단 시간 평균 {f(lat[0])}초·P95 {f(lat[1])}초/장(RTX 5070, 모델 적재 제외)." if lat else "."), ""]
    L_ += ["## 6. 주장하지 않는 것", "",
           "임의의 새 장비 보장, 모든 실패 감지(여러 예측이 함께 놓친 이물은 남는다), 자동 확정 오류율이 항상 5% 이하, 정상 제품 자동 합격, 물리적 시험편 완전 분리, "
           "shortcut 부재. QUALIFIED는 경험적 운영 등급이고, LTT 확률 보장은 교환가능성이 확인되지 않아 주장하지 않는다. 남은 한계는 [LIMITATIONS](docs/LIMITATIONS.md)에 있다.", "",
           "## 7. 실행", "", "```bash", "pip install -r requirements.txt",
           "python run_all.py predict12     # XGD-1.2: 평가 79장 재판정 + 봉인된 최종 예측과 비교(다르면 실패)",
           "python -m src.followup_analyzer --image sample.bmp --metadata meta.json --profile dfl_src12 --out result.json   # 영상 1장",
           "python -m pytest -q tests -p no:cacheprovider", "python run_all.py               # XGD-1.1 기준선 경로(보존용)", "```", "",
           "- `meta.json` 예: `{\"calibration_id\": \"3호기\", \"preprocessed\": false}`. 1·2호기는 `dfl_src3_2`, 3호기는 `dfl_src12` profile을 쓴다. 다른 장비의 보정으로 라우팅하지 않는다.",
           "- 재학습 경로: DFL `python -m src.train_dfl --all`, 흔적 0 대조 `python -m experiments.train_controls v3only --seeds 17 29 43`, 흔적 0.5배 `python -m experiments.followup.train_trace05_resume`.",
           "- 후속 실험 전체: `python -m experiments.followup.run_resume`(완료된 불변 캐시는 건너뜀). 결과 문서는 `python -m experiments.followup.docs12`, `python -m experiments.final_docs`로 다시 만든다.",
           "- 새로 설치한 의존성 venv(기존 site-packages 미사용)에서 회귀 테스트와 동결 추론을 재현했다(`outputs/followup/reproduction/FRESH_REPRODUCTION.json`). 다른 GPU는 검증하지 않았다.", "",
           "## 8. 폴더 구성", "", "| 경로 | 내용 |", "| --- | --- |",
           "| `src/followup_analyzer.py`, `src/followup_safety.py` | XGD-1.2 운영 코드(완결성 확인·해상도 등록부·서명) |",
           "| `src/xgd_analyzer.py` 등 | XGD-1.1 분석기·전처리·학습·지표(보존) |",
           "| `config.yaml` | 공통 설정(1.1 값). 1.2 설정은 `outputs/followup/deployment/CONFIG.json` |",
           "| `weights/` | DFL 6개(profile 2 × seed 17·29·43). 대조·흔적 0.5배 가중치는 `outputs/controls/v3only/`, `outputs/followup/train/trace05/` |",
           "| `data/` | 원본 BMP, 개발 TXT, 메타데이터, 분할, 전처리 영상 |",
           "| `experiments/followup/` | 1.2 후속 실험(캐시·예측·비교·선택·보정·보고) |",
           "| `outputs/followup/deployment/` | 1.2 설정·보정·평가 79장 예측(JSON·CSV·YOLO TXT) |",
           "| `outputs/followup/` | 비교 결과, 선택·검증 기록, 재현 기록, 제출 목록 |",
           "| `outputs/experiments/` | 1.1 검증 실험 E1~E9, 후속 진단 E10·E11 |",
           "| `docs/` | 결과 요약, 후속 실험, trade-off, 최종 선택, 해상도, 변경 기록, 지적 대응표, 서면평가 근거, 한계, 보고서 반영 자료 |", "",
           "## 9. 문서 읽는 순서", "",
           "1. [FINAL_SUMMARY](docs/FINAL_SUMMARY.md): 1.2 결과 요약과 1.1 기준선.",
           "2. [FOLLOWUP_EXPERIMENTS](docs/FOLLOWUP_EXPERIMENTS.md) → [FINAL_SELECTION](docs/FINAL_SELECTION.md) → [RISK_COVERAGE_TRADEOFF](docs/RISK_COVERAGE_TRADEOFF.md) → [RESOLUTION_COMPATIBILITY](docs/RESOLUTION_COMPATIBILITY.md).",
           "3. [ISSUE_RESPONSE_TABLE](docs/ISSUE_RESPONSE_TABLE.md), [RUBRIC_EVIDENCE](docs/RUBRIC_EVIDENCE.md), [LIMITATIONS](docs/LIMITATIONS.md), [CHANGELOG_XGD-1.2](docs/CHANGELOG_XGD-1.2.md), [사례 그림](docs/FAILURE_EXAMPLES_XGD-1.2.md).",
           "4. 보고서·발표 반영: [REPORT_MATERIAL_XGD-1.2](docs/REPORT_MATERIAL_XGD-1.2.md). 제출 목록: `outputs/followup/SUBMISSION_ALLOWLIST.json`."]
    (L.ROOT / "README.md").write_text("\n".join(L_) + "\n", encoding="utf8")
    print("README.md written", flush=True)


if __name__ == "__main__":
    main()
