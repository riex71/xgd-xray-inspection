"""Readable XGD-1.2 documents built only from completed follow-up artifacts (no new values, no selection).

Overwrites the terse versions written by report.py:
  docs/FOLLOWUP_EXPERIMENTS.md, docs/RISK_COVERAGE_TRADEOFF.md, docs/FINAL_SELECTION.md, docs/RESOLUTION_COMPATIBILITY.md,
  docs/CHANGELOG_XGD-1.2.md, docs/REPORT_MATERIAL_XGD-1.2.md
Missing artifacts are reported as 미완료 instead of being filled in.

python -m experiments.followup.docs12
"""
import collections
import json
import os
from pathlib import Path

import numpy as np

from experiments import lib as L
from experiments.followup import policy as P
from experiments.followup import tables as T

OUT = L.ROOT / "outputs" / "followup"
DOCS = Path(os.environ.get("DOCS12_OUT", L.ROOT / "docs"))
FAM = T.FAM
STAGE = os.environ.get("DOCS12_STAGE", "final")

NAMES = {"P0": "기준선(1.1 정책을 새 설계로 재보정)", "P1_low0.05": "잔여 후보 확인(점수 0.05~0.25)", "P1_low0.1": "잔여 후보 확인(0.10~0.25)",
         "P1_low0.15": "잔여 후보 확인(0.15~0.25)", "P1b_score0.15": "출력 점수 0.15", "P1b_score0.35": "출력 점수 0.35", "E0_2": "중복 억제 2px",
         "E0_6": "중복 억제 6px", "P2_seed_3": "seed 불일치(3px)", "P2_seed_6": "seed 불일치(6px)", "P2_view_3": "view 불일치(3px)",
         "P2_view_6": "view 불일치(6px)", "P2_v3only_3": "대조 모델(v3only) 불일치(3px)", "P2_v3only_6": "대조 모델 불일치(6px)",
         "P2_bundle_3": "seed·view·대조 불일치 묶음(3px)", "P2_bundle_6": "seed·view·대조 불일치 묶음(6px)", "P3_always": "추가 view 합집합(항상)",
         "P3_triggered": "추가 view 합집합(의심 영상만)", "L1_flip": "뒤집기 TTA 중심 중앙값", "L2_scale": "배율 ±32px TTA 중심 중앙값",
         "L3_source_bias": "원천 장비 중심 편향 보정", "L4_view_spread": "view 중심 흩어짐 > 1px REVIEW", "Q_3_8": "품질 REVIEW 3 / FULL 8",
         "Q_4_8": "품질 REVIEW 4 / FULL 8", "Q_5_8": "품질 REVIEW 5 / FULL 8", "Q_only_Q2": "품질 신호 Q2만", "Q_only_Q4": "품질 신호 Q4만",
         "GRADE_G1": "등급: 보정 절반 안 촬영일 LODO", "GRADE_G2": "등급: 분할 평가 없이 보정 절반 기준"}


def load(rel):
    p = OUT / rel
    try:
        return json.loads(p.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return None


def f(v, p=3):
    if v is None or v == "N/A":
        return "N/A"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    return f"{v:.{p}f}"


def write(name, text):
    (DOCS / name).write_text(text.rstrip() + "\n", encoding="utf8")


def label(family, cid):
    if family == "dfl":
        return f"{cid} {NAMES.get(cid, '')}".strip()
    return {"v3only": "흔적 증강 0(v3only) + P0", "trace05": "흔적 증강 0.5배(trace05) + P0"}.get(family, family) + ("" if cid == "P0" else f" + {cid}")


class Art:
    def __init__(self):
        self.sel = load("FINAL_SELECTION.json")
        self.config = load("deployment/CONFIG.json")
        self.cal = load("deployment/CALIBRATION.json")
        self.pred = load("deployment/TEST_PREDICTIONS.json")
        self.res = load("RESOLUTION_COMPATIBILITY.json")
        self.ltt = load("LTT_PROCEDURE_DIAGNOSTICS.json")
        self.syn = load("SYNTHESIS_GEOMETRY_PHOTOMETRY_AUDIT.json")
        self.dist = load("DFL_DISTRIBUTION_DIAGNOSTIC.json")
        self.curve = load("RISK_COVERAGE_DIAGNOSTIC.json")
        self.contract = load("COMPLETED_PIPELINE_CONTRACT.json")
        self.fresh = load("reproduction/FRESH_REPRODUCTION.json")
        self.install = load("reproduction/INSTALL_STATUS.json")
        self.allow = load("SUBMISSION_ALLOWLIST.json")
        self.ready = load("PACKAGE_READINESS.json")
        self.done = load("FOLLOWUP_COMPLETED.json")
        self.e10 = json.loads((L.OUT / "E10_localization_bias/RESULTS.json").read_text(encoding="utf8")) if (L.OUT / "E10_localization_bias/RESULTS.json").exists() else None
        self.e11 = json.loads((L.OUT / "E11_quality_signals/RESULTS.json").read_text(encoding="utf8")) if (L.OUT / "E11_quality_signals/RESULTS.json").exists() else None
        self.rows = T.candidate_rows(STAGE)
        self.base11 = json.loads((L.ROOT / "outputs/test_predictions.json").read_text(encoding="utf8")) if (L.ROOT / "outputs/test_predictions.json").exists() else None

    def r(self, family, cid, prof):
        return next((x for x in self.rows if x["family"] == family and x["id"] == cid and x["profile"] == prof), None)

    def item(self, prof):
        return (self.sel or {}).get("selection", {}).get("profiles", {}).get(prof) if self.sel else None

    def final_item(self, prof):
        """Item after fixed validation (may have reverted family/policy)."""
        hist = (self.sel or {}).get("validation_history", {}).get(prof, [])
        passed = [h for h in hist if h.get("passed")]
        return passed[-1]["item"] if passed else None


def candidate_table(a, ids, extra_cols=()):
    head = ("| 후보 | 장비군 | 정상 R@3px | F1@3px | 오탐/장 | 최저 seed R@3px | 정상 자동 확정 A | 14조건 평균 전체 위험 | 놓친 이물 자동 확정 위험(8px 진단) | "
            "합성 조건 놓침 자동 확정(등급 고정 진단) | 비용50 | 모델 호출/장 |")
    lines = [head, "| " + " | ".join(["---"] * 12) + " |"]
    for fam, cid in ids:
        for prof in T.PROFILES:
            x = a.r(fam, cid, prof)
            if not x:
                continue
            lines.append(f"| {label(fam, cid)} | {FAM[prof]} | {f(x['R3'], 4)} | {f(x['F13'], 4)} | {f(x['FP'])} | {f(x['R3_worst'], 4)} | {f(x['clean_A'])} | "
                         f"{f(x['risk_mean'], 4)} | {f(x['FN8_risk_mean'], 4)} | {f(x['synth_FN8_risk_diag'])} | {f(x['cost50'])} | {f(x['calls'], 1)} |")
    return lines


def selection_lines(a):
    if not a.sel:
        return ["- 최종 선택 미완료(FINAL_SELECTION.json 없음)."]
    S = a.sel["selection"]
    out = []
    for target, tab in S["selection_tables"].items():
        src = S["profiles"][target]["source_selection_profile"]
        ok = [k for k, v in tab["policies"].items() if v["admissible"]]
        best = S["original_source_selected_policies"][target]["id"]
        out.append(f"- **{FAM[target]} 대상 정책은 {FAM[src]} 보류 자료로 선택:** 통과 후보 {len(ok)}개({', '.join(sorted(ok)) or '없음'}) → 선택 `{best}`.")
        fails = collections.Counter()
        for k, v in tab["policies"].items():
            for reason in v["reasons"]:
                fails[reason.split(":")[-1].strip()] += 1
        if fails:
            out.append("  - 탈락 사유(후보 수): " + "; ".join(f"{k} {n}" for k, n in fails.most_common(8)))
        for fam, v in tab["families"].items():
            out.append(f"  - 흔적 강도 후보 {fam}: {'통과' if v['admissible'] else '탈락'}" + (f" — {'; '.join(v['reasons'])}" if v["reasons"] else "") +
                       f" (합성 짝 실제 놓침 비율 {f(v['key'][0])}, 이동/원위치 R3 {f(-v['key'][1])}, 정상 오탐 {f(v['key'][2])})")
    if not a.sel["selection"].get("same_policy_across_profiles", True) or any("conflict_resolution" in v for v in S["profiles"].values()):
        out.append("- 두 방향의 선택이 달라 사전 규칙(관찰된 실패 포착이 큰 값, 동률이면 비용이 작은 값)으로 하나를 공통 적용했다. 이 값을 독립적으로 고른 방향만 독립 선택이다.")
    for prof in T.PROFILES:
        hist = a.sel["validation_history"].get(prof, [])
        for i, h in enumerate(hist):
            it = h["item"]
            out.append(f"- {FAM[prof]} 고정 검증 {i + 1}: {it['family']} / {it['policy']['id']} → {'통과' if h['passed'] else '탈락'}" +
                       (f" ({'; '.join(h['reasons'][:6])}{' …' if len(h['reasons']) > 6 else ''})" if h["reasons"] else ""))
    return out


def changed_values(a):
    """Fixed values of the final item versus XGD-1.1."""
    base = P.default()
    lines = []
    for prof in T.PROFILES:
        it = a.final_item(prof)
        if not it:
            continue
        pol = it["policy"]
        diffs = [f"{k}: {base[k]} → {pol[k]}" for k in base if k != "id" and pol.get(k) != base[k]]
        lines.append(f"- {FAM[prof]}: 검출기 {it['family']}, 정책 `{pol['id']}`" + (f"; 바뀐 값 — {', '.join(diffs)}" if diffs else "; 정책 값 변경 없음(기준선 유지)"))
    return lines


def transitions(a):
    if not a.pred:
        return ["- 평가 79장 1.2 예측 미완료."]
    tr = a.pred.get("baseline_to_new_transition", {})
    return [f"- 1.2 판정 분포: {a.pred.get('decision_counts')}",
            f"- 1.1 → 1.2 전이: " + ", ".join(f"{k} {v}" for k, v in sorted(tr.items())),
            "- 정답이 없으므로 정확도 비교가 아니다. 분포를 보고 규칙을 바꾸지 않았다."]


def main():
    a = Art()
    done = bool(a.done)
    status = "완료" if done else "미완료(최종 단계 진행 중)"
    ids_complete = [("dfl", c) for c in ("P0", "P1_low0.05", "P1_low0.1", "P1_low0.15", "P1b_score0.15", "P1b_score0.35", "P2_seed_3", "P2_seed_6",
                                        "P2_view_3", "P2_view_6", "P2_v3only_3", "P2_v3only_6", "P2_bundle_3", "P2_bundle_6", "P3_always", "P3_triggered")]
    ids_loc = [("dfl", c) for c in ("P0", "L1_flip", "L2_scale", "L3_source_bias", "L4_view_spread", "E0_2", "E0_6")]
    ids_q = [("dfl", c) for c in ("P0", "Q_3_8", "Q_4_8", "Q_5_8", "Q_only_Q2", "Q_only_Q4", "GRADE_G1", "GRADE_G2")]
    ids_fam = [("dfl", "P0"), ("trace05", "P0"), ("v3only", "P0")]
    # ------------------------------------------------------------------ FOLLOWUP_EXPERIMENTS
    L_ = [f"# XGD-1.2 후속 실험 ({status})", "",
          "이 문서는 완료된 결과 파일에서 자동으로 만든다. 모든 수치는 같은 공급처 개발 자료 안의 내부 결과이며 독립 외부 검증이 아니다. "
          "seed 17·29·43은 같은 영상의 반복이므로 독립 표본으로 세지 않는다. **ACCEPT는 검출한 이물 위치만 확인한다는 뜻이며 제품 합격이나 영상 전체 완결성 보장이 아니다.**", "",
          "## 실행 계약", "",
          "- 사전 등록 `docs/PREREG_FOLLOWUP.md`(2026-10-06 03:03 KST), 선택 계약 `docs/FOLLOWUP_SELECTION_CONTRACT.md`와 실행 잠금 `outputs/followup/ANALYSIS_EXECUTION_LOCK.json`(03:41, 결과 전).",
          "- 결과 전 구현 수정 등록 4건(`outputs/followup/preanalysis_amendment_01~04`)과 재개 수정 3건: "
          "`RESUME_AMENDMENT_01.json`(05:38, 흔적 0.5배 학습 데이터 비교의 tuple/list 정규화만, 학습 전), "
          "`RESUME_AMENDMENT_02.json`(11:39, 판정에 쓰지 않는 DFL 분포 진단의 CPU/GPU head 불일치 수정, 선택·검증 결과 기록 뒤), "
          "`RESUME_AMENDMENT_03.json`(11:46, 제출 목록 생성의 문자 인코딩 이름 오류 수정, 모든 판정·보정·재현 단계 완료 뒤). 봉인 파일은 고치지 않고 별도 실행 파일로 감쌌다.",
          "- 이미 본 자료: XGD-1.1의 E1~E9 전부, 평가 79장의 판정 분포·해상도·Q4(정답 미사용). 재개 세션은 05:3x에 저렴한 비교(screen) 요약을 읽었지만 선택 규칙·후보·허용폭은 03:41 봉인 그대로다.",
          "- 자료 역할: 정책 후보값은 **반대 장비군** 보류 자료로 고르고(3호기용은 1·2호기 자료로), 대상 장비군 안에서는 촬영일 절반 + 고신뢰 근접 중복 그룹(NCC ≥ 0.998)을 분리해 "
          "보정 절반으로 품질 기준·λ·등급을 만들고 평가 절반을 판정한다. 그룹을 지키려고 3호기 13장을 다른 절반으로 옮겨 일부 촬영일이 두 절반에 걸친다(한계).",
          "- 바뀐 파이프라인은 모두 다시 보정했다(기존 λ 재사용 없음).", ""]
    L_ += ["## 1. 놓친 이물을 자동 확정하는 문제(최우선)", "",
           "**관찰.** 1.1의 판정 신뢰도는 검출한 이물의 평균 점수라서, 이물 하나를 놓쳐도 나머지가 확실하면 자동 확정됐다(E9: 합성 M2 이물 조건에서 자동 확정 실패의 대부분이 놓침).",
           "**가설.** 평균을 최소로 바꾸는 것으로는 해결되지 않는다(놓친 이물은 어떤 집계에도 들어가지 않음). 대신 영상 전체의 검출 완결성을 따로 확인하면 놓침이 있는 영상을 REVIEW로 돌릴 수 있다.",
           "**후보.** 잔여 저점수 후보(P1), 출력 점수 변경(P1b, 대조), seed·view·대조 모델 불일치(P2), 의심 영상의 추가 view 합집합(P3).", ""]
    L_ += candidate_table(a, ids_complete)
    L_ += ["", "- 열 설명: '정상 자동 확정 A'와 '전체 위험'은 날짜·그룹 분리 보정 설계(주 결과), '등급 고정 진단'은 자격 조건을 QUALIFIED로 고정한 진단으로 운영 성능이 아니다. "
           "'놓친 이물 자동 확정 위험'은 8px 안에 예측이 아예 없는 정답이 있는 영상이 자동 확정될 확률(전체 영상 분모)이다. 비용50은 가정 비용(놓침 자동 확정 50, 오탐·위치 오차 자동 확정 10, REVIEW 1, FULL 3)의 14조건 평균이다.",
           "- 해석: 불일치 확인은 검출 좌표를 바꾸지 않고 자동 확정만 REVIEW로 돌린다. 그래서 정상 조건 검출률은 그대로이고 위험은 줄거나 같다. "
           "'전부 동의'에도 공통 놓침이 남을 수 있다. 합성 이물 결과는 실제 현장 놓침 빈도의 증거가 아니다.", ""]
    L_ += ["## 2. 고신뢰도 3~4px 위치 오차", "",
           "**관찰.** 1.1 정상 조건 실패는 대부분 정답 중심에서 3.0~4.0px 벗어난 위치 오차였다(E8).",
           (f"**오차 분해(E10).** 3호기 보류: dx {a.e10['profiles']['dfl_src12']['held']['all']['dx_median']:+.2f}px, dy {a.e10['profiles']['dfl_src12']['held']['all']['dy_median']:+.2f}px의 일정한 치우침 "
            f"(같은 객체가 세 seed 모두 같은 dx 부호 {a.e10['profiles']['dfl_src12']['held']['same_sign_dx_all_seeds']:.0%}); 1·2호기 보류: dx {a.e10['profiles']['dfl_src3_2']['held']['all']['dx_median']:+.2f}px. "
            f"원천(학습) 장비에서는 ±0.1px 수준이라 원천 편향 보정으로는 대상 장비의 치우침을 설명하지 못한다. seed 불일치로는 이런 공통 편향을 잡기 어렵다." if a.e10 else "**오차 분해(E10)** 미완료."),
           "**후보.** 뒤집기 TTA(L1), 배율 TTA(L2), 원천 편향 보정(L3), view 흩어짐 REVIEW(L4), 중복 억제 거리(E0 2/6).", ""]
    L_ += candidate_table(a, ids_loc)
    if a.dist:
        sp = a.dist.get("conditional_held_error_Spearman", {})
        L_ += ["", f"- DFL 분포 진단(동결 DFL, 일부 영상): 변 분포 엔트로피·분산 proxy와 8px 조건부 오차의 순위상관 {', '.join(f'{k} {f(v)}' for k, v in sp.items())}. "
               "보정된 중심 불확실성이 아니며, 검출이 없는 객체는 측정되지 않아 놓침 신호로 쓰지 않았다."]
    L_ += ["", "- 4/5px 성공은 현장 허용 오차가 달라졌을 때의 시나리오이며 모델 개선이 아니다(위 표는 3px 기준).", ""]
    L_ += ["## 3. 위치·문맥·흔적과 합성 방법", "",
           "**관찰.** 1.1에서 흔적 무력화 없는 대조(v3only)는 3호기 M2 합성 이물 검출 0.987, 최종 모델은 0.563이었고, 실제 정상 검출률은 0.982 대 0.988이었다. M1은 위치와 무관, M2는 이동하면 크게 하락했다.",
           "**조치.** (1) M2 합성 점검: seamlessClone의 짝수 크기 마스크에서 붙여넣기 ROI가 1px 어긋나는 문제를 찾아 ROI만 정합한 M2v2를 만들었다(대비 정합은 아님). "
           "(2) 흔적 증강 강도 0 / 0.5배 / 1.0을 같은 조리법·분할·seed로 비교(0.5배 6개 신규 학습).", ""]
    if a.syn:
        L_ += [f"- 합성 점검: 짝 {a.syn['pairs']}개 중 1.0 M2의 ROI 어긋남 {a.syn['legacy_shifted_pairs']}개. 원본 대비 국소 대비 비(P10/P50/P90): "
               + "; ".join(f"{k} {', '.join(f(v, 2) for v in d['P10_P50_P90'])}" for k, d in a.syn["contrast_ratios"].items()), ""]
    lines_fam = ["| 검출기 | 장비군 | 정상 R@3px | F1@3px | 오탐/장 | 최저 seed R@3px | 합성(M1·M2·M2v2) 실제 놓침 비율 | 정상 자동 확정 A | 비용50 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for fam, cid in ids_fam:
        for prof in T.PROFILES:
            x = a.r(fam, cid, prof)
            if x:
                lines_fam.append(f"| {label(fam, cid)} | {FAM[prof]} | {f(x['R3'], 4)} | {f(x['F13'], 4)} | {f(x['FP'])} | {f(x['R3_worst'], 4)} | {f(x['synth_actual_missing_rate'])} | {f(x['clean_A'])} | {f(x['cost50'])} |")
    L_ += lines_fam + ["", "- 허용 손실(탐색 제안): 정상 3px 재현율·F1 평균 손실 ≤ 0.5%p, 최저 seed 손실 ≤ 1.0%p. 이 범위 안에서 합성 짝 실제 놓침 → 이동/원위치 → 오탐 순으로 비교했다. 합성 성능만으로 운영 가중치를 바꾸지 않는다.", ""]
    L_ += ["## 4. 촬영 변화와 과차단", "",
           "**관찰.** 약한 열화에도 대부분 FULL이 되는 과차단이 있었다. E11 진단에서 운영 신호 Q4는 실패가 적은 열화의 대부분에 경보를 냈다.",
           "**후보.** FULL 경계만 8로(자동 확정 경계 유지), REVIEW 경계 4·5, 신호 하나만(Q2/Q4).", ""]
    L_ += candidate_table(a, ids_q[:6])
    if a.e11:
        L_ += ["", "- E11(채택 없음): " + "; ".join(f"{q} 정상 오경보 {np.mean([v['clean_false_flag_rate_z3_LOSO'][q] for v in a.e11['devices'].values()]):.3f}, "
                                                f"실패 적은 열화 경보 {f(a.e11['mean_flag_rate_on_low_failure_stresses'][q])}, 실패 많은 열화 경보 {f(a.e11['mean_flag_rate_on_high_failure_stresses'][q])}"
                                                for q in ("Q2", "Q4", "SBR", "SHARP"))]
    L_ += ["", "## 5. 해상도 호환성", ""]
    if a.res:
        for sd, v in a.res["per_seed"].items():
            L_.append(f"- seed {sd}: 1호기 개발 {v['images']}장, 가운데 316px crop: R3 변화 {f(v['crop']['R3_delta'], 4)}, 오탐 변화 {f(v['crop']['FP_per_image_delta'], 4)}, 품질 상태 일치 {f(v['crop']['quality_state_agreement'])}"
                     f" → {'통과' if v['crop']['compatibility_diagnostic_pass'] else '불통과'}; 352 canvas: R3 변화 {f(v['canvas']['R3_delta'], 4)}, 품질 일치 {f(v['canvas']['quality_state_agreement'])} → {'통과' if v['canvas']['compatibility_diagnostic_pass'] else '불통과'}")
        L_ += [f"- 정답이 crop 밖으로 나간 객체 {a.res.get('GT_objects_outside_crop')}개. 평가 316×332 영상은 ROI·촬영 설정을 알 수 없어 자동 등록하지 않았다(규칙: 등록부에 없는 native 해상도는 FULL)."]
    else:
        L_.append("- 해상도 호환성 시험 미완료.")
    L_ += ["", "## 6. 작은 보정 자료와 LTT", ""]
    L_ += candidate_table(a, [("dfl", "P0"), ("dfl", "GRADE_G1"), ("dfl", "GRADE_G2")])
    if a.ltt:
        mz = a.ltt["minimum_zero_failure_accepted"]
        L_ += ["", f"- 실패 0건일 때 필요한 자동 확정 독립 영상 수(δ 0.10): α 0.05 → K=10 {mz['0.05']['10']}, K=3 {mz['0.05']['3']}, K=1 {mz['0.05']['1']}; α 0.10 → {mz['0.1']['10']}/{mz['0.1']['3']}/{mz['0.1']['1']}. 총 보정 영상 수가 아니다."]
        for prof, rows in a.ltt["profiles"].items():
            for row in rows:
                res = row["alphas"]["0.05"]
                L_.append(f"- {FAM[prof]} 보정 절반({row['unique_images']}장, 촬영일 {row['date_groups']}개), α 0.05: " + "; ".join(
                    f"{name} 통과 {len(v['procedure_only']['bonferroni_passed']) if name != 'fixed_sequence' else len(v['procedure_only']['fixed_sequence']['passed'])}점" for name, v in res.items()))
        L_.append("- 날짜 의존·시험편 ID 부재·반복 노출로 교환가능성이 확인되지 않아 어떤 절차도 확률 보장으로 주장하지 않는다.")
    L_ += ["", "## 선택과 검증", ""] + selection_lines(a) + ["", "## 최종 채택(바뀐 고정값)", ""] + (changed_values(a) or ["- 미완료"])
    write("FOLLOWUP_EXPERIMENTS.md", "\n".join(L_))
    write("RISK_COVERAGE_TRADEOFF.md", tradeoff_doc(a, status))
    write("FINAL_SELECTION.md", selection_doc(a, status))
    write("RESOLUTION_COMPATIBILITY.md", resolution_doc(a, status))
    write("CHANGELOG_XGD-1.2.md", changelog_doc(a, status))
    write("REPORT_MATERIAL_XGD-1.2.md", report_material_doc(a, status))
    print("XGD-1.2 documents written", flush=True)


def tradeoff_doc(a, status):
    L_ = [f"# 위험–자동 확정–부담 trade-off ({status})", "",
          "- 자동 확정 비율 A = P(ACCEPT). 전체 위험 = P(ACCEPT ∧ 실패) = A × 조건부 위험이라서, 보류를 늘리기만 해도 전체 위험은 줄어든다.",
          "- 그래서 세 가지를 따로 비교한다: 같은 A에서 위험이 더 낮은지, 같은 위험에서 A가 더 큰지, 같은 A·위험에서 재검사·연산 부담이 더 적은지.",
          "- 실패 = 3px 안에서 놓친 정답이 있거나 점수 ≥ 0.25 오탐이 있음. FN과 FP가 함께 있는 영상은 한 번만 센다.",
          "- 비용은 가정 시나리오다.",
          "  - 놓침이 있는 자동 확정: 10/50/100 (주 시나리오 50)",
          "  - 오탐·위치 오차만 있는 자동 확정: 10",
          "  - REVIEW: 1",
          "  - FULL: 3",
          "  - 실제 작업자 정확도·시간은 측정하지 않았다.",
          "- 값은 날짜·그룹 분리 보정 설계에서 seed 17·29·43의 평균이다. 14조건 = 정상 1 + 합성 이물 12 + 흔적 1."]
    worst = []
    for prof in T.PROFILES:
        fin = a.final_item(prof)
        q = OUT / f"SELECTED_VALIDATION_{prof}_{fin['family']}_{fin['policy']['id']}.json" if fin else None
        if q is not None and q.exists():
            cand = json.loads(q.read_text(encoding="utf8"))["candidate_results"]
            w = max(((T.dec(cand[c], "joint_risk") or 0.0, c) for c in cand if c.startswith("STRESS_")), default=(None, None))
            if w[1]:
                worst.append(f"{FAM[prof]} {f(w[0])}({w[1]})")
    L_ += ["- '열화 최대 위험' N/A: 비교 단계에서는 열화 17조건을 기준선·흔적 강도·품질·등급 후보에만 계산했다. 최종 선택의 열화 조건은 선택 뒤 고정 검증에서 계산했다"
           + (f"(최대 전체 위험 {', '.join(worst)}; FINAL_SELECTION)." if worst else "."), ""]
    for prof in T.PROFILES:
        rows = sorted([x for x in a.rows if x["profile"] == prof], key=lambda x: x["cost50"])
        front = set(T.pareto(a.rows, prof))
        L_ += [f"## {FAM[prof]}", "", f"- Pareto 비지배 후보(정상 A↑, 놓침 자동 확정 위험↓, 비용50↓; 정상 A가 기준선의 절반 미만인 '전부 보류' 정책은 제외): {', '.join(sorted(front))}", "",
               "| 후보 | Pareto | 정상 A | 정상 REVIEW | 정상 FULL | 정상 올바른 결과 보류 | 14조건 전체 위험 | 놓침 자동 확정 위험 | 열화 최대 위험 | 비용10 | 비용50 | 비용100 | 호출/장 |",
               "| " + " | ".join(["---"] * 13) + " |"]
        for x in rows:
            key = f"{x['family']}/{x['id']}"
            L_.append(f"| {label(x['family'], x['id'])} | {'●' if key in front else ''} | {f(x['clean_A'])} | {f(x['clean_review'])} | {f(x['clean_full'])} | "
                      f"{f(x['clean_hold_ok'])} | {f(x['risk_mean'], 4)} | {f(x['FN8_risk_mean'], 4)} | {f(x['stress_risk_max'])} | {f(x['cost10'])} | "
                      f"{f(x['cost50'])} | {f(x['cost100'])} | {f(x['calls'], 1)} |")
        base = a.r("dfl", "P0", prof)
        if base:
            same_a = [x for x in rows if x["clean_A"] is not None and abs(x["clean_A"] - base["clean_A"]) <= 0.01 and not (x["family"] == "dfl" and x["id"] == "P0")]
            better = sorted([x for x in same_a if x["risk_mean"] < base["risk_mean"]], key=lambda x: x["risk_mean"])[:5]
            L_ += ["", "- 같은 A(±0.01)에서 기준선보다 위험이 낮은 후보(상위 5): " +
                   "; ".join(f"{label(x['family'], x['id'])} 위험 {f(x['risk_mean'], 4)}(기준 {f(base['risk_mean'], 4)}), 호출 {f(x['calls'], 1)}" for x in better)]
            more_a = [x for x in rows if x["clean_A"] is not None and x["clean_A"] > base["clean_A"] + 0.01]
            L_.append("- 기준선보다 A가 큰 후보: " + ("; ".join(f"{label(x['family'], x['id'])} A {f(x['clean_A'])}, 위험 {f(x['risk_mean'], 4)}(기준 {f(base['risk_mean'], 4)})"
                                                      for x in more_a) or "없음"))
        L_.append("")
    L_ += ["## 위험–자동 확정 곡선", "",
           "- `outputs/followup/risk_coverage.png`와 `outputs/followup/RISK_COVERAGE_DIAGNOSTIC.json`에 있다. 기준선과 최종 선택에서 신뢰도 임계값을 바꿔 가며 그린 기술적 곡선이다.",
           "- 등급·품질 gate를 고정하지 않은 진단이므로 운영 정책을 대신하지 않는다."]
    return "\n".join(L_)


def validation_table(a, prof):
    """Fixed post-selection validation of the final item: every condition, candidate versus baseline (seed means, group-date design)."""
    it = a.final_item(prof)
    if not it:
        return ["- 미완료"]
    p = OUT / f"SELECTED_VALIDATION_{prof}_{it['family']}_{it['policy']['id']}.json"
    if not p.exists():
        return ["- 미완료"]
    v = json.loads(p.read_text(encoding="utf8"))
    cand, base = v["candidate_results"], v["baseline_results"]
    L_ = [f"| 조건 | A 기준→1.2 | 전체 위험 기준→1.2 | 놓침 자동 확정 위험 기준→1.2 | FULL 기준→1.2 | 3px 재현율 기준→1.2 |", "| --- | --- | --- | --- | --- | --- |"]
    agg = collections.defaultdict(list)
    for c in cand:
        cb, cc = base[c], cand[c]
        vals = [(T.dec(cb, k), T.dec(cc, k)) for k in ("coverage", "joint_risk", "FN8_joint_risk", "full_rate")] + [(T.mean(cb, "R3"), T.mean(cc, "R3"))]
        group = "열화" if c.startswith("STRESS_") else ("정상" if c == "dev_clean" else "합성·통제")
        for i, pair in enumerate(vals):
            agg[(group, i)].append(pair)
        if c == "dev_clean" or not c.startswith("STRESS_"):
            L_.append(f"| {c} | " + " | ".join(f"{f(x)}→{f(y)}" for x, y in vals) + " |")
    worst_stress = max(((T.dec(cand[c], 'joint_risk') or 0.0, c) for c in cand if c.startswith("STRESS_")), default=(None, None))
    L_ += ["", "열화 17조건 요약(평균): " + "; ".join(
        f"{name} {f(np.mean([x for x, _ in agg[('열화', i)] if x is not None]))}→{f(np.mean([y for _, y in agg[('열화', i)] if y is not None]))}"
        for i, name in enumerate(("A", "전체 위험", "놓침 자동 확정 위험", "FULL", "3px 재현율")) if agg[("열화", i)]),
        f"열화 조건 최대 전체 위험(1.2): {f(worst_stress[0])} ({worst_stress[1]})", f"고정 검증 탈락 사유: {'; '.join(v['rejection_reasons']) or '없음'}"]
    return L_


def rejected_tradeoffs(a):
    """Validated-then-rejected items (fixed post-selection validation): what they improved and which pre-registered check stopped them."""
    out = []
    if not a.sel:
        return out
    for prof in T.PROFILES:
        for h in a.sel["validation_history"].get(prof, []):
            if h["passed"]:
                continue
            it = h["item"]
            p = OUT / f"SELECTED_VALIDATION_{prof}_{it['family']}_{it['policy']['id']}.json"
            if not p.exists():
                continue
            v = json.loads(p.read_text(encoding="utf8"))
            cand, base = v["candidate_results"], v["baseline_results"]
            m2 = [c for c in cand if ("M2" in c) and not c.startswith("R2a_same")]
            def avg(R, k):
                vals = [T.dec(R[c], k) for c in m2]
                vals = [x for x in vals if x is not None]
                return float(np.mean(vals)) if vals else None
            line = (f"- **{FAM[prof]} {it['family']} + {it['policy']['id']}(탈락):** 사유 — {'; '.join(h['reasons'])}. "
                    f"정상 3px 재현율 {f(T.mean(base['dev_clean'], 'R3'), 4)} → {f(T.mean(cand['dev_clean'], 'R3'), 4)}, "
                    f"정상 A {f(T.dec(base['dev_clean'], 'coverage'))} → {f(T.dec(cand['dev_clean'], 'coverage'))}, "
                    f"M2 계열 평균 전체 위험 {f(avg(base, 'joint_risk'), 4)} → {f(avg(cand, 'joint_risk'), 4)}, 놓침 자동 확정 위험 {f(avg(base, 'FN8_joint_risk'), 4)} → {f(avg(cand, 'FN8_joint_risk'), 4)} "
                    "(기준 = 1.1 정책 DFL).")
            fin = a.final_item(prof)
            q = OUT / f"SELECTED_VALIDATION_{prof}_{fin['family']}_{fin['policy']['id']}.json" if fin else None
            if q is not None and q.exists() and fin["family"] != it["family"]:
                adopted = json.loads(q.read_text(encoding="utf8"))["candidate_results"]
                line += (f" 같은 정책의 채택안({fin['family']}) 대비: 정상 A {f(T.dec(cand['dev_clean'], 'coverage'))} 대 {f(T.dec(adopted['dev_clean'], 'coverage'))}, "
                         f"M2 계열 전체 위험 {f(avg(cand, 'joint_risk'), 4)} 대 {f(avg(adopted, 'joint_risk'), 4)}, 놓침 자동 확정 위험 {f(avg(cand, 'FN8_joint_risk'), 4)} 대 {f(avg(adopted, 'FN8_joint_risk'), 4)}, "
                         f"M2 계열 A {f(avg(cand, 'coverage'))} 대 {f(avg(adopted, 'coverage'))}.")
            out.append(line + " 사전 규칙상 채택하지 않았고 결과를 본 뒤 규칙을 바꾸지 않았다.")
    if any(h["item"]["family"] == "trace05" and not h["passed"] for hs in a.sel["validation_history"].values() for h in hs):
        out.append("- 흔적 0.5배: 기준 정책(P0)끼리 비교하면 합성 짝 실제 놓침이 줄었다(FOLLOWUP_EXPERIMENTS 3절). 그러나 묶음 확인과 함께 쓰면 채택안보다 위험이 낮아지지 않았고 "
                   "M2 계열 자동 확정만 늘었으며, 한 조건에서 영상 1장 수준의 위험이 늘어 탈락했다. 자동 확정 증가를 목표로 할 다음 버전 후보이며, 그때는 단일 영상 잡음을 다루는 "
                   "비열등성 기준(예: 신뢰구간 또는 허용 영상 수)을 결과 전에 등록해 다시 검증한다(유형 2 trade-off 후보, 이번 버전 미채택).")
    return out


def selection_doc(a, status):
    L_ = [f"# XGD-1.2 최종 선택 ({status})", "",
          "## 선택 규칙(03:41 봉인, `docs/FOLLOWUP_SELECTION_CONTRACT.md`)", "",
          "1. 정상 조건 관문: 3px 재현율·F1 평균 손실 ≤ 0.005, 최저 seed 재현율 손실 ≤ 0.01, 오탐/장 증가 ≤ 0.02.",
          "2. 정상 조건 자동 확정 비율은 기준선의 절반 이상이어야 한다.",
          "3. 14조건 각각에서 전체 위험과 놓침 자동 확정 위험이 기준선보다 늘지 않아야 한다. 일부 후보에는 추가 조건이 있다.",
          "   - 품질 후보: 열화 17개 조건 모두에서 전체 위험 ≤ 0.05",
          "   - 등급 후보: 조건별 위험 ≤ 0.05, seed별 위험 ≤ 0.08",
          "4. 비용50 개선이 5% 이상인 후보 중에서 비용 → 모델 호출 수 → ID 순으로 고른다. 각 대상 장비군의 값은 반대 장비군 자료로 고른다.",
          "5. 흔적 강도(0/0.5/1)는 같은 정상 관문을 먼저 적용한다. 그다음 합성 짝(M1/M2/M2v2)의 실제 놓침 → 이동/원위치 R3 → 오탐 순으로 비교한다.",
          "6. 선택 뒤 고정 검증에서 어느 조건이든 위험이 늘면 흔적 강도 → 정책 순으로 기준선에 복귀한다. 이 과정에서 값을 새로 만들지 않는다.", "",
          "## 선택 결과", ""] + selection_lines(a) + ["", "## 최종 채택과 바뀐 고정값", ""] + (changed_values(a) or ["- 미완료"])
    for prof in T.PROFILES:
        L_ += ["", f"## 고정 검증 — {FAM[prof]}(선택 뒤, 내부 검증)", ""] + validation_table(a, prof)
    if a.cal:
        L_ += ["", "## 재보정", "",
               "선택이 끝난 뒤 보류 개발 라벨 전체로 만든 배포용 보정이다. 새로운 검증 결과가 아니다.", ""]
        for prof, P_ in a.cal["profiles"].items():
            for dev, pkg in P_["devices"].items():
                lam = {sd: v.get("lambda") for sd, v in pkg["seeds"].items()}
                L_.append(f"- {FAM[prof]} / {dev}: 보정 {len(pkg['cal_stems'])}장, 촬영일 {len(pkg['dates'])}개, 운영 등급 {pkg['grade']}, λ " +
                          ", ".join(f"seed {sd} {f(v) if isinstance(v, (int, float)) else '∞'}" for sd, v in lam.items()))
    if a.pred:
        lat = a.pred.get("latency_mean_P95_seconds")
        calls = []
        for p_ in T.PROFILES:
            it = a.final_item(p_)
            x = a.r(it["family"], it["policy"]["id"], p_) if it else None
            calls.append(f"{FAM[p_]} {f(x['calls'], 1) if x else 'N/A'}")
        L_ += ["", "## 연산 비용", "",
               (f"- 평가 79장 종단 처리 시간(원본 읽기~판정, 모델 적재 제외): 평균 {f(lat[0])}초, P95 {f(lat[1])}초" if lat else "- 처리 시간 미기록"),
               "- 모델 호출 수/장: " + ", ".join(calls)]
    L_ += ["", "## 실패하거나 채택하지 않은 후보", ""] + rejected_tradeoffs(a) + [
           "- 후보별 탈락 사유는 위 '선택 결과'와 `outputs/followup/FINAL_SELECTION.json`(selection_tables)에 있다.",
           "- 유형 3(기준 완화): α 0.10과 4/5px 성공은 채택 기준으로 쓰지 않았다.",
           "- 유형 4(기준선 유지): 관문을 넘지 못했거나 위험이 늘어난 후보는 기준선을 유지했다(위 표 참조)."]
    return "\n".join(L_)


def resolution_doc(a, status):
    L_ = [f"# 해상도 호환성 ({status})", "", "## 문제", "",
          "1.1은 장비별 보정(품질 기준·CRC λ)을 그 장비 개발 영상의 해상도에서 만들었지만, 다른 해상도로 들어온 같은 장비 영상을 막는 규칙이 없었다. "
          "평가 1호기 32장(316×332, 개발 1호기는 352×332)은 품질 gate에서 멈췄을 뿐이다(E7).", "", "## 수정(XGD-1.2)", "",
          "- **보정 등록부:** 장비마다 아래 항목을 기록한다.",
          "  - 등록된 native 해상도(폭×높이, 방향 포함)",
          "  - 보정 촬영일",
          "  - 전처리 버전·해시",
          "  - 유효 입력 배율 범위",
          "  - 파이프라인 서명",
          "  - ROI·촬영 설정·물리 배율은 'unknown'으로 둔다.",
          "- **규칙:** 등록부에 없는 native 해상도나 서명이 다른 등록부는 FULL_REINSPECTION으로 보내고 사유를 남긴다. 가로를 늘려 기존 크기에 맞추는 처리는 하지 않는다.", ""]
    if a.cal:
        L_ += ["## 등록부", "", "| profile | 장비 | 등록 해상도 | 보정 촬영일 | ROI | 촬영 설정 | 새 해상도 자동 승인 |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for prof, P_ in a.cal["profiles"].items():
            for dev, e in P_["registry"].items():
                L_.append(f"| {prof} | {dev} | {', '.join('×'.join(map(str, r)) for r in e['native_resolutions'])} | {', '.join(e['calibration_dates'])} | "
                          f"{e['ROI']} | {e['acquisition_settings']} | {e['automatic_new_resolution_approval']} |")
    L_ += ["", "## 개발 자료 호환성 시험", "", "1호기 352px 영상을 가운데 316px로 crop한 뒤 다시 352px canvas에 넣어 비교했다.", ""]
    if a.res:
        L_ += ["| seed | 영상 | crop R3 변화 | crop 오탐 변화 | crop 품질 일치 | crop 통과 | canvas R3 변화 | canvas 오탐 변화 | canvas 품질 일치 | canvas 통과 |",
               "| " + " | ".join(["---"] * 10) + " |"]
        for sd, v in a.res["per_seed"].items():
            c, k = v["crop"], v["canvas"]
            L_.append(f"| {sd} | {v['images']} | {f(c['R3_delta'], 4)} | {f(c['FP_per_image_delta'], 4)} | {f(c['quality_state_agreement'])} | {c['compatibility_diagnostic_pass']} | "
                      f"{f(k['R3_delta'], 4)} | {f(k['FP_per_image_delta'], 4)} | {f(k['quality_state_agreement'])} | {k['compatibility_diagnostic_pass']} |")
        L_ += ["", f"- crop 밖으로 나간 정답 객체: {a.res.get('GT_objects_outside_crop')}개. canvas는 {a.res.get('canvas_fill')}(으)로 채웠다.",
               f"- 결론: {a.res.get('reason')}",
               f"- 그래서 평가 316×332 영상은 등록하지 않았다(자동 승인: {a.res.get('actual_evaluation316_native_auto_approved')})."]
    else:
        L_.append("- 미완료")
    if a.pred:
        d1 = [v for v in a.pred["images"].values() if v["device"] == "1호기"]
        reasons = collections.Counter(r for v in d1 for r in v["reason"] if "해상도" in r)
        L_ += ["", "## 평가 79장 1호기에 미친 영향(정답 미사용)", "",
               f"- 1호기 {len(d1)}장 판정: {dict(collections.Counter(v['decision'] for v in d1))}",
               f"- 해상도 관련 사유: {dict(reasons)}"]
    L_ += ["", "## 남은 한계", "",
           "- 해상도가 같아도 Q4·잡음 스펙트럼·PSF가 바뀌면 다른 위험이 남는다. 이 부분은 품질 gate가 따로 본다.",
           "- 316×332 1호기를 쓰려면 그 조건의 라벨 보정 자료나 ROI·촬영 설정 정보가 필요하다."]
    return "\n".join(L_)


def changelog_doc(a, status):
    L_ = [f"# XGD-1.2 변경 기록 ({status})", "", "| 구분 | XGD-1.1.0 | XGD-1.2.0 |", "| --- | --- | --- |"]
    base = P.default()
    fam_name = {"dfl": "DFL(흔적 증강 1.0)", "trace05": "흔적 증강 0.5배 DFL", "v3only": "흔적 증강 0 DFL(v3only)"}
    for prof in T.PROFILES:
        it = a.final_item(prof)
        if not it:
            continue
        pol = it["policy"]
        L_.append(f"| {FAM[prof]} 검출기 | DFL(흔적 증강 1.0) | {fam_name.get(it['family'], it['family'])} |")
        L_.append(f"| {FAM[prof]} 정책 | P0 | {pol['id']} {NAMES.get(pol['id'], '')} |")
        for k in ("score", "e0", "z_review", "z_full", "signals", "grade", "localization", "safety", "radius", "low"):
            if pol.get(k) != base[k]:
                L_.append(f"| {FAM[prof]} {k} | {base[k]} | {pol[k]} |")
    L_ += ["| 품질 기준(X12) 자료 범위 | 장비별 개발 영상 전체. 2호기는 1·2호기용 검출기 학습에 쓴 8/31·9/1 촬영 32장 포함(132장) | "
           "CRC와 같은 보류 개발 영상만(2호기 6월 100장; 1호기 124장·3호기 100장은 범위 동일). CRC는 1.1에서도 보류 영상만 썼다. "
           "03:41 봉인 계약에서 정했고 평가 79장 분포를 보고 바꾸지 않았다 |",
           "| 해상도 | 규칙 없음 | 등록부에 없는 native 해상도 → FULL |",
           "| 보정 서명 | 가중치·전처리·규칙·소스 해시 | 여기에 정책·추가 모델 가중치·의존성 버전까지 서명하고, 불일치 시 FULL |",
           "| 출력 | decision, detections_confirmed | 추가 필드: confirmation_scope(검출 위치만), completeness_status(NOT_CERTIFIED/INCONSISTENT), product_pass_certified=false, 불일치 신호 |",
           "| 운영 코드 | src/xgd_analyzer.py | src/followup_analyzer.py(1.1 코드는 그대로 보존) |", "", "## 평가 79장 판정 변화", ""] + transitions(a)
    L_ += ["", "## 보존", "",
           "- 1.1 코드·설정·보정 산출물·평가 79장 예측·문서: `versions/XGD-1.1.0/`(SNAPSHOT.json에 해시).",
           "- 원본 BMP·TXT·동결 가중치 6개는 바꾸지 않았다. 흔적 0.5배 가중치는 `outputs/followup/train/trace05/`에 따로 있다."]
    return "\n".join(L_)


def report_material_doc(a, status):
    L_ = [f"# 결과보고서·발표 반영 자료(XGD-1.2, {status})", "",
          "프로젝트에 있는 기존 결과보고서·발표자료(`00_과거버전_보관/제출물_4번/`)는 9/28 Champion 모델 기준이다. 이전 버전 보관 원칙에 따라 수정하지 않았다.",
          "XGD-1.2 보고서·발표를 만들 때는 아래 본문·표를 옮기고 체크리스트를 확인한다. 대회 양식(6개 장)과 블라인드 규정(소속·로고 금지)을 지킨다.", ""]
    chap = [("제1장 데이터 이해 및 진단(15)",
             ["영상 1장 = 완제품 1개의 검사다. 개발 356장은 모두 이물이 있고, 평가 79장은 정답을 쓰지 않았으며, 정상 제품은 0장이다.",
              "장비 3대·해상도 3종이다. 같은 날 여러 장비로 촬영해 시험편을 공유했을 가능성이 있지만 물리 ID가 없다.",
              "근접 중복: NCC ≥ 0.98 기준은 공통 배경 때문에 1·2호기 327장이 한 덩어리가 된다. 그래서 분리 단위는 NCC ≥ 0.998 고신뢰 그룹만 쓰고, 약한 연결은 누수 민감도에만 쓴다.",
              "평가 1호기는 9월에 촬영 형식이 바뀌었다(316×332, 잡음 특성). 정답 없이 원인만 분해했다(E7)."]),
            ("제2장 AI 예측모델 개발 및 성능평가(40)",
             ["같은 조건의 일반 YOLO26s 대비 DFL: 3호기 3px F1 0.923 → 0.988.",
              "XGD-1.2에서 채택한 변경과 기준선 대비 수치는 FINAL_SELECTION 표를 그대로 옮긴다. 3px가 주 기준이고 5px·IoU는 민감도다.",
              "흔적 증강 강도 0/0.5/1 비교 표(FOLLOWUP_EXPERIMENTS 3절): " + "; ".join(
                  f"{FAM[p]} 정상 3px 재현율 1.0 {f(a.r('dfl', 'P0', p)['R3'], 4)}·0.5배 {f(a.r('trace05', 'P0', p)['R3'], 4)}·0 {f(a.r('v3only', 'P0', p)['R3'], 4)}, "
                  f"합성 실제 놓침 {f(a.r('dfl', 'P0', p)['synth_actual_missing_rate'])}·{f(a.r('trace05', 'P0', p)['synth_actual_missing_rate'])}·{f(a.r('v3only', 'P0', p)['synth_actual_missing_rate'])}"
                  for p in T.PROFILES if all(a.r(fm, 'P0', p) for fm in ('dfl', 'trace05', 'v3only'))) +
              ". 흔적을 줄이면 합성 놓침은 줄지만 3호기 흔적 0은 정상 검출을 잃었고, 0.5배는 묶음 확인과 결합했을 때 위험 이득이 없어 채택하지 않았다(흔적 1.0 유지)."]),
            ("제3장 영향요인 및 오류분석(15)",
             ["정상 조건 실패의 기하: 대부분 3~4px 위치 오차(E8)이고, 장비별로 일정한 중심 치우침이 있다(E10).",
              "놓친 이물의 자동 확정(E9)과, 완결성 확인으로 줄어든 위험(FOLLOWUP_EXPERIMENTS 1절).",
              "위치·대비·촬영 조건별 미탐(E4 표)과 합성 방법 M1/M2/M2v2 짝 비교."]),
            ("제4장 현장 활용방안(10)",
             ["ACCEPT는 검출 위치만 자동으로 확인한다는 뜻이다(제품 합격·완결성 아님). 완결성 불일치가 있으면 위치 후보를 표시한 채 REVIEW로 보낸다.",
              "해상도 미등록·품질 이상·보정 불일치는 FULL로 보내고 장비·촬영 조건을 점검한다.",
              "자동 확정·REVIEW·FULL 비율과 처리 시간, 가정 작업 시간 시나리오."]),
            ("제5장 창의성 및 차별성(10)",
             ["검출 완결성 확인(seed·view·대조 모델 불일치)을 위험–자동 확정 비용 기준으로 선택했다.",
              "흔적 shortcut 대조·입력 배율 정규화·해상도 등록부·보정 서명을 검증하며 결합했다. 기존 방법(DFL·CRC·LTT)은 인용하고, 타 팀보다 우월하다는 주장은 하지 않는다."]),
            ("제6장 코드 구성 및 재현성(10)",
             ["새로 의존성을 설치한 venv에서 회귀 테스트와 동결 추론 재현을 확인했다(FRESH_REPRODUCTION).",
              "전처리 → 학습(흔적 0.5배 6개 포함) → 보정 → 평가 79장 예측을 단일 진입점으로 돌린다. 불변 캐시와 완료 manifest를 쓰고, 다른 GPU는 검증하지 못했다."])]
    for title, items in chap:
        L_ += [f"## {title}", ""] + [f"- {x}" for x in items] + [""]
    L_ += ["## 반영 체크리스트", "",
           "- [ ] 모델 이름·버전을 XGD-1.2.0으로 바꾸고, 검출기·정책·해상도 규칙을 FINAL_SELECTION과 같게 적었다.",
           "- [ ] CRC α 0.05는 기대 전체 위험 P(ACCEPT∧실패)이며 ACCEPT 안 실패율 보장이 아니라고 적었다. LTT·QUALIFIED도 확률 보장이 아니다.",
           "- [ ] ACCEPT는 제품 합격이 아니며 영상 전체 완결성을 인증하지 않는다고 적었다.",
           "- [ ] 정상 조건 수치는 3px 주 기준으로 적고, 4/5px는 허용 오차 시나리오로 구분했다.",
           "- [ ] 합성 시험 결과를 실제 현장 놓침 빈도로 쓰지 않았다.",
           "- [ ] 평균과 실제 최저(조건×seed)를 구분했다. 자동 확정이 0인 경우 조건부 위험은 N/A로 적었다.",
           "- [ ] 평가 79장은 판정 분포만 적고 정확도로 쓰지 않았다.",
           "- [ ] 소속·학교명·로고·개인 경로가 없다.",
           "- [ ] 기존 보고서의 Champion 수치를 XGD-1.2 수치로 바꿨다."]
    return "\n".join(L_)


if __name__ == "__main__":
    main()
