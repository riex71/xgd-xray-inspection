"""E7. Label-free decomposition of why evaluation-79 images were not auto-confirmed, plus a development-only counterfactual
(docs/PREREG_E7_EVAL79_REASONS.md, recorded 2026-10-06 01:12 KST before this script ran; post-hoc motivated diagnostic).

Reads analyzer outputs (outputs/test_predictions.json - no evaluation labels exist in the package), the deployment quality reference and
development images. Changes no model, threshold, gate, rule, device label or routing.

python -m experiments.e7_eval79_reasons   -> outputs/experiments/E7_eval79_reasons/RESULTS.json, SUMMARY.md
"""
import collections
import json

import cv2
import numpy as np

from experiments import lib as L
from src import xgd_analyzer as X
from src.preprocess import clean_v3

QG = L.CFG["quality_gate"]
W_NEW = 316                                   # width of the evaluation 1호기 images (development 1호기: 352)
STATES = ("NORMAL", "REVIEW", "FULL_REINSPECTION")


def stop_conditions(v):
    a = v["assurance"]
    lam, t = a.get("crc_lambda"), v.get("trust")
    c = []
    if v.get("input_problems"):
        c.append("input")
    if a.get("compatible") is False:
        c.append("incompatible")
    if a.get("operating_grade") != "QUALIFIED":
        c.append("grade")
    if not v["scale"].get("envelope_ok"):
        c.append("scale")
    q = v["quality"]["state"]
    if q == "FULL_REINSPECTION":
        c.append("quality_FULL")
    elif q == "REVIEW":
        c.append("quality_REVIEW")
    if not v.get("object_count"):
        c.append("zero_detections")
    if lam is None or t is None or not t >= lam:
        c.append("trust_below_lambda")
    return c


def transforms(raw):
    h, w = raw.shape[:2]
    off = (w - W_NEW) // 2
    arms = {"original": raw,
            "resample_area": cv2.resize(raw, (W_NEW, h), interpolation=cv2.INTER_AREA),
            "resample_linear": cv2.resize(raw, (W_NEW, h), interpolation=cv2.INTER_LINEAR),
            "crop_center": raw[:, off:off + W_NEW], "crop_left": raw[:, :W_NEW], "crop_right": raw[:, w - W_NEW:]}
    return {k: np.ascontiguousarray(v) for k, v in arms.items()}


def signals_of(clean, pm):
    return X.quality_signals(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY), pm)


def dist(v):
    v = np.asarray(v, float)
    return {"n": int(v.size), "p05": float(np.percentile(v, 5)), "median": float(np.median(v)), "p95": float(np.percentile(v, 95))} if v.size else None


def verdict(full_rate):
    if full_rate >= 0.9:
        return "SUFFICIENT_ALONE"
    if full_rate <= 0.1:
        return "NOT_SUFFICIENT_ALONE"
    return "PARTIAL"


def main():
    out = {"prereg": "docs/PREREG_E7_EVAL79_REASONS.md (2026-10-06 01:12 KST)", "evaluation79_label_reads": 0,
           "changes_to_model_or_rules": "none"}
    pred = json.loads((L.ROOT / "outputs/test_predictions.json").read_text(encoding="utf8"))["images"]
    qall = json.loads((L.ROOT / "outputs/calibration/quality_reference.json").read_text(encoding="utf8"))
    out["quality_reference_version"] = qall.get("version")
    qref = qall["devices"]
    pm = L.product_mask()

    # ---- E7a: stop conditions from the analyzer outputs
    combos = collections.defaultdict(collections.Counter)
    only_quality = collections.Counter()
    for v in pred.values():
        c = stop_conditions(v)
        combos[f"{v['device']}|{v['decision']}"][" + ".join(c) if c else "(none)"] += 1
        if c and all(x.startswith("quality_") for x in c):
            only_quality[f"{v['device']}|{v['decision']}"] += 1
    out["E7a_stop_conditions"] = {k: dict(v) for k, v in sorted(combos.items())}
    out["E7a_quality_is_the_only_stop_condition"] = dict(only_quality)
    out["E7a_resolution_by_device"] = {
        f"{role}|{d}": dict(collections.Counter(f"{m['resolution'][0]}x{m['resolution'][1]}" for s, m in L.META.items()
                                                if m["device"] == d and (s in L.DEV) == (role == "development")))
        for role in ("development", "evaluation79") for d in ("1호기", "2호기", "3호기")}
    top = collections.Counter()
    for v in pred.values():
        z = v["quality"].get("z") or {}
        if z:
            top[f"{v['device']}|{max(z, key=lambda q: abs(z[q]))}"] += 1
    out["E7a_largest_abs_z_signal"] = dict(top)

    # ---- E7b: development 1호기 counterfactual
    d1 = [s for s in L.DEV if L.META[s]["device"] == "1호기"]
    session = {s: L.META[s]["date"] for s in d1}
    sig0 = {s: signals_of(L.clean_img(s), pm) for s in d1}                          # stored preprocessed images (as calibrate())
    loso = {sess: X.build_quality_reference(d1, sig0, session, exclude_session=sess) for sess in sorted(set(session.values()))}
    arms = collections.defaultdict(lambda: {"deployment": collections.Counter(), "loso": collections.Counter(), "Q2": [], "Q4": [],
                                            "z_Q2_dep": [], "z_Q4_dep": []})
    pixel_equal = 0
    for s in d1:
        raw = X.read_image(L.XC.raw_path(L.CFG, s))
        for name, im in transforms(raw).items():
            clean = clean_v3(im)
            if name == "original":
                pixel_equal += int(np.array_equal(clean, L.clean_img(s)))
            sg = signals_of(clean, pm)
            A = arms[name]
            st_dep, _, z_dep = X.quality_state(sg, qref["1호기"], QG)
            st_loso = X.quality_state(sg, loso[session[s]], QG)[0]
            A["deployment"][st_dep] += 1
            A["loso"][st_loso] += 1
            A["Q2"].append(sg["Q2"])
            A["Q4"].append(sg["Q4"])
            A["z_Q2_dep"].append(z_dep["Q2"] if z_dep else float("nan"))
            A["z_Q4_dep"].append(z_dep["Q4"] if z_dep else float("nan"))
    n1 = len(d1)
    out["E7b_counterfactual_dev_1호기"] = {
        "images": n1, "original_arm_pixel_equal_to_stored_preprocessed": pixel_equal,
        "arms": {k: {"deployment_reference": {st: A["deployment"][st] for st in STATES},
                     "loso_reference": {st: A["loso"][st] for st in STATES},
                     "full_rate_deployment": A["deployment"]["FULL_REINSPECTION"] / n1,
                     "full_rate_loso": A["loso"]["FULL_REINSPECTION"] / n1,
                     "verdict_prereg": verdict(A["deployment"]["FULL_REINSPECTION"] / n1) if k != "original" else "CONTROL",
                     "Q2": dist(A["Q2"]), "Q4": dist(A["Q4"]), "z_Q2_deployment": dist(A["z_Q2_dep"]),
                     "z_Q4_deployment": dist(A["z_Q4_dep"])} for k, A in arms.items()}}

    # ---- E7c: signal characteristics across devices (descriptive)
    d2 = [s for s in L.DEV if L.META[s]["device"] == "2호기"]
    sig2 = {s: signals_of(L.clean_img(s), pm) for s in d2}
    ev = {d: [v for v in pred.values() if v["device"] == d] for d in ("1호기", "2호기")}
    def states(sigs, ref):
        c = collections.Counter(X.quality_state(sg, ref, QG)[0] for sg in sigs)
        return {st: c[st] for st in STATES}
    out["E7c_signal_distributions"] = {
        "dev_1호기": {q: dist([sig0[s][q] for s in d1]) for q in ("Q2", "Q4")},
        "dev_2호기": {q: dist([sig2[s][q] for s in d2]) for q in ("Q2", "Q4")},
        "eval_1호기": {q: dist([v["quality"]["signals"][q] for v in ev["1호기"]]) for q in ("Q2", "Q4")},
        "eval_2호기": {q: dist([v["quality"]["signals"][q] for v in ev["2호기"]]) for q in ("Q2", "Q4")}}
    out["E7c_cross_reference_states"] = {
        "dev_2호기_under_1호기_reference": states(sig2.values(), qref["1호기"]),
        "dev_1호기_under_2호기_reference": states(sig0.values(), qref["2호기"]),
        "eval_1호기_under_1호기_reference(analyzer)": states([v["quality"]["signals"] for v in ev["1호기"]], qref["1호기"]),
        "eval_1호기_under_2호기_reference": states([v["quality"]["signals"] for v in ev["1호기"]], qref["2호기"]),
        "eval_2호기_under_2호기_reference(analyzer)": states([v["quality"]["signals"] for v in ev["2호기"]], qref["2호기"])}
    # added after the first E7b / E7c read-out (2026-10-06 01:15 KST; descriptive only, no decision rule)
    def by_session(items):
        g = collections.defaultdict(list)
        for date, sg in items:
            g[date].append(sg)
        return {d: {"n": len(v), "Q2_median": float(np.median([x["Q2"] for x in v])), "Q4_median": float(np.median([x["Q4"] for x in v]))}
                for d, v in sorted(g.items())}
    out["E7c_session_medians_added_after_first_readout"] = {
        "dev_1호기": by_session((session[s], sig0[s]) for s in d1),
        "dev_2호기": by_session((L.META[s]["date"], sig2[s]) for s in d2),
        "eval_1호기": by_session((v["date"], v["quality"]["signals"]) for v in ev["1호기"]),
        "eval_2호기": by_session((v["date"], v["quality"]["signals"]) for v in ev["2호기"])}
    out["calibration_resolution_check"] = ("XGD-1.1.0 has no explicit rule that the image resolution must match the resolution of the device's "
                                           "calibration images; the effective-scale envelope is profile-level. Recorded as a limitation, not changed "
                                           "(motivated by evaluation outputs).")
    out["interpretation_limits"] = [
        "How the provider produced 316x332 1호기 images (resampling, cropping or another acquisition setting) is unknown.",
        "Without evaluation labels, a stop cannot be called a justified block or an over-block.",
        "Cross-reference states describe signal similarity only; they are not evidence of a device-label error and are not used for routing."]
    L.jdump(L.OUT / "E7_eval79_reasons" / "RESULTS.json", out)

    b = out["E7b_counterfactual_dev_1호기"]["arms"]
    lines = ["# E7. 평가 79장 자동 확정 중단 원인 분해 (정답 미사용)", "",
             f"- 사전 등록: {out['prereg']} — 사후 동기의 진단. 규칙·임계값·라우팅 변경 없음.",
             f"- 해상도(역할|장비): {out['E7a_resolution_by_device']}", "", "## E7a. 켜진 중단 조건(분석기 출력)", ""]
    for k, v in out["E7a_stop_conditions"].items():
        lines.append(f"- {k}: {v}")
    lines += [f"- 품질만이 유일한 중단 조건인 영상: {out['E7a_quality_is_the_only_stop_condition']}",
              f"- |z|가 가장 큰 품질 신호: {out['E7a_largest_abs_z_signal']}", "",
              "## E7b. 개발 1호기 124장을 316×332로 바꿨을 때 품질 판정(배포 기준 / LOSO 기준)", "",
              "| 변환 | 배포 기준 N/R/F | LOSO 기준 N/R/F | FULL 비율(배포) | 사전 기준 판정 | Q4 중앙값 | Q4 z 중앙값(배포) |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, A in b.items():
        dd, ll = A["deployment_reference"], A["loso_reference"]
        lines.append(f"| {k} | {dd['NORMAL']}/{dd['REVIEW']}/{dd['FULL_REINSPECTION']} | {ll['NORMAL']}/{ll['REVIEW']}/{ll['FULL_REINSPECTION']} | "
                     f"{A['full_rate_deployment']:.3f} | {A['verdict_prereg']} | {A['Q4']['median']:.4f} | {A['z_Q4_deployment']['median']:.2f} |")
    lines += ["", f"- 원본 arm의 전처리 결과가 저장된 전처리 영상과 픽셀 단위로 같은 수: {out['E7b_counterfactual_dev_1호기']['original_arm_pixel_equal_to_stored_preprocessed']}/{n1}",
              "", "## E7c. 신호 분포와 교차 기준 판정(기술 통계)", ""]
    for k, v in out["E7c_signal_distributions"].items():
        lines.append(f"- {k}: Q2 중앙값 {v['Q2']['median']:.4f}, Q4 중앙값 {v['Q4']['median']:.4f} (Q4 p05~p95 {v['Q4']['p05']:.4f}~{v['Q4']['p95']:.4f})")
    for k, v in out["E7c_cross_reference_states"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "촬영일별 중앙값(첫 결과를 본 뒤 추가한 기술 통계, 판정 기준 없음):", ""]
    for k, v in out["E7c_session_medians_added_after_first_readout"].items():
        lines.append(f"- {k}: " + ", ".join(f"{d}({x['n']}장) Q4 {x['Q4_median']:.3f}" for d, x in v.items()))
    lines += ["", f"- 보정 해상도 검사: {out['calibration_resolution_check']}", "", "## 해석 제한", ""] + [f"- {x}" for x in out["interpretation_limits"]]
    (L.OUT / "E7_eval79_reasons" / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("E7 written", {k: (A["full_rate_deployment"], A["verdict_prereg"]) for k, A in b.items()}, flush=True)


if __name__ == "__main__":
    main()
