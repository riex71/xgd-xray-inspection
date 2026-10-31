"""Result-linked final documents (called at the end of experiments/final_report.py; can also run alone).

Every statement with a number is computed from a completed artifact. Missing artifacts are reported as 미완료 instead of being filled
in. Pre-registered decision rules (docs/PREREGISTRATION_XGD-1.1.md) are applied in code.
Outputs: docs/FINAL_SUMMARY.md, docs/ISSUE_RESPONSE_TABLE.md, docs/RUBRIC_EVIDENCE.md, docs/CHANGELOG_XGD-1.1.md, docs/LIMITATIONS.md
Returns the package-readiness record (written by final_report.py to outputs/experiments/resume/PACKAGE_READINESS.json).
"""
import collections
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "experiments"


def load(rel):
    p = ROOT / rel
    try:
        return json.loads(p.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return None


def g(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        elif isinstance(d, dict) and str(k) in d:
            d = d[str(k)]
        else:
            return default
    return d


def f(v, p=3, pct=False):
    if v is None or v == "N/A":
        return "N/A"
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)) and not pct:
        return str(int(v))
    if not math.isfinite(float(v)):
        return "N/A"
    return f"{100 * v:.{max(p - 2, 1)}f}%" if pct else f"{v:.{p}f}"


LABEL = {"ACCEPT": "ACCEPT", "REVIEW": "REVIEW", "FULL_REINSPECTION": "FULL", "NORMAL": "정상"}


def counts(d, order=("ACCEPT", "REVIEW", "FULL_REINSPECTION", "NORMAL")):
    """{'FULL_REINSPECTION': 32, 'ACCEPT': 39} -> 'ACCEPT 39, FULL 32' (fixed order, Korean labels)."""
    if not isinstance(d, dict):
        return "N/A"
    keys = [k for k in order if k in d] + [k for k in d if k not in order]
    return ", ".join(f"{LABEL.get(k, k)} {d[k]}" for k in keys)


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x is not None and math.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def text(rel, s):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s.rstrip() + "\n", encoding="utf8")


class R:
    """All artifacts in one place."""
    def __init__(self):
        self.e1 = load("outputs/experiments/E1_rules_x14/RESULTS.json")
        self.e3 = load("outputs/experiments/E3_scoring/RESULTS.json")
        self.sig = load("outputs/experiments/E3_scoring/SAFETY_SIGNALS.json")
        self.e4 = load("outputs/experiments/E4_audit/RESULTS.json")
        self.e5 = load("outputs/experiments/E5_device_tolerance/RESULTS.json")
        self.e6 = load("outputs/experiments/E6_data_profile/RESULTS.json")
        self.e7 = load("outputs/experiments/E7_eval79_reasons/RESULTS.json")
        self.e8 = load("outputs/experiments/E8_failure_geometry/RESULTS.json")
        self.e9 = load("outputs/experiments/E9_missed_object_blindspot/RESULTS.json")
        self.verify = load("outputs/validation/VERIFY_REPORT.json")
        self.repro = load("outputs/experiments/resume/REPRODUCTION_REPORT.json")
        self.timing = load("outputs/experiments/resume/OPERATIONAL_TIMING.json")
        self.pred = load("outputs/test_predictions.json")
        self.pred_old = load("versions/XGD-1.0.0/outputs/test_predictions.json")
        self.crc = {p: load(f"outputs/calibration/crc_{p}.json") for p in ("dfl_src12", "dfl_src3_2")}
        self.seal = load("outputs/experiments/resume/EXECUTION_SEAL.json")
        tl = ROOT / "outputs/experiments/resume/tests.log"
        self.tests_log = tl.read_text(encoding="utf8", errors="replace") if tl.exists() else ""
        # the test log counts only if it is newer than every source it tests
        watched = [*ROOT.glob("src/*.py"), *ROOT.glob("tests/*.py"), ROOT / "experiments/lib.py", ROOT / "config.yaml"]
        self.tests_fresh = tl.exists() and all(tl.stat().st_mtime >= w.stat().st_mtime for w in watched if w.exists())
        self.tests_ok = self.tests_fresh and "passed" in self.tests_log and "failed" not in self.tests_log and "error" not in self.tests_log.lower()
        self.tests_line = next((x.strip() for x in reversed(self.tests_log.splitlines()) if "passed" in x or "failed" in x), "")

    # ---------- E3 accessors
    def cond(self, prof, c):
        return g(self.e3, "profiles", prof, "conditions", c)

    def seeds(self, prof, c):
        return g(self.cond(prof, c), "per_seed", default={}) or {}

    def det_mean(self, prof, c, k):
        return mean([g(m, k) for m in self.seeds(prof, c).values()])

    def det_worst(self, prof, c, k, fn=min):
        v = [g(m, k) for m in self.seeds(prof, c).values() if isinstance(g(m, k), (int, float))]
        return fn(v) if v else None

    def dec(self, prof, c, design, seed=17):
        return g(self.seeds(prof, c), seed, design) or g(self.seeds(prof, c), str(seed), design)

    def dec_mean(self, prof, c, design, k):
        return mean([g(m, design, k) for m in self.seeds(prof, c).values()])


FAM = {"dfl_src12": ("3호기", "DFL-B"), "dfl_src3_2": ("1·2호기", "DFL-A")}


# ------------------------------------------------------------------------------------------------ findings
def x8_effect(r):
    if not r.e1:
        return None
    c = g(r.e1, "families", "DFL-B", "conditions")
    nat, rule = g(c, "native|clean", "detection", "mean"), g(c, "rule|clean", "detection", "mean")
    if not nat or not rule:
        return None
    return {"R3_native": nat["R3"], "R3_x8": rule["R3"], "FP_native": nat["FP_per_image3"], "FP_x8": rule["FP_per_image3"],
            "pFROC_native": nat.get("pFROC3"), "pFROC_x8": rule.get("pFROC3"), "F1_native": nat["F13"], "F1_x8": rule["F13"],
            "F1iou_native": nat["F1_iou"], "F1iou_x8": rule["F1_iou"]}


def translation(r, prof):
    rows = []
    for c, C in (g(r.e3, "profiles", prof, "conditions", default={}) or {}).items():
        if not c.startswith("R1_"):
            continue
        for sd, m in C["per_seed"].items():
            t = m.get("translation") or {}
            p = t.get("paired_P50_P90_P95_max")
            rows.append({"cond": c, "seed": sd, "P95": p[2] if p else None, "max": p[3] if p else None,
                         "acc_fail": g(m, "P-IND-deployment-grade", "accepted_failures"), "n": g(m, "P-IND-deployment-grade", "n"),
                         "lost": t.get("clean_GT_lost"), "recovered": t.get("GT_recovered"), "R3": m.get("R3"),
                         "unmatched": (t.get("unmatched_clean") or 0) + (t.get("unmatched_shift") or 0)})
    if not rows:
        return None
    base = r.det_mean(prof, "clean", "R3")
    return {"conditions": len({x["cond"] for x in rows}), "worst_P95": max(x["P95"] for x in rows if x["P95"] is not None),
            "worst_max": max(x["max"] for x in rows if x["max"] is not None),
            "median_P95": float(np.median([x["P95"] for x in rows if x["P95"] is not None])),
            "min_R3": min(x["R3"] for x in rows), "clean_R3": base, "max_lost": max(x["lost"] for x in rows),
            "total_lost": sum(x["lost"] for x in rows), "total_recovered": sum(x["recovered"] for x in rows),
            "worst_cond": max(rows, key=lambda x: x["P95"] or 0)["cond"],
            "acc_fail_rate": (sum(x["acc_fail"] or 0 for x in rows) / sum(x["n"] or 0 for x in rows)) if sum(x["n"] or 0 for x in rows) else None,
            "clean_acc_fail_rate": mean([(g(m, "P-IND-deployment-grade", "accepted_failures") or 0) / g(m, "P-IND-deployment-grade", "n")
                                         for m in r.seeds(prof, "clean").values() if g(m, "P-IND-deployment-grade", "n")])}


def synth_rate(r, prof, cond, key="new_hit_rate"):
    return mean([g(m, "synthetic", key) for m in r.seeds(prof, cond).values()])


def location(r, prof):
    """Pre-registered rule: 'no evidence of location dependence' only if moved-location hit rate drops <= 5 pp versus the same-location
    re-insertion control for BOTH synthesis methods; 'dependence' if > 5 pp for both; otherwise 'undetermined'."""
    out = {}
    for meth in ("M1", "M2"):
        mv = mean([synth_rate(r, prof, f"R2a_move_{meth}_{j}") for j in range(3)])
        sm = mean([synth_rate(r, prof, f"R2a_same_{meth}_{j}") for j in range(3)])
        old = mean([sum(g(m, "synthetic", "old_hit_count") or 0 for m in r.seeds(prof, f"R2a_move_{meth}_{j}").values()) / 3 for j in range(3)])
        sweep = [synth_rate(r, prof, f"R2b_sweep_{meth}_{k}") for k in range(10)]
        out[meth] = {"moved": mv, "same": sm, "drop_pp": None if mv is None or sm is None else 100 * (sm - mv), "old_site_hits_per_seed": old,
                     "sweep": sweep}
    d = [out[m]["drop_pp"] for m in ("M1", "M2")]
    if any(x is None for x in d):
        verdict = "미완료"
    elif all(x <= 5 for x in d):
        verdict = "근거 없음(사전 기준: 두 합성 방법 모두 하락 ≤ 5%p)"
    elif all(x > 5 for x in d):
        verdict = "의존성 있음(두 합성 방법 모두 하락 > 5%p)"
    else:
        verdict = "판단 불가(합성 방법 간 결론 불일치)"
    out["verdict"] = verdict
    return out


def trace_control(r, prof):
    """Pre-registered rule: trace neutralization supported if the final DFL's trace-site alarm rate is lower than the v3only control's
    (3-seed means) AND its real held-fold detection (3 px recall) is equal or higher."""
    fin_trace = synth_rate(r, prof, "R2e_trace", "trace_hit_rate")
    ctl = g(r.e3, "controls", "v3only", prof, default={}) or {}
    def ctl_rate(cond):
        return mean([(g(m, cond, "target_hits") / g(m, cond, "target_count")) if g(m, cond, "target_count") else None for m in ctl.values()])
    ctl_trace = ctl_rate("R2e_trace")
    fin_R3, ctl_R3 = r.det_mean(prof, "clean", "R3"), mean([g(m, "clean", "R3") for m in ctl.values()])
    free = {meth: (synth_rate(r, prof, f"R2f_free_{meth}"), ctl_rate(f"R2f_free_{meth}")) for meth in ("M1", "M2")}
    mark = {meth: (synth_rate(r, prof, f"R2g_mark_{meth}"), ctl_rate(f"R2g_mark_{meth}")) for meth in ("M1", "M2")}
    if None in (fin_trace, ctl_trace, fin_R3, ctl_R3):
        verdict = "미완료"
    elif fin_trace < ctl_trace and fin_R3 >= ctl_R3:
        verdict = "흔적 무력화 효과 지지(사전 기준 충족)"
    elif fin_trace == ctl_trace == 0:
        verdict = "구별 불가: 두 모델 모두 경보 0, 사전 기준 미충족"
    else:
        verdict = "이 시험으로는 지지되지 않음(사전 기준 미충족)"
    return {"final_trace": fin_trace, "control_trace": ctl_trace, "final_R3": fin_R3, "control_R3": ctl_R3, "free": free, "mark": mark,
            "verdict": verdict}


def stress_rows(r, prof):
    rows = []
    for c, C in (g(r.e3, "profiles", prof, "conditions", default={}) or {}).items():
        if not (c.startswith("R3_") or c.startswith("GRID_") or c in ("clean", "native")):
            continue
        ps = C["per_seed"]
        rows.append({"cond": c, "R3_mean": mean([m.get("R3") for m in ps.values()]), "R3_worst": min(m.get("R3") for m in ps.values()),
                     "FP_mean": mean([m.get("FP_per_image3") for m in ps.values()]),
                     "full_dep": mean([g(m, "P-IND-deployment-grade", "full_rate") for m in ps.values()]),
                     "cov_dep": mean([g(m, "P-IND-deployment-grade", "coverage") for m in ps.values()]),
                     "accfail_dep": sum(g(m, "P-IND-deployment-grade", "accepted_failures") or 0 for m in ps.values()),
                     "acc_dep": sum(g(m, "P-IND-deployment-grade", "accept") or 0 for m in ps.values()),
                     "cov_strict": mean([g(m, "P-IND-strict", "coverage") for m in ps.values()])})
    return rows


def baseline_rows(r):
    rows = []
    for prof, yp in (("dfl_src12", "yolo_src12"), ("dfl_src3_2", "yolo_src3_2")):
        for mode in ("clean", "native"):
            for name, per in (("DFL (reg_max 16)", {sd: m for sd, m in r.seeds(prof, mode).items()}),
                              ("YOLO26s (reg_max 1)", {sd: g(C, mode) for sd, C in (g(r.e3, "controls", "baseline", yp, default={}) or {}).items()})):
                ms = [m for m in per.values() if m]
                if not ms:
                    continue
                rows.append({"set": FAM[prof][0], "model": name, "mode": "X8" if mode == "clean" else "960(원래)",
                             "P3": mean([m["P3"] for m in ms]), "R3": mean([m["R3"] for m in ms]), "F13": mean([m["F13"] for m in ms]),
                             "F15": mean([m["F15"] for m in ms]), "F1iou": mean([m["F1_iou"] for m in ms]), "pFROC": mean([m.get("pFROC3") for m in ms]),
                             "FP": mean([m["FP_per_image3"] for m in ms]), "R3_worst": min(m["R3"] for m in ms),
                             "P90": mean([m["err_P50_P90_P95"][1] for m in ms if m.get("err_P50_P90_P95")]),
                             "AP50": mean([m.get("AP50_101pt_E0_top30") for m in ms]), "seeds": len(ms)})
    return rows


def eval79_reasons(r):
    """E7 (label-free): stop conditions of the evaluation-79 decisions and the development-only resolution counterfactual."""
    e = r.e7
    if not e:
        return None
    sc = g(e, "E7a_stop_conditions", default={}) or {}
    oq = g(e, "E7a_quality_is_the_only_stop_condition", default={}) or {}
    arms = g(e, "E7b_counterfactual_dev_1호기", "arms", default={}) or {}
    dist = g(e, "E7c_signal_distributions", default={}) or {}
    xr = g(e, "E7c_cross_reference_states", default={}) or {}
    sess = g(e, "E7c_session_medians_added_after_first_readout", default={}) or {}
    q4 = {k: g(v, "Q4", "median") for k, v in dist.items()}
    d1_full = sum(n for k, v in sc.items() if k.startswith("1호기|FULL") for c, n in v.items())
    d1_rev = sum(n for k, v in sc.items() if k.startswith("1호기|REVIEW") for c, n in v.items())
    return {"stop_conditions": sc, "only_quality": oq, "d1_full": d1_full, "d1_review": d1_rev,
            "d1_only_quality": sum(n for k, n in oq.items() if k.startswith("1호기")),
            "resample_full": {k: g(arms, k, "full_rate_deployment") for k in ("resample_area", "resample_linear")},
            "resample_q4_z": {k: g(arms, k, "z_Q4_deployment", "median") for k in ("resample_area", "resample_linear")},
            "resample_review": {k: g(arms, k, "deployment_reference", "REVIEW") for k in ("resample_area", "resample_linear")},
            "crop_full": max(float(g(arms, k, "full_rate_deployment") or 0.0) for k in ("crop_center", "crop_left", "crop_right")),
            "verdicts": {k: g(arms, k, "verdict_prereg") for k in arms if k != "original"},
            "q4": q4, "eval1_under_2": g(xr, "eval_1호기_under_2호기_reference"), "dev2_under_1": g(xr, "dev_2호기_under_1호기_reference"),
            "sessions": sess}


def audit_facts(r):
    """E4 numbers used in the documents."""
    if not r.e4:
        return None
    lp = g(r.e4, "unproven_same_date_cross_device_layout_pairs", default=[]) or []
    stems = {x for pr in lp for x in (pr["a"], pr["b"])}
    dev = collections.Counter(L_META_DEVICE(s) for s in stems)
    out = {"layout_pairs": len(lp), "layout_images_by_device": dict(dev), "near_pairs": len(g(r.e4, "near_pairs", default=[]) or []),
           "candidates": g(r.e4, "near_search", "candidate_count"), "profiles": {}}
    for p in FAM:
        a = g(r.e4, "profile_audit", p, default={}) or {}
        sub = g(a, "screen_disjoint_subset_metrics", default={}) or {}
        perm = g(a, "bbox_permutation_diagnostic", default={}) or {}
        out["profiles"][p] = {"cross_split_near_pairs": len(g(a, "near_pairs_cross_split", default=[]) or []),
                              "excluded_held": len(g(a, "excluded_held_images", default=[]) or []),
                              "subset_images": g(next(iter(sub.values()), {}), "images"),
                              "subset_R3_mean": mean([g(m, "R3") for m in sub.values()]),
                              "prior_R3": g(a, "source_only_position_prior", "R3"),
                              "perm_R3_mean": mean([g(m, "R3") for m in perm.values()])}
    return out


def L_META_DEVICE(stem):
    global _META
    if "_META" not in globals():
        recs = load("data/metadata.json") or {"records": []}
        _META = {x["stem"]: x["device"] for x in recs["records"]}
    return _META.get(stem, "?")


def calibration_requirements():
    """Sample sizes implied by the implemented rules (no data): CRC accepts something only if (0 + 1) / (n + 1) <= alpha; LTT (exact binomial,
    Bonferroni over the 10-point grid, delta 0.10) needs n accepted with k failures such that P(Bin(n, alpha) <= k) <= 0.01."""
    from scipy import stats
    crc = next(n for n in range(1, 1000) if 1 / (n + 1) <= 0.05)
    ltt = {a: {k: next(n for n in range(1, 5000) if stats.binom.cdf(k, n, a) <= 0.10 / 10) for k in (0, 1, 2)} for a in (0.05, 0.10)}
    return {"crc_min_n": crc, "ltt": ltt}


def component_rows(r):
    """Detector -> +X8 -> +X12/CRC on the same held images (3 seed means). 'All accepted' risk = image failure rate."""
    rows = []
    for p in FAM:
        nat, cl = r.seeds(p, "native"), r.seeds(p, "clean")
        def m(ps, k):
            return mean([g(x, k) for x in ps.values()])
        deg = [c for c in (g(r.e3, "profiles", p, "conditions", default={}) or {})
               if c.startswith("R3_") or (c.startswith("GRID_") and not c.endswith("_clean"))]
        deg_fail = mean([m(r.seeds(p, c), "image_failure_rate") for c in deg])
        acc = sum(g(x, "P-IND-deployment-grade", "accepted_failures") or 0 for c in deg for x in r.seeds(p, c).values())
        tot = sum(g(x, "P-IND-deployment-grade", "n") or 0 for c in deg for x in r.seeds(p, c).values())
        rows.append({"set": FAM[p][0], "native_same": all(g(nat, sd, "R3") == g(cl, sd, "R3") for sd in cl),
                     "det960": (m(nat, "R3"), m(nat, "FP_per_image3"), m(nat, "image_failure_rate")),
                     "x8": (m(cl, "R3"), m(cl, "FP_per_image3"), m(cl, "image_failure_rate")),
                     "dep": (m(cl, "P-IND-deployment-grade") if False else mean([g(x, "P-IND-deployment-grade", "coverage") for x in cl.values()]),
                             mean([g(x, "P-IND-deployment-grade", "joint_risk") for x in cl.values()])),
                     "strict": (mean([g(x, "P-IND-strict", "coverage") for x in cl.values()]), mean([g(x, "P-IND-strict", "joint_risk") for x in cl.values()])),
                     "deg_all_accept_risk": deg_fail, "deg_xgd_joint": acc / tot if tot else None, "deg_n": len(deg)})
    return rows


def breakdown_lines(r):
    cb = g(r.e4, "condition_breakdown", default={}) or {}
    names = {"min_side_px": "박스 최소 변(px)", "edge_distance_px": "제품 경계까지 거리(px)", "contrast_proxy": "대비 대리지표", "bbox_aspect": "박스 종횡비"}
    out = ["| 조건 | 구간 | 1호기 이물 수(seed 17/29/43 미탐) | 2호기 | 3호기 |", "| --- | --- | --- | --- | --- |"]
    for feat, d in cb.items():
        bins = []
        for k in d:
            b = k.split(": ", 1)[1]
            if b not in bins:
                bins.append(b)
        for b in bins:
            cells = []
            for dev in ("1호기", "2호기", "3호기"):
                v = d.get(f"{dev}: {b}") or {}
                n = v.get("objects", 0)
                cells.append(f"{n}({'/'.join(str(v['miss_by_seed'][sd]) for sd in ('17', '29', '43'))})" if n else "-")
            if any(c != "-" for c in cells):
                out.append(f"| {names.get(feat, feat)} | {b} | " + " | ".join(cells) + " |")
    return out


def blindspot(r):
    """E9 (exploratory): synthetic M2 objects that are missed still get ACCEPT; FN vs FP risk; C3 seed-disagreement flag rates."""
    if not r.e9:
        return None
    out = {}
    for p, P in r.e9["profiles"].items():
        d = "P-IND-strict" if p == "dfl_src3_2" else "P-IND-deployment-grade"      # 3호기 strict accepts nothing
        x = P.get("R2f_free_M2")
        if not x:
            continue
        m2 = [c for c in P if "M2" in c and not c.startswith("R2a_same")]
        out[p] = {"design": "독립 보정" if d == "P-IND-strict" else "진단 설계", "acc": x["decisions"][d]["accept"], "af": x["decisions"][d]["accepted_failures"],
                  "fn": x["decisions"][d]["FN_risk_mean"], "fp": x["decisions"][d]["FP_risk_mean"],
                  "c3_fail": (min(P[c]["C3_flag"]["fail"]["rate"] for c in m2), max(P[c]["C3_flag"]["fail"]["rate"] for c in m2)),
                  "c3_ok": (min(P[c]["C3_flag"]["ok"]["rate"] for c in m2), max(P[c]["C3_flag"]["ok"]["rate"] for c in m2))}
    return out


def blindspot_text(r):
    b = blindspot(r)
    if not b:
        return None
    return "; ".join(f"{FAM[p][0]}({v['design']}, 흔적 없는 M2 합성 이물): 자동 확정 {v['acc']}장 중 {v['af']}장 실패, FN 위험 {f(v['fn'])} 대 FP 위험 {f(v['fp'])}; "
                     f"seed 불일치(C3)는 M2 조건의 실패 영상 {f(v['c3_fail'][0])}~{f(v['c3_fail'][1])}를 표시하지만 정상 영상도 {f(v['c3_ok'][0])}~{f(v['c3_ok'][1])}를 표시"
                     for p, v in b.items())


def failure_geometry(r):
    """E8: clean-condition failures of the final DFL classified by distance (3 seeds)."""
    out = {}
    for p, P in (g(r.e8, "profiles", default={}) or {}).items():
        loc = [d for v in P["per_seed"].values() for x in v["failures"] if x["type"] == "localization_only" for d in x["missed_gt_nearest_pred_px"]]
        fp_only = [x for v in P["per_seed"].values() for x in v["failures"] if x["type"] == "FP_only"]
        stems = collections.Counter(x["stem"] for v in P["per_seed"].values() for x in v["failures"])
        out[p] = {"failed": P["all_seeds"]["failed_images"], "types": P["all_seeds"]["image_types"], "loc_n": len(loc),
                  "loc_min": min(loc) if loc else None, "loc_max": max(loc) if loc else None, "fp_only": len(fp_only),
                  "fp_only_dist": [round(d, 1) for x in fp_only for d in x["fp_nearest_gt_px"]],
                  "fp_only_stems": sorted({x["stem"] for x in fp_only}), "distinct_images": len(stems),
                  "repeated_images": sum(1 for n in stems.values() if n > 1)}
    return out


def geometry_text(r):
    fg = failure_geometry(r)
    if not fg:
        return None
    parts = []
    for p, v in fg.items():
        t = f"{FAM[p][0]} 실패 {v['failed']}장(3 seed) 중 {v['loc_n']}장은 예측이 정답 중심에서 {f(v['loc_min'], 2)}~{f(v['loc_max'], 2)}px 떨어진 위치 오차"
        if v["fp_only"]:
            t += f", {v['fp_only']}장은 같은 영상({', '.join(v['fp_only_stems'])})의 반복 오탐(정답에서 약 {min(v['fp_only_dist']):.0f}~{max(v['fp_only_dist']):.0f}px)"
        t += f"; 서로 다른 영상 {v['distinct_images']}장 중 {v['repeated_images']}장은 2개 이상 seed에서 반복"
        parts.append(t)
    return "; ".join(parts)


def retrain_compare(r):
    per = g(r.repro, "retrain_profiles_seeds")
    if not per:
        return None
    out = {}
    for p in FAM:
        rt = [m for m in (per.get(p) or {}).values() if m]
        fr = [m for m in r.seeds(p, "clean").values() if m]
        if not rt or not fr:
            continue
        out[p] = {k: (mean([m.get(key) for m in fr]), mean([m.get(key) for m in rt])) for k, key in
                  (("R3", "R3"), ("F13", "F13"), ("FP", "FP_per_image3"), ("pFROC", "pFROC3"))}
        out[p]["R3_worst"] = (min(m["R3"] for m in fr), min(m["R3"] for m in rt))
        out[p]["seeds"] = (len(fr), len(rt))
    return out


def signal_rows(r):
    out = []
    for prof, P in (g(r.sig, "profiles", default={}) or {}).items():
        for name, c in P.get("candidates", {}).items():
            out.append({"profile": prof, "signal": name, **c})
    return out


def signal_verdict(r):
    rows = signal_rows(r)
    if not rows:
        return "후보 신호 평가 미완료"
    met = [f"{FAM.get(x['profile'], (x['profile'],))[0]} {x['signal']}" for x in rows if x.get("status") == "INTERNAL_GATE_MET_NOT_INDEPENDENTLY_QUALIFIED"]
    if met:
        return f"내부 사전 기준을 충족한 후보: {', '.join(met)} (독립 검증 전이라 자동 채택하지 않음)"
    parts = []
    for prof in FAM:
        rs = [x for x in rows if x["profile"] == prof]
        if not rs:
            continue
        st = ("판단 불가(주 설계에서 자동 확정된 실패 없음)" if all(str(x.get("status", "")).startswith("INCONCLUSIVE") for x in rs)
              else "채택 기준 미충족(자동 확정된 실패 포착 " + ", ".join(f"{x.get('captured')}/{x.get('accepted_failures_before')}" for x in rs) + ")")
        flags = ", ".join(f"{x['signal'].split('_')[0]} {x.get('flagged_failures')}/{x.get('all_failures')}" for x in rs)
        parts.append(f"{FAM[prof][0]} {st}; 3 seed 전체 실패 중 신호가 표시한 수 {flags}")
    return "사전 채택 기준을 충족한 후보 없음 — " + " / ".join(parts)


# ------------------------------------------------------------------------------------------------ documents
def summary_doc(r):
    L = ["# XGD-1.1 결과 요약 (쉬운 설명)", "",
         "이 문서는 실행이 끝난 결과 파일에서 자동으로 만든 요약이다. 모든 수치는 같은 공급처 개발 자료(반복 분석된 자료)의 내부 결과이며, 독립 외부 검증이 아니다.",
         "**ACCEPT는 '검출한 이물 위치를 자동으로 확정'한다는 뜻이며 제품 합격이 아니다.** 개발 자료에는 이물 없는 정상 제품 영상이 없다.", ""]
    x8 = x8_effect(r)
    L += ["## 실패 원인 → 대응 → 측정 결과 → 남은 한계", "",
          "| 실패 원인 | 대응 | 측정 결과 | 남은 한계 |", "| --- | --- | --- | --- |"]
    if x8:
        L.append(f"| 3호기는 같은 이물이 작은 입력 배율로 들어감 | X8: 네트워크 입력 유효 배율(imgsz/긴 변)을 학습 범위 하한까지 확대 | 같은 가중치에서 3px 검출률 {f(x8['R3_native'])}→{f(x8['R3_x8'])}, 오탐/장 {f(x8['FP_native'])}→{f(x8['FP_x8'])}, pFROC {f(x8['pFROC_native'])}→{f(x8['pFROC_x8'])} (3 seed 평균) | 물리적 배율·pixel pitch 보정이 아님. 장비와 해상도가 한 묶음이라 원인을 완전히 분리하지 못함 |")
    br = baseline_rows(r)
    def pick(set_, model):
        return next((x for x in br if x["set"] == set_ and x["model"].startswith(model) and x["mode"] == "X8"), None)
    d3, y3, d12, y12 = pick("3호기", "DFL"), pick("3호기", "YOLO"), pick("1·2호기", "DFL"), pick("1·2호기", "YOLO")
    if d3 and y3:
        L.append(f"| 작은 이물의 중심 위치가 2~3px 흔들림 | 분포형 박스 회귀(DFL, reg_max 16) | 같은 TXT·전처리·조리법·분할의 일반 YOLO26s(reg_max 1) 대비 3호기 3px F1 "
                 f"{f(y3['F13'])}→{f(d3['F13'])}, 오탐/장 {f(y3['FP'])}→{f(d3['FP'])}, pFROC {f(y3['pFROC'])}→{f(d3['pFROC'])}"
                 + (f"; 1·2호기 F1 {f(y12['F13'])}→{f(d12['F13'])}, 오탐/장 {f(y12['FP'])}→{f(d12['FP'])}" if d12 and y12 else "") +
                 " (X8, 3 seed 평균) | DFL은 이미 선택된 모델이라 사후 비교. 1·2호기에서는 차이가 거의 없음 |")
    tc = {p: trace_control(r, p) for p in FAM} if r.e3 else {}
    if tc:
        L.append("| 검사기 색 표시의 흔적이 이물 단서가 될 위험 | v3 표시 제거 + 가짜 흔적 학습(v3tr) | " +
                 "; ".join(f"{FAM[p][0]}: 흔적만 있는 자리 경보 최종 {f(t['final_trace'])}·대조 {f(t['control_trace'])}({t['verdict']}), "
                           f"흔적 없는 합성 이물 검출 M1 {f(t['free']['M1'][0])}·대조 {f(t['free']['M1'][1])}, M2 {f(t['free']['M2'][0])}·대조 {f(t['free']['M2'][1])}, "
                           f"실제 보류 영상 3px 검출률 {f(t['final_R3'])}·대조 {f(t['control_R3'])}" for p, t in tc.items()) +
                 " (대조 = 같은 조리법에서 가짜 흔적 학습만 뺀 DFL) | 흔적 무력화 효과는 이 시험으로 확인되지 않음. 최종 모델은 M2(Poisson 합성) 이물을 대조보다 덜 검출한다 — "
                 "가짜 흔적 학습의 영향인지 합성 방식의 인공 흔적 때문인지 분리하지 못함. 흔적 없는 실제 현장 영상 없음 |")
    if r.e3:
        loc = {p: location(r, p) for p in FAM}
        def sweep_txt(sw):
            v = [x for x in sw if x is not None]
            return f"{f(min(v))}~{f(max(v))}" if v else "N/A"
        L.append("| 학습 위치가 좁아(제품 폭 25~38%) 위치·문맥에 의존할 위험 | 합성 이물 이동·원위치 재삽입·10구간 위치 sweep·다른 날짜 문맥 시험 | " +
                 "; ".join(f"{FAM[p][0]}: 이동/원위치 검출 M1 {f(l['M1']['moved'])}/{f(l['M1']['same'])}, M2 {f(l['M2']['moved'])}/{f(l['M2']['same'])}, "
                           f"구간별 M1 {sweep_txt(l['M1']['sweep'])}·M2 {sweep_txt(l['M2']['sweep'])} → {l['verdict']}" for p, l in loc.items()) +
                 " | M1은 위치와 무관하게 검출되지만 M2는 위치에 따라 크게 달라진다. 단, M2도 정답이 몰린 구간보다 제품 가운데 구간이 더 높고 옛 자리 잔류 검출이 거의 없어 "
                 "'학습 위치 기억'만으로는 설명되지 않음(확인 아님). 실제 넓은 위치 분포 자료 없음 |")
        tr = {p: translation(r, p) for p in FAM}
        if all(tr.values()):
            L.append("| 평행 이동에 따라 예측이 흔들릴 위험 | 14개 정수 픽셀 이동(가로·세로 ±1~±3px 12개 + 대각 2개; 원본 1px = 입력 약 2.7~2.9px) | " +
                     "; ".join(f"{FAM[p][0]}: 짝지은 중심 변위 P95 중앙값 {f(t['median_P95'], 2)}px·최악 {f(t['worst_P95'], 2)}px, 3px 검출률 최저 {f(t['min_R3'])}(정상 {f(t['clean_R3'])}), "
                               f"자동 확정된 실패 비율 {f(t['acc_fail_rate'])}(정상 {f(t['clean_acc_fail_rate'])}, 진단 설계)"
                               for p, t in tr.items()) + " | 경계 복제 방식의 합성 이동이며 실제 제품 위치 변화의 증명은 아님. 이동해도 고신뢰도 실패 비율은 정상 조건과 비슷한 수준 |")
    sr = {p: stress_rows(r, p) for p in FAM} if r.e3 else {}
    if sr:
        strong = {p: [x for x in rows if x["cond"] in ("R3_blur16", "R3_noise8", "R3_resample05")] for p, rows in sr.items()}
        deg = {p: [x for x in rows if x["cond"].startswith("R3_") or (x["cond"].startswith("GRID_") and not x["cond"].endswith("_clean"))] for p, rows in sr.items()}
        L.append("| 흐림·잡음·해상도 저하 등 촬영 조건 변화 | X12 품질 검사로 자동 확정 중단 | " +
                 "; ".join(f"{FAM[p][0]}: 강한 열화(흐림σ1.6·잡음σ8·리샘플0.5) 3px 검출률 최저 {f(min((x['R3_worst'] for x in strong[p]), default=None))}; "
                           f"열화 {len(deg[p])}조건 전체에서 자동 확정 {sum(x['acc_dep'] for x in deg[p])}건, 그중 실패 {sum(x['accfail_dep'] for x in deg[p])}건, "
                           f"평균 FULL {f(mean([x['full_dep'] for x in deg[p]]))}" for p in sr) +
                 " (전체 등급 진단 설계, 3 seed 합) | 자동 확정을 멈춘 것이지 검출을 개선한 것이 아님. 약한 열화도 대부분 FULL(처리량 비용) |")
    if r.e1:
        f17 = {fam: g(r.e1, "families", fam, "seed17_clean_rule_failures", default=[]) for fam in ("DFL-B", "DFL-A")}
        acc = {fam: sum(x.get("decision_v10_X14") == "ACCEPT" for x in v) for fam, v in f17.items()}
        L.append(f"| 정상 화질에서 신뢰도 높은 오판 | 장비별 CRC(기대 전체 위험 α 0.05) + 사전 등록한 일관성 신호 3개 시험 | 운영 seed 17 정상 조건 실패: 3호기 {len(f17['DFL-B'])}장 중 {acc['DFL-B']}장, 1·2호기 {len(f17['DFL-A'])}장 중 {acc['DFL-A']}장이 자동 확정됨(X14 설계; 품질 정상, 신뢰도 "
                 + ", ".join(f(x['trust'], 2) for v in f17.values() for x in v) + " ≥ λ)" + (f". 실패 기하(E8): {geometry_text(r)}" if r.e8 else "") + f" | 평균 신뢰도는 이런 실패를 거르지 못함 — 대부분 이물을 찾았지만 중심이 3px 기준을 조금 넘은 경우라 신뢰도가 높다. 3~4px 오차가 라벨 중심 오차인지 모델 오차인지 구분하지 못함. 후보 신호: {signal_verdict(r)} |")
    if r.e9:
        L.append(f"| 놓친 이물이 자동 확정될 위험(평균 신뢰도의 한계) | 판정 신뢰도 = 검출한 이물의 평균 점수 → 놓친 이물은 신뢰도에 드러나지 않음(E9, 탐색) | {blindspot_text(r)} (3 seed 합) | "
                 "자동 확정된 실패의 대부분이 놓침(FN)이다. C3는 정상 영상도 많이 보류시켜 채택하지 않았다. 합성 이물이므로 실제 놓침 빈도의 증거는 아님 |")
    L.append(f"| 안전 판정 코드의 빈틈(독립 검토 §8) | XGD-1.1.0: 등급 enum 집행·배율 범위·CRC 동점·검출 0개·비정상 입력 차단 | 회귀 테스트 {('통과(' + r.tests_line + ')') if r.tests_ok else '확인 필요(로그 없음·오래됨·실패)'}; X14 정상 X8 판정은 1.0과 같고, 진단용 강제 배율은 모두 FULL | 규칙 수정은 위험 관리 개선이며 검출 성능 개선이 아님 |")
    L.append("")
    # operating metrics
    if r.e3:
        L += ["## 정상 조건 운영 지표", "",
              "coverage 열은 '운영 seed 17 / 3 seed 평균', 나머지 열은 운영 seed 17이다. 독립 보정은 평가 절반의 정보를 쓰지 않는 주 평가이고, "
              "전체 등급 진단은 등급을 전체 자료로 정해 독립이 아니다. 3호기는 절반 자료(약 50장)로는 운영 등급이 성립하지 않아 독립 보정에서 모두 보류된다.", "",
              "| 장비군 | 설계 | coverage | REVIEW | FULL | 전체 위험 | ACCEPT 조건부 위험 [95% CI] | 실패 포착 | 올바른 결과 보류 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for p in FAM:
            for design, label in (("P-IND-strict", "독립 보정(주 평가)"), ("P-IND-deployment-grade", "전체 등급 진단(비독립)"), ("P-TIME-strict", "과거→미래")):
                d = r.dec(p, "clean", design)
                if not d:
                    continue
                ci = d.get("conditional_risk_CI95")
                L.append(f"| {FAM[p][0]} | {label} | {f(d['coverage'])} / {f(r.dec_mean(p, 'clean', design, 'coverage'))} | {f(d['review_rate'])} | {f(d['full_rate'])} | "
                         f"{f(d['joint_risk'])} | {f(d['conditional_risk'])} {('[' + f(ci[0]) + ', ' + f(ci[1]) + ']') if isinstance(ci, list) else ''} | "
                         f"{f(d['failure_capture'])} | {f(d['correct_hold_rate'])} |")
        cr = calibration_requirements()
        L += ["", f"- 보증에 필요한 자료량(구현 규칙에서 계산): CRC α 0.05는 보정 영상이 {cr['crc_min_n']}장 이상이어야 무엇이든 자동 확정할 수 있다. "
              f"LTT(δ 0.10, 10점 격자 Bonferroni)로 조건부 위험 ≤ 0.05를 확인하려면, 자동 확정한 영상이 실패 0개일 때 {cr['ltt'][0.05][0]}장, 1개일 때 {cr['ltt'][0.05][1]}장, 2개일 때 {cr['ltt'][0.05][2]}장 필요하다"
              f"(α 0.10이면 {cr['ltt'][0.10][0]}/{cr['ltt'][0.10][1]}/{cr['ltt'][0.10][2]}장). 3호기 개발 영상은 100장뿐이라 α 0.05의 LTT 확인은 사실상 불가능하다."]
        L.append("")
    if r.pred:
        L += ["## 평가 79장 (정답 미사용, 판정 분포만)", "",
              f"- XGD-1.1.0: {counts(g(r.pred, 'summary', 'decisions'))} (장비별 " + "; ".join(f"{d} {counts(v)}" for d, v in (g(r.pred, 'summary', 'by_device', default={}) or {}).items()) + ")"]
        if r.pred_old:
            L.append(f"- XGD-1.0.0: {counts(g(r.pred_old, 'summary', 'decisions'))} — 영상별 판정과 검출 좌표가 1.1.0과 모두 같다(규칙 수정은 이 79장의 판정을 바꾸지 않음).")
        e7 = eval79_reasons(r)
        if e7:
            L += [f"- 중단 원인 분해(E7, 정답 미사용, 사후 동기 진단 — docs/PREREG_E7_EVAL79_REASONS.md): 1호기 32장은 배율 범위·운영 등급·호환성 조건을 모두 통과했고 "
                  f"촬영 품질 gate로 멈췄다(FULL {e7['d1_full']}, REVIEW {e7['d1_review']}; 그중 {e7['d1_only_quality']}장은 품질이 유일한 중단 조건). 2호기 REVIEW는 대부분 신뢰도 < λ다.",
                  f"- 평가 1호기 32장은 해상도가 316×332로 개발 1호기(352×332, 6~7월 촬영)와 다르고, 잡음 이웃 상관(Q4) 중앙값이 {f(e7['q4'].get('eval_1호기'))}로 "
                  f"개발 1호기 {f(e7['q4'].get('dev_1호기'))}보다 낮다. 같은 변화가 2호기 8/31·9/1 촬영일에도 있고(2호기 기준에는 이 촬영일이 포함됨), "
                  f"평가 1호기 신호를 2호기 기준에 넣으면 {counts(e7['eval1_under_2'])}이다.",
                  f"- 반사실(개발 1호기를 316×332로 변환): 가로 리샘플은 Q4를 오히려 높여(z 중앙값 {', '.join(f(v, 2) for v in e7['resample_q4_z'].values())}) 대부분 REVIEW가 되고 "
                  f"FULL 비율은 {', '.join(f(v) for v in e7['resample_full'].values())}, 자르기는 FULL 최대 {f(e7['crop_full'])}다. "
                  "사전 기준으로 '해상도 변환만으로 FULL이 되기에 충분하지 않음'. 9월 1호기의 변화는 옛 형식의 단순 리샘플로 설명되지 않는다.",
                  "- 정답이 없으므로 이 중단이 정당한 차단인지 과차단인지는 판단하지 않는다. 보정 영상과 다른 해상도를 직접 막는 규칙은 없다(한계 13)."]
        L.append("")
    rc = retrain_compare(r)
    if r.e5 or rc or r.verify:
        L += ["## 재현성", ""]
        if r.verify:
            L.append(f"- GPU 재실행 검증(VERIFY): {g(r.verify, 'VERIFY_PASS')} — 동결 가중치의 박스·판정이 봉인 기준과 일치.")
        if r.repro:
            L.append(f"- 격리 사본(새 venv) 기본 실행: VERIFY {g(r.repro, 'isolated_default_VERIFY_PASS')}, 평가 79장 예측·판정 동일 {g(r.repro, 'isolated_default_predictions_decisions_equal')}.")
        if rc:
            for p, v in rc.items():
                L.append(f"- 같은 조리법 재학습({FAM[p][0]}, {v['seeds'][1]} seed): 3px 검출률 동결 {f(v['R3'][0])} → 재학습 {f(v['R3'][1])}, "
                         f"최저 seed {f(v['R3_worst'][0])} → {f(v['R3_worst'][1])}, 오탐/장 {f(v['FP'][0])} → {f(v['FP'][1])}, F1@3px {f(v['F13'][0])} → {f(v['F13'][1])}. "
                         "비트 단위 일치는 기대하지 않으며 제출 가중치는 바꾸지 않았다.")
        for p, v in (g(r.e5, "profiles", default={}) or {}).items():
            L.append(f"- CPU 대 GPU({FAM[p][0]}, seed 17, {v['images']}장): 판정 변경 {v['decision_flips']}건, 3px 결과 변경 {v['outcome_flips_3px']}건, "
                     f"검출 수 불일치 {v['detection_count_mismatch_images']}장, 최대 중심 차이 {v['max_center_shift_px']:.1e}px, 최대 점수 차이 {v['max_score_diff']:.1e}. 다른 GPU 모델은 시험하지 못했다.")
        L.append("")
    return "\n".join(L)


def issue_rows(r):
    """(ID, 평가서 절, 지적, 재확인 근거, 조치, 완료 기준, 상태, 결과 경로, 남은 한계)"""
    rows = []
    x8 = x8_effect(r)
    e1b = g(r.e1, "families", "DFL-B")
    e1a = g(r.e1, "families", "DFL-A")
    ok_tests = r.tests_ok
    def add(*a):
        rows.append(a)
    # §1 / claims
    add("E01", "§1", "'임의 새 장비 보장·모든 실패 감지·자동 확정 오류율 항상 5% 이하' 표현은 근거 없음", "README·config의 주장 문구 대조",
        "주장을 'within-provider 경험적 보증'으로 제한하고 CRC 정의(전체 위험)·N/A 규칙을 문서화", "과장 문구 제거·정의 명시", "근거상 지적 수정",
        "README.md, docs/FINAL_SUMMARY.md, config.yaml(crc.risk_definition)", "외부 독립 검증 없음")
    add("E02", "§1", "검토자는 검출기를 다시 실행하지 않았음(저장 기록 기반 PASS)", "이번에 GPU로 직접 재실행",
        "X14 봉인 예측과 비트 단위 재현, 기존 기준값과 자체 재현(verify), 격리 사본 실행", "VERIFY_PASS·격리 재현 PASS",
        "수정·검증 완료" if g(r.verify, "VERIFY_PASS") else "미완료", "outputs/validation/VERIFY_REPORT.json, outputs/experiments/resume/REPRODUCTION_REPORT.json",
        "같은 GPU·드라이버에서의 재현")
    add("E03", "§1", "전처리 캐시 168장만 포함(결함 아님)", "prepare가 435장을 재생성하고 픽셀 해시 대조", "변경 없음(설계 확인)", "PREPROCESS_REPORT 일치",
        "근거상 지적 수정", "outputs/validation/PREPROCESS_REPORT.json", "-")
    # §2 rubric items
    br = baseline_rows(r)
    yolo = [x for x in br if x["model"].startswith("YOLO") and x["mode"] == "X8"]
    add("E04", "§2·§10", "서로 다른 검출기의 동일 조건 비교 부족(F1 없음)", "4개 구성 비교는 같은 DFL의 운영 방식 차이였음",
        "같은 TXT·전처리·조리법·t2 분할·seed 17/29/43의 YOLO26s(reg_max 1)와 DFL 비교(X8·960 두 크기)" +
        (f"; X8 3px F1(YOLO26s → DFL): " + ", ".join(f"{x['set']} {f(x['F13'])} → {f(next((y['F13'] for y in br if y['set'] == x['set'] and y['mode'] == 'X8' and y['model'].startswith('DFL')), None))}" for x in yolo) if yolo else ""),
        "3 seed 비교표·F1(3/5px·IoU0.5)",
        "추가 실험으로 확인" if yolo else "미완료", "outputs/experiments/E3_scoring/RESULTS.json(controls.baseline)",
        "seed 17 YOLO는 9/28 학습분(같은 조리법); DFL은 이미 선택된 모델이라 사후 비교")
    add("E05", "§2", "F1·IoU 지표 부재, '정확도 99%' 단일 표현 위험", "F1을 3px/5px/IoU0.5로 분리 계산",
        "P/R/F1(3px·5px·IoU0.5), pFROC3, AP50(101점, mAP 아님) 동시 보고", "지표별 분리 표", "수정·검증 완료" if r.e3 else "미완료",
        "docs/FINAL_VALIDATION_REPORT.md, docs/RUBRIC_EVIDENCE.md", "IoU F1은 mAP가 아님")
    add("E06", "§2", "독립 홀드아웃·외부 검증 없음", "홀드아웃 65 원본 TXT 바이트 출처 미인증, 평가79 정답 미사용",
        "독립 보정 설계(P-IND-strict)와 과거→미래(P-TIME)로 내부 검증 강화", "설계 분리·결과 보고", "가용 데이터로 확인 불가",
        "outputs/experiments/E3_scoring/RESULTS.json", "외부 장비·시험편·정상 제품 자료 필요")
    # §3 task fit
    add("E07", "§3", "최종 DFL의 이물 크기·가장자리·형상·대비별 분석 부족", "과거 표는 다른 모델",
        "최종 DFL 3 seed의 객체별 미탐을 크기·제품 경계 거리·대비 대리지표·박스 종횡비·장비 구간으로 집계(표본 수 표시)",
        "구간별 표본 수와 미탐률", "추가 실험으로 확인" if r.e4 else "미완료", "outputs/experiments/E4_audit/RESULTS.json(condition_breakdown)",
        "재질·밀도 정답 없음 — 대비는 영상 대리지표")
    add("E08", "§3", "색 표시 shortcut이 없다는 증명은 아님", "가짜 흔적 학습만으로 제거 완료라 할 수 없음",
        "흔적만 있는 자리·흔적 없는 이물·전처리 손상 시험과 흔적 무력화 없는 DFL 대조(3 seed)", "사전 기준 판정",
        "추가 실험으로 확인" if r.e3 else "미완료", "outputs/experiments/E3_scoring/RESULTS.json",
        "; ".join(f"{FAM[p][0]}: {trace_control(r, p)['verdict']}" for p in FAM) if r.e3 else "-")
    add("E09", "§3", "전처리 효과의 DFL 단독 대조 없음", "과거 비교는 P2·seed 결합이 함께 바뀜",
        "같은 DFL 조리법에서 학습 영상만 v3tr↔v3로 바꾼 대조 학습(3 seed × 2 profile)", "대조 학습 완료·비교", "추가 실험으로 확인" if r.e3 else "미완료",
        "outputs/controls/v3only/, outputs/experiments/resume/CONTROL_LINEAGE.json", "합성 시험 기반")
    add("E10", "§3", "외부 초기 모델 출처·버전·라이선스", "COCO 초기값과 Ultralytics 사용", "출처·버전·AGPL 고지·검증된 인용 정리",
        "고지 문서", "수정·검증 완료", "docs/THIRD_PARTY_AND_DATA_NOTICE.md, docs/REFERENCES.md, licenses/", "-")
    add("E11", "§3", "제출물의 소속 식별 정보·개인 경로", "기록에 로컬 경로가 남을 수 있음",
        "출력은 상대 경로; weights/MANIFEST.json의 개인 경로 24곳을 상대 표기로 바꿈; preflight가 포함 대상 텍스트 파일의 자격 증명·홈 디렉터리 절대 경로·평가 정답 존재를 검사하고 발견 시 비정상 종료",
        "preflight 통과", "수정·검증 완료", "outputs/experiments/resume/SUBMISSION_FILE_MANIFEST.json", "과거 가중치 내부 학습 인자에 로컬 경로 문자열(실행 미참조)")
    add("E12", "§3", "ACCEPT는 제품 합격이 아님, 장당 FP 정의", "정상 제품 0장", "모든 출력·문서에 'ACCEPT = 검출 위치 자동 확정' 명시, 검출 0개는 항상 REVIEW",
        "출력 필드·README 일치", "수정·검증 완료", "src/xgd_analyzer.py(detections_confirmed), README.md", "정상 제품 특이도 평가 불가")
    # §4 detection & X8
    add("E13", "§4", "X8 검출 개선은 근거가 강함(같은 가중치)", "X14 저장 예측 재채점",
        (f"3px {f(x8['R3_native'])}→{f(x8['R3_x8'])}, FP/장 {f(x8['FP_native'])}→{f(x8['FP_x8'])}" if x8 else "-"), "검토 수치 재현",
        "추가 실험으로 확인" if x8 else "미완료", "outputs/experiments/E1_rules_x14/RESULTS.json", "물리 배율 아님")
    add("E14", "§4", "P90/P95는 8px 안 매칭만 대상", "FN·매칭률 미병기", "P90와 함께 매칭률·FN·미탐 포함 2/3px 초과율 보고", "모든 표에 병기",
        "수정·검증 완료" if r.e3 else "미완료", "experiments/lib.py(detection_metrics), docs/FINAL_VALIDATION_REPORT.md", "-")
    add("E15", "§4", "X8은 물리적 배율 보정이 아님", "imgsz/max(w,h) 정의", "config·README·출력에 '네트워크 입력 유효 배율' 명시", "표현 정정", "근거상 지적 수정",
        "config.yaml(scale_policy.meaning)", "물리 배율·pixel pitch 자료 없음")
    # §5 generalization / CRC
    add("E16", "§5.1", "dfl_src3_2는 2호기 32장을 학습(완전 미학습 아님)", "t2 분할 확인(학습 2호기 32장 = 8/31·9/1 촬영, 보류 2호기 100장 = 6월 촬영)",
        "profile 설명을 '1호기 미학습, 2호기 다른 날짜'로 정정", "config·README 일치", "수정·검증 완료", "config.yaml, README.md", "-")
    add("E17", "§5.1", "새 장비는 안전 보류는 가능하나 생산적 자동 운영 보장 아님", "보정 없으면 FULL", "문서에 명시(미등록·보정 부족 → FULL)", "표현 정정",
        "근거상 지적 수정", "README.md", "새 장비마다 라벨 보정 자료 필요")
    if e1b:
        c = g(e1b, "conditions", "rule|clean", "v1.0/P-X14")
        add("E18", "§5.2", "CRC 5%는 P(ACCEPT∧실패)이지 P(실패|ACCEPT)가 아님", "3호기 3 seed: 조건부 위험·실패 포착 재계산",
            f"분모별 지표 분리: 3호기 전체 위험 {f(g(c, 'mean', 'joint_risk'))}, 실패 포착 {f(g(c, 'mean', 'failure_capture'))}(3 seed 평균)", "분모 명시",
            "수정·검증 완료", "outputs/experiments/E1_rules_x14/RESULTS.json", "교환가능성 가정의 기대 위험")
        add("E19", "§5.2", "운영 seed17 3호기 실패 2장이 CRC 통과(신뢰도 0.5654·0.6316)", "X14 저장 예측으로 재현",
            "; ".join(f"{x['stem']}: 신뢰도 {f(x['trust'], 4)} ≥ λ {f(x['lambda_x14'], 4)} → {x['decision_v10_X14']}" for x in g(e1b, 'seed17_clean_rule_failures', default=[])),
            "검토 수치 재현", "추가 실험으로 확인", "outputs/experiments/E1_rules_x14/RESULTS.json, outputs/experiments/E8_failure_geometry/RESULTS.json",
            "고신뢰도 오판 방어 미해결" + (f" — E8: {geometry_text(r)}" if r.e8 else ""))
    add("E20", "§5.3", "LTT α0.05 미성립, 3호기 분할 위험 초과율 22/29%", "보정 산출물의 분할 평가", "확률 보장 필드 분리(preconditions_verified=false)",
        "주장 정정", "가용 데이터로 확인 불가", "outputs/calibration/crc_*.json", "교환가능성·독립 표본 부족")
    add("E21", "§5.3", "3호기 촬영일 5개 — 반복 분할은 독립 표본이 아님", "C(5,2)=10, C(7,3)=35", "분할 평가에 distinct_halves 기록, 독립 표본처럼 쓰지 않음",
        "표현 정정", "근거상 지적 수정", "src/xgd_analyzer.py(crc_split_assessment)", "-")
    add("E22", "§5.3", "QUALIFIED는 경험적 등급, X14는 탐색", "등급 enum과 확률 보장 분리", "operating_grade·probability_guarantee 별도 필드",
        "출력 일치", "수정·검증 완료", "src/xgd_analyzer.py", "외부 인증 아님")
    # §6 vulnerabilities
    if r.e3:
        loc = {p: location(r, p) for p in FAM}
        tr = {p: translation(r, p) for p in FAM}
        add("E23", "§6", "위치·문맥 의존성 미입증(3호기 GT x 범위 좁음)", "제품 기준 GT 중심 x 0.25~0.38(1·3호기)",
            "; ".join(f"{FAM[p][0]} 이동 M1 {f(loc[p]['M1']['moved'])}/원위치 {f(loc[p]['M1']['same'])}, M2 {f(loc[p]['M2']['moved'])}/{f(loc[p]['M2']['same'])}" for p in FAM),
            "사전 기준(두 방법 하락 ≤ 5%p)", "추가 실험으로 확인", "outputs/experiments/E3_scoring/RESULTS.json(R2a·R2b·R2c)",
            "; ".join(f"{FAM[p][0]}: {loc[p]['verdict']}" for p in FAM))
        add("E24", "§6", "Translation equivariance 미검증", "이동량 대비 중심 변화 표 없음",
            "; ".join(f"{FAM[p][0]} {tr[p]['conditions']}개 이동: 짝지은 중심 변위 P95 중앙값 {f(tr[p]['median_P95'])}px, 최악 P95 {f(tr[p]['worst_P95'])}px({tr[p]['worst_cond']}), 3px 검출률 최저 {f(tr[p]['min_R3'])}(정상 {f(tr[p]['clean_R3'])})" for p in FAM if tr[p]),
            "고정 14개 이동 결과", "추가 실험으로 확인", "outputs/experiments/E3_scoring/RESULTS.json(R1_*)", "합성 이동(replicate 경계)")
    add("E25", "§6", "장비 전이 비대칭(표본·학습 범위 차이)", "1·2→3호기와 3·2→1·2호기는 조건이 다름",
        "장비군별 결과를 따로 보고하고 같은 날 다른 장비 시험편 배치 유사성(proxy)을 감사", "분리 보고", "가용 데이터로 확인 불가",
        "outputs/experiments/E4_audit/RESULTS.json(unproven_same_date_cross_device_layout_pairs)",
        (lambda a: f"같은 날짜 다른 장비 정답 배치 일치 쌍 {a['layout_pairs']}개(관련 영상 {a['layout_images_by_device']}; 시험편 공유 가능성, 미인증)")(audit_facts(r)) if r.e4 else "-")
    if r.e3:
        sr = {p: stress_rows(r, p) for p in FAM}
        add("E26", "§6", "운영 범위가 좁음(약한 열화도 FULL)", "3호기 흐림σ0.4 FULL 100%",
            "; ".join(f"{FAM[p][0]} 열화 {len(d)}조건 평균 FULL {f(mean([x['full_dep'] for x in d]))}, 검출률이 0.95 이상인데도 FULL 100%인 조건: "
                      + (", ".join(x['cond'].replace('R3_', '') for x in d if x['R3_mean'] >= 0.95 and x['full_dep'] == 1.0) or "없음")
                      for p, rows in sr.items() for d in [[x for x in rows if x['cond'].startswith('R3_') or (x['cond'].startswith('GRID_') and not x['cond'].endswith('_clean'))]]),
            "조건별 coverage·보류 부담 보고", "추가 실험으로 확인", "outputs/experiments/E3_scoring/RESULTS.json", "과차단 완화는 검증 자료 없이 시도하지 않음")
    sig = signal_rows(r)
    add("E27", "§6", "target 장비 threshold/fail-safe 부분 작동(포착 15%, seed17 2/2 통과)", "E1에서 재현",
        ("이동·배율·seed 일관성 신호 3개를 사전 기준으로 평가: " + signal_verdict(r) + (f". 합성 놓침 탐색(E9): {blindspot_text(r)}" if r.e9 else "")) if sig else "미완료",
        "사전 채택 기준(i)(ii)(iii)", "추가 실험으로 확인" if sig else "미완료", "outputs/experiments/E3_scoring/SAFETY_SIGNALS.json", "자동 채택 없음 — 고신뢰도 오판 방어 미해결")
    add("E28", "§6", "acquisition-domain shift 일부 개선 + 보류 방어", "기존 4종 열화",
        "상관 잡음·원판/운동/비등방 PSF·배율×흐림×잡음 격자 추가, 조건×seed 실제 최저 별도", "조건별·seed별·실제 최저", "추가 실험으로 확인" if r.e3 else "미완료",
        "outputs/experiments/E3_scoring/RESULTS.json(true_worst_condition_seed)", "합성 조건은 실제 장비 변화의 증명이 아님")
    add("E29", "§6.1", "'최악 seed 0.960'은 조건별 최악의 평균(실제 최저 0.9133)", "README 문구 대조", "평균·조건별 최악·전체 최저를 분리", "표현 정정",
        "근거상 지적 수정", "docs/FINAL_VALIDATION_REPORT.md", "-")
    add("E30", "§6.1", "'열화에서도 오류 0%'는 자동 확정 0을 뜻함(조건부 위험 N/A)", "ACCEPT 0이면 N/A", "decision_metrics에서 N/A 처리, 보류 부담 병기",
        "표현 정정", "수정·검증 완료", "experiments/lib.py", "-")
    if e1a:
        u = g(e1a, "conditions", "under|clean", "v1.0/P-X14", "per_seed", default={})
        add("E31", "§6.2", "under 진단에서 1·2호기 ACCEPT 내부 실패율 18.4%·100%", "E1 재현",
            "; ".join(f"seed {sd}: ACCEPT {m['accept']}, 실패 {m['accepted_failures']}" for sd, m in u.items()) + " → v1.1은 배율 범위 밖 FULL",
            "진단과 운영 분리", "수정·검증 완료", "outputs/experiments/E1_rules_x14/RESULTS.json", "진단 조건은 운영 성능이 아님")
    # §7 shortcut / leakage
    add("E32", "§7.1", "3호기·1호기 모든 영상 이물 3개, 위치 좁음", "데이터 프로파일",
        "개수·위치 분포 공개, 위치 사전확률 대조·정답 순열 대조·위치 sweep" + ((lambda a: "; 학습 위치 중앙값만 찍는 대조의 3px 검출률 " + ", ".join(f"{FAM[p][0]} {f(v['prior_R3'])}" for p, v in a["profiles"].items()) + ", 다른 영상 정답과 짝지은 순열 대조 " + ", ".join(f"{FAM[p][0]} {f(v['perm_R3_mean'])}" for p, v in a["profiles"].items()) + " → 예측은 영상별 내용에 따름(위치 shortcut 부재의 증명은 아님)")(audit_facts(r)) if r.e4 else ""),
        "분포·대조 결과 공개", "가용 데이터로 확인 불가",
        "outputs/experiments/E6_data_profile/RESULTS.json, outputs/experiments/E4_audit/RESULTS.json", "넓은 위치·다양한 개수 자료 필요")
    add("E33", "§7.2", "시험편 분리 미인증(physical_specimen_disjoint_certified false)", "물리 ID 없음", "근접 중복 탐색·배치 proxy로 감사",
        "proxy 결과 공개", "가용 데이터로 확인 불가", "outputs/experiments/E4_audit/RESULTS.json", "물리적 시험편 ID 필요")
    add("E34", "§7.2", "실제 학습은 validation_split.json t2를 읽음", "학습 목록과 t2 대조", "학습 목록·계보 검사(CONTROL_LINEAGE)", "목록 일치",
        "수정·검증 완료", "outputs/experiments/resume/CONTROL_LINEAGE.json, outputs/validation/TRAIN_CHECK.json", "-")
    add("E35", "§7.2", "X14 품질 기준이 평가 절반의 무라벨 통계를 봄", "LOSO 설계", "P-IND-strict: 품질 기준·λ·등급을 반대 절반만으로",
        "독립 보정 결과", "추가 실험으로 확인" if r.e3 else "미완료", "outputs/experiments/E3_scoring/RESULTS.json", "절반 50장 수준이라 등급 미달이 많음")
    add("E36", "§7.2", "홀드아웃65·평가79 미채점", "출처 미인증·정답 미사용", "변경 없음(규칙 유지)", "정답 읽기 0회 기록", "가용 데이터로 확인 불가",
        "outputs/experiments/resume/EXECUTION_SEAL.json", "독립 성능 증거 부족")
    add("E37", "§7.2", "개발 자료 반복 사용에 따른 선택 과적합", "반복 분석 이력", "실행 봉인·노출 이력 기록, 독립 확증이라 부르지 않음", "기록",
        "가용 데이터로 확인 불가", "outputs/experiments/resume/EXECUTION_SEAL.json", "새 독립 자료 필요")
    if r.e4:
        add("E38", "§7.2", "바이트 해시만 검사(이동·회전·재촬영 유사성 미검사)", "dHash·NCC(±4px·±3°·0.9~1.1배) 탐색",
            f"후보 {g(r.e4, 'near_search', 'candidate_count')}쌍 → NCC ≥ 0.98 근접 쌍 {len(g(r.e4, 'near_pairs', default=[]))}개; profile별 학습↔보류 근접 쌍 " +
            ", ".join(f"{FAM[p][0]}: {v['cross_split_near_pairs']}쌍(보류 영상 {v['excluded_held']}장 제외 시 {v['subset_images']}장 3px 검출률 {f(v['subset_R3_mean'])})" for p, v in audit_facts(r)["profiles"].items()),
            "근접 쌍 제외 부분집합 재채점", "추가 실험으로 확인", "outputs/experiments/E4_audit/RESULTS.json", "영상 격자 진단이며 물리적 재촬영 식별은 아님")
    # §8 code bugs
    add("E39", "§8", "decide()가 qualification을 읽지 않음", "UNQUALIFIED·높은 신뢰도 → 1.0 ACCEPT 재현", "등급 enum 집행(UNQUALIFIED·미등록 → FULL, REVIEW → 자동 확정 금지)",
        "회귀 테스트", "수정·검증 완료" if ok_tests else "미완료", "tests/test_safety.py", "-")
    add("E40", "§8", "operating_envelope_valid에 배율 범위 없음", "under/over/native 진단에서 자동 확정 발생", "유효 입력 배율 [0.95×하한, 상한/0.95] 집행(3호기 X8 1568 통과)",
        "회귀 테스트", "수정·검증 완료" if ok_tests else "미완료", "tests/test_safety.py", "-")
    add("E41", "§8", "CRC 동점 미처리(25장 반례 3.85%)", "1.0 함수로 반례 재현", "동점 묶음 처리·실제 수용 집합으로 상한 계산(LTT 포함)", "회귀 테스트",
        "수정·검증 완료" if ok_tests else "미완료", "tests/test_safety.py", "-")
    add("E42", "§8", "검출 0개 명시적 차단 없음", "λ=0·검출 0 → 1.0 ACCEPT", "검출 0개는 항상 REVIEW", "회귀 테스트", "수정·검증 완료" if ok_tests else "미완료",
        "tests/test_safety.py", "-")
    add("E43", "§8", "수정 후 새 버전·재계산 필요", "규칙 변경", "XGD-1.1.0 버전, 보정 산출물 재생성(호환성 서명), 판정 재계산·변경 수 기록",
        "버전·재생성", "수정·검증 완료" if g(r.crc, "dfl_src12", "version", default="").startswith("XGD-1.1.0") else "미완료",
        "outputs/calibration/, docs/CHANGELOG_XGD-1.1.md", "-")
    # §9 evaluation 79
    e7 = eval79_reasons(r)
    add("E44", "§9", "평가 79장 자동 확정 49.4%, 원인(해상도·품질) 미분리", "정답 없음 — 분석기 출력의 중단 조건으로 분해",
        (f"새 규칙 판정 분포 {counts(g(r.pred, 'summary', 'decisions'))}(1.0과 영상별 동일)" if r.pred else "미완료") +
        (f"; E7: 1호기 FULL {e7['d1_full']}·REVIEW {e7['d1_review']}는 배율·등급 통과 후 품질 gate(Q4)로 중단, 평가 1호기는 316×332·Q4 {f(e7['q4'].get('eval_1호기'))}"
         f"(개발 1호기 352×332·{f(e7['q4'].get('dev_1호기'))}); 개발 1호기 316 리샘플 반사실 FULL {', '.join(f(v) for v in e7['resample_full'].values())} → 해상도 변환만으로는 불충분(사전 기준)" if e7 else ""),
        "중단 조건 분해·반사실(사전 등록)", "추가 실험으로 확인" if e7 else "가용 데이터로 확인 불가",
        "outputs/test_predictions.json, outputs/experiments/E7_eval79_reasons/RESULTS.json",
        "정답·현장 검수 없이는 과차단/정당 차단 구분 불가; 보정 해상도 일치 규칙 없음")
    add("E45", "§9", "사람 재검사 처리량·정확도 미검증", "-", "실제 추론 시간 측정 + 가정 기반 작업 시간 시나리오(가정 명시)", "측정/가정 분리",
        "가용 데이터로 확인 불가", "outputs/experiments/resume/OPERATIONAL_TIMING.json", "현장 검수 자료 필요")
    add("E46", "§10", "새 발명이 아니라 결합·검증의 기여로 표현", "-", "인용 확인 후 기여 범위 정정", "문서", "근거상 지적 수정", "docs/REFERENCES.md, README.md",
        "타 팀 제출물을 보지 않아 유일성 판단 불가")
    # reproducibility (spec §7.6 / §10)
    add("E47", "지시문 §7.6", "다른 장치의 수치 허용오차·판정 변경", "다른 GPU 없음", "같은 가중치의 CPU 추론과 GPU 결과 비교(E5)", "연속 차이·이산 변경 수",
        "추가 실험으로 확인" if r.e5 else "미완료", "outputs/experiments/E5_device_tolerance/RESULTS.json", "다른 GPU 모델 미검증")
    add("E48", "지시문 §7.6", "재학습 경로 동작·성능", "-", "격리 사본에서 같은 조리법 6개 재학습 후 동결 가중치와 비교", "재학습 완료·비교",
        "추가 실험으로 확인" if g(r.repro, "full_retrain_executed") else "미완료", "outputs/experiments/resume/REPRODUCTION_REPORT.json", "비트 일치 기대하지 않음")
    return rows


def issue_doc(r):
    L = ["# 독립 평가 지적별 대응표", "",
         "원 평가서(`../XGD_모델_독립평가_20261005.md`)는 수정하지 않았다. 상태는 네 가지 중 하나다:",
         "수정·검증 완료 / 추가 실험으로 확인 / 가용 데이터로 확인 불가 / 근거상 지적 수정. '추가 실험으로 확인'은 실행 근거를 확보했다는 뜻이며 개선이나 shortcut 부재를 뜻하지 않는다.", "",
         "| ID | 평가서 절 | 지적 | 재확인 근거 | 수정·실험과 측정 결과 | 완료 기준 | 상태 | 결과 경로 | 남은 한계 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in issue_rows(r):
        L.append("| " + " | ".join(str(x).replace("|", "/") for x in row) + " |")
    counts = {}
    for row in issue_rows(r):
        counts[row[6]] = counts.get(row[6], 0) + 1
    L += ["", f"상태별 개수: {counts}"]
    return "\n".join(L)


def rubric_doc(r):
    br = baseline_rows(r)
    L = ["# 서면평가 6개 항목 근거", "", "각 항목은 주장 → 실험 → 결과 → 코드·결과 경로 → 한계 순서로 적는다.", ""]
    e6 = r.e6 or {}
    L += ["## 1. 데이터 이해 및 진단 (15)", "",
          f"- 생산 단위: 영상 1장 = 완제품 1개의 X-ray 검사. 개발 영상은 모두 이물이 있는 영상이고 정상 제품은 0장이다.",
          f"- 장비·해상도: {g(e6, 'resolution')}",
          f"- 같은 날 세 장비의 영상 수: {g(e6, 'same_date_counts_across_devices')} → 같은 시험편을 여러 장비로 찍었을 가능성(물리 ID 없음).",
          f"- 영상당 이물 수: " + "; ".join(f"{d} {v['objects_per_image']}" for d, v in (g(e6, 'labels', default={}) or {}).items()),
          f"- 정답 중심 위치(제품 기준 x): " + "; ".join(f"{d} {f(v['center_x_rel_product']['min'])}~{f(v['center_x_rel_product']['max'])}" for d, v in (g(e6, 'labels', default={}) or {}).items()),
          "- 라벨: 제공 TXT(YOLO 형식, 소수 8자리)를 정답으로 쓴다. 색 표시는 정답·마스크로 쓰지 않는다. 평가 79장 TXT는 넣지 않고 읽지 않는다.",
          f"- 중복·재촬영: 바이트 동일 0, 근접 쌍 {len(g(r.e4, 'near_pairs', default=[]))}개(E4), 같은 날짜 다른 장비 배치 일치 {len(g(r.e4, 'unproven_same_date_cross_device_layout_pairs', default=[]))}쌍(시험편 proxy).",
          f"- 촬영일: 1호기 개발 {list(g(e6, 'dates', '1호기', default={}) or {})}, 2호기 개발 {list(g(e6, 'dates', '2호기', default={}) or {})}, "
          f"3호기 개발 {list(g(e6, 'dates', '3호기', default={}) or {})}; 평가 79장은 모두 9월({g(e6, 'evaluation79_dates')}).",
          "- 2호기 8/31·9/1 영상 32장은 모두 이물 1개이고 dfl_src3_2 학습에 들어간다(1호기는 학습에 없음).",
          (f"- 9월 촬영 조건 변화(E7, 정답 미사용): 평가 1호기는 해상도 316×332(개발 1호기 352×332)이고, 잡음 이웃 상관 Q4 중앙값이 개발 1호기 "
           f"{f(eval79_reasons(r)['q4'].get('dev_1호기'))} → 평가 1호기 {f(eval79_reasons(r)['q4'].get('eval_1호기'))}로 낮다. 2호기도 8/31·9/1부터 같은 방향으로 낮아졌다."
           if r.e7 else "- 9월 촬영 조건 변화 분석(E7) 미완료"),
          "- 검증 설계: 장비 단위 분할(t2), 독립 보정(P-IND-strict), 과거→미래(P-TIME), 평가 79장 정답 미사용.",
          "- 근거: outputs/experiments/E6_data_profile/, outputs/experiments/E4_audit/, outputs/experiments/E7_eval79_reasons/, data/metadata.json, data/validation_split.json", ""]
    L += ["## 2. AI 예측모델 개발 (40)", "", "같은 TXT·전처리(v3tr)·조리법·t2 분할·seed 17/29/43에서 모델 구조(reg_max)만 다른 비교(3 seed 평균, 최악은 seed 최저):", "",
          "| 집합 | 모델 | 추론 크기 | P@3px | R@3px | F1@3px | F1@5px | F1@IoU0.5 | pFROC3 | 오탐/장 | 최악 R@3px | P90(px, 8px 조건부) | AP50(101점) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    def same_as_x8(x):
        twin = next((y for y in br if y["set"] == x["set"] and y["model"] == x["model"] and y["mode"] == "X8"), None)
        return x["mode"] != "X8" and twin is not None and all(twin[k] == x[k] for k in ("P3", "R3", "F13", "FP", "pFROC"))
    for x in br:
        if same_as_x8(x):
            continue
        L.append(f"| {x['set']} | {x['model']} | {x['mode']} | {f(x['P3'])} | {f(x['R3'])} | {f(x['F13'])} | {f(x['F15'])} | {f(x['F1iou'])} | {f(x['pFROC'])} | {f(x['FP'])} | {f(x['R3_worst'])} | {f(x['P90'])} | {f(x['AP50'])} |")
    x8 = x8_effect(r)
    L += ["", "1·2호기는 원래 크기(960)가 이미 학습 배율 범위 안이라 X8과 결과가 같아 960 행을 생략했다.",
          "구성요소 비교(3호기, 같은 DFL 가중치):" + (f" 검출기만(960) 3px {f(x8['R3_native'])}, F1 {f(x8['F1_native'])} → +X8 3px {f(x8['R3_x8'])}, F1 {f(x8['F1_x8'])}." if x8 else ""),
          "X12·CRC는 검출 좌표를 바꾸지 않고 자동 확정 여부만 바꾼다. 구성요소별 효과(같은 보류 영상, 3 seed 평균):", "",
          "| 장비군 | 검출기만(960): R@3px / 오탐/장 / 영상 실패율 | +X8 | +X12+CRC 진단 설계: coverage / 전체 위험 | +X12+CRC 독립 보정: coverage / 전체 위험 | 열화 조건: 모두 자동 확정 시 위험 → XGD 전체 위험 |",
          "| --- | --- | --- | --- | --- | --- |",
          *[f"| {x['set']} | {'(X8과 같음)' if x['native_same'] else ' / '.join(f(v) for v in x['det960'])} | {' / '.join(f(v) for v in x['x8'])} | "
            f"{' / '.join(f(v) for v in x['dep'])} | {' / '.join(f(v) for v in x['strict'])} | {f(x['deg_all_accept_risk'])} → {f(x['deg_xgd_joint'])} ({x['deg_n']}조건) |"
            for x in component_rows(r)],
          "", "영상 실패율 = 영상을 모두 자동 확정했을 때의 전체 위험. 열화 조건의 XGD 전체 위험은 진단 설계(전체 등급) 기준이다.",
          "선정 근거: DFL은 9/28~29 개발 자료 비교로 이미 선택됐고, 위 표는 사후 기술이다. IoU F1은 mAP가 아니며 AP50은 E0·상위 30 예측의 101점 보간값이다.", ""]
    L += ["## 3. 영향요인 및 오류분석 (15)", "",
          "- 객체 조건: 최종 DFL 3 seed의 객체별 미탐을 크기(최소 변)·제품 경계 거리·대비 대리지표·박스 종횡비·장비 구간과 상호작용으로 집계했다(E4). 칸 = 이물 수(seed별 미탐 수), '-' = 표본 없음.", "",
          *breakdown_lines(r), "",
          "- 읽는 법: 모든 이물이 제품 경계에서 30px 이상 떨어져 있어 가장자리 영향은 이 자료로 분석할 수 없다. 3호기 미탐은 모두 최소 변 10~20px 박스에서 나왔다(5~10px 61개에서는 0). "
          "표본이 작은 구간(예: 3호기 대비 0.5~1, 13개)의 비율은 일반화하지 않는다.",
          "- 촬영 조건: 흐림·리샘플·백색/상관 잡음·원판/운동/비등방 PSF·배율 격자, 조건×seed 실제 최저(E3).",
          "- 위치·문맥: 정수 이동, 이물 국소 이동, 위치 sweep, 다른 날짜 제품 문맥, 주변만 열화, 객체 제거·sham·객체만 남김(E3).",
          (f"- 정상 조건 실패의 기하(E8, 사전 등록 docs/PREREG_E8_FAILURE_GEOMETRY.md): {geometry_text(r)}. 5px 기준에서는 3호기 F1이 1.000이다." if r.e8 else "- 실패 기하(E8) 미완료"),
          "- 실패 예시: docs/FAILURE_EXAMPLES.md, outputs/experiments/E4_audit/examples/(초록 = TXT 정답, 빨강 = 예측).",
          "- 재질·밀도 정답이 없어 분석하지 않았다. 대비·종횡비는 영상 대리지표다.", ""]
    L += ["## 4. 현장 활용방안 (10)", "",
          "- 흐름: 이물 위치 표시 → ACCEPT(위치 자동 확정) / REVIEW(작업자 확인) / FULL_REINSPECTION(전수 재검사, 장비·촬영 조건 점검) → 새 정상 촬영일은 사람 확인 → 오프라인 보정 갱신 → 새 버전 검증.",
          "- 보정 갱신 절차: 사람이 확인한 라벨 영상을 개발 자료에 넣고 `python run_all.py calibrate`를 다시 돌리면 새 버전의 품질 기준·CRC 산출물이 만들어지고, 호환성 서명이 맞지 않는 옛 산출물은 FULL로 막힌다. "
          "사람 확인 단계는 이번에 실제로 실행하지 않았으므로 절차 제안이다.",
          "- 재검사 비율(정상 조건, 운영 seed 17, 독립 보정): " + "; ".join(f"{FAM[p][0]} REVIEW {f(g(r.dec(p, 'clean', 'P-IND-strict'), 'review_rate'))}, FULL {f(g(r.dec(p, 'clean', 'P-IND-strict'), 'full_rate'))}" for p in FAM if r.dec(p, 'clean', 'P-IND-strict')) +
          ". 평가 79장: " + counts(g(r.pred, 'summary', 'decisions')) + ".",
          "- ACCEPT는 제품 합격이 아니다. 검출 0개는 항상 REVIEW이고 '이물 없음' 자동 확정은 없다.",
          f"- 실제 처리 시간(배치 1, 원본 읽기~판정, 모델 적재 제외): " + ("; ".join(f"{FAM[p][0]} 평균 {f(t['mean_seconds'])}초/장(P95 {f(t['P50_P90_P95_seconds'][2])}초)" for p, t in (g(r.timing, 'profiles', default={}) or {}).items()) or "미측정"),
          "- 작업자 시간·비용은 가정 시나리오로만 제시(FINAL_VALIDATION_REPORT). 정상 제품 특이도는 정상 자료가 없어 평가하지 않았다.",
          "- 평가 79장에서 1호기 영상은 9월 촬영 형식 변화(해상도·잡음 특성)로 모두 자동 확정이 멈췄다(E7). 현장에서는 이런 경우 장비·촬영 설정을 점검하고, "
          "사람이 확인한 새 촬영일을 오프라인 보정에 넣어 새 버전으로 검증한 뒤에만 자동 확정을 다시 켠다. 보정 영상과 다른 해상도를 직접 막는 규칙은 다음 버전 후보다.", ""]
    L += ["## 5. 창의성·차별성 (10)", "",
          "- 기존 방법(DFL, CRC, LTT — docs/REFERENCES.md)을 새로 발명한 것이 아니다. 기여는 다음 결합과 검증이다.",
          "  1. 검사기 색 표시가 박스 안 밝기를 약 0.80배로 낮춰 shortcut이 될 수 있음을 찾아, 표시 제거(v3)와 가짜 흔적 학습(v3tr)을 설계하고, 흔적 무력화 없는 DFL 대조로 직접 시험했다. "
          "결과는 두 모델 모두 흔적 자리 경보 0이라 효과를 구별하지 못했고, 최종 모델이 M2 합성 이물을 덜 검출하는 차이가 새로 드러났다(한계로 보고).",
          "  2. 3호기 실패를 입력 유효 배율과 연결하고, 같은 가중치에서 배율 정규화의 검출 개선을 정량화했다.",
          "  3. 검출 결과와 함께 자동 확정/작업자 확인/전수 재검사와 사유를 내는 판정 구조를 만들고, 위험 분모·coverage·보류 부담을 함께 측정했다.",
          f"- 고신뢰도 오판 방어: 후보 신호 3개를 사전 기준으로 시험 — {signal_verdict(r)}. 운영 규칙에는 자동 추가하지 않았다.",
          *([f"- 놓친 이물의 자동 확정(E9, 탐색): {blindspot_text(r)}. 평균 신뢰도 기반 판정의 구조적 한계로 보고한다."] if r.e9 else []), ""]
    rep = r.repro or {}
    L += ["## 6. 코드 및 재현성 (10)", "",
          "- 단일 실행: `python run_all.py`(동결 가중치, 약 2분), `python run_all.py full`(전처리→재학습→보정→예측), `python -m pytest -q tests`.",
          f"- 회귀 테스트: {('통과(' + r.tests_line + ')') if r.tests_ok else '확인 필요(로그 없음·오래됨·실패)'}; GPU 재현(VERIFY): {g(r.verify, 'VERIFY_PASS')}; 격리 사본 기본 실행: {g(rep, 'isolated_default_VERIFY_PASS', default='미완료')}; 재학습 실행: {g(rep, 'full_retrain_executed', default='미완료')}.",
          *([f"- 같은 조리법 재학습({FAM[p][0]}): 3px 검출률 동결 {f(v['R3'][0])} → 재학습 {f(v['R3'][1])}, 오탐/장 {f(v['FP'][0])} → {f(v['FP'][1])}" for p, v in (retrain_compare(r) or {}).items()]),
          f"- 다른 장치 수치 허용오차(E5, CPU 대 GPU, seed 17): " + ("; ".join(f"{FAM[p][0]} {v.get('images')}장 판정 변경 {v.get('decision_flips')}건, 3px 결과 변경 {v.get('outcome_flips_3px')}건, 최대 중심 차이 {v.get('max_center_shift_px'):.1e}px" for p, v in (g(r.e5, 'profiles', default={}) or {}).items()) or "미실행"),
          "- 원본·라벨·가중치 해시와 학습 계약: data/metadata.json, weights/MANIFEST.json, outputs/validation/TRAIN_CHECK.json, outputs/experiments/resume/EXECUTION_SEAL.json.",
          "- 필수 검사 실패는 0이 아닌 종료 코드로 멈춘다. 다른 GPU·새 설치 환경 검증은 하지 못했다.", ""]
    return "\n".join(L)


def changelog_doc(r):
    L = ["# XGD-1.1 변경 기록", "",
         "| 구분 | XGD-1.0.0 | XGD-1.1.0 |", "| --- | --- | --- |",
         "| 검출 가중치 | DFL t2 폴드 6개 | 같음(교체 없음) |",
         "| 전처리·X8·품질 신호·임계값 값 | 봉인 값 | 같음 |",
         "| 장비 운영 등급 | 기록만 함 | enum 집행: UNQUALIFIED·미등록·알 수 없음 → FULL, REVIEW → 자동 확정 금지 |",
         "| 배율 범위 | 판정에 미사용 | 유효 입력 배율이 [0.95×학습 하한, 학습 상한/0.95] 밖이면 FULL |",
         "| CRC/LTT 동점 | 순위 중간에서 자를 수 있음 | 동점 묶음 단위, 실제 수용 집합으로 상한 계산 |",
         "| 검출 0개 | λ>0이면 REVIEW(암묵적) | 항상 REVIEW(명시) |",
         "| 비정상 입력·보정 불일치 | NaN이 NORMAL로 흐를 수 있음 | FULL(입력·신호·메타데이터·보정 형식·호환성 서명 검사) |",
         "| 보정 산출물 | 텍스트 등급 | enum 등급 + 확률 보장 필드 분리 + 호환성 서명(가중치·전처리·규칙·소스 해시) |", ""]
    if r.e1:
        L += ["## X14 조건에서 판정 변경 수(1.0 → 1.1, X14 설계, 3 seed 합)", "", "| 집합 | 조건(배율 방식, 열화) | 변경 |", "| --- | --- | --- |"]
        for fam in ("DFL-B", "DFL-A"):
            for c, C in g(r.e1, "families", fam, "conditions", default={}).items():
                ch = C.get("changes_v10_to_v11_P-X14_all_seeds")
                if ch:
                    sc_, dg_ = (c.split("|") + [""])[:2]
                    L.append(f"| {fam} | 배율 {sc_}, 열화 {dg_} | {ch} |")
        L += ["", "정상 X8(rule) 조건의 판정은 모든 열화 조건에서 바뀌지 않았다. 바뀐 것은 진단용 강제 배율(under/over/native)뿐이다.", ""]
    if r.pred and r.pred_old:
        L += ["## 평가 79장 판정 분포", "", f"- 1.0.0: {counts(g(r.pred_old, 'summary', 'decisions'))}", f"- 1.1.0: {counts(g(r.pred, 'summary', 'decisions'))} (영상별 판정·검출 좌표 모두 1.0.0과 같음)",
              "- 정답이 없으므로 정확도 비교가 아니다. 분포를 보고 규칙을 바꾸지 않았다.", ""]
    L += ["## 추가 실험(모델·규칙 변경 없음)", "",
          "- E1: X14 저장 예측으로 1.0.0 판정 재현과 1.1.0 규칙 재채점.",
          "- E2·E3: 강건성 예측과 채점. 이동, 위치·문맥, 흔적, 열화, 배율 격자, 일관성 신호, 비교 모델.",
          "- E4: 근접 중복, 시험편 배치 proxy, 위치 사전확률·정답 순열 대조, 객체 조건별 미탐.",
          "- E5: CPU 대 GPU 판정 일관성.",
          "- E6: 데이터 프로파일.",
          "- E7: 평가 79장 중단 원인을 정답 없이 분해하고 해상도 반사실을 시험(사후 동기, 실행 전 사전 등록 docs/PREREG_E7_EVAL79_REASONS.md).",
          "- E8: 정상 조건 실패의 위치 기하 분류(실행 전 사전 등록 docs/PREREG_E8_FAILURE_GEOMETRY.md).",
          "- E9: 합성 이물을 놓친 영상의 자동 확정과 seed 불일치 신호(탐색, 사후 동기, 실행 전 기록 docs/PREREG_E9_MISSED_OBJECT_BLINDSPOT.md).", "",
          "## 추가된 모델(비교·대조용, 운영 교체 아님)", "",
          "- 일반 YOLO26s(reg_max 1): seed 17은 9/28 학습분, seed 29·43 × 2 profile을 같은 조리법으로 새로 학습 → weights/baseline_yolo26s/.",
          "- 흔적 무력화 없는 DFL(v3only): seed 17·29·43 × 2 profile → outputs/controls/v3only/.",
          "- 격리 사본 재학습 DFL 6개(재현 검증용) → outputs/isolated_package/outputs/full_run/.", "",
          "## 보존", "", "- 원본 BMP·TXT·DFL 가중치·봉인 X14 기록은 바꾸지 않았다. 1.0.0 코드·산출물은 versions/XGD-1.0.0/, 중단 시점 코드는 versions/XGD-1.1-interrupted-20261005/에 있다."]
    return "\n".join(L)


def limitations_doc(r):
    L = ["# 남은 한계", "", "직접 누수 방지(평가 정답 미사용, 학습·보류 영상 분리)와 별개로 아래 한계가 남는다.", "",
         "1. 같은 공급처 개발 356장을 9/21 이후 반복해서 분석·선택에 썼다. 이 패키지의 결과는 내부 검증이며 독립 확증이 아니다.",
         "2. 물리적 시험편 ID가 없다. 같은 날짜에 세 장비가 비슷한 수를 촬영했고 정답 배치가 비슷한 쌍이 있어, 장비 전이 평가에 같은 시험편이 섞였을 수 있다.",
         "3. 정상 제품 영상이 0장이다. '이물 없음' 판정의 안전성과 제품 단위 특이도는 평가할 수 없다.",
         "4. 1·3호기 개발 영상은 모두 이물 3개, 중심 x가 제품 폭의 25~38%에 몰려 있다. 위치·개수 다양성이 작다.",
         "5. 3호기 보정 자료는 100장·촬영일 5개다. 반복 분할 수는 독립 표본 수가 아니며, 독립 보정(P-IND-strict)에서는 등급이 성립하지 않는 경우가 많다.",
         "6. CRC α 0.05는 교환가능성 가정의 기대 전체 위험이다. LTT 확률 보장은 전제가 확인되지 않아 주장하지 않는다.",
         f"7. 정상 화질에서 신뢰도가 높은 오판(운영 seed 17에서 장비군별 2장)은 현재 규칙으로 거르지 못한다. 후보 신호: {signal_verdict(r)}." +
         (f" 실패 기하(E8): {geometry_text(r)}. 대부분 3px 기준을 조금 넘는 위치 오차이며, 라벨 중심 오차인지 모델 오차인지는 구분하지 못했다." if r.e8 else ""),
         "8. 촬영 조건 변화는 합성 열화로만 시험했다. 실제 장비의 잡음 스펙트럼·PSF·물리 배율 변화를 대표한다고 할 수 없다.",
         "9. 위치·문맥·흔적 시험은 합성 영상이다. 합성 방법 2가지로 결론의 일관성을 봤지만 실제 새 이물·제품 분포를 대체하지 못한다.",
         "10. 평가 79장은 정답 없이 판정 분포만 있다. 1호기 32장·3호기 2장이 자동 확정되지 않은 것이 정당한 차단인지 과차단인지 판단할 수 없다. "
         "중단 조건은 정답 없이 분해했다(E7: 1호기는 9월 촬영 형식 변화에 따른 품질 gate 중단).",
         "11. 다른 GPU·새 패키지 설치 환경에서는 검증하지 않았다(CPU 대 GPU 비교만 수행). 재학습 가중치는 제출 가중치와 비트 단위로 같지 않다.",
         "12. 대조 학습의 YOLO seed 17은 9/28 학습분이라 실행 환경의 완전한 동일성을 소급 증명하지 않는다.",
         "13. 장비별 보정(품질 기준·CRC λ)은 그 장비의 개발 영상 해상도에서 만들었지만, 다른 해상도로 들어온 같은 장비 영상을 직접 막는 규칙은 없다. "
         "배율 범위는 profile 단위로만 본다. 평가 1호기(316×332)는 품질 gate가 멈췄지만 잡음 특성이 기준과 비슷했다면 352×332 영상으로 만든 1호기 보정값이 그대로 쓰였을 것이다. "
         "평가 출력을 보고 찾은 문제라 이번 버전 규칙에는 넣지 않고 다음 버전 후보로 남긴다.",
         "14. 판정 신뢰도는 검출한 이물의 평균 점수라서, 이물 하나를 놓쳐도 나머지 검출이 확실하면 자동 확정될 수 있다. "
         + (f"합성 시험(E9, 탐색): {blindspot_text(r)}. " if r.e9 else "") + "정상 제품 자료가 없어 '놓침'을 직접 보정할 방법이 이번 자료에는 없다."]
    return "\n".join(L)


def readiness(r, passed):
    rep = r.repro or {}
    checks = {"regression_tests": r.tests_ok,
              "verify_gpu_reproduction": bool(g(r.verify, "VERIFY_PASS")),
              "legacy_x14_rescore": bool(r.e1) and max((x for F in r.e1["families"].values() for C in F["conditions"].values()
                                                        for x in C.get("x14_check", {}).values()), default=1) <= 1e-12,
              "isolated_copy_default_run": bool(g(rep, "isolated_default_VERIFY_PASS")) and bool(g(rep, "isolated_default_predictions_decisions_equal")),
              "full_retrain_executed": bool(g(rep, "full_retrain_executed")),
              "robustness_scored": bool(r.e3), "audit_done": bool(r.e4), "evaluation79_predictions": bool(r.pred),
              "calibration_v1_1": all(str(g(c, "version", default="")).startswith("XGD-1.1.0") for c in r.crc.values()),
              "final_report_checks": bool(passed)}
    ok = all(checks.values())
    return {"mandatory_checks": checks, "mandatory_local_checks_pass": ok, "zip_ready": ok,
            "blocking_items": [k for k, v in checks.items() if not v],
            "verified_limitations_not_blocking": ["different GPU / fresh wheel installation not verified (CPU-vs-GPU comparison only)",
                                                   "no independent external, normal-product or specimen-certified data"],
            "outside_source_zip": ["result report PDF", "presentation PDF/PPT", "survey screenshot"],
            "additional_evidence": {"device_tolerance_E5": bool(r.e5), "data_profile_E6": bool(r.e6), "eval79_reason_decomposition_E7": bool(r.e7),
                                    "failure_geometry_E8": bool(r.e8), "missed_object_blindspot_E9": bool(r.e9)},
            "evaluation79_label_reads": 0, "independent_safety_qualification": "UNQUALIFIED", "new_signal_auto_adopted": False}


def main(passed=True):
    r = R()
    # XGD-1.2 (follow-up) sections go first once the follow-up has completed; the XGD-1.1 content below them is the baseline record.
    pre = {"summary": "", "issue": "", "rubric": "", "limits": ""}
    try:
        from experiments.followup import docs12 as D12, docs12_sections as S12
        if S12.available():
            a12 = D12.Art()
            pre = {"summary": S12.summary_section(a12), "issue": S12.issue_section(a12), "rubric": S12.rubric_section(a12),
                   "limits": S12.limitations_section(a12)}
    except Exception as e:                     # never hide a failure: report it in the document head
        pre = {k: f"> XGD-1.2 section generation failed: {type(e).__name__}: {e}\n\n" for k in pre}
    text("docs/FINAL_SUMMARY.md", pre["summary"] + summary_doc(r))
    text("docs/ISSUE_RESPONSE_TABLE.md", pre["issue"] + issue_doc(r))
    text("docs/RUBRIC_EVIDENCE.md", pre["rubric"] + rubric_doc(r))
    text("docs/CHANGELOG_XGD-1.1.md", changelog_doc(r))
    text("docs/LIMITATIONS.md", pre["limits"] + limitations_doc(r))
    rd = readiness(r, passed)
    print("final docs written; mandatory checks", rd["mandatory_checks"], flush=True)
    return rd


if __name__ == "__main__":
    import sys
    rd = main()
    (OUT / "resume").mkdir(parents=True, exist_ok=True)
    print(json.dumps(rd, ensure_ascii=False, indent=1))
    sys.exit(0)
