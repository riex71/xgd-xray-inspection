"""Candidate / trade-off tables for the XGD-1.2 documents (read-only over sealed comparison files; no selection, no new values).

Every number is a mean over seeds 17/29/43 of the group-date calibration design unless the column says 'diagnostic' (grade fixed to
QUALIFIED, not an operating result). Seeds are repetitions of the same images, not independent samples.
"""
import json
from functools import lru_cache

import numpy as np

from experiments import lib as L
from experiments.followup import policy as P

DESIGN = "group_date_calibration"
DIAG = "QUALIFIED_FIXED_DIAGNOSTIC"
PROFILES = ("dfl_src12", "dfl_src3_2")
FAM = {"dfl_src12": "3호기", "dfl_src3_2": "1·2호기"}
SCREEN = ("dev_clean", "R2f_free_M1", "R2f_free_M2", "R2f_free_M2v2", "R2a_move_M1_0", "R2a_move_M2_0", "R2a_move_M2v2_0", "R2a_same_M1_0",
          "R2a_same_M2_0", "R2a_same_M2v2_0", "R2c_context_M1", "R2c_context_M2", "R2c_context_M2v2", "R2e_trace")
SYNTH_M = ("R2f_free_M1", "R2f_free_M2", "R2f_free_M2v2", "R2a_move_M1_0", "R2a_move_M2_0", "R2a_move_M2v2_0", "R2c_context_M1", "R2c_context_M2",
           "R2c_context_M2v2")


@lru_cache(maxsize=256)
def comparison(stage, family, cid, prof):
    p = L.ROOT / f"outputs/followup/comparisons/{stage}_{family}_{cid}_{prof}.json"
    return json.loads(p.read_text(encoding="utf8")) if p.exists() else None


def seeds(r):
    return list(r["per_seed"].values())


def mean(r, k):
    v = [x[k] for x in seeds(r) if isinstance(x.get(k), (int, float))]
    return float(np.mean(v)) if v else None


def worst(r, k):
    v = [x[k] for x in seeds(r) if isinstance(x.get(k), (int, float))]
    return float(min(v)) if v else None


def dec(r, k, label=DESIGN):
    v = [x["decisions"][label][k] for x in seeds(r) if isinstance(x["decisions"][label].get(k), (int, float))]
    return float(np.mean(v)) if v else None


def row(stage, family, cid, prof):
    C = comparison(stage, family, cid, prof)
    if C is None:
        return None
    clean = C["dev_clean"]
    conds = [c for c in SCREEN if c in C]
    synth = [c for c in SYNTH_M if c in C]
    stress = [c for c in C if c.startswith("STRESS_")]
    miss_rate = [mean(C[c], "actual_missing8") / mean(C[c], "objects") for c in synth if mean(C[c], "objects")]
    return {"stage": stage, "family": family, "id": cid, "profile": prof,
            "R3": mean(clean, "R3"), "F13": mean(clean, "F13"), "FP": mean(clean, "FP_per_image3"), "R3_worst": worst(clean, "R3"),
            "R5": mean(clean, "R5"), "R4": mean(clean, "R4"), "success2px": mean(clean, "success2px"), "pFROC": mean(clean, "pFROC3"),
            "P90": float(np.mean([x["err_P50_P90_P95"][1] for x in seeds(clean) if x.get("err_P50_P90_P95")])) if seeds(clean) else None,
            "matched8": mean(clean, "matched8_rate"), "miss_or_err_gt3": mean(clean, "miss_or_err_gt3px"),
            "clean_A": dec(clean, "coverage"), "clean_risk": dec(clean, "joint_risk"), "clean_cond_risk": dec(clean, "conditional_risk"),
            "clean_review": dec(clean, "review_rate"), "clean_full": dec(clean, "full_rate"), "clean_hold_ok": dec(clean, "correct_hold_rate"),
            "clean_A_diag": dec(clean, "coverage", DIAG),
            "risk_mean": float(np.mean([dec(C[c], "joint_risk") for c in conds])),
            "FN8_risk_mean": float(np.mean([dec(C[c], "FN8_joint_risk") for c in conds])),
            "synth_FN8_risk_diag": float(np.mean([dec(C[c], "FN8_joint_risk", DIAG) for c in synth])) if synth else None,
            "synth_A_diag": float(np.mean([dec(C[c], "coverage", DIAG) for c in synth])) if synth else None,
            "synth_capture_diag": float(np.mean([v for v in (dec(C[c], "failure_capture", DIAG) for c in synth) if v is not None])) if synth else None,
            "synth_actual_missing_rate": float(np.mean(miss_rate)) if miss_rate else None,
            "stress_risk_max": max((dec(C[c], "joint_risk") for c in stress), default=None),
            "stress_full_mean": float(np.mean([dec(C[c], "full_rate") for c in stress])) if stress else None,
            "stress_review_mean": float(np.mean([dec(C[c], "review_rate") for c in stress])) if stress else None,
            "cost10": float(np.mean([dec(C[c], "assumed_cost_10") for c in conds])),
            "cost50": float(np.mean([dec(C[c], "assumed_cost_50") for c in conds])),
            "cost100": float(np.mean([dec(C[c], "assumed_cost_100") for c in conds])),
            "calls": dec(clean, "model_calls_mean"),
            "trace_alarm": None}


def candidate_rows(stage):
    out = []
    for cfg in P.candidates():
        for prof in PROFILES:
            r = row(stage, "dfl", cfg["id"], prof)
            if r:
                out.append(r)
    for fam in ("v3only", "trace05"):
        for prof in PROFILES:
            r = row(stage, fam, "P0", prof)
            if r:
                out.append(r)
    return out


def pareto(rows, prof, x="clean_A", y="FN8_risk_mean", z="cost50", min_fraction=0.5):
    """Non-dominated set on (higher clean_A, lower FN8 risk, lower cost50) within one profile. Policies whose clean_A is below
    min_fraction x the baseline's (the registered usability floor, e.g. 'hold everything') are excluded: zero risk by holding all
    images is not a usability improvement."""
    base = next((r for r in rows if r["profile"] == prof and r["family"] == "dfl" and r["id"] == "P0"), None)
    floor = min_fraction * base[x] if base and base[x] is not None else 0.0
    rr = [r for r in rows if r["profile"] == prof and None not in (r[x], r[y], r[z]) and r[x] >= floor]
    front = []
    for a in rr:
        dominated = any(b is not a and b[x] >= a[x] and b[y] <= a[y] and b[z] <= a[z] and (b[x] > a[x] or b[y] < a[y] or b[z] < a[z]) for b in rr)
        if not dominated:
            front.append(f"{a['family']}/{a['id']}")
    return front


def fmt(v, p=3):
    return "N/A" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
