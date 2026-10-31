"""E1. Decision rules on the sealed X14 conditions (no GPU: the stored X14 predictions are bitwise equal to this analyzer's predictions).

Three decision settings on the same predictions:
  v1.0 / P-X14  XGD-1.0.0 rules, X14 design (LOSO quality reference, CRC from the other date half)   -> must reproduce X14
  v1.1 / P-X14  XGD-1.1.0 rules, same design (isolates the rule changes)
  v1.1 / P-IND-strict  quality reference, CRC lambda and grade from the other date half only (primary)
  P-IND-deployment-grade  same independent reference/lambda but full-data operating grade (diagnostic, not independent)
Also re-checks the review's figures (failure capture, operating seed 17 accepted failures, under-diagnostic acceptances).
python -m experiments.e1_rules_on_x14   -> outputs/experiments/E1_rules_x14/RESULTS.json, SUMMARY.md
"""
import collections
import json
import math

import numpy as np

from experiments import lib as L
from src import xgd_analyzer as X

REC = L.ROOT / "tools" / "x14_records"
QUALITIES = ["clean", "blur04", "blur08", "resample085", "resample07", "noise2", "noise4", "composition"]
FAMS = [("DFL-B", "dfl_src12"), ("DFL-A", "dfl_src3_2")]
RUNS = {17: "dfl_orig", 29: "dfl_orig_s29", 43: "dfl_orig_s43"}


def arm_scale(profile, stem, arm):
    w, h = L.size_of(stem)
    rng = X.XC.profile_scale_range(L.CFG, profile)
    t = X.XC.letterbox_ratio(w, h)
    if arm == "rule":
        return X.scale_policy(w, h, rng, L.CFG["scale_policy"], "x8")
    if arm == "native":
        return X.scale_policy(w, h, rng, L.CFG["scale_policy"], "native")
    f = 1.0 if arm == "under" else 1.5 * max(t, rng["min_ratio"])
    return X.scale_policy(w, h, rng, L.CFG["scale_policy"], "fixed", f)


def main():
    hd = next(f for f in json.loads((REC / "OUTER_INNER_SPLITS_V2.json").read_text(encoding="utf8"))["folds"] if f["held_device"].startswith("3"))["held_device"]
    assert hd == L.X14_HD, hd
    x14 = json.loads((REC / "X14_ANALYSIS.json").read_text(encoding="utf8"))
    crc = L.CFG["crc"]
    result = {"families": {}}
    for fam, prof in FAMS:
        tg = L.targets(prof)
        gts = {s: L.gt(s) for s in tg}
        clean = {s: L.clean_img(s) for s in tg}
        sig = {q: {s: L.signals_of(clean[s] if q == "clean" else L.degrade_any(clean[s], q, s)) for s in tg} for q in QUALITIES}
        devs = {L.META[s]["device"] for s in tg}
        dev_all = [s for s in L.DEV if L.META[s]["device"] in devs]
        sig_all = {s: (sig["clean"][s] if s in sig["clean"] else L.signals_of(L.clean_img(s))) for s in dev_all}
        loso = L.loso_references(dev_all, sig_all)          # X14 design: all dev sessions of the device except the image's own
        arms = ["under", "rule", "over"] + (["native"] if prof == "dfl_src12" else [])
        preds = {sd: json.loads((REC / "pred" / f"{fam}_{run}.json").read_text(encoding="utf8"))["predictions"] for sd, run in RUNS.items()}
        if "native" in arms:
            for sd, run in RUNS.items():
                x1 = json.loads((REC / "x1" / f"{run}.json").read_text(encoding="utf8"))["native"]
                for q in QUALITIES:
                    preds[sd][q]["native"] = {s: (X.e0(x1[s], *L.size_of(s)) if q == "clean" else None) for s in tg}
        recs = {sd: {q: {a: {s: L.image_eval(preds[sd][q][a][s], gts[s], *L.size_of(s)) for s in tg if preds[sd][q][a][s] is not None}
                         for a in arms} for q in QUALITIES} for sd in RUNS}
        clean_rule = {sd: recs[sd]["clean"]["rule"] for sd in RUNS}
        # X14 design: lambda from the other date half (v1.0 and v1.1 threshold functions)
        lam = {"v1.0": {}, "v1.1": {}}
        for sd in RUNS:
            for v, fn in (("v1.0", X.crc_lambda_v1_0), ("v1.1", X.crc_lambda)):
                lam[v][sd] = {}
                for d in sorted({L.META[s]["device"] for s in tg}):
                    hv = L.date_halves([s for s in tg if L.META[s]["device"] == d])
                    for h in (0, 1):
                        other = hv[1 - h]
                        l_, b_ = fn([clean_rule[sd][x]["trust"] for x in other], [clean_rule[sd][x]["fail"] for x in other], crc["alpha"])
                        for s in hv[h]:
                            lam[v][sd][s] = {"N": len(other), "lambda": l_, "bound": b_, "grade": "QUALIFIED"}
        ind_strict = L.design(prof, clean_rule, sig["clean"], "IND", "strict")
        ind = L.design(prof, clean_rule, sig["clean"], "IND", "deployment")
        tim = L.design(prof, clean_rule, sig["clean"], "TIME", "deployment")
        fam_out = {"profile": prof, "targets": len(tg), "conditions": {},
                   "ind_strict_grades": sorted({(L.META[s]["device"], p["grade"], tuple(p["cal_dates"]), p["n_cal"]) for s, p in ind_strict.items()}),
                   "time_split": sorted({(L.META[s]["device"], tuple(p["cal_dates"]), p["n_cal"]) for s, p in tim.items()})}
        for q in QUALITIES:
            for a in arms:
                if not recs[17][q][a]:
                    continue
                cond = {}
                for setting in ("v1.0/P-X14", "v1.1/P-X14", "v1.1/P-IND-deployment-grade", "v1.1/P-IND-strict", "v1.1/P-TIME-deployment-grade"):
                    per_seed_dec, per_seed_det = {}, {}
                    for sd in RUNS:
                        R = recs[sd][q][a]
                        dec = {}
                        for s in R:
                            sc = arm_scale(prof, s, a)
                            if setting == "v1.1/P-TIME-deployment-grade" and s not in tim:
                                continue
                            if setting == "v1.0/P-X14":
                                g = X.quality_state_v1_0(sig[q][s], loso[s], L.CFG["quality_gate"])[0]
                                dec[s] = X.decide_v1_0(g, lam["v1.0"][sd][s], R[s]["trust"], crc)[0]
                            elif setting == "v1.1/P-X14":
                                g = X.quality_state(sig[q][s], loso[s], L.CFG["quality_gate"])[0]
                                ok, why = X.scale_envelope(sc, L.CFG["scale_policy"])
                                dec[s] = X.decide(g, lam["v1.1"][sd][s], R[s]["trust"], R[s]["n_det"], crc, ok, why)[0]
                            else:
                                pk = {"v1.1/P-IND-deployment-grade": ind, "v1.1/P-IND-strict": ind_strict, "v1.1/P-TIME-deployment-grade": tim}[setting][s]
                                dec[s] = L.decide_image(pk, sd, sig[q][s], R[s], sc)[0]
                        per_seed_dec[sd] = L.decision_metrics(dec, {s: R[s] for s in dec})
                        if setting == "v1.0/P-X14":
                            per_seed_det[sd] = L.detection_metrics(R, {s: preds[sd][q][a][s] for s in R}, gts)
                            per_seed_dec[sd]["_decisions"] = dec
                    cond[setting] = L.summarize_seeds({sd: {k: v for k, v in m.items() if k != "_decisions"} for sd, m in per_seed_dec.items()},
                                                      keys_min=("coverage",), keys_max=("joint_risk", "accepted_failures"))
                    if setting == "v1.0/P-X14":
                        cond["detection"] = L.summarize_seeds(per_seed_det, keys_min=("R3", "F13", "pFROC3"), keys_max=("FP_per_image3", "FN"))
                        cond["_v10_decisions"] = {sd: m["_decisions"] for sd, m in per_seed_dec.items()}
                # equality with the sealed X14 decision metrics (integrated mode)
                if a in ("under", "rule", "over"):
                    ref = x14["families"][fam]["conditions"][f"{a}|{q}"]["decision"]["integrated"]
                    mine = cond["v1.0/P-X14"]["mean"]
                    cond["x14_check"] = {"ACCEPT_rate": abs(ref["ACCEPT_rate"] - mine["coverage"]), "unsafe_ACCEPT_joint": abs(ref["unsafe_ACCEPT_joint"] - mine["joint_risk"])}
                # decision changes 1.0 -> 1.1 (P-X14), operating seed 17
                v10 = cond.pop("_v10_decisions")
                changes = collections.Counter()
                for sd in RUNS:
                    R = recs[sd][q][a]
                    for s in R:
                        sc = arm_scale(prof, s, a)
                        g = X.quality_state(sig[q][s], loso[s], L.CFG["quality_gate"])[0]
                        ok, why = X.scale_envelope(sc, L.CFG["scale_policy"])
                        d11 = X.decide(g, lam["v1.1"][sd][s], R[s]["trust"], R[s]["n_det"], crc, ok, why)[0]
                        if d11 != v10[sd][s]:
                            changes[f"{v10[sd][s]}->{d11}"] += 1
                cond["changes_v10_to_v11_P-X14_all_seeds"] = dict(changes)
                fam_out["conditions"][f"{a}|{q}"] = cond
        # operating seed 17 accepted failures, clean + rule (review §5.2)
        R17 = recs[17]["clean"]["rule"]
        af = []
        for s in R17:
            if R17[s]["fail"]:
                g = X.quality_state_v1_0(sig["clean"][s], loso[s], L.CFG["quality_gate"])[0]
                d10 = X.decide_v1_0(g, lam["v1.0"][17][s], R17[s]["trust"], crc)[0]
                d11 = L.decide_image(ind[s], 17, sig["clean"][s], R17[s], arm_scale(prof, s, "rule"))[0]
                af.append({"stem": s, "device": L.META[s]["device"], "tp": R17[s]["tp3"], "n": R17[s]["n"], "fp": R17[s]["fp3"], "trust": R17[s]["trust"],
                           "lambda_x14": lam["v1.0"][17][s]["lambda"], "decision_v10_X14": d10, "decision_v11_IND": d11,
                           "lambda_ind": ind[s]["seeds"][17]["lambda"], "gate_ind": X.quality_state(sig["clean"][s], ind[s]["reference"], L.CFG["quality_gate"])[0]})
        fam_out["seed17_clean_rule_failures"] = af
        fam_out["seed17_clean_rule_counts"] = {"TP": sum(r["tp3"] for r in R17.values()), "GT": sum(r["n"] for r in R17.values()),
                                               "FP": sum(r["fp3"] for r in R17.values())}
        result["families"][fam] = fam_out
        print("E1 done", fam, flush=True)
    L.jdump(L.OUT / "E1_rules_x14" / "RESULTS.json", result)
    write_summary(result)


def f(v, p=3):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return f"{v:.{p}f}"


def write_summary(res):
    L_ = ["# E1. X14 조건에서 판정 규칙 비교 (저장 예측 재사용, GPU 불필요)", "",
          "열: 운영 seed 17 / 3 seed 평균. 전체 위험 = P(ACCEPT ∧ 실패) (분모 전체), 조건부 위험 = P(실패 | ACCEPT) (ACCEPT 0이면 N/A).", ""]
    for fam, F in res["families"].items():
        L_ += [f"## {fam} ({F['profile']}, 보류 {F['targets']}장)", "",
               f"P-IND-strict 보정 등급(장비, 등급, 보정 촬영일, 보정 장수): {F['ind_strict_grades']}", "",
               f"P-TIME 보정(장비, 보정 촬영일, 보정 장수): {F['time_split']}", "",
               "| 조건 | 설정 | coverage (s17/평균) | 전체 위험 | 조건부 위험 s17 [95% CI] | 실패 포착 s17 | 올바른 결과 보류 s17 | ACCEPT된 실패 s17 | REVIEW/FULL s17 |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for c, C in F["conditions"].items():
            for st in ("v1.0/P-X14", "v1.1/P-X14", "v1.1/P-IND-deployment-grade", "v1.1/P-IND-strict", "v1.1/P-TIME-deployment-grade"):
                s17, mn = C[st]["seed17"], C[st]["mean"]
                ci = s17["conditional_risk_CI95"]
                L_.append(f"| {c} | {st} | {f(s17['coverage'])} / {f(mn.get('coverage'))} | {f(s17['joint_risk'])} / {f(mn.get('joint_risk'))} | "
                          f"{f(s17['conditional_risk'])} {('[' + f(ci[0]) + ', ' + f(ci[1]) + ']') if isinstance(ci, list) else ''} | {f(s17['failure_capture'])} | "
                          f"{f(s17['correct_hold_rate'])} | {s17['accepted_failures']}/{s17['failures']} | {s17['review']}/{s17['full']} |")
        L_ += ["", "운영 seed 17, 정상 + X8 실패 영상:", ""]
        for r in F["seed17_clean_rule_failures"]:
            L_.append(f"- {r['stem']} ({r['device']}): TP {r['tp']}/{r['n']}, FP {r['fp']}, 신뢰도 {r['trust']:.4f}, X14 λ {f(r['lambda_x14'], 4)} → v1.0 {r['decision_v10_X14']}; "
                      f"P-IND λ {f(r['lambda_ind'], 4)}, 품질 {r['gate_ind']} → v1.1 {r['decision_v11_IND']}")
        L_.append("")
    (L.OUT / "E1_rules_x14" / "SUMMARY.md").write_text("\n".join(L_) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
