"""XGD Analyzer regression test against X14 + final 4-configuration evaluation. No training, no threshold tuning.

Stage A (analyzer process; never imports Phase G code): runs the XGD Analyzer on the exact X14 conditions
  (DFL t2 fold-B = 3호기 primary, DFL t2 fold-A = 1·2호기 secondary; seeds 17/29/43; qualities clean, blur04, blur08, resample085,
  resample07, noise2, noise4, composition; scale arms under / rule(X8) / over, plus native 960 for fold B). Regression mode uses the
  same leakage-free references as X14: LOSO-session X12 references and CRC lambda calibrated on the OTHER date half of the device.
  Degradation and its RNG are re-implemented operation-for-operation (validated by bitwise equality with the X14 predictions).
  Also checks the full raw-BMP path (inspection-mark removal inside the analyzer) against the preprocessed path.
Stage B (evaluation process; sealed evaluators): bitwise comparison with X14 predictions, outcome / decision-metric / qualification
  equality with X14_ANALYSIS.json, and the final table: detector-only, DFL + X8, DFL + X8 + X12, FULL XGD Analyzer.
Self-contained: the X14 records (stored predictions, X14_ANALYSIS.json, X1 native predictions, split record, development-set truth)
are in tools/x14_records/, and the sealed evaluators are verbatim copies in tools/x14_sealed.py (SHA256: x14_records/SOURCES.json).
Usage (from this model folder): python -u tools/regression_x14.py run      (stage A -> deployment calibration -> stage B)
Outputs: outputs/validation/REGRESSION_REPORT.json, FINAL_EVALUATION.json/.md; deployment artifacts in outputs/calibration.
"""
import collections
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent                      # this model folder
REC = HERE / "x14_records"               # X14 records copied from the research repository
OUT = MODEL / "outputs" / "validation"
ART = MODEL / "outputs" / "calibration"
SEEDS = [17, 29, 43]
QUALITIES = ["clean", "blur04", "blur08", "resample085", "resample07", "noise2", "noise4", "composition"]
GROUPS = {"clean": ["clean"], "mild": ["blur04", "resample085", "noise2"], "moderate": ["blur08", "resample07", "noise4"], "compound": ["composition"]}
FAMILIES = [("DFL-B", "dfl_src12", 1, "primary"), ("DFL-A", "dfl_src3_2", 0, "secondary")]
NOISE = {"noise2": 2, "noise4": 4, "noise8": 8, "noise12": 12}
BLUR = {"blur04": .4, "blur08": .8, "blur16": 1.6, "blur24": 2.4}
RESAMPLE = {"resample085": .85, "resample07": .7, "resample05": .5, "resample035": .35}


def phase_rng(seed, device, epoch, stem, purpose):
    h = hashlib.sha256(f"PhaseG/{seed}/{device}/{epoch}/{stem}/{purpose}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def degrade(im, suite, rng):
    import cv2
    h, w = im.shape[:2]
    if suite == "baseline":
        return im
    if suite in NOISE:
        return np.clip(im.astype(float) + rng.normal(0, NOISE[suite], (h, w, 1)), 0, 255).astype(np.uint8)
    if suite in BLUR:
        return cv2.GaussianBlur(im, (0, 0), BLUR[suite], borderType=cv2.BORDER_REFLECT_101)
    if suite in RESAMPLE:
        q = RESAMPLE[suite]
        return cv2.resize(cv2.resize(im, (max(1, round(w * q)), max(1, round(h * q))), interpolation=cv2.INTER_AREA), (w, h),
                          interpolation=cv2.INTER_LINEAR)
    if suite == "composition":
        im = cv2.GaussianBlur(im, (0, 0), 1.2, borderType=cv2.BORDER_REFLECT_101)
        return np.clip((im.astype(float) - 127.5) * .7 + 127.5 + rng.normal(0, 6, (h, w, 1)), 0, 255).astype(np.uint8)
    raise ValueError(suite)


def halves(stems, meta):
    cl = collections.defaultdict(list)
    for s in stems:
        cl[meta[s]["date"] + meta[s]["device"]].append(s)
    keys = sorted(cl)
    return [sorted(s for i, k in enumerate(keys) if i % 2 == h for s in cl[k]) for h in (0, 1)]


# ------------------------------------------------------------------------------------------------ stage A
def stage_a():
    sys.path.insert(0, str(MODEL))
    import cv2
    from src import xgd_analyzer as XAN
    from src import xgd_config as XC
    cfg = XC.load()
    _, product_mask, centre_matches = XAN.protocol(cfg)
    meta = XC.provider_metadata(cfg)
    dev = XC.dev_stems(cfg)
    hd = next(f for f in json.loads((REC / "OUTER_INNER_SPLITS_V2.json").read_text(encoding="utf8"))["folds"]
              if f["held_device"].startswith("3"))["held_device"]
    gray = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    clean = {s: XAN.read_image(XC.clean_path(cfg, s)) for s in dev}
    sig0 = {s: XAN.quality_signals(gray(clean[s]), product_mask) for s in dev}
    session = {s: meta[s]["date"] for s in dev}
    loso = {}
    for d in sorted({meta[s]["device"] for s in dev}):
        ss = [s for s in dev if meta[s]["device"] == d]
        for sess in sorted({session[s] for s in ss}):
            r = XAN.build_quality_reference(ss, sig0, session, exclude_session=sess)
            for s in ss:
                if session[s] == sess:
                    loso[s] = r
    qg, crc, thr = cfg["quality_gate"], cfg["crc"], cfg["detector"]["decision_threshold"]
    a = XC.split(cfg)["schemes"]["t2"]["assignment"]
    result = {"families": {}}
    for fam, profile, fold, role in FAMILIES:
        targets = sorted(s for s, f in a.items() if f == fold)
        rng_ = XC.profile_scale_range(cfg, profile)
        lo = rng_["min_ratio"]
        imgs = {q: {s: (clean[s] if q == "clean" else degrade(clean[s], q, phase_rng(20261002, hd, 0, s, q))) for s in targets} for q in QUALITIES}
        sig = {q: ({s: sig0[s] for s in targets} if q == "clean" else {s: XAN.quality_signals(gray(imgs[q][s]), product_mask) for s in targets}) for q in QUALITIES}
        gate = {q: {s: XAN.quality_state_v1_0(sig[q][s], loso[s], qg) for s in targets} for q in QUALITIES}
        arms = ["under", "rule", "over"] + (["native"] if fold == 1 else [])
        fam_out = {"profile": profile, "role": role, "scale_range": rng_, "targets": targets,
                   "gate": {q: {s: [g[0], g[1]] for s, g in gate[q].items()} for q in QUALITIES}, "seeds": {}}
        for sd in SEEDS:
            an = XAN.XGDAnalyzer(cfg, profile, sd, load_artifacts=False)
            recs = {q: {arm: {} for arm in arms} for q in QUALITIES}
            for s in targets:
                w, h = meta[s]["resolution"]
                gt = XAN.read_truth(XC.label_path(cfg, s))
                t = XC.letterbox_ratio(w, h)
                for q in QUALITIES:
                    for arm in arms:
                        kw = {"scale_mode": "x8"} if arm == "rule" else ({"scale_mode": "native"} if arm == "native" else
                                                                       {"scale_mode": "fixed", "fixed_scale": 1.0 if arm == "under" else 1.5 * max(t, lo)})
                        r = an.analyze(image=imgs[q][s], preprocessed=True, signals=sig[q][s], reference=loso[s],
                                       calibration={"N": 0, "lambda": math.inf}, raw_boxes=True, **kw)
                        o = XAN.outcome(r["_boxes_normalized_e0"], gt, w, h, centre_matches, thr)
                        recs[q][arm][s] = {"boxes": r["_boxes_normalized_e0"], "trust": r["trust"], "imgsz": r["scale"]["imgsz"], **o}
            # CRC lambda: device-wise other date half, clean + rule (= X14); decisions with the analyzer's decision function
            lam = {}
            for d in sorted({meta[s]["device"] for s in targets}):
                hv = halves([s for s in targets if meta[s]["device"] == d], meta)
                for hh, part in enumerate(hv):
                    other = hv[1 - hh]
                    l_, b_ = XAN.crc_lambda_v1_0([recs["clean"]["rule"][x]["trust"] for x in other], [recs["clean"]["rule"][x]["fail"] for x in other], crc["alpha"])
                    for s in part:
                        lam[s] = {"lambda": l_, "N": len(other), "bound": b_}
            for q in QUALITIES:
                for arm in arms:
                    for s in targets:
                        rr = recs[q][arm][s]
                        rr["decision"] = XAN.decide_v1_0(gate[q][s][0], lam[s], rr["trust"], crc)[0]
            # replicated qualification (validated split assessment + fixed rule) on clean + rule, LOSO gate-lexicographic trust
            qual = {}
            for d in sorted({meta[s]["device"] for s in targets}):
                ss = [s for s in targets if meta[s]["device"] == d]
                cluster = {s: meta[s]["date"] + meta[s]["device"] for s in ss}
                rc = recs["clean"]["rule"]
                assess = XAN.crc_split_assessment_v1_0({s: rc[s]["trust"] - (1. if gate["clean"][s][0] != "NORMAL" else 0.) for s in ss}, {s: rc[s]["fail"] for s in ss}, ss, cluster)
                qual[d] = {"success3": sum(rc[s]["tp"] for s in ss) / sum(rc[s]["n"] for s in ss), "FP_per_image": float(np.mean([rc[s]["fp"] for s in ss])),
                           "crc05": assess["0.05"]["CRC"]}
            fam_out["seeds"][str(sd)] = {"records": recs, "lambda": lam, "qualification_inputs": qual}
            print("stage A", fam, sd, flush=True)
            del an
        result["families"][fam] = fam_out
    # full raw-BMP path (inspection-mark removal in the analyzer) vs preprocessed path: fold B, seed 17, clean, X8
    an = XAN.XGDAnalyzer(cfg, "dfl_src12", 17, load_artifacts=False)
    tb = result["families"]["DFL-B"]["targets"]
    rec = result["families"]["DFL-B"]["seeds"]["17"]["records"]["clean"]["rule"]
    diffs = []
    for s in tb:
        r = an.analyze(image_path=XC.raw_path(cfg, s), preprocessed=False, reference=loso[s], calibration={"N": 0, "lambda": math.inf}, raw_boxes=True)
        A_, B_ = np.asarray(r["_boxes_normalized_e0"], float).reshape(-1, 5), np.asarray(rec[s]["boxes"], float).reshape(-1, 5)
        diffs.append(float(np.abs(A_ - B_).max()) if A_.shape == B_.shape and A_.size else (0. if A_.shape == B_.shape else float("inf")))
    result["raw_path_check"] = {"images": len(tb), "max_abs_diff": max(diffs), "identical_images": int(sum(d == 0 for d in diffs))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "regression_stageA.json").write_text(json.dumps(result, ensure_ascii=False, default=float), encoding="utf8")
    print("stage A done | raw path", result["raw_path_check"], flush=True)


# ------------------------------------------------------------------------------------------------ stage B
def stage_b():
    sys.path.insert(0, str(HERE))
    from x14_sealed import AP, EV, META, read
    S = json.loads((OUT / "regression_stageA.json").read_text(encoding="utf8"))
    X14 = read(REC / "X14_ANALYSIS.json")
    rep = {"families": {}}
    final = {}
    for fam, profile, fold, role in FAMILIES:
        F = S["families"][fam]
        stems = F["targets"]
        runs = ["dfl_orig" if sd == 17 else f"dfl_orig_s{sd}" for sd in SEEDS]
        x14p = [read(REC / "pred" / f"{fam}_{r}.json")["predictions"] for r in runs]
        # 1) bitwise predictions vs X14
        n_cmp, n_count_mismatch, maxdiff = 0, 0, 0.
        for i, sd in enumerate(SEEDS):
            R = F["seeds"][str(sd)]["records"]
            for q in QUALITIES:
                for arm in ("under", "rule", "over"):
                    for s in stems:
                        a_, b_ = np.asarray(R[q][arm][s]["boxes"], float).reshape(-1, 5), np.asarray(x14p[i][q][arm][s], float).reshape(-1, 5)
                        n_cmp += 1
                        if a_.shape != b_.shape:
                            n_count_mismatch += 1
                        elif a_.size:
                            maxdiff = max(maxdiff, float(np.abs(a_ - b_).max()))
        fr = {"prediction_equality_vs_X14": {"compared_image_conditions": n_cmp, "box_count_mismatches": n_count_mismatch, "max_abs_diff": maxdiff}}
        if fold == 1:   # native 960 vs X1 recomputed native
            md = 0.
            mm = 0
            for i, sd in enumerate(SEEDS):
                x1 = read(REC / "x1" / f"{runs[i]}.json")["native"]
                for s in stems:
                    a_ = np.asarray(F["seeds"][str(sd)]["records"]["clean"]["native"][s]["boxes"], float).reshape(-1, 5)
                    b_ = np.asarray(EV.e0(x1[s], *META[s]["resolution"]), float).reshape(-1, 5)
                    if a_.shape != b_.shape:
                        mm += 1
                    elif a_.size:
                        md = max(md, float(np.abs(a_ - b_).max()))
            fr["native_equality_vs_X1"] = {"box_count_mismatches": mm, "max_abs_diff": md}
        # 2) outcome replication vs sealed per_image; decision metrics vs X14
        out_mm = 0
        cond = {}
        recs_ap = {}
        for q in QUALITIES:
            for arm in ("under", "rule", "over") + (("native",) if fold == 1 else ()):
                per = []
                for sd in SEEDS:
                    R = F["seeds"][str(sd)]["records"][q][arm]
                    ap = AP.per_image({s: R[s]["boxes"] for s in stems}, stems)
                    out_mm += sum(1 for s in stems if ap[s]["fail"] != R[s]["fail"] or abs(ap[s]["conf"] - R[s]["trust"]) > 1e-12)
                    per.append((R, ap))
                recs_ap[(q, arm)] = per
        fr["outcome_mismatches_vs_sealed_per_image"] = out_mm

        def modes(q, arm):
            out = {}
            for mode in ("detector_only", "gate_only", "crc_only", "integrated"):
                acc_r, rev_r, full_r, uns, cap, frev = [], [], [], [], [], []
                for i, (R, ap) in enumerate(recs_ap[(q, arm)]):
                    lam = F["seeds"][str(SEEDS[i])]["lambda"]
                    dec = {}
                    for s in stems:
                        g = F["gate"][q][s][0]
                        g = "ACCEPT" if g == "NORMAL" else g
                        crc_ok = ap[s]["conf"] >= lam[s]["lambda"]
                        dec[s] = ("ACCEPT" if mode == "detector_only" else ("ACCEPT" if g == "ACCEPT" else g) if mode == "gate_only" else
                                  ("ACCEPT" if crc_ok else "REVIEW") if mode == "crc_only" else (g if g != "ACCEPT" else ("ACCEPT" if crc_ok else "REVIEW")))
                        if mode == "integrated":
                            assert dec[s] == R[s]["decision"].replace("NORMAL", "ACCEPT"), "analyzer decision differs from the X14 policy"
                    fails = [s for s in stems if ap[s]["fail"]]
                    passes = [s for s in stems if not ap[s]["fail"]]
                    acc = [s for s in stems if dec[s] == "ACCEPT"]
                    acc_r.append(len(acc) / len(stems))
                    rev_r.append(sum(dec[s] == "REVIEW" for s in stems) / len(stems))
                    full_r.append(sum(dec[s] == "FULL_REINSPECTION" for s in stems) / len(stems))
                    uns.append(sum(ap[s]["fail"] for s in acc) / len(stems))
                    cap.append(sum(dec[s] != "ACCEPT" for s in fails) / len(fails) if fails else None)
                    frev.append(sum(dec[s] != "ACCEPT" for s in passes) / len(passes) if passes else None)
                m = lambda v: float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else None
                out[mode] = {"ACCEPT_rate": m(acc_r), "REVIEW_rate": m(rev_r), "FULL_rate": m(full_r), "unsafe_ACCEPT_joint": m(uns), "failure_capture": m(cap), "false_REVIEW": m(frev)}
            return out
        worst = 0.
        for q in QUALITIES:
            for arm in ("under", "rule", "over"):
                mine = modes(q, arm)
                ref = X14["families"][fam]["conditions"][f"{arm}|{q}"]["decision"]
                for mode in mine:
                    for k, v in mine[mode].items():
                        r_ = ref[mode][k]
                        if v is None or r_ is None:
                            if (v is None) != (r_ is None):
                                worst = float("inf")
                            continue
                        worst = max(worst, abs(v - r_))
                cond[f"{arm}|{q}"] = mine
        fr["decision_metric_max_abs_diff_vs_X14"] = worst
        # 3) qualification: sealed crc_ltt (LOSO gate-lexicographic trust) vs analyzer replication vs X14
        qual = {}
        for d in sorted({META[s]["device"] for s in stems}):
            ss = [s for s in stems if META[s]["device"] == d]
            succ, fps, cov, st, rep_cov = [], [], [], [], []
            for i, sd in enumerate(SEEDS):
                R, ap = recs_ap[("clean", "rule")][i]
                apd = {s: ap[s] for s in ss}
                trust = {s: ap[s]["conf"] - (1. if F["gate"]["clean"][s][0] != "NORMAL" else 0.) for s in ss}
                c = AP.crc_ltt(apd, ss, trust)["0.05"]["CRC"]
                succ.append(sum(ap[s]["tp"] for s in ss) / sum(ap[s]["n"] for s in ss))
                fps.append(float(np.mean([ap[s]["fp"] for s in ss])))
                cov.append(c["median_coverage"])
                st.append(c["status"])
                rq = F["seeds"][str(sd)]["qualification_inputs"][d]["crc05"]
                rep_cov.append(abs(rq["median_coverage"] - c["median_coverage"]) + abs((rq["mean_realized_risk"] or 0) - (c["mean_realized_risk"] or 0)))
            cov05, st05 = float(np.median(cov)), collections.Counter(st).most_common(1)[0][0]
            q_ = AP.qualification(min(succ), float(np.mean(fps)), cov05, st05)
            x = X14["families"][fam]["qualification"][d]
            qual[d] = {"worst_seed_3px": min(succ), "FP_per_image": float(np.mean(fps)), "CRC05_coverage": cov05, "CRC05_status": st05, "qualification": q_,
                       "equals_X14": bool(q_ == x["qualification"] and abs(cov05 - x["CRC05_coverage"]) < 1e-12 and abs(min(succ) - x["worst_seed_3px"]) < 1e-12),
                       "analyzer_replication_max_abs_diff": max(rep_cov)}
        fr["qualification"] = qual
        rep["families"][fam] = fr

        # 4) final evaluation: 4 configurations on real acquisitions
        def det_metrics(q, arm):
            ms = [EV.score({s: R[s]["boxes"] for s in stems}, stems) for R, _ in recs_ap[(q, arm)]]
            return {"3px": float(np.mean([m["radii"]["3"]["success"] for m in ms])), "worst_seed_3px": float(min(m["radii"]["3"]["success"] for m in ms)),
                    "FP_per_image": float(np.mean([m["radii"]["3"]["FP_per_image"] for m in ms])), "FN": float(np.mean([m["radii"]["3"]["FN"] for m in ms])),
                    "P90": float(np.mean([m["P50_P90_P95"][1] for m in ms if m["P50_P90_P95"]])) if all(m["P50_P90_P95"] for m in ms) else None,
                    "P95": float(np.mean([m["P50_P90_P95"][2] for m in ms if m["P50_P90_P95"]])) if all(m["P50_P90_P95"] for m in ms) else None,
                    "pFROC3": float(np.mean([m["pFROC3"] for m in ms]))}
        native_arm = "native" if fold == 1 else "rule"
        configs = {"detector-only (DFL native 960)": (native_arm, "detector_only"), "DFL + X8": ("rule", "detector_only"),
                   "DFL + X8 + X12": ("rule", "gate_only"), "FULL XGD Analyzer": ("rule", "integrated")}
        table = {}
        for cname, (arm, mode) in configs.items():
            grp = {}
            for g, qs in GROUPS.items():
                dm = [det_metrics(q, arm) for q in qs]
                dc = [modes(q, arm)[mode] for q in qs]
                avg = lambda L, k: (float(np.mean([x[k] for x in L if x[k] is not None])) if any(x[k] is not None for x in L) else None)
                grp[g] = {**{k: avg(dm, k) for k in dm[0]}, **{k: avg(dc, k) for k in dc[0]}}
                grp[g]["CRC_realized_risk"] = grp[g]["unsafe_ACCEPT_joint"]
            table[cname] = grp
        final[fam] = {"role": role, "table": table, "qualification": {d: v["qualification"] for d, v in qual.items()} if True else None}
    # deployment artifacts consistency (qualification from the full calibration sets must equal the regression / X14 qualification)
    dep = {}
    for fam, profile, fold, role in FAMILIES:
        p = ART / f"crc_{profile}.json"
        if p.exists():
            c = json.loads(p.read_text(encoding="utf8"))
            dep[profile] = {d: {"N": v["N"], "level": v["level"], "qualification": v["qualification"],
                                "equals_regression": v["qualification"] == rep["families"][fam]["qualification"].get(d, {}).get("qualification"),
                                "lambda_by_seed": {k: s_["lambda"] for k, s_ in v["seeds"].items()},
                                "estimated_risk_by_seed": {k: s_["estimated_risk"] for k, s_ in v["seeds"].items()},
                                "ltt_by_seed": {k: s_["ltt"] for k, s_ in v["seeds"].items()}} for d, v in c["devices"].items()}
    rep["deployment_artifacts"] = dep
    rep["raw_path_check"] = S["raw_path_check"]
    ok = all(f["prediction_equality_vs_X14"]["box_count_mismatches"] == 0 and f["prediction_equality_vs_X14"]["max_abs_diff"] == 0 and
             f["outcome_mismatches_vs_sealed_per_image"] == 0 and f["decision_metric_max_abs_diff_vs_X14"] < 1e-12 and
             all(v["equals_X14"] for v in f["qualification"].values()) for f in rep["families"].values())
    rep["REGRESSION_PASS"] = bool(ok and S["raw_path_check"]["max_abs_diff"] == 0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REGRESSION_REPORT.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf8")
    (OUT / "FINAL_EVALUATION.json").write_text(json.dumps(final, ensure_ascii=False, indent=1, default=float), encoding="utf8")
    f3 = lambda v: "-" if v is None else f"{v:.3f}"
    L = ["# XGD Analyzer — 최종 평가 (X14 조건, 같은 provider 데이터 안의 경험적 결과)", "",
         f"회귀 검증: **{'PASS' if rep['REGRESSION_PASS'] else 'FAIL'}** (X14 예측과 비트 단위 비교, 판정 지표·보증 상태 일치, 원본 BMP 경로 확인).", ""]
    for fam, v in final.items():
        L += [f"## {fam} ({v['role']}) — 보증 상태 {v['qualification']}", "",
              "| 구성 | 품질 조건 | 3px | 최악 seed | FP/장 | FN | P90 | P95 | pFROC | ACCEPT | REVIEW | FULL | 위험한 자동 통과 | 실패 포착 | 정상 REVIEW |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for cname, grp in v["table"].items():
            for g, r in grp.items():
                L.append(f"| {cname} | {g} | {f3(r['3px'])} | {f3(r['worst_seed_3px'])} | {f3(r['FP_per_image'])} | {r['FN']:.1f} | {f3(r['P90'])} | {f3(r['P95'])} | {f3(r['pFROC3'])} | "
                         f"{f3(r['ACCEPT_rate'])} | {f3(r['REVIEW_rate'])} | {f3(r['FULL_rate'])} | {f3(r['unsafe_ACCEPT_joint'])} | {f3(r['failure_capture'])} | {f3(r['false_REVIEW'])} |")
        L.append("")
    (OUT / "FINAL_EVALUATION.md").write_text("\n".join(L) + "\n", encoding="utf8")
    print("REGRESSION_PASS", rep["REGRESSION_PASS"], {f: {k: v for k, v in r.items() if k != "qualification"} for f, r in rep["families"].items()}, flush=True)


def run():
    py = sys.executable
    for args in (["stage_a"], None, ["stage_b"]):
        if args is None:
            for profile in ("dfl_src12", "dfl_src3_2"):
                subprocess.run([py, "-u", "-m", "src.xgd_analyzer", "calibrate", "--profile", profile], check=True, cwd=MODEL)
            continue
        subprocess.run([py, "-u", str(Path(__file__)), *args], check=True, cwd=MODEL)


if __name__ == "__main__":
    {"stage_a": stage_a, "stage_b": stage_b, "run": run}[sys.argv[1]]()
