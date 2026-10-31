"""E5. Numerical tolerance and discrete-decision stability across compute devices (GPU vs CPU), frozen operating seed 17.

A different GPU could not be tested here; CPU inference with the same weights, preprocessing, X8 scale, E0 and decision rules is the
available proxy for "another device". Reports box / score differences, image-outcome flips (3 px) and decision flips under the
deployment calibration (outputs/calibration, XGD-1.1.0). No threshold or rule is changed. The GPU side is read from the sealed clean
caches of E2 (seal-checked by e3_score.load_cache).

python -m experiments.e5_device_tolerance   -> outputs/experiments/E5_device_tolerance/RESULTS.json
"""
import json
import math
import time

import numpy as np

from experiments import lib as L
from experiments.e3_score import load_cache, pairs
from src import xgd_analyzer as X


def main():
    out = {"device_gpu": None, "device_cpu": "cpu", "seed": 17, "profiles": {}}
    import torch
    out["device_gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    for prof in ("dfl_src12", "dfl_src3_2"):
        gpu = load_cache(f"{prof}_s17", "clean")
        an = X.XGDAnalyzer(L.CFG, prof, 17, device="cpu")          # deployment artifacts (quality reference, CRC) loaded
        tg = L.targets(prof)
        rows, t0 = [], time.perf_counter()
        for s in tg:
            img = L.clean_img(s)
            dev = L.META[s]["device"]
            rc = an.analyze(image=img, preprocessed=True, calibration_id=dev, raw_boxes=True)
            bc = rc["_boxes_normalized_e0"]
            bg = gpu[s]["boxes"]
            w, h = L.size_of(s)
            g = L.gt(s)
            # same decision function and artifacts for the GPU boxes (signals do not depend on the device)
            eg = L.image_eval(bg, g, w, h)
            ec = L.image_eval(bc, g, w, h)
            calib = an._calibration(dev)
            ref, _ = an._reference(dev)
            gate = X.quality_state(rc["quality"]["signals"], ref, L.CFG["quality_gate"])[0]
            sc = X.scale_policy(w, h, an.scale_range, L.CFG["scale_policy"])
            ok, why = X.scale_envelope(sc, L.CFG["scale_policy"])
            dg = X.decide(gate, calib, eg["trust"], eg["n_det"], L.CFG["crc"], ok, why, compatible=True)[0]
            pm, ua, ub = pairs(bg, bc, w, h, 3.0)
            A, B = L.centers_px(bg, w, h), L.centers_px(bc, w, h)
            dxy = [e for _, _, e in pm]
            dscore = [abs(A[i][2] - B[j][2]) for i, j, _ in pm]
            rows.append({"stem": s, "n_det_gpu": eg["n_det"], "n_det_cpu": ec["n_det"], "unmatched_gpu": ua, "unmatched_cpu": ub,
                         "max_center_shift_px": max(dxy) if dxy else 0.0, "max_score_diff": max(dscore) if dscore else 0.0,
                         "trust_diff": abs(eg["trust"] - ec["trust"]), "fail_gpu": eg["fail"], "fail_cpu": ec["fail"],
                         "decision_gpu": dg, "decision_cpu": rc["decision"]})
        sec = (time.perf_counter() - t0) / len(tg)
        cs = [r["max_center_shift_px"] for r in rows]
        sd = [r["max_score_diff"] for r in rows]
        out["profiles"][prof] = {
            "images": len(rows), "cpu_seconds_per_image": sec,
            "detection_count_mismatch_images": sum(r["n_det_gpu"] != r["n_det_cpu"] or r["unmatched_gpu"] or r["unmatched_cpu"] for r in rows),
            "max_center_shift_px": max(cs), "p99_center_shift_px": float(np.percentile(cs, 99)),
            "max_score_diff": max(sd), "p99_score_diff": float(np.percentile(sd, 99)), "max_trust_diff": max(r["trust_diff"] for r in rows),
            "outcome_flips_3px": sum(r["fail_gpu"] != r["fail_cpu"] for r in rows),
            "decision_flips": sum(r["decision_gpu"] != r["decision_cpu"] for r in rows),
            "decision_counts_gpu": {d: sum(r["decision_gpu"] == d for r in rows) for d in X.STATES},
            "decision_counts_cpu": {d: sum(r["decision_cpu"] == d for r in rows) for d in X.STATES},
            "flipped": [r for r in rows if r["decision_gpu"] != r["decision_cpu"] or r["fail_gpu"] != r["fail_cpu"]]}
        print("E5", prof, {k: v for k, v in out["profiles"][prof].items() if k != "flipped"}, flush=True)
        del an
    out["interpretation"] = ("Bit-identical boxes are not expected across devices. Reported here: magnitude of continuous differences and "
                             "the number of discrete outcome / decision changes on the development held folds (seed 17).")
    L.jdump(L.OUT / "E5_device_tolerance" / "RESULTS.json", out)


if __name__ == "__main__":
    main()
