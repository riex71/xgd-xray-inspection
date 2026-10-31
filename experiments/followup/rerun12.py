"""Re-run the fixed XGD-1.2 runtime on the evaluation images and compare with the sealed final predictions (reproduction check).

Uses src/followup_analyzer.FollowupAnalyzer with outputs/followup/deployment/CONFIG.json + CALIBRATION.json (signature-checked; a changed
model, policy, preprocessing, calibration or dependency version makes every decision FULL_REINSPECTION). Routing: 3호기 -> dfl_src12,
1·2호기 -> dfl_src3_2, seed 17. Evaluation labels are not read. Results go to outputs/followup/rerun/ (the sealed
outputs/followup/deployment/TEST_PREDICTIONS.json is never overwritten). Exit code 1 if any decision, detection count or centre differs.

python -m experiments.followup.rerun12            (or: python run_all.py predict12)
"""
import datetime
import json
import sys

from experiments import lib as L
from experiments.followup import common as F
from src.followup_analyzer import FollowupAnalyzer

ROUTES = {"3호기": "dfl_src12", "1호기": "dfl_src3_2", "2호기": "dfl_src3_2"}


def main():
    sealed = json.loads((F.OUT / "deployment/TEST_PREDICTIONS.json").read_text(encoding="utf8"))["images"]
    out, mism = {}, []
    for prof in ("dfl_src12", "dfl_src3_2"):
        an = FollowupAnalyzer(prof, 17)
        for s in L.XC.test_stems(L.CFG):
            if ROUTES[L.META[s]["device"]] != prof:
                continue
            r = an.analyze(image_path=L.XC.raw_path(L.CFG, s), metadata={"calibration_id": L.META[s]["device"], "session": L.META[s]["date"], "preprocessed": False})
            out[s] = {"decision": r["decision"], "object_count": r["object_count"], "detections": r["detections"], "reason": r["reason"],
                      "completeness_status": r.get("completeness_status"), "pipeline_signature": r.get("pipeline_signature")}
            ref = sealed[s]
            centers = [d["center"] for d in r["detections"]]
            ref_c = [d["center"] for d in ref["detections"]]
            delta = max((max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(sorted(centers), sorted(ref_c))), default=0.0)
            if r["decision"] != ref["decision"] or r["object_count"] != ref["object_count"] or delta > 0.05:
                mism.append({"image": s, "decision": [ref["decision"], r["decision"]], "objects": [ref["object_count"], r["object_count"]], "max_center_delta_px": delta})
        del an
    report = {"time": datetime.datetime.now().astimezone().isoformat(), "images": len(out), "mismatches": mism, "status": "PASS" if not mism else "FAIL",
              "compared_with": "outputs/followup/deployment/TEST_PREDICTIONS.json", "evaluation79_label_reads": 0, "results": out}
    F.atomic_json(F.OUT / "rerun/TEST_PREDICTIONS_RERUN.json", report)
    print("XGD-1.2 rerun", report["status"], len(out), "images,", len(mism), "mismatches", flush=True)
    if mism:
        sys.exit(1)


if __name__ == "__main__":
    main()
