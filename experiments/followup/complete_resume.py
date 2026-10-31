"""Resume the registered completion stage after the 08:34 KST stop (RESUME_AMENDMENT_02).

complete.main() had already written SELECTION_LOCK.json, the fixed validations, FINAL_SELECTION.json, the synthesis audit and the LTT report
when the registered DFL distribution diagnostic failed (device mismatch). This runner:
  1. verifies the sealed execution lock (experiments.followup.execution_lock.verify),
  2. checks that FINAL_SELECTION.json is complete (every profile has a passed validation item) — it is not recomputed or changed,
  3. runs the corrected diagnostic (experiments.followup.distribution_probe_resume) in place of the failed one,
  4. then calls the registered complete.py functions in their registered order and unchanged:
     calibrate -> post_selection_tests -> completed_contract -> resolution_test -> fresh_reproduce -> predict_final -> report.main,
  5. writes FOLLOWUP_COMPLETED.json exactly as complete.main() does.

python -m experiments.followup.complete_resume
"""
import datetime
import json

from experiments.followup import analyze as A
from experiments.followup import common as F
from experiments.followup import complete as C


def main():
    from experiments.followup.execution_lock import verify
    verify()
    A.initialize()
    final = json.loads((F.OUT / "FINAL_SELECTION.json").read_text(encoding="utf8"))
    selection = final["selection"]
    for prof, item in selection["profiles"].items():
        hist = final["validation_history"][prof]
        if not item.get("final_validation_passed") or not hist or not hist[-1]["passed"] or hist[-1]["item"]["policy"] != item["policy"] \
                or hist[-1]["item"]["family"] != item["family"]:
            raise RuntimeError(f"FINAL_SELECTION.json incomplete or inconsistent for {prof}; not resumed")
    C.task("DFL_distribution_diagnostic_resume", ["-m", "experiments.followup.distribution_probe_resume"])
    config = C.calibrate(selection)
    C.task("post_selection_tests", ["-m", "pytest", "-q", "tests", "-p", "no:cacheprovider"])
    C.completed_contract(config)
    C.resolution_test(config)
    C.fresh_reproduce()
    C.predict_final(config)
    from experiments.followup.report import main as report
    report()
    F.atomic_json(F.OUT / "FOLLOWUP_COMPLETED.json", {"time": datetime.datetime.now().astimezone().isoformat(),
                  "pipeline_signature": config["pipeline_signature"], "evaluation79_GT_reads": 0, "formal_qualification": "UNQUALIFIED",
                  "status": "COMPLETED_INTERNAL_RESEARCH_AND_REPRODUCTION", "submission_zip_created": False,
                  "resumed_by": "experiments/followup/complete_resume.py (RESUME_AMENDMENT_02)"})
    print("Follow-up final selection, recalibration, reproduction, and reporting completed", flush=True)


if __name__ == "__main__":
    main()
