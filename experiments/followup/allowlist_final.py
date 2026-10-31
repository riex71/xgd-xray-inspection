"""Final submission allowlist (no ZIP is created). RESUME_AMENDMENT_03.

The sealed experiments/followup/report.submission_manifest() stopped with LookupError: unknown encoding 'utf8-sig' (Python accepts 'utf-8-sig').
This script reproduces its registered include patterns and exclusions verbatim, reads text with the valid codec name, and adds the evidence
files the regenerated documents cite (E5/E7/E10/E11, example figures, amendments, locks, reproduction records, 1.1 baseline predictions).
Every included text file is scanned for absolute home-directory paths and credential patterns; SHA-256 and sizes are recorded.
Writes outputs/followup/SUBMISSION_ALLOWLIST.json and outputs/followup/PACKAGE_READINESS.json; exits non-zero if anything is flagged.

python -m experiments.followup.allowlist_final
"""
import datetime
import json
import re
import sys

from experiments import lib as L
from experiments.followup import common as F

ROOT = F.ROOT
# registered patterns of report.submission_manifest (verbatim)
PATTERNS = ("src/*.py", "experiments/*.py", "experiments/followup/*.py", "tests/*.py", "tools/*.py", "docs/*.md",
            "requirements.txt", "README.md", "config.yaml", "data/metadata.json", "data/split.csv", "data/validation_split.json",
            "data/v3_pixel_sha256.json", "data/models/*.yaml", "data/initial/*.pt", "data/raw/*.bmp", "data/labels_orig/*.txt",
            "data/clean_v3/images/*.png", "data/clean_v3/labels/*.txt", "data/v3tr_orig/images/*.png", "data/v3tr_orig/labels/*.txt",
            "weights/*.pt", "weights/MANIFEST.json", "outputs/calibration/*.json", "outputs/followup/deployment/*.json",
            "outputs/followup/deployment/*.csv", "outputs/followup/deployment/test_predictions_yolo/*.txt")
NAMED = ("FINAL_SELECTION.json", "ANALYSIS_EXECUTION_LOCK.json", "COMPLETED_PIPELINE_CONTRACT.json", "RESOLUTION_COMPATIBILITY.json",
         "LTT_PROCEDURE_DIAGNOSTICS.json", "SYNTHESIS_GEOMETRY_PHOTOMETRY_AUDIT.json", "DFL_DISTRIBUTION_DIAGNOSTIC.json", "RISK_COVERAGE_DIAGNOSTIC.json",
         "risk_coverage.png")
OLD_RESULTS = ("E1_rules_x14", "E3_scoring", "E4_audit", "E6_data_profile", "E8_failure_geometry", "E9_missed_object_blindspot")
EXCLUDED = ["AGENTS.md", "CLAUDE.md", "versions/", "outputs/followup/cache/", "outputs/followup/reproduction/fresh_venv/", "__pycache__/", ".env",
            "credentials", "intermediate epoch weights", "absolute-path training logs/args", "evaluation GT"]
# evidence the regenerated documents cite, beyond the registered list
EXTRA = ("outputs/experiments/*/SUMMARY.md", "outputs/experiments/E5_device_tolerance/RESULTS.json", "outputs/experiments/E7_eval79_reasons/RESULTS.json",
         "outputs/experiments/E10_localization_bias/RESULTS.json", "outputs/experiments/E11_quality_signals/RESULTS.json",
         "outputs/followup/examples/*.png", "outputs/followup/FINAL_FAILURE_ROWS.json", "outputs/followup/RESUME_AMENDMENT_0*.json",
         "outputs/followup/SELECTION_LOCK.json", "outputs/followup/RAW_COLLECTION_LOCK.json", "outputs/followup/train/trace05/TRAIN_LOCK.json",
         "outputs/followup/reproduction/FRESH_REPRODUCTION.json", "outputs/followup/reproduction/INSTALL_STATUS.json",
         "outputs/followup/reproduction/REQUIREMENTS_RESOLVED.txt", "outputs/followup/rerun/TEST_PREDICTIONS_RERUN.json", "outputs/validation/*.json",
         "licenses/*.txt", "run_all.py", "outputs/test_predictions.json", "outputs/test_predictions_images.csv", "outputs/test_predictions_boxes.csv")
HOME = re.compile(r"(?i)(?:\b[a-z]:[\\/]+" + "us" + r"ers[\\/]+[^\\/\s\"]+|/" + "ho" + r"me/[^/\s\"]+|/" + "Us" + r"ers/[^/\s\"]+)")
SECRET = re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|sk-[A-Za-z0-9]{30,}")


def main():
    config = json.loads((F.OUT / "deployment/CONFIG.json").read_text(encoding="utf8"))
    files = set()
    for pattern in PATTERNS + EXTRA:
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    files.update(F.OUT / n for n in NAMED)
    for key in config["weights"]:
        fam, prof, sd = key.split("/")
        files.add(F.checkpoint(fam, prof, int(sd)))
    if any(i["family"] == "trace05" for i in config["profiles"].values()):
        files.update(p for p in (ROOT / "data/followup_trace05").rglob("*") if p.is_file())
        files.update(p for p in (F.OUT / "train/trace05").glob("*.json") if p.is_file())
    for prof, item in config["profiles"].items():
        files.add(F.OUT / f"SELECTED_VALIDATION_{prof}_{item['family']}_{item['policy']['id']}.json")
    for old in OLD_RESULTS:
        p = L.OUT / old / "RESULTS.json"
        if p.exists():
            files.add(p)
    rows, flags = [], []
    for p in sorted(files):
        rel = p.relative_to(ROOT).as_posix()
        if p.name in ("AGENTS.md", "CLAUDE.md") or rel.startswith(("versions/", "outputs/followup/cache/", "outputs/followup/reproduction/fresh_venv/")) \
                or p.suffix == ".log" or "__pycache__" in rel:
            continue
        if not p.exists():
            flags.append({"path": rel, "reasons": ["missing required file"]})
            continue
        reasons = []
        if p.suffix.lower() in (".md", ".py", ".txt", ".csv", ".json", ".yaml", ".yml") and p.stat().st_size < 60_000_000:
            text = p.read_text(encoding="utf-8-sig", errors="replace")
            if HOME.search(text):
                reasons.append("absolute home-directory path")
            if SECRET.search(text):
                reasons.append("credential pattern")
        if reasons:
            flags.append({"path": rel, "reasons": reasons})
        rows.append({"path": rel, "bytes": p.stat().st_size, "sha256": F.sha_file(p)})
    status = "READY_FOR_REVIEW" if not flags else "BLOCKED_SANITIZATION_REQUIRED"
    now = datetime.datetime.now().astimezone().isoformat()
    allow = {"status": status, "zip_created": False, "files": rows, "excluded": EXCLUDED, "flags_paths_only": flags, "time": now,
             "total_bytes": sum(r["bytes"] for r in rows),
             "generated_by": "experiments/followup/allowlist_final.py (registered report.submission_manifest patterns + cited evidence; RESUME_AMENDMENT_03)",
             "blind_scan_scope": "absolute home-directory paths and high-confidence credential patterns; human name/logo visual review still required",
             "data_rights": "KAMP licensed project data; do not infer redistribution rights from private/public repository status",
             "checkpoint_selection": "registered final/representative checkpoints only (DFL x6 + v3only control x6 for the selected policy); no epoch sweep artifacts"}
    F.atomic_json(F.OUT / "SUBMISSION_ALLOWLIST.json", allow)
    fresh = json.loads((F.OUT / "reproduction/FRESH_REPRODUCTION.json").read_text(encoding="utf8"))
    contract = json.loads((F.OUT / "COMPLETED_PIPELINE_CONTRACT.json").read_text(encoding="utf8"))
    pred = json.loads((F.OUT / "deployment/TEST_PREDICTIONS.json").read_text(encoding="utf8"))
    F.atomic_json(F.OUT / "PACKAGE_READINESS.json", {"status": status, "zip_created": False, "fresh_install_inference": fresh["status"],
                  "completed_contract": contract["status"], "formal_qualification": "UNQUALIFIED", "evaluation79_GT_reads": 0,
                  "evaluation79_images": pred["n_images"], "blocking_paths": flags, "different_GPU": "UNTESTED", "files": len(rows),
                  "total_megabytes": round(allow["total_bytes"] / 1e6, 1), "pipeline_signature": config["pipeline_signature"], "time": now})
    print("allowlist", status, len(rows), "files", round(allow["total_bytes"] / 1e6, 1), "MB", flush=True)
    if flags:
        for x in flags[:20]:
            print("  flagged", x, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
