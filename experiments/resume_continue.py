"""Continue the sealed validation queue of experiments/resume_validation.py after its 2026-10-05 23:57 stop.

Cause of the stop: a transient Windows file lock. The read-only progress monitor was reading a prediction cache at the moment
e2_predict atomically replaced it (PermissionError [WinError 5] on os.replace). No model, data, rule, threshold or sealed source
changes. The first supervisor completed tests, the E1 re-score, the control-training wait, the control collection and the execution
seal. Those steps are not repeated (the seal is never overwritten). This runner executes the remaining steps with the SAME commands,
in the same order, and resumes the provenance-sealed prediction caches.

Changes versus the first supervisor:
  - The progress monitor is not started (it caused the race); progress is read from predictions.log instead.
  - The prediction step is retried (at most 6 times) only when its last traceback is that transient PermissionError. Caches resume
    from the last flush, and the per-cache source / model / input seal is still enforced by e2_predict and e3_score.
  - Any other failure stops with a nonzero exit (no silent skip).
  - --from <step> restarts at a later step after a fix.

  python -m experiments.resume_continue [--from score]
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "experiments" / "resume"
STEPS = [
    ("predictions", ["-m", "experiments.e2_predict", "--groups", "clean", "r1", "r2", "r2b", "r3", "grid", "c2", "native", "baseline", "v3only"]),
    ("score", ["-m", "experiments.e3_score"]),
    ("data_error_audit", ["-m", "experiments.e4_audit"]),
    ("deployment_calibrate_12", ["-m", "src.xgd_analyzer", "calibrate", "--profile", "dfl_src12"]),
    ("deployment_calibrate_3_2", ["-m", "src.xgd_analyzer", "calibrate", "--profile", "dfl_src3_2"]),
    ("verify", ["-m", "src.verify"]),
    ("predict79", ["-m", "src.predict_test"]),
    ("isolated_reproduction", ["-m", "experiments.isolated_reproduction"]),
    ("final_report", ["-m", "experiments.final_report"]),
]


def now():
    return datetime.datetime.now().astimezone().isoformat()


def state(stage, status="RUNNING", **kw):
    p = OUT / "STATUS.json"
    old = json.loads(p.read_text(encoding="utf8")) if p.exists() else {}
    old.update({"stage": stage, "status": status, "updated": now(), "evaluation79_label_reads": 0, "runner": "resume_continue", **kw})
    old.pop("error", None) if status != "FAILED" else None
    p.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf8")
    with open(OUT / "continue.log", "a", encoding="utf8") as f:
        f.write(f"{now()} {stage} {status} {json.dumps(kw, ensure_ascii=False)}\n")
    print(stage, status, kw, flush=True)


def transient_lock(log):
    t = log.read_text(encoding="utf8", errors="replace")[-6000:] if log.exists() else ""
    last = t[t.rfind("Traceback"):] if "Traceback" in t else ""
    return "PermissionError" in last and ("WinError 5" in last or "WinError 32" in last) and ".json" in last


def call(name, args, retries=0):
    for attempt in range(retries + 1):
        state(name, attempt=attempt + 1)
        log = OUT / f"{name}.log"
        with open(log, "a", encoding="utf8") as f:
            f.write(f"\n===== {now()} resume_continue attempt {attempt + 1}\n")
            f.flush()
            p = subprocess.run([sys.executable, "-u", *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if p.returncode == 0:
            state(name, "STEP_COMPLETE", attempts=attempt + 1)
            return
        if attempt < retries and transient_lock(log):
            state(name, "RETRY_TRANSIENT_FILE_LOCK", attempt=attempt + 1)
            time.sleep(10)
            continue
        raise RuntimeError(f"{name} exited {p.returncode}; see {name}.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="predictions", choices=[s for s, _ in STEPS])
    a = ap.parse_args()
    if not (OUT / "EXECUTION_SEAL.json").exists():
        raise SystemExit("No execution seal: the first supervisor stages are not complete; do not continue.")
    lock = OUT / "PROCESS_LOCK_continue.json"
    try:
        with lock.open("x", encoding="utf8") as f:
            json.dump({"pid": os.getpid(), "start": now(), "from": a.start}, f)
    except FileExistsError:
        raise SystemExit("Continuation lock exists. Check STATUS/continue.log and the live PID; do not start a duplicate.")
    try:
        names = [s for s, _ in STEPS]
        for name, args in STEPS[names.index(a.start):]:
            call(name, args, retries=6 if name == "predictions" else 0)
        state("FINISHED", "COMPLETE")
    except BaseException as e:
        state("STOPPED", "FAILED", error=str(e))
        raise
    finally:
        rec = json.loads(lock.read_text(encoding="utf8"))
        rec["finished"] = now()
        lock.write_text(json.dumps(rec, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
