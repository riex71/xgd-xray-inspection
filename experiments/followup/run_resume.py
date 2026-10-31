"""Resume the single follow-up GPU queue after the 05:09 KST stop (trace05 manifest comparison).

Completed stages of experiments/followup/run.py (tests, preflight, synthetic sets, DFL / v3only raw predictions, sealed screen
comparisons) are not re-run: their results are immutable, hash-checked caches. Remaining registered stages, in the registered order:
  trace05_training  -> experiments.followup.train_trace05_resume (wrapper, RESUME_AMENDMENT_01)
  trace05_raw       -> experiments.followup.predict --family trace05 --views base --conds dev_clean synth
  final_comparison  -> experiments.followup.analyze --stage final (then complete(): selection, validation, recalibration, contract,
                       resolution diagnostic, fresh-install reproduction, evaluation-79 predictions, report)

python -m experiments.followup.run_resume [--from STAGE]
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

from experiments.followup import common as F

STEPS = [("trace05_training", ["-m", "experiments.followup.train_trace05_resume"]),
         ("trace05_raw", ["-m", "experiments.followup.predict", "--family", "trace05", "--views", "base", "--conds", "dev_clean", "synth"]),
         ("final_comparison", ["-m", "experiments.followup.analyze", "--stage", "final"])]


def call(name, args):
    F.status(stage=name, status="RUNNING", evaluation79_GT_reads=0, runner="run_resume")
    with (F.OUT / f"{name}.log").open("a", encoding="utf8") as log:
        log.write(f"\n===== {datetime.datetime.now().astimezone().isoformat()} run_resume {name}\n")
        log.flush()
        result = subprocess.run([sys.executable, "-u", *args], cwd=F.ROOT, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if result.returncode:
        raise RuntimeError(f"{name} exit {result.returncode}; inspect preserved log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=STEPS[0][0], choices=[s for s, _ in STEPS])
    a = ap.parse_args()
    lock = F.OUT / "PROCESS_LOCK.json"
    with lock.open("x", encoding="utf8") as f:
        json.dump({"pid": os.getpid(), "start": datetime.datetime.now().astimezone().isoformat(), "runner": "run_resume", "from": a.start}, f)
    try:
        names = [s for s, _ in STEPS]
        for name, args in STEPS[names.index(a.start):]:
            call(name, args)
        done = F.OUT / "FOLLOWUP_COMPLETED.json"
        if not done.exists():
            raise RuntimeError("final stage returned without FOLLOWUP_COMPLETED.json")
        final = json.loads(done.read_text(encoding="utf8"))
        F.status(stage="FOLLOWUP_COMPLETED", status="COMPLETE", evaluation79_GT_reads=0, runner="run_resume",
                 pipeline_signature=final.get("pipeline_signature"))
    except BaseException as e:
        F.status(stage="STOPPED", status="FAILED", error=str(e), evaluation79_GT_reads=0, runner="run_resume")
        raise
    finally:
        rec = json.loads(lock.read_text(encoding="utf8"))
        rec["finished"] = datetime.datetime.now().astimezone().isoformat()
        lock.write_text(json.dumps(rec, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
