"""Resume wrapper for the registered trace05 training (implementation fix only; outputs/followup/RESUME_AMENDMENT_01.json).

The registered runner (train_trace05.py, sealed in ANALYSIS_EXECUTION_LOCK / TRAIN_LOCK) republishes DATA_MANIFEST.json before training.
common.publish compares the newly generated in-memory records with the JSON manifest already on disk; the new records hold mark
positions as Python tuples while the stored JSON holds lists, so an identical manifest was rejected ("Completed cache is immutable",
05:09 KST). This wrapper applies the same JSON normalization that atomic_json applies when writing, then calls the unchanged registered
main(). An identical manifest is therefore accepted without rewriting, and any real difference is still rejected by the same check.
No registered source file, image, label, recipe, seed, lock or cache is modified.

python -m experiments.followup.train_trace05_resume
"""
import json

from experiments.followup import common as F
from experiments.followup import train_trace05 as T5

_registered_publish = F.publish


def _publish_normalized(path, obj, signature):
    return _registered_publish(path, json.loads(json.dumps(obj, ensure_ascii=False, allow_nan=False)), signature)


def main():
    F.publish = _publish_normalized
    try:
        T5.main()
    finally:
        F.publish = _registered_publish


if __name__ == "__main__":
    main()
