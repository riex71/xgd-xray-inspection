"""XGD Analyzer configuration loader (config.yaml) and data-provenance helpers. All paths are relative to this model folder."""
import csv
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load(path=None):
    cfg = yaml.safe_load(Path(path or CONFIG_PATH).read_text(encoding="utf8"))
    cfg["_root"] = ROOT
    cfg["_artifacts"] = ROOT / cfg["paths"]["artifacts"]
    return cfg


def checkpoint(cfg, profile, seed):
    return cfg["_root"] / cfg["profiles"][profile]["checkpoints"][int(seed)]


def rel(cfg, p):
    """Path relative to the model folder when inside it (outputs never carry absolute personal paths)."""
    p = Path(p)
    return p.relative_to(cfg["_root"]).as_posix() if p.is_relative_to(cfg["_root"]) else p.as_posix()


def provider_metadata(cfg):
    """stem -> {device, date, resolution} (data/metadata.json, copied from the provider snapshot metadata)."""
    recs = json.loads((cfg["_root"] / cfg["paths"]["metadata"]).read_text(encoding="utf8"))["records"]
    return {r["stem"]: {"device": r["device"], "date": r["date"], "resolution": tuple(r["resolution"])} for r in recs}


def split(cfg):
    return json.loads((cfg["_root"] / "data" / "validation_split.json").read_text(encoding="utf8"))


def split_rows(cfg):
    with open(cfg["_root"] / "data" / "split.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sizes(cfg):
    return {r["stem"]: (int(r["width"]), int(r["height"])) for r in split_rows(cfg)}


def letterbox_ratio(w, h, canvas=960):
    return min(canvas / w, canvas / h)


def profile_training_stems(cfg, profile):
    ts = cfg["profiles"][profile]["training_split"]
    a = split(cfg)["schemes"][ts["scheme"]]["assignment"]
    return sorted(s for s, f in a.items() if f not in (ts["held_fold"], -1))


def profile_scale_range(cfg, profile):
    """Training effective-scale range of a detector profile (letterbox ratios of its own training images)."""
    sz = sizes(cfg)
    canvas = cfg["scale_policy"]["canvas"]
    r = [letterbox_ratio(*sz[s], canvas) for s in profile_training_stems(cfg, profile)]
    return {"min_ratio": min(r), "max_ratio": max(r), "n_training_images": len(r)}


def dev_stems(cfg):
    a = split(cfg)["schemes"]["t1"]["assignment"]
    return sorted(s for s, f in a.items() if f != -1)


def test_stems(cfg):
    return sorted(r["stem"] for r in split_rows(cfg) if r["partition"] == "test")


def label_path(cfg, stem):
    return cfg["_root"] / "data" / "labels_orig" / f"{stem}.txt"


def raw_path(cfg, stem):
    return cfg["_root"] / "data" / "raw" / f"{stem}.bmp"


def clean_path(cfg, stem):
    return cfg["_root"] / "data" / "clean_v3" / "images" / f"{stem}.png"
