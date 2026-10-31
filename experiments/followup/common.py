"""Shared pieces of the XGD-1.2 follow-up (docs/PREREG_FOLLOWUP.md).

Immutable, content-addressed prediction cache under outputs/followup/cache/:
  pred/<model key>/<view>/<condition>.json      raw detector boxes (before E0), deployment predict settings
  pred/<model key>/<view>/<condition>.done.json completion manifest: signature + SHA-256 of the result file
A result is read only through read_done(): the manifest must exist, its signature must equal the requested one and the file hash must
match. Results are written to a temporary file and published with os.replace (bounded retries for Windows sharing violations).
Monitors read outputs/followup/STATUS.json only, never the result files.
"""
import hashlib
import json
import math
import os
import time
import importlib.metadata as md
from pathlib import Path

import numpy as np

from experiments import lib as L
from src import xgd_analyzer as X

ROOT = L.ROOT
OUT = ROOT / "outputs" / "followup"
CACHE = OUT / "cache"
PRED = CACHE / "pred"
SYNTH = CACHE / "synth"
PC = L.CFG["detector"]["predict"]
VIEWS = ("base", "hflip", "vflip", "hvflip", "sm32", "sp32")


# ------------------------------------------------------------------------------------------------ files
def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def _replace(src, dst, tries=40):
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(0.25 * (1 + i))
    raise PermissionError(f"could not publish {dst} after {tries} attempts")


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    def finite(v):
        if isinstance(v,dict):return {str(k):finite(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [finite(x) for x in v]
        if isinstance(v,(float,np.floating)):return float(v) if math.isfinite(v) else None
        if isinstance(v,np.integer):return int(v)
        return v
    tmp.write_text(json.dumps(finite(obj), ensure_ascii=False, allow_nan=False), encoding="utf8")
    _replace(tmp, path)


def publish(path, obj, signature):
    """Write the result, then its completion manifest (signature + content hash). The manifest is the 'done' flag."""
    path = Path(path)
    if path.with_name(path.stem+'.done.json').exists():
        previous=read_done(path,signature)
        if previous!=obj:raise RuntimeError(f'Completed cache is immutable: {path}')
        return
    atomic_json(path, obj)
    atomic_json(path.with_name(path.stem + ".done.json"), {"signature": signature, "sha256": sha_file(path), "n": len(obj),
                                                           "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})


def read_done(path, signature=None):
    """Completed result or None. Raises if a completed result exists under the same path with a different signature (no silent reuse)."""
    path = Path(path)
    man = path.with_name(path.stem + ".done.json")
    if not man.exists() or not path.exists():
        return None
    m = json.loads(man.read_text(encoding="utf8"))
    if signature is not None and m["signature"] != signature:
        raise RuntimeError(f"cache signature mismatch at {path}; existing result preserved, use a new key")
    if sha_file(path) != m["sha256"]:
        raise RuntimeError(f"cache content hash mismatch at {path}; not reused")
    result=json.loads(path.read_text(encoding="utf8"))
    if m.get('n')!=len(result):raise RuntimeError(f'cache cardinality mismatch: {path}')
    return result


def status(**kw):
    atomic_json(OUT / "STATUS.json", {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **kw})


# ------------------------------------------------------------------------------------------------ models
def checkpoint(family, profile, seed):
    if family == "dfl":
        return L.XC.checkpoint(L.CFG, profile, seed)
    if family == "v3only":
        return ROOT / f"outputs/controls/v3only/retrain/{profile}_s{seed}/weights/last.pt"
    if family == "trace05":
        return ROOT / f"outputs/followup/train/trace05/retrain/{profile}_s{seed}/weights/last.pt"
    raise KeyError(family)


def model_key(family, profile, seed):
    return f"{family}_{profile}_s{seed}_{sha_file(checkpoint(family,profile,seed))[:12]}"


def load_model(family, profile, seed):
    ck = checkpoint(family, profile, seed)
    rel = Path(ck).resolve().relative_to(ROOT).as_posix()
    cfg = L.cfg_with_checkpoints({profile: {seed: rel}})
    return L.Model(profile, seed, cfg=cfg, key=model_key(family, profile, seed))


# ------------------------------------------------------------------------------------------------ views
def view_imgsz(model, w, h, view):
    z = model.scale(w, h)["imgsz"]
    return z - 32 if view == "sm32" else z + 32 if view == "sp32" else z


def view_image(img, view):
    if view in ("hflip", "hvflip"):
        img = img[:, ::-1]
    if view in ("vflip", "hvflip"):
        img = img[::-1, :]
    return np.ascontiguousarray(img)


def view_inverse(boxes, view):
    """Normalized xyxy boxes predicted on the transformed image -> original image coordinates."""
    out = []
    for b in boxes:
        x0, y0, x1, y1, s = b[:5]
        if view in ("hflip", "hvflip"):
            x0, x1 = 1 - x1, 1 - x0
        if view in ("vflip", "hvflip"):
            y0, y1 = 1 - y1, 1 - y0
        out.append([x0, y0, x1, y1, s])
    return out


def predict_view(model, img, view):
    h, w = img.shape[:2]
    z = view_imgsz(model, w, h, view)
    raw = X.detect_raw(model.an.model, view_image(img, view), z, PC, model.an.device)
    if L.X.input_problems(img,'development-audit',boxes=raw):raise RuntimeError('invalid raw detector output')
    return view_inverse(raw, view), z


def signature(model, view, condition, stems, input_digest):
    return {"checkpoint_sha256": model.an.ckpt_sha256, "view": view, "condition": condition, "predict": PC,
            "stems_sha256": sha_bytes("\n".join(stems).encode()), "inputs_sha256": input_digest,
            "transform_source_sha256": sha_file(Path(__file__)), "scale_range": model.rng,
            'sources':{f:sha_file(ROOT/f) for f in ('experiments/followup/predict.py','experiments/followup/synth.py','experiments/e2_predict.py',
                                                    'experiments/lib.py','src/xgd_analyzer.py','src/preprocess.py')},
            'inputs':{f:sha_file(ROOT/f) for f in ('config.yaml','data/metadata.json','data/validation_split.json','data/v3_pixel_sha256.json')},
            'runtime':{k:md.version(k) for k in ('torch','ultralytics','numpy','opencv-python')}}


# ------------------------------------------------------------------------------------------------ geometry
def centers(boxes, w, h, thr):
    return [((b[0] + b[2]) / 2 * w, (b[1] + b[3]) / 2 * h, b[4]) for b in boxes if b[4] >= thr]


def e0(boxes, w, h, radius):
    return X.e0(boxes, w, h, radius)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])
