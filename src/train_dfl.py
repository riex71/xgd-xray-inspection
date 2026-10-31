"""DFL 폴드 검출기 학습(재현용).

제출 가중치(weights/*.pt)는 원 protocol 의 src.validation.train.run("t2", ["dfl_orig", "dfl_orig_s29", "dfl_orig_s43"])
(= detectors.train_fold) 로 학습한 t2 폴드 모델이다. 이 스크립트는 같은 학습 목록·같은 초기값·같은 모델 구성·같은 학습 인자로
다시 학습한다. 제출 가중치를 덮어쓰지 않고 outputs/retrain/ 에 만든다. GPU 결정론 설정을 그대로 쓰지만 학습 결과가
제출 가중치와 비트 단위로 같다고 보장하지는 않는다(재학습 비교는 하지 않았음).

  python -m src.train_dfl --check                          학습 없이 계약 확인: 학습 목록·학습 인자·초기값·모델 구성·
                                                           학습 영상 픽셀 해시·학습 라벨을 weights/MANIFEST.json 과 대조
  python -m src.train_dfl --profile dfl_src12 --seed 17    모델 하나 학습 (12GB GPU 에서 약 8분)
  python -m src.train_dfl --all                            6개 모두 학습
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from . import xgd_config as XC
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import xgd_config as XC

PATH_KEYS = ("model", "data", "project", "name", "save_dir")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def train_args(cfg, seed):
    """원 detectors.train_args(cfg, spec) + dfl_orig 덮어쓰기(scale 0.40) 와 같은 값."""
    t = cfg["train"]
    return dict(epochs=t["epochs"], imgsz=t["imgsz"], batch=t["batch"], optimizer=t["optimizer"],
                lr0=t["lr0"], weight_decay=t["weight_decay"], box=t["box"], cls=t["cls"],
                mosaic=t["mosaic"], scale=t["scale"], close_mosaic=0, degrees=t["degrees"],
                translate=t["translate"], hsv_h=0, hsv_s=0, hsv_v=t["hsv_v"],
                fliplr=t["fliplr"], flipud=0, mixup=0, amp=False, workers=0, patience=0,
                deterministic=True, seed=seed, save=True, verbose=False, plots=False, cache=False, exist_ok=True)


def stems(cfg, profile):
    ts = cfg["profiles"][profile]["training_split"]
    a = XC.split(cfg)["schemes"][ts["scheme"]]["assignment"]
    held = sorted(s for s, f in a.items() if f == ts["held_fold"])
    return XC.profile_training_stems(cfg, profile), held


def write_lists(cfg, profile, seed, out):
    """학습 목록: 파일명 순(배치 구성을 정하므로 고정). 학습 = config train.train_input(기본 v3tr_orig), 보류 폴드 = clean_v3
    (val=False 라 채점에 쓰지 않음)."""
    root = cfg["_root"]
    train, held = stems(cfg, profile)
    lists = out / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    name = f"{profile}_s{seed}"
    tr, va = lists / f"{name}_train.txt", lists / f"{name}_held.txt"
    src = cfg["train"].get("train_input", "v3tr_orig")
    tr.write_text("\n".join((root / "data" / src / "images" / f"{s}.png").as_posix() for s in train) + "\n", encoding="utf8")
    va.write_text("\n".join((root / "data" / "clean_v3" / "images" / f"{s}.png").as_posix() for s in held) + "\n", encoding="utf8")
    y = lists / f"{name}.yaml"
    y.write_text(f"path: {root.as_posix()}\ntrain: {tr.as_posix()}\nval: {va.as_posix()}\nnames:\n  0: foreign_object\n", encoding="utf8")
    return y


def runtime(cfg):
    """원 protocol common.init_runtime 과 같은 실행 환경."""
    out = cfg["_artifacts"] / "runtime"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update({"YOLO_CONFIG_DIR": str(out), "YOLO_OFFLINE": "true", "WANDB_MODE": "disabled",
                       "OMP_NUM_THREADS": str(cfg["threads"]), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    import torch

    torch.set_num_threads(cfg["threads"])
    if not getattr(torch.utils.data.DataLoader, "_pin_off", False):
        original = torch.utils.data.DataLoader.__init__

        def init(self, *a, **kw):
            kw["pin_memory"] = False
            return original(self, *a, **kw)

        torch.utils.data.DataLoader.__init__ = init
        torch.utils.data.DataLoader._pin_off = True
    from ultralytics import settings

    settings.update({k: False for k in ["sync", "wandb", "mlflow", "clearml", "comet"] if k in settings})
    # Import-time telemetry sampling consumes Python RNG; complete it before training seeds.
    import importlib
    importlib.import_module("ultralytics.utils.events")


def train(cfg, profile, seed):
    target = cfg["_artifacts"] / "retrain" / f"{profile}_s{seed}" / "weights" / "last.pt"
    if target.exists():
        raise RuntimeError(f"Existing checkpoint preserved: {target}. Use a separately registered output directory instead of overwriting.")
    runtime(cfg)
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds

    out = cfg["_artifacts"] / "retrain"
    data = write_lists(cfg, profile, seed, out)
    init_seeds(seed, deterministic=True)       # 새로 초기화되는 DFL 층보다 먼저(트레이너는 나중에 다시 시드)
    m = cfg["_root"] / cfg["train"]["model"]
    model = YOLO(str(m) if m.exists() else cfg["train"]["model"]).load(str(cfg["_root"] / cfg["train"]["init_weights"]))
    started = time.time()
    model.train(data=str(data), device=cfg["device"], project=str(out), name=f"{profile}_s{seed}", val=False, **train_args(cfg, seed))
    w = out / f"{profile}_s{seed}" / "weights" / "last.pt"
    rec = {"profile": profile, "seed": seed, "minutes": round((time.time() - started) / 60, 1),
           "weights": (w.relative_to(cfg["_root"]) if w.is_relative_to(cfg["_root"]) else w).as_posix()}
    print("학습 완료", rec, flush=True)
    return rec


def check(cfg):
    """학습 없이 재현 계약 확인. 하나라도 다르면 AssertionError."""
    import cv2

    man = json.loads((cfg["_root"] / "weights" / "MANIFEST.json").read_text(encoding="utf8"))
    root = cfg["_root"]
    assert sha(root / cfg["train"]["init_weights"]) == man["init_weights"]["sha256"], "초기값 불일치"
    assert sha(root / cfg["train"]["model"]) == man["model_yaml"]["sha256"], "모델 구성 불일치"
    assert XC.split(cfg)["sha256"] == man["split_sha256"], "분할 불일치"
    pix = json.loads((root / "data" / "v3_pixel_sha256.json").read_text(encoding="utf8"))["v3tr"]
    seen, rep = set(), {}
    for key, m in man["checkpoints"].items():
        prof, seed = m["profile"], m["seed"]
        assert sha(root / m["file"]) == m["sha256"], f"{key}: 가중치 해시 불일치"
        assert str(XC.checkpoint(cfg, prof, seed).relative_to(root).as_posix()) == m["file"], f"{key}: config 경로 불일치"
        train_s, held_s = stems(cfg, prof)
        assert train_s == m["training_stems"], f"{key}: 학습 목록 불일치"
        assert len(held_s) == m["held_images"], f"{key}: 보류 폴드 크기 불일치"
        mine = train_args(cfg, seed)
        rec = m["ultralytics_args"]
        diff = {k: (v, rec.get(k)) for k, v in mine.items() if k != "verbose" and rec.get(k) != v}
        assert not diff, f"{key}: 학습 인자 불일치 {diff}"
        assert rec.get("val") is False and str(rec.get("device")) == str(cfg["device"]), f"{key}: val/device 불일치"
        for s in train_s:
            if s in seen:
                continue
            g = cv2.cvtColor(cv2.imdecode(np.fromfile(str(root / "data" / "v3tr_orig" / "images" / f"{s}.png"), np.uint8),
                                          cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
            assert hashlib.sha256(np.ascontiguousarray(g).tobytes()).hexdigest() == pix[s], f"학습 영상 불일치: {s}"
            assert (root / "data" / "v3tr_orig" / "labels" / f"{s}.txt").read_bytes() == XC.label_path(cfg, s).read_bytes(), f"학습 라벨 불일치: {s}"
            seen.add(s)
        rep[key] = {"train_images": len(train_s), "held_images": len(held_s), "args_equal_keys": len(mine) - 1,
                    "sha256": m["sha256"][:16], "source_run": m["source_run"]}
    out = {"status": "PASS", "checked": rep, "training_images_pixel_and_label_checked": len(seen),
           "note": "학습 목록·학습 인자(ultralytics args.yaml)·초기값·모델 구성·분할·학습 영상 픽셀 해시·학습 라벨이 제출 가중치의 학습 기록과 일치. 재학습은 하지 않음."}
    p = cfg["_artifacts"] / "validation" / "TRAIN_CHECK.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf8")
    print("학습 재현 계약 PASS:", {k: v["train_images"] for k, v in rep.items()}, "학습 영상", len(seen), "장", flush=True)
    return out


def _cli():
    ap = argparse.ArgumentParser(description="DFL 폴드 검출기 학습(재현용)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--profile")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = XC.load(a.config)
    if a.check:
        check(cfg)
    elif a.all:
        for prof in [p for p, v in cfg["profiles"].items() if v.get("role", "deployment") == "deployment"]:
            for sd in cfg["profiles"][prof]["checkpoints"]:
                train(cfg, prof, int(sd))
    else:
        train(cfg, a.profile or cfg["detector"]["default_profile"], a.seed or cfg["detector"]["default_seed"])


if __name__ == "__main__":
    _cli()
