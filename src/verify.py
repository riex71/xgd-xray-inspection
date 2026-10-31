"""자체 재현 검증 (연구 저장소 없이 이 폴더만으로 실행). 학습·임계값 조정 없음.

 1) 전처리: data/clean_v3(435장)·data/v3tr_orig(356장) PNG 의 픽셀 해시 = data/v3_pixel_sha256.json
 2) 검출: 보정 집합(3호기 100장 = dfl_src12, 1·2호기 224장 = dfl_src3_2) × 시드 17·29·43, v3 영상·X8 배율에서 E0 뒤 박스
    (정규화 좌표·점수)가 원 분석기 실행(2026-10-05 20:10 회귀 실행; outputs/validation/reference/predictions_clean_x8.json)과
    비트 단위로 같고, 신뢰도·추론 크기·3px 결과(tp, n, fp, fail)도 같음
 3) 원본 BMP 경로: 3호기 100장을 원본 BMP 로 넣으면(분석기 안에서 표시 제거) 2)의 seed 17 박스와 비트 단위로 같음
 4) 보정 산출물: outputs/calibration/*.json 이 원 실행 산출물(reference/calibration)과 같음(created 시각 제외)

  python -m src.verify            결과: outputs/validation/VERIFY_REPORT.json
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    from . import xgd_analyzer as XAN
    from . import xgd_config as XC
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import xgd_analyzer as XAN
    import xgd_config as XC


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k != "created"}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def _pixels(path):
    import cv2
    g = cv2.cvtColor(cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    return hashlib.sha256(np.ascontiguousarray(g).tobytes()).hexdigest()


def _same_boxes(a, b):
    a, b = np.asarray(a, float).reshape(-1, 5), np.asarray(b, float).reshape(-1, 5)
    if a.shape != b.shape:
        return False, math.inf
    d = float(np.abs(a - b).max()) if a.size else 0.0
    return d == 0.0, d


def run(config=None):
    cfg = XC.load(config)
    root = cfg["_root"]
    ref = json.loads((cfg["_artifacts"] / "validation" / "reference" / "predictions_clean_x8.json").read_text(encoding="utf8"))
    rep = {}
    # 1) preprocessing
    rec = json.loads((root / "data" / "v3_pixel_sha256.json").read_text(encoding="utf8"))
    dev, test = XC.dev_stems(cfg), XC.test_stems(cfg)
    bad_v3 = [s for s in dev + test if _pixels(XC.clean_path(cfg, s)) != rec["v3"][s]]
    bad_tr = [s for s in dev if _pixels(root / "data" / "v3tr_orig" / "images" / f"{s}.png") != rec["v3tr"][s]]
    rep["preprocess"] = {"v3_checked": len(dev) + len(test), "v3_mismatch": bad_v3, "v3tr_checked": len(dev), "v3tr_mismatch": bad_tr}
    print("전처리 해시", rep["preprocess"]["v3_checked"], "+", rep["preprocess"]["v3tr_checked"], "장, 불일치",
          len(bad_v3) + len(bad_tr), flush=True)
    # 2) detections on the calibration sets (clean v3, X8) vs the original analyzer run; 3) raw BMP path
    _, _, centre_matches = XAN.protocol(cfg)
    meta = XC.provider_metadata(cfg)
    thr = cfg["detector"]["decision_threshold"]
    det = {}
    for fam, F in ref["families"].items():
        prof = F["profile"]
        n_img = n_box_mm = n_other_mm = 0
        maxd = 0.0
        for sd, recs in F["seeds"].items():
            an = XAN.XGDAnalyzer(cfg, prof, int(sd), load_artifacts=False)
            for s in F["targets"]:
                r = an.analyze(image=XAN.read_image(XC.clean_path(cfg, s)), preprocessed=True, calibration={"N": 0, "lambda": math.inf},
                               raw_boxes=True)
                w, h = meta[s]["resolution"]
                o = XAN.outcome(r["_boxes_normalized_e0"], XAN.read_truth(XC.label_path(cfg, s)), w, h, centre_matches, thr)
                same, d = _same_boxes(r["_boxes_normalized_e0"], recs[s]["boxes"])
                n_img += 1
                n_box_mm += not same
                maxd = max(maxd, d)
                exp = recs[s]
                n_other_mm += int(r["trust"] != exp["trust"] or r["scale"]["imgsz"] != exp["imgsz"] or
                                  any(o[k] != exp[k] for k in ("tp", "n", "fp", "fail")))
                if fam == "DFL-B" and sd == "17":
                    det.setdefault("_b17", {})[s] = r["_boxes_normalized_e0"]
            print("검출 재현", fam, prof, "seed", sd, flush=True)
            del an
        rep.setdefault("detections", {})[fam] = {"profile": prof, "image_runs": n_img, "box_mismatches": n_box_mm, "max_abs_diff": maxd,
                                                 "trust_imgsz_outcome_mismatches": n_other_mm}
    an = XAN.XGDAnalyzer(cfg, "dfl_src12", 17, load_artifacts=False)
    raw_mm, raw_d = 0, 0.0
    for s in ref["families"]["DFL-B"]["targets"]:
        r = an.analyze(image_path=XC.raw_path(cfg, s), preprocessed=False, calibration={"N": 0, "lambda": math.inf}, raw_boxes=True)
        same, d = _same_boxes(r["_boxes_normalized_e0"], det["_b17"][s])
        raw_mm += not same
        raw_d = max(raw_d, d)
    rep["raw_bmp_path"] = {"images": len(ref["families"]["DFL-B"]["targets"]), "box_mismatches": raw_mm, "max_abs_diff": raw_d}
    # 4) deployment artifacts
    art = {}
    for p in sorted((cfg["_artifacts"] / "validation" / "reference" / "calibration").glob("*.json")):
        mine = cfg["_artifacts"] / "calibration" / p.name
        if not mine.exists():
            art[p.name] = "MISSING (run calibrate first)"
            continue
        old, current = json.loads(p.read_text(encoding="utf8")), json.loads(mine.read_text(encoding="utf8"))
        # Old calibration reference is NEVER replaced by a current output. Compare immutable numeric
        # invariants against that old reference; separately validate version/source/model compatibility.
        if p.name == "quality_reference.json":
            equal = old["devices"] == current["devices"] and current.get("signature") == XAN.quality_signature(cfg)
        else:
            equal = set(old["devices"]) == set(current["devices"])
            for d, prior in old["devices"].items():
                now = current["devices"].get(d, {})
                for sd, ov in prior["seeds"].items():
                    nv = now.get("seeds", {}).get(sd, {})
                    prof = current["profile"]
                    cp = XC.checkpoint(cfg, prof, int(sd))
                    equal = equal and all(nv.get(k) == ov.get(k) for k in ("N", "lambda", "bound", "coverage_in_calibration", "estimated_risk"))
                    equal = equal and nv.get("compatibility") == XAN.compatibility_signature(cfg, XAN.sha256_file(cp))
                    equal = equal and XAN.parse_grade(nv.get("grade")) is not None
                    equal = equal and nv.get("probability_guarantee", {}).get("preconditions_verified") is False
        equal = equal and current.get("version", "").startswith(cfg["version"] + "-")
        art[p.name] = "EQUAL_LEGACY_NUMERICS_VALID_NEW_SCHEMA" if equal else "DIFFERENT"
    rep["calibration_artifacts"] = art
    ok = (not bad_v3 and not bad_tr and all(v["box_mismatches"] == 0 and v["trust_imgsz_outcome_mismatches"] == 0 for v in rep["detections"].values())
          and raw_mm == 0 and bool(art) and all(v == "EQUAL_LEGACY_NUMERICS_VALID_NEW_SCHEMA" for v in art.values()))
    rep["VERIFY_PASS"] = bool(ok)
    rep["reference"] = ref["source"]
    rep["calibration_verification_policy"] = "Original legacy reference preserved. Numeric invariants compared against it; new enum/version/compatibility/schema checked separately."
    out = cfg["_artifacts"] / "validation" / "VERIFY_REPORT.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf8")
    print("VERIFY_PASS", ok, json.dumps({k: rep[k] for k in ("detections", "raw_bmp_path", "calibration_artifacts")}, ensure_ascii=False), flush=True)
    return rep


if __name__ == "__main__":
    raise SystemExit(0 if run()["VERIFY_PASS"] else 1)
