"""평가 79장 예측 결과 파일 (data/split.csv 의 partition = test). 정답 라벨은 폴더에 없고 읽지 않으며, 점수도 매기지 않는다.

입력: 원본 BMP (검사기 표시 제거를 분석기 안에서 수행 = 현장 경로). 장비·촬영일은 data/metadata.json.
profile: 장비마다 CRC 보정 정보가 있는 profile (config deployment.routing) — 3호기 = dfl_src12, 1·2호기 = dfl_src3_2.
seed: config detector.default_seed (17). 판정 규칙·임계값은 config 그대로(평가셋을 보고 바꾼 값 없음).

출력 (outputs/):
  test_predictions_images.csv        영상별 판정(ACCEPT / REVIEW / FULL_REINSPECTION), 자동 판정 결과, 검출 수, 최고 점수, 품질 z, CRC 임계값 등
  test_predictions_boxes.csv         운영 점수(0.25) 이상 검출의 픽셀 좌표·중심·점수
  test_predictions_yolo/<영상>.txt   같은 검출을 제공 라벨 형식(class cx cy w h, 정규화) + 점수로
  test_predictions.json              영상별 분석기 전체 출력(판정 이유 포함)

  python -m src.predict_test
"""
import collections
import csv
import json
import sys
from pathlib import Path

try:
    from . import xgd_analyzer as XAN
    from . import xgd_config as XC
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import xgd_analyzer as XAN
    import xgd_config as XC

ACTION = {"ACCEPT": "자동 판정 확정", "REVIEW": "작업자 확인(재검사 대기열)", "FULL_REINSPECTION": "전수 재검사(자동 판정 중단)"}


def route(cfg):
    """장비 -> profile: 그 장비의 CRC 보정 정보가 있는 profile (여럿이면 default_profile)."""
    cover = collections.defaultdict(list)
    for prof in [p for p, v in cfg["profiles"].items() if v.get("role", "deployment") == "deployment"]:
        p = cfg["_artifacts"] / "calibration" / f"crc_{prof}.json"
        if p.exists():
            for d in json.loads(p.read_text(encoding="utf8"))["devices"]:
                cover[d].append(prof)
    default = cfg["detector"]["default_profile"]
    return {d: (default if default in ps else ps[0]) for d, ps in cover.items()}


def _csv(path, rows, fields):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run(config=None):
    cfg = XC.load(config)
    out = cfg["_artifacts"]
    meta = XC.provider_metadata(cfg)
    stems = XC.test_stems(cfg)
    routing = route(cfg)
    seed = int(cfg["detector"]["default_seed"])
    thr = cfg["detector"]["decision_threshold"]
    groups = collections.defaultdict(list)
    for s in stems:
        groups[routing.get(meta[s]["device"], cfg["detector"]["default_profile"])].append(s)
    res, img_rows, box_rows = {}, [], []
    yolo = out / "test_predictions_yolo"
    yolo.mkdir(parents=True, exist_ok=True)
    for prof, ss in groups.items():
        an = XAN.XGDAnalyzer(cfg, prof, seed)
        for s in ss:
            md = {"calibration_id": meta[s]["device"], "session": meta[s]["date"], "preprocessed": False}
            r = an.analyze(image_path=XC.raw_path(cfg, s), metadata=md, raw_boxes=True)
            cand = r.pop("_boxes_normalized_e0")
            res[s] = {"device": meta[s]["device"], "date": meta[s]["date"], **r}
            w, h = r["scale"]["original_resolution"]
            a, q = r["assurance"] or {}, r["quality"]
            img_rows.append({
                "image": s, "device": meta[s]["device"], "date": meta[s]["date"], "width": w, "height": h,
                "profile": prof, "seed": seed, "inference_imgsz": r["scale"]["imgsz"], "scale_normalized": r["scale"]["normalized"],
                "quality_state": q["state"], "quality_max_abs_z": None if q["max_abs_z"] is None else round(q["max_abs_z"], 4),
                "calibration_n": a.get("calibration_n"), "calibration_level": a.get("level"), "crc_lambda": a.get("crc_lambda"),
                "trust": round(r["trust"], 6), "object_count": r["object_count"],
                "top_score": round(max((b[4] for b in cand), default=0.0), 6),
                "decision": r["decision"],
                "detections_confirmed": r["detections_confirmed"],
                "candidate_result": "이물 후보 있음" if r["object_count"] else "검출 후보 없음",
                "auto_result": ("검출 위치 자동 확정" if r["detections_confirmed"] else
                                "미확정 이물 후보 있음" if r["object_count"] else "검출 후보 없음(미확정)") ,
                "action": ACTION[r["decision"]], "reason": " | ".join(r["reason"])})
            lines = []
            for d in r["detections"]:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                box_rows.append({"image": s, "decision": r["decision"], "confirmed": r["detections_confirmed"],
                                 "x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2),
                                 "cx": round(d["center"][0], 2), "cy": round(d["center"][1], 2), "confidence": round(d["confidence"], 6)})
                lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f} {d['confidence']:.6f}")
            (yolo / f"{s}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf8")
        del an
    img_rows.sort(key=lambda r: r["image"])
    _csv(out / "test_predictions_images.csv", img_rows, list(img_rows[0]))
    _csv(out / "test_predictions_boxes.csv", sorted(box_rows, key=lambda r: (r["image"], -r["confidence"])),
         ["image", "decision", "confirmed", "x1", "y1", "x2", "y2", "cx", "cy", "confidence"])
    summary = {"n_images": len(stems), "routing": routing, "seed": seed, "decision_threshold": thr,
               "decisions": dict(collections.Counter(r["decision"] for r in img_rows)),
               "by_device": {d: dict(collections.Counter(r["decision"] for r in img_rows if r["device"] == d))
                             for d in sorted({r["device"] for r in img_rows})},
               "auto_result": dict(collections.Counter(r["auto_result"] for r in img_rows)),
               "note": "평가 79장의 정답 라벨은 폴더에 없고 읽지 않았다. 판정 규칙·임계값은 config 그대로이며 이 결과로 바꾸지 않는다."}
    (out / "test_predictions.json").write_text(json.dumps({"summary": summary, "images": res}, ensure_ascii=False, indent=1, default=float),
                                               encoding="utf8")
    print("평가 79장 예측 완료", json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="평가 79장 예측 결과 파일")
    ap.add_argument("--config")
    run(ap.parse_args().config)
