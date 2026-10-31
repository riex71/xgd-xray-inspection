"""E6. Data profile for the data-understanding criterion: production unit, device / date / resolution structure, labels, imbalance,
positions and sizes. Reads metadata, split, development labels and clean images only (never evaluation79 labels, never prediction caches).

python -m experiments.e6_data_profile   -> outputs/experiments/E6_data_profile/RESULTS.json, SUMMARY.md
"""
import collections
import json

import numpy as np

from experiments import lib as L


def q(v):
    v = np.asarray(v, float)
    return {"n": int(v.size), "min": float(v.min()), "p05": float(np.percentile(v, 5)), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())} if v.size else None


def main():
    recs = json.loads((L.ROOT / "data/metadata.json").read_text(encoding="utf8"))["records"]
    meta = {r["stem"]: r for r in recs}
    split = L.XC.split(L.CFG)
    out = {"production_unit": "one X-ray image of one finished product (one inspection); every development image contains >= 1 foreign object (NG images); no normal (object-free) product images",
           "evaluation79_label_reads": 0}
    out["images"] = {f"{r}|{d}": n for (r, d), n in sorted(collections.Counter((m["role"], m["device"]) for m in recs).items())}
    out["resolution"] = {f"{r}|{d}|{tuple(m['resolution'])}": n for (r, d, m_), n in []}
    res = collections.Counter((m["role"], m["device"], tuple(m["resolution"])) for m in recs)
    out["resolution"] = {f"{a}|{b}|{c[0]}x{c[1]}": n for (a, b, c), n in sorted(res.items())}
    out["dates"] = {d: dict(sorted(collections.Counter(m["date"] for m in recs if m["device"] == d and m["role"] == "development356").items()))
                    for d in sorted({m["device"] for m in recs})}
    out["evaluation79_dates"] = {d: dict(sorted(collections.Counter(m["date"] for m in recs if m["device"] == d and m["role"] == "test79").items()))
                                 for d in sorted({m["device"] for m in recs})}
    same_day = collections.defaultdict(dict)
    for d, c in out["dates"].items():
        for day, n in c.items():
            same_day[day][d] = n
    out["same_date_counts_across_devices"] = dict(sorted(same_day.items()))
    # labels (development only)
    per_dev = collections.defaultdict(lambda: {"objects_per_image": [], "cx": [], "cy": [], "w_px": [], "h_px": [], "cx_rel_product": [],
                                               "cy_rel_product": [], "decimals": collections.Counter(), "classes": collections.Counter()})
    border = 0
    for s in L.DEV:
        m = meta[s]
        w, h = m["resolution"]
        lines = [l for l in L.XC.label_path(L.CFG, s).read_text(encoding="utf8").splitlines() if l.strip()]
        D = per_dev[m["device"]]
        D["objects_per_image"].append(len(lines))
        g8 = L.gray(L.clean_img(s))
        pm = L.product_mask()(g8, empty=0)
        ys, xs = np.nonzero(pm)
        for l in lines:
            parts = l.split()
            D["classes"][parts[0]] += 1
            D["decimals"][max(len(p.split(".")[1]) if "." in p else 0 for p in parts[1:])] += 1
            cx, cy, bw, bh = map(float, parts[1:])
            D["cx"].append(cx)
            D["cy"].append(cy)
            D["w_px"].append(bw * w)
            D["h_px"].append(bh * h)
            if len(xs):
                D["cx_rel_product"].append((cx * w - xs.min()) / max(1, xs.max() - xs.min()))
                D["cy_rel_product"].append((cy * h - ys.min()) / max(1, ys.max() - ys.min()))
            border += int(cx - bw / 2 <= 0 or cy - bh / 2 <= 0 or cx + bw / 2 >= 1 or cy + bh / 2 >= 1)
    out["labels"] = {d: {"images": len(D["objects_per_image"]), "objects": int(sum(D["objects_per_image"])),
                         "objects_per_image": dict(sorted(collections.Counter(D["objects_per_image"]).items())),
                         "class_ids": dict(D["classes"]), "decimal_places": dict(D["decimals"]),
                         "center_x_norm": q(D["cx"]), "center_y_norm": q(D["cy"]), "center_x_rel_product": q(D["cx_rel_product"]),
                         "center_y_rel_product": q(D["cy_rel_product"]), "box_w_px": q(D["w_px"]), "box_h_px": q(D["h_px"])}
                     for d, D in sorted(per_dev.items())}
    out["boxes_touching_image_border"] = border
    out["missing_or_excluded"] = {"hold65": "not included (original TXT byte provenance not re-certified for this package)",
                                  "evaluation79_labels": "not included and never read; predictions only",
                                  "normal_product_images": 0, "physical_specimen_id": "not provided",
                                  "physical_magnification_or_pixel_pitch": "not provided", "object_material_or_density": "not provided"}
    out["splits"] = {k: dict(collections.Counter(v["assignment"].values())) for k, v in split["schemes"].items()}
    out["repeated_use"] = "development356 has been analysed repeatedly since 2026-09-21 (model selection, X-series audits); all results are internal development evidence"
    L.jdump(L.OUT / "E6_data_profile" / "RESULTS.json", out)
    lines = ["# E6. 데이터 프로파일 (개발 356장 라벨 + 메타데이터)", "",
             f"- 생산 단위: {out['production_unit']}", f"- 영상 수(역할|장비): {out['images']}", f"- 해상도: {out['resolution']}",
             f"- 촬영일별 영상 수(장비): {out['dates']}", f"- 같은 날짜의 장비별 영상 수: {out['same_date_counts_across_devices']}",
             f"- 평가 79장 촬영일: {out['evaluation79_dates']}", f"- 경계에 닿은 박스: {border}", ""]
    for d, v in out["labels"].items():
        lines.append(f"- {d}: 영상 {v['images']}, 이물 {v['objects']}, 영상당 이물 수 {v['objects_per_image']}, 소수 자릿수 {v['decimal_places']}, "
                     f"중심 x(제품 기준) {v['center_x_rel_product']}, 박스 폭 px {v['box_w_px']}")
    (L.OUT / "E6_data_profile" / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("E6 data profile written", flush=True)


if __name__ == "__main__":
    main()
