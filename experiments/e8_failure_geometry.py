"""E8. Geometry of the clean-condition image failures of the final DFL (3 seeds, development held folds).

For every GT missed at 3 px: distance from its centre to the nearest prediction (score >= 0.25). For every unmatched prediction: distance to
the nearest GT centre. Fixed bins, descriptive only (docs/PREREG_E8_FAILURE_GEOMETRY.md, recorded 2026-10-06 01:22 KST before this ran).
No threshold, label, failure definition or rule change. Predictions come from the sealed E2 clean caches.

python -m experiments.e8_failure_geometry   -> outputs/experiments/E8_failure_geometry/RESULTS.json, SUMMARY.md
"""
import collections
import math

from experiments import lib as L
from experiments.e3_score import load_cache
from src.metrics import centre_matches

THR = L.CFG["detector"]["decision_threshold"]


def miss_bin(d):
    return "3-5px" if d is not None and d <= 5 else ("5-8px" if d is not None and d <= 8 else ">8px_or_none")


def fp_bin(d):
    return "<=5px" if d is not None and d <= 5 else ("5-8px" if d is not None and d <= 8 else ">8px")


def analyse(boxes, g, w, h):
    gc = [((q[0] + q[2]) / 2 * w, (q[1] + q[3]) / 2 * h) for q in g]
    rows = centre_matches({"x": boxes}, {"x": g}, {"x": (w, h)}, 3.0)
    preds = sorted(boxes, key=lambda p: -p[4])[:30]
    matched_gt = {k for sc, hit, _, k in rows if hit and sc >= THR}
    pc = [((p[0] + p[2]) / 2 * w, (p[1] + p[3]) / 2 * h) for p in preds if p[4] >= THR]
    hit_flags = [hit for sc, hit, _, k in rows if sc >= THR]
    fps = [c for c, hit in zip(pc, hit_flags) if not hit]
    misses = [gc[k] for k in range(len(gc)) if k not in matched_gt]
    md = [min((math.hypot(x - a, y - b) for a, b in pc), default=None) for x, y in misses]
    fd = [min((math.hypot(x - a, y - b) for a, b in gc), default=None) for x, y in fps]
    return md, fd


def image_type(md, fd):
    if not md and not fd:
        return None
    if md and not fd:
        return "FN_only"
    if fd and not md:
        return "FP_only"
    if all(d is not None and d <= 8 for d in md) and all(d is not None and d <= 8 for d in fd):
        return "localization_only"
    return "mixed"


def main():
    out = {"prereg": "docs/PREREG_E8_FAILURE_GEOMETRY.md (2026-10-06 01:22 KST)", "failure_definition": "unchanged (3 px, score >= 0.25)",
           "profiles": {}}
    lines = ["# E8. 정상 조건 실패의 위치 기하 (최종 DFL, 3 seed, 개발 보류 폴드)", "",
             f"- 사전 등록: {out['prereg']}. 기술 통계이며 실패 정의·라벨·규칙은 바꾸지 않았다.", ""]
    for prof in ("dfl_src12", "dfl_src3_2"):
        P = {"per_seed": {}}
        tot_t, tot_m, tot_f = collections.Counter(), collections.Counter(), collections.Counter()
        for sd in L.SEEDS:
            cache = load_cache(f"{prof}_s{sd}", "clean")
            types, mb, fb, failed = collections.Counter(), collections.Counter(), collections.Counter(), []
            for s in L.targets(prof):
                w, h = L.size_of(s)
                md, fd = analyse(cache[s]["boxes"], L.gt(s), w, h)
                t = image_type(md, fd)
                if t is None:
                    continue
                types[t] += 1
                for d in md:
                    mb[miss_bin(d)] += 1
                for d in fd:
                    fb[fp_bin(d)] += 1
                failed.append({"stem": s, "type": t, "missed_gt_nearest_pred_px": md, "fp_nearest_gt_px": fd})
            P["per_seed"][str(sd)] = {"failed_images": sum(types.values()), "image_types": dict(types), "missed_gt_bins": dict(mb),
                                      "fp_bins": dict(fb), "failures": failed}
            tot_t.update(types)
            tot_m.update(mb)
            tot_f.update(fb)
        P["all_seeds"] = {"failed_images": sum(tot_t.values()), "image_types": dict(tot_t), "missed_gt_bins": dict(tot_m), "fp_bins": dict(tot_f)}
        out["profiles"][prof] = P
        lines += [f"## {prof}", "", f"- 3 seed 합: 실패 영상 {P['all_seeds']['failed_images']}장, 유형 {P['all_seeds']['image_types']}",
                  f"- 놓친 정답의 가장 가까운 예측 거리: {P['all_seeds']['missed_gt_bins']}", f"- 오탐의 가장 가까운 정답 거리: {P['all_seeds']['fp_bins']}"]
        for sd, v in P["per_seed"].items():
            lines.append(f"- seed {sd}: " + "; ".join(f"{x['stem']} {x['type']} 놓침→예측 {[round(d, 2) if d is not None else None for d in x['missed_gt_nearest_pred_px']]} "
                                                      f"오탐→정답 {[round(d, 2) if d is not None else None for d in x['fp_nearest_gt_px']]}" for x in v["failures"]))
        lines.append("")
    lines += ["## 해석 제한", "", "- 정답 TXT 기준이다. 3~5px 위치 오차가 라벨 중심 오차인지 모델 오차인지는 이 분석으로 구분하지 않는다."]
    L.jdump(L.OUT / "E8_failure_geometry" / "RESULTS.json", out)
    (L.OUT / "E8_failure_geometry" / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("E8 written", {p: v["all_seeds"] for p, v in out["profiles"].items()}, flush=True)


if __name__ == "__main__":
    main()
