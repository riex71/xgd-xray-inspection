"""E10. Sign and distribution of centre errors of the frozen DFL (descriptive; spec §4 first step).

Reads the sealed follow-up raw prediction cache (base view, deployment predict settings) through analyze.raw (hash/signature checked),
applies the operating E0 4 px and score 0.25, and pairs predictions with TXT GT one-to-one within 8 px (diagnostic radius; 3 px stays the
success criterion). Reports signed dx/dy (prediction - GT, original pixels) by device, date, seed, object size, product position and
role (held fold vs. the profile's own training images). No threshold, label or rule is changed; nothing here is used for selection.

python -m experiments.e10_localization_bias   -> outputs/experiments/E10_localization_bias/RESULTS.json, SUMMARY.md
"""
import collections
import json
import math

import numpy as np

from experiments import lib as L
from experiments.followup import analyze as A
from experiments.followup import common as F

PROFILES = ("dfl_src12", "dfl_src3_2")


def pairs(boxes, gt, w, h, radius=8.0, thr=0.25):
    pred = sorted([b for b in boxes if b[4] >= thr], key=lambda b: -b[4])
    g = [((q[0] + q[2]) / 2 * w, (q[1] + q[3]) / 2 * h, (q[2] - q[0]) * w, (q[3] - q[1]) * h) for q in gt]
    used, out = set(), []
    for b in pred:
        cx, cy = (b[0] + b[2]) / 2 * w, (b[1] + b[3]) / 2 * h
        cand = [(math.hypot(cx - x, cy - y), k) for k, (x, y, _, _) in enumerate(g) if k not in used and math.hypot(cx - x, cy - y) <= radius]
        if cand:
            d, k = min(cand)
            used.add(k)
            out.append({"k": k, "dx": cx - g[k][0], "dy": cy - g[k][1], "err": d, "gw": g[k][2], "gh": g[k][3], "score": b[4],
                        "gx": g[k][0], "gy": g[k][1]})
    return out


def summary(rows):
    if not rows:
        return None
    dx, dy, e = (np.array([r[k] for r in rows]) for k in ("dx", "dy", "err"))
    return {"n": len(rows), "dx_median": float(np.median(dx)), "dy_median": float(np.median(dy)), "dx_mean": float(dx.mean()),
            "dy_mean": float(dy.mean()), "err_median": float(np.median(e)), "err_P90": float(np.percentile(e, 90)),
            "share_err_gt3": float(np.mean(e > 3)), "share_err_3to5": float(np.mean((e > 3) & (e <= 5)))}


def main():
    pm = L.product_mask()
    out = {"source": "outputs/followup/cache/pred (base view, sealed); E0 4 px, score >= 0.25, one-to-one 8 px diagnostic pairing",
           "success_criterion_unchanged": "3 px", "evaluation79_label_reads": 0, "profiles": {}}
    lines = ["# E10. 중심 오차의 부호와 분포 (동결 DFL, 개발 자료, 기술 통계)", "",
             "- 예측 − 정답(원본 px). 8px 안 일대일 짝만 포함한다(3px 성공 기준은 그대로). 선택·보정에 쓰지 않았다.", ""]
    for prof in PROFILES:
        held, train = set(L.targets(prof)), set(L.XC.profile_training_stems(L.CFG, prof))
        rows = []
        for sd in L.SEEDS:
            cache = A.raw("dfl", prof, sd, "base", "dev_clean")
            for s, e in cache.items():
                if s not in held and s not in train:
                    continue
                w, h = L.size_of(s)
                g8 = L.gray(L.clean_img(s))
                ys, xs = np.nonzero(pm(g8, empty=0))
                x0, x1 = (xs.min(), xs.max()) if len(xs) else (0, w)
                y0, y1 = (ys.min(), ys.max()) if len(ys) else (0, h)
                for p in pairs(F.e0(e["boxes"], w, h, 4.0), L.gt(s), w, h):
                    rows.append({**p, "seed": sd, "stem": s, "device": L.META[s]["device"], "date": L.META[s]["date"],
                                 "role": "held" if s in held else "training",
                                 "x_rel": (p["gx"] - x0) / max(1, x1 - x0), "y_rel": (p["gy"] - y0) / max(1, y1 - y0),
                                 "size_bin": "min side <10px" if min(p["gw"], p["gh"]) < 10 else "min side >=10px"})
        P = {}
        for role in ("held", "training"):
            rr = [r for r in rows if r["role"] == role]
            P[role] = {"all": summary(rr),
                       "by_device": {k: summary([r for r in rr if r["device"] == k]) for k in sorted({r["device"] for r in rr})},
                       "by_seed": {str(k): summary([r for r in rr if r["seed"] == k]) for k in L.SEEDS},
                       "by_date": {k: summary([r for r in rr if r["date"] == k]) for k in sorted({r["date"] for r in rr})},
                       "by_size": {k: summary([r for r in rr if r["size_bin"] == k]) for k in sorted({r["size_bin"] for r in rr})},
                       "by_x_rel": {k: summary([r for r in rr if (r["x_rel"] < .33, .33 <= r["x_rel"] < .66, r["x_rel"] >= .66).index(True) == i])
                                    for i, k in enumerate(("x<.33", ".33-.66", ">=.66"))},
                       "by_y_rel": {k: summary([r for r in rr if (r["y_rel"] < .33, .33 <= r["y_rel"] < .66, r["y_rel"] >= .66).index(True) == i])
                                    for i, k in enumerate(("y<.33", ".33-.66", ">=.66"))}}
            # sign consistency of the same object across seeds (systematic, seed-independent offsets)
            per_obj = collections.defaultdict(dict)
            for r in rr:
                per_obj[(r["stem"], r["k"])][r["seed"]] = (r["dx"], r["dy"], r["err"])
            full = [v for v in per_obj.values() if len(v) == 3]
            P[role]["objects_paired_in_all_seeds"] = len(full)
            P[role]["same_sign_dx_all_seeds"] = float(np.mean([len({np.sign(t[0]) for t in v.values()}) == 1 for v in full])) if full else None
            P[role]["same_sign_dy_all_seeds"] = float(np.mean([len({np.sign(t[1]) for t in v.values()}) == 1 for v in full])) if full else None
            big = [v for v in full if max(t[2] for t in v.values()) > 3]
            P[role]["objects_err_gt3_any_seed"] = len(big)
            P[role]["objects_err_gt3_all_seeds"] = sum(min(t[2] for t in v.values()) > 3 for v in full)
        out["profiles"][prof] = P
        a = P["held"]["all"]
        t = P["training"]["all"]
        lines += [f"## {prof} (보류 장비군 {', '.join(P['held']['by_device'])})", "",
                  f"- 보류: 짝 {a['n']}개, dx 중앙값 {a['dx_median']:+.2f}px, dy 중앙값 {a['dy_median']:+.2f}px, 오차 중앙값 {a['err_median']:.2f}px, "
                  f"P90 {a['err_P90']:.2f}px, 3px 초과 {a['share_err_gt3']:.1%}",
                  f"- 학습 영상(원천 장비): dx {t['dx_median']:+.2f}px, dy {t['dy_median']:+.2f}px, 오차 중앙값 {t['err_median']:.2f}px, 3px 초과 {t['share_err_gt3']:.1%}",
                  f"- 같은 객체가 세 seed 모두에서 같은 dx 부호: {P['held']['same_sign_dx_all_seeds']:.1%}, dy 부호: {P['held']['same_sign_dy_all_seeds']:.1%} "
                  f"(세 seed 모두 짝 있는 객체 {P['held']['objects_paired_in_all_seeds']}개; 3px 초과가 한 seed라도 있는 객체 {P['held']['objects_err_gt3_any_seed']}개, 세 seed 모두 {P['held']['objects_err_gt3_all_seeds']}개)"]
        for key, title in (("by_date", "촬영일"), ("by_seed", "seed"), ("by_size", "크기"), ("by_x_rel", "제품 내 x 위치"), ("by_y_rel", "제품 내 y 위치")):
            lines.append(f"- {title}별 (dx/dy 중앙값, 3px 초과): " + "; ".join(
                f"{k} {v['dx_median']:+.2f}/{v['dy_median']:+.2f}, {v['share_err_gt3']:.1%} (n={v['n']})" for k, v in P["held"][key].items() if v))
        lines.append("")
    lines += ["## 해석 제한", "", "- 정답 TXT 기준이다. 일정한 부호의 오차가 라벨 작성 관례인지 모델 편향인지는 이 분석으로 구분하지 않는다.",
              "- seed는 같은 영상의 반복이며 독립 표본이 아니다."]
    L.jdump(L.OUT / "E10_localization_bias" / "RESULTS.json", out)
    (L.OUT / "E10_localization_bias" / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("E10 written", flush=True)


if __name__ == "__main__":
    main()
