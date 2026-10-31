"""E11. Additional acquisition-quality signals (diagnostic only; spec §6 "소수만 시험").

Signals on the clean v3 image inside the eroded product mask (same region as the X12 signals):
  Q2, Q4 (operating X12 signals), Q4x / Q4y (directional lag-1 autocorrelation of the 1-px high-pass residual), Q4aniso = Q4x - Q4y,
  SBR = log(high-band / mid-band spectral energy) of the high-pass residual, SHARP = log(P90 gradient magnitude / product intensity IQR).
Per device, a multi-session reference (location = mean of session medians, scale = sqrt(within-session robust SD^2 + SD of session
medians^2), as X12) is built from all development clean images; clean false flags use leave-one-session-out references. Stress images
are the sealed follow-up STRESS_* sets (held folds). Failure association uses the frozen-DFL P0 rows of the sealed screen comparison
(seed 17, group-date design). Nothing is adopted; no threshold or rule changes.

python -m experiments.e11_quality_signals   -> outputs/experiments/E11_quality_signals/RESULTS.json, SUMMARY.md
"""
import collections
import json
import math

import cv2
import numpy as np

from experiments import lib as L
from experiments.followup import synth as S

SIG = ("Q2", "Q4", "Q4x", "Q4y", "Q4aniso", "SBR", "SHARP")


def signals(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = cv2.erode((L.product_mask()(g.astype(np.uint8), empty=0) > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if m.sum() < 500:
        m = np.ones_like(m, bool)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    mx, my = m[:, :-1] & m[:, 1:], m[:-1, :] & m[1:, :]
    cx = float(np.corrcoef(hp[:, :-1][mx], hp[:, 1:][mx])[0, 1])
    cy = float(np.corrcoef(hp[:-1, :][my], hp[1:, :][my])[0, 1])
    q2 = float(np.log(hp[m].var() / (g[m].var() + 1e-6) + 1e-9))
    ys, xs = np.nonzero(m)
    y0, x0 = int(np.median(ys)) - 32, int(np.median(xs)) - 32
    patch = hp[max(0, y0):max(0, y0) + 64, max(0, x0):max(0, x0) + 64]
    if patch.shape != (64, 64):
        patch = cv2.resize(patch, (64, 64))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(patch * np.outer(np.hanning(64), np.hanning(64))))) ** 2
    fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(64)), np.fft.fftshift(np.fft.fftfreq(64)), indexing="ij")
    r = np.hypot(fx, fy)
    sbr = float(np.log((spec[(r >= .25) & (r <= .5)].sum() + 1e-6) / (spec[(r >= .08) & (r < .25)].sum() + 1e-6)))
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    iqr = float(np.subtract(*np.percentile(g[m], [75, 25]))) + 1e-6
    sharp = float(np.log(np.percentile(grad[m], 90) / iqr + 1e-6))
    return {"Q2": q2, "Q4": (cx + cy) / 2, "Q4x": cx, "Q4y": cy, "Q4aniso": cx - cy, "SBR": sbr, "SHARP": sharp}


def reference(values, sessions):
    by = collections.defaultdict(list)
    for v, s in zip(values, sessions):
        by[s].append(v)
    med = {s: float(np.median(v)) for s, v in by.items()}
    resid = [v - med[s] for s, vv in by.items() for v in vv]
    within = 1.4826 * float(np.median(np.abs(resid))) if resid else 0.0
    between = float(np.std(list(med.values()), ddof=1)) if len(med) > 1 else 0.0
    return float(np.mean(list(med.values()))), math.sqrt(within ** 2 + between ** 2) or 1e-6


def auroc(pos, neg):
    if not pos or not neg:
        return None
    a = np.array(pos)[:, None]
    b = np.array(neg)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))


def main():
    dev = sorted(L.DEV)
    clean = {s: signals(L.clean_img(s)) for s in dev}
    out = {"signals": SIG, "evaluation79_label_reads": 0, "adopted": False, "devices": {}, "stress": {}}
    refs, loso_z = {}, {}
    for d in sorted({L.META[s]["device"] for s in dev}):
        ss = [s for s in dev if L.META[s]["device"] == d]
        refs[d] = {q: reference([clean[s][q] for s in ss], [L.META[s]["date"] for s in ss]) for q in SIG}
        for s in ss:
            others = [t for t in ss if L.META[t]["date"] != L.META[s]["date"]]
            loso_z[s] = {}
            for q in SIG:
                loc, sc = reference([clean[t][q] for t in others], [L.META[t]["date"] for t in others])
                loso_z[s][q] = (clean[s][q] - loc) / sc
        out["devices"][d] = {"images": len(ss), "clean_false_flag_rate_z3_LOSO": {q: float(np.mean([abs(loso_z[s][q]) > 3 for s in ss])) for q in SIG},
                             "reference": {q: {"loc": refs[d][q][0], "scale": refs[d][q][1]} for q in SIG}}
    # failure labels from the sealed screen comparison (frozen DFL, P0, seed 17)
    fail = {}
    for prof in ("dfl_src12", "dfl_src3_2"):
        comp = json.loads((L.ROOT / f"outputs/followup/comparisons/screen_dfl_P0_{prof}.json").read_text(encoding="utf8"))
        for cond, C in comp.items():
            for r in C["per_seed"]["17"]["rows"]:
                fail[(cond, r["stem"])] = r["fail"]
    zs, labels = collections.defaultdict(list), []
    for prof in ("dfl_src12", "dfl_src3_2"):
        for s in L.targets(prof):
            d = L.META[s]["device"]
            labels.append(("dev_clean", s, fail.get(("dev_clean", s))))
            for q in SIG:
                zs[q].append(abs(loso_z[s][q]))
        for cond in [c for c in S.CONDS if c.startswith("STRESS_")]:
            meta = S.verified_meta(prof, cond)
            flags = {q: [] for q in SIG}
            F = []
            for s, m in meta.items():
                if m is None:
                    continue
                d = L.META[s]["device"]
                v = signals(S.load_image(prof, cond, s))
                z = {q: (v[q] - refs[d][q][0]) / refs[d][q][1] for q in SIG}
                for q in SIG:
                    flags[q].append(abs(z[q]) > 3)
                    zs[q].append(abs(z[q]))
                labels.append((cond, s, fail.get((cond, s))))
                F.append(fail.get((cond, s)))
            key = f"{prof}|{cond}"
            out["stress"][key] = {"images": len(F), "P0_seed17_failure_rate": float(np.mean([x for x in F if x is not None])) if F else None,
                                  "flag_rate_z3": {q: float(np.mean(flags[q])) for q in SIG}}
    y = [l[2] for l in labels]
    out["failure_AUROC_abs_z"] = {q: auroc([z for z, t in zip(zs[q], y) if t == 1], [z for z, t in zip(zs[q], y) if t == 0]) for q in SIG}
    # over-blocking view: stress conditions whose failure rate stays <= 5% but get flagged
    low = {k: v for k, v in out["stress"].items() if v["P0_seed17_failure_rate"] is not None and v["P0_seed17_failure_rate"] <= .05}
    high = {k: v for k, v in out["stress"].items() if v["P0_seed17_failure_rate"] is not None and v["P0_seed17_failure_rate"] >= .2}
    out["mean_flag_rate_on_low_failure_stresses"] = {q: float(np.mean([v["flag_rate_z3"][q] for v in low.values()])) if low else None for q in SIG}
    out["mean_flag_rate_on_high_failure_stresses"] = {q: float(np.mean([v["flag_rate_z3"][q] for v in high.values()])) if high else None for q in SIG}
    out["low_failure_stresses"], out["high_failure_stresses"] = sorted(low), sorted(high)
    L.jdump(L.OUT / "E11_quality_signals" / "RESULTS.json", out)
    lines = ["# E11. 추가 품질 신호 진단 (채택 없음)", "",
             "- 신호: Q2·Q4(운영), Q4x·Q4y·Q4aniso(잡음 상관 방향성), SBR(고/중주파 에너지 비), SHARP(경계 선명도).",
             "- 기준: 장비별 개발 정상 영상의 다중 촬영일 기준(X12와 같은 방식). 정상 오경보는 자기 촬영일 제외(LOSO).",
             "- 실패: 봉인된 비교의 동결 DFL P0, seed 17, 날짜·그룹 분리 설계 행.", "",
             "| 신호 | 정상 오경보(|z|>3, 장비 평균) | 실패 적은 열화(≤5%)에서 경보 | 실패 많은 열화(≥20%)에서 경보 | |z|의 실패 AUROC |", "| --- | --- | --- | --- | --- |"]
    for q in SIG:
        ff = np.mean([v["clean_false_flag_rate_z3_LOSO"][q] for v in out["devices"].values()])
        lo, hi, au = out["mean_flag_rate_on_low_failure_stresses"][q], out["mean_flag_rate_on_high_failure_stresses"][q], out["failure_AUROC_abs_z"][q]
        lines.append(f"| {q} | {ff:.3f} | {lo:.3f} | {hi:.3f} | {au:.3f} |" if None not in (lo, hi, au) else f"| {q} | {ff:.3f} | {lo} | {hi} | {au} |")
    lines += ["", f"- 실패가 적은 열화(P0 seed17 실패율 ≤ 5%): {', '.join(out['low_failure_stresses'])}",
              f"- 실패가 많은 열화(≥ 20%): {', '.join(out['high_failure_stresses'])}", "",
              "## 해석 제한", "", "- 합성 열화이며 실제 장비 변화의 증명이 아니다. 품질 신호의 변화는 실패 확률이 아니며, 실패 확률로 쓰려면 라벨 있는 보정 근거가 필요하다.",
              "- 이 진단으로 운영 규칙을 바꾸지 않았다."]
    (L.OUT / "E11_quality_signals" / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("E11 written", flush=True)


if __name__ == "__main__":
    main()
