"""Frozen copies of the sealed evaluators used by the X14 regression (tools/regression_x14.py stage B).

Every function below is a verbatim copy, generated from the research-repository source files (SHA256 of each source file and of each
function text: x14_records/SOURCES.json). Only the data they read is redirected to x14_records/ (development-set targets only;
no hold65 / test79 labels). This lets the regression run without the research repository, while stage B still checks the analyzer
against independent, sealed evaluator code rather than against the analyzer's own functions.
"""
import collections
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent
REC = HERE / "x14_records"
R = REC                                    # phase_g_evaluate.score reads R / META[s]["label"]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))
from src.metrics import centre_matches     # verbatim copy of the protocol centre_matches (src/metrics.py)


def read(p):                               # = phase_g_provider_audit.read
    return json.loads(Path(p).read_text(encoding="utf8"))


def image_sizes():                         # = protocol common.image_sizes (data/split.csv)
    with open(MODEL / "data" / "split.csv", encoding="utf-8-sig") as f:
        return {r["stem"]: (int(r["width"]), int(r["height"])) for r in csv.DictReader(f)}


META = read(REC / "META_targets.json")["records"]


# --- 4번 분석 모델/src/metrics.py (2026-10-05부터 00_과거버전_보관/4번 분석 모델/)


def partial_froc_auc(curve, limit=1.0):
    """영상당 오탐 1/128~1까지 로그 간격 8개 지점의 평균 민감도 (주 지표)."""
    f = np.asarray(curve["fp_per_image"])
    s = np.asarray(curve["sensitivity"])
    vals = []
    for point in np.geomspace(max(limit / 128, 1e-3), limit, 8):
        ok = np.nonzero(f <= point)[0]
        vals.append(float(s[ok].max()) if len(ok) else 0.0)
    return float(np.mean(vals))

# --- 4번 분석 모델/src/validation/evaluate.py (2026-10-05부터 00_과거버전_보관/4번 분석 모델/)


def _blocks(rows):
    """점수 내림차순 매칭 결과를 같은 점수 묶음으로 나눈다(임계값은 묶음 단위로만 움직일 수 있다)."""
    out, cur = [], []
    for r in rows:
        if cur and r[0] != cur[-1][0]:
            out.append(cur)
            cur = []
        cur.append(r)
    return out + ([cur] if cur else [])


def froc_rows(rows, n_obj, n_img):
    """동점 묶음 단위 FROC(묶음 끝에서만 점을 찍는다 — 같은 점수 안의 정답·오탐 순서에 결과가 좌우되지 않게)."""
    th, sens, fpi, tp, fp = [], [], [], 0, 0
    for blk in _blocks(rows):
        tp += sum(1 for r in blk if r[1])
        fp += sum(1 for r in blk if not r[1])
        th.append(blk[0][0]), sens.append(tp / max(n_obj, 1)), fpi.append(fp / max(n_img, 1))
    return {"threshold": th, "sensitivity": sens, "fp_per_image": fpi}


def pair_errors(pred, gt, stems, threshold, radius=8.0, mode="score"):
    """정답별 짝 예측의 중심 오차 (오차, dx, dy) 목록과 정답 수.
    mode='score'(1차, 2026-09-28 P0): 임계값 이상 예측을 점수 내림차순으로 보며 반경 안의 아직 짝 없는 가장 가까운 정답과
      일대일로 묶는다 — 3px 맞힘 판정(metrics.centre_matches, 영상당 상위 30개)과 같은 규칙, 반경만 다름.
    mode='nearest'(옛 정의, 진단용): 정답마다 가장 가까운 예측(한 예측이 여러 정답에 쓰일 수 있음)."""
    sizes = image_sizes()
    out, n = [], 0
    for s in stems:
        w, h = sizes[s]
        gs = [((g[0] + g[2]) / 2 * w, (g[1] + g[3]) / 2 * h) for g in gt[s]]
        n += len(gs)
        ps = sorted([p for p in pred.get(s, []) if p[4] >= threshold], key=lambda p: -p[4])
        cs = [((p[0] + p[2]) / 2 * w, (p[1] + p[3]) / 2 * h) for p in ps]
        if mode == "nearest":
            for gx, gy in gs:
                if cs:
                    x, y = min(cs, key=lambda c: np.hypot(c[0] - gx, c[1] - gy))
                    if np.hypot(x - gx, y - gy) <= radius:
                        out.append((float(np.hypot(x - gx, y - gy)), x - gx, y - gy))
            continue
        used = set()
        for x, y in cs[:30]:
            near = [(np.hypot(x - gx, y - gy), k) for k, (gx, gy) in enumerate(gs) if k not in used and np.hypot(x - gx, y - gy) <= radius]
            if near:
                e, k = min(near)
                used.add(k)
                out.append((float(e), x - gs[k][0], y - gs[k][1]))
    return out, n

# --- analysis/phase_g_h0_audit.py
SITE_R, MATCH_R, THR = 8.0, 8.0, .25


def image_counts(preds, gt, wh, shift=(0, 0)):
    """3px one-to-one score-order counts for one image (same rule as centre_matches), plus <=8px matched centres per GT index."""
    w, h = wh
    g = [((b[0] + b[2]) / 2 * w + shift[0], (b[1] + b[3]) / 2 * h + shift[1]) for b in gt]
    rows = centre_matches({"x": preds}, {"x": [[(c[0] - .5) / w, (c[1] - .5) / h, (c[0] + .5) / w, (c[1] + .5) / h] for c in g]},
                          {"x": (w, h)}, 3.0) if g else []
    tp = sum(1 for z in rows if z[1] and z[0] >= THR)
    fp = sum(1 for p in preds if p[4] >= THR) - tp
    used, cen = set(), {}
    for p in sorted([p for p in preds if p[4] >= THR], key=lambda p: -p[4]):
        x, y = (p[0] + p[2]) / 2 * w, (p[1] + p[3]) / 2 * h
        d = sorted((math.hypot(x - c[0], y - c[1]), i) for i, c in enumerate(g) if i not in used)
        if d and d[0][0] <= MATCH_R:
            used.add(d[0][1])
            cen[d[0][1]] = (x - shift[0], y - shift[1])
    return tp, len(g), fp, cen, rows


A = SimpleNamespace(image_counts=image_counts, THR=THR, MATCH_R=MATCH_R)

# --- analysis/assurance_pipeline_20261004.py
COVERAGE = [round(x, 2) for x in np.arange(.1, 1.01, .1)]
ALPHAS, DELTA, SPLITS, RNG = [.05, .10], .10, 200, 20261004
TRUTH = read(REC / "TRUTH_baseline_targets.json")["baseline"]
WH = {s: tuple(META[s]["resolution"]) for s in META}


def per_image(preds, stems):
    """tp, n_gt, fp, matched errors (<=8px one-to-one), S1 image confidence, detections>=.25 (px centres)."""
    rec = {}
    for s in stems:
        w, h = WH[s]
        gt = TRUTH[s]
        tp, n, fp, cen, _ = A.image_counts(preds[s], gt, (w, h))
        err = []
        for i, c in cen.items():
            g = gt[i]
            err.append(math.hypot(c[0] - (g[0] + g[2]) / 2 * w, c[1] - (g[1] + g[3]) / 2 * h))
        dets = [((p[0] + p[2]) / 2 * w, (p[1] + p[3]) / 2 * h, p[4]) for p in preds[s] if p[4] >= A.THR]
        rec[s] = {"tp": tp, "n": n, "fp": fp, "err": err, "cen": cen, "conf": float(np.mean([d[2] for d in dets])) if dets else 0.0, "dets": dets,
                  "fail": int(tp < n or fp > 0)}
    return rec


def crc_threshold(trust_cal, fail_cal, alpha):
    """Conformal risk control on L(lam)=1{trust>=lam and failure} (non-increasing in lam, bound B=1).
    Accept the top-k calibration images by trust; the largest k with (cumfail_k + 1)/(n+1) <= alpha gives lam = k-th trust; inf = accept nothing."""
    t, f = np.asarray(trust_cal, float), np.asarray(fail_cal, float)
    n = len(t)
    order = np.argsort(-t, kind="stable")
    cf = np.cumsum(f[order])
    ok = np.nonzero(cf <= alpha * (n + 1) - 1)[0]
    return float(t[order][ok[-1]]) if len(ok) else math.inf


def ltt_threshold(trust_cal, fail_cal, alpha, delta=DELTA):
    """Learn-then-Test with Bonferroni over the 10-point coverage grid: reject H_c: P(failure | accepted top-c) > alpha when the exact
    binomial p-value <= delta/10; return the threshold of the largest rejected coverage (inf if none)."""
    t, f = np.asarray(trust_cal, float), np.asarray(fail_cal, float)
    n = len(t)
    order = np.argsort(-t, kind="stable")
    cf = np.cumsum(f[order])
    best = math.inf
    for c in COVERAGE:
        k = max(1, int(round(c * n)))
        if stats.binom.cdf(cf[k - 1], k, alpha) <= delta / len(COVERAGE):
            best = float(t[order][k - 1])
    return best


def evaluate_threshold(trust, fail, lam):
    t, f = np.asarray(trust, float), np.asarray(fail, float)
    acc = t >= lam
    k = int(acc.sum())
    return {"coverage": k / len(t), "joint_false_safe_risk": float(f[acc].sum() / len(t)), "conditional_failure_risk": float(f[acc].mean()) if k else None}


def crc_ltt(rec, stems, trust):
    dates = collections.defaultdict(list)
    for s in stems:
        dates[META[s]["date"] + META[s]["device"]].append(s)
    keys = sorted(dates)
    gen = np.random.default_rng(RNG)
    out = {}
    for alpha in ALPHAS:
        rows = {"CRC": [], "LTT": []}
        for _ in range(SPLITS):
            perm = list(gen.permutation(keys))
            cal = [s for k in perm[:len(keys) // 2] for s in dates[k]]
            test = [s for k in perm[len(keys) // 2:] for s in dates[k]]
            tc, fc = [trust[s] for s in cal], [rec[s]["fail"] for s in cal]
            tt, ft = [trust[s] for s in test], [rec[s]["fail"] for s in test]
            rows["CRC"].append(evaluate_threshold(tt, ft, crc_threshold(tc, fc, alpha)))
            rows["LTT"].append(evaluate_threshold(tt, ft, ltt_threshold(tc, fc, alpha)))
        res = {}
        for m, rr in rows.items():
            cov = np.array([r["coverage"] for r in rr])
            if m == "CRC":
                viol = np.mean([r["joint_false_safe_risk"] > alpha for r in rr])
                realized = float(np.mean([r["joint_false_safe_risk"] for r in rr]))
            else:
                cr = [r["conditional_failure_risk"] for r in rr if r["conditional_failure_risk"] is not None]
                viol = float(np.mean([c > alpha for c in cr])) if cr else 0.0
                realized = float(np.mean(cr)) if cr else None
            # CRC guarantees EXPECTED risk (mean over calibration draws); LTT guarantees risk <= alpha with probability >= 1-delta.
            ok = (realized is not None and realized <= alpha) if m == "CRC" else (viol <= DELTA)
            status = "FULL_REINSPECTION" if np.median(cov) == 0 else ("QUALIFIED_DEVICE_CONDITIONAL" if ok else "GUARANTEE_NOT_MET")
            res[m] = {"median_coverage": float(np.median(cov)), "mean_coverage": float(cov.mean()), "violation_rate": float(viol), "mean_realized_risk": realized, "status": status}
        # small calibration budgets (k images, date-separated): CRC needs n >= 1/alpha - 1
        small = {}
        for k in (5, 10, 20):
            covs = []
            for d in range(SPLITS):
                g2 = np.random.default_rng(RNG + d)
                perm = list(g2.permutation(keys))
                cal = []
                for kk in perm:
                    for s in sorted(dates[kk]):
                        if len(cal) < k:
                            cal.append(s)
                test = [s for s in stems if s not in set(cal)]
                lam = crc_threshold([trust[s] for s in cal], [rec[s]["fail"] for s in cal], alpha)
                covs.append(evaluate_threshold([trust[s] for s in test], [rec[s]["fail"] for s in test], lam)["coverage"])
            small[str(k)] = {"median_coverage": float(np.median(covs)), "status": "FULL_REINSPECTION" if np.median(covs) == 0 else "CALIBRATED"}
        res["CRC_small_k"] = small
        out[str(alpha)] = res
    return out


def qualification(worst_seed, fp, crc05_cov, crc05_status):
    """Reporting classification fixed before H2 is added (not an H2 gate).
    QUALIFIED: worst-seed 3px >= .95, FP/image <= .05, CRC expected false-safe risk <= .05 with coverage >= .5 on this device.
    REVIEW: CRC controlled at .05 with 0 < coverage < .5, or worst-seed in [.90, .95) -> partial automation, rest to human review.
    UNQUALIFIED: CRC not controlled / zero coverage, or worst-seed < .90 -> full reinspection on this device."""
    ctrl = crc05_status == "QUALIFIED_DEVICE_CONDITIONAL"
    if not ctrl or crc05_cov == 0 or worst_seed < .90:
        return "UNQUALIFIED (full reinspection on this device)"
    if worst_seed >= .95 and fp <= .05 and crc05_cov >= .5:
        return "QUALIFIED within envelope (device-conditional)"
    return "REVIEW (partial automation)"

# --- analysis/phase_g_evaluate.py


def e0(boxes,w,h):
 keep=[]
 for b in sorted(boxes,key=lambda v:-v[4]):
  x,y=(b[0]+b[2])*w/2,(b[1]+b[3])*h/2
  if all(math.hypot(x-(a[0]+a[2])*w/2,y-(a[1]+a[3])*h/2)>4 for a in keep):keep.append(b)
 return keep


def score(pred,stems):
 truth={};sizes={}
 for s in stems:
  r=META[s];w,h=r['resolution'];sizes[s]=(w,h);b=[]
  for line in (R/r['label']).read_text(encoding='utf8').splitlines():
   if not line.strip():continue
   _,x,y,bw,bh=map(float,line.split());b.append([x-bw/2,y-bh/2,x+bw/2,y+bh/2])
  truth[s]=b
 gt=sum(map(len,truth.values()));radii={}
 for rad in (1,2,3):
  rr=centre_matches(pred,truth,sizes,float(rad));tp=sum(z[1] and z[0]>=.25 for z in rr);fp=sum(not z[1] and z[0]>=.25 for z in rr)
  radii[str(rad)]={'TP':tp,'FN':gt-tp,'FP':fp,'success':tp/gt if gt else None,'FP_per_image':fp/len(stems)}
 rows=centre_matches(pred,truth,sizes,3.);curve=froc_rows(rows,gt,len(stems));errors,n=pair_errors(pred,truth,stems,.25)
 a=np.asarray(errors,dtype=float);dist=a[:,0] if len(a) else np.array([])
 return {'images':len(stems),'GT_objects':gt,'pFROC3':partial_froc_auc(curve),'radii':radii,
  'P50_P90_P95':np.percentile(dist,[50,90,95]).tolist() if len(dist) else None,
  'signed_dx':float(a[:,1].mean()) if len(a) else None,'signed_dy':float(a[:,2].mean()) if len(a) else None,
  'matched8':len(errors),'unmatched8':n-len(errors),'whole_object_error_CDF':sorted(dist.tolist())+[None]*(n-len(errors)),
  'P90_conditioning':'Matched<=8px; always interpret with FN/FP and whole-object failure distribution'}


AP = SimpleNamespace(per_image=per_image, crc_ltt=crc_ltt, qualification=qualification)
EV = SimpleNamespace(e0=e0, score=score)
