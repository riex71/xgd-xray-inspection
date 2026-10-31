"""XGD Generalization Assurance Analyzer (version in config.yaml).

A conventional detector maps image -> prediction. XGD maps
    image -> input-scale normalization (X8: effective network input magnification) -> DFL detection -> acquisition-quality check (X12)
          -> device-conditioned conformal risk control (CRC) -> ACCEPT / REVIEW / FULL_REINSPECTION (+ human-readable reasons).
ACCEPT means "the detected foreign-object positions are confirmed automatically"; it is never a product pass.
Claim: within-provider empirical assurance under a validated operating envelope. No universal / arbitrary-unseen-device guarantee, no
external validation, and no probability guarantee unless an LTT test is reported as established AND its preconditions hold.

Rules version XGD-1.1.0 (2026-10-05) fixes the safety gaps of XGD-1.0.0 found by an independent review:
  - the device operating grade (enum) is enforced: UNQUALIFIED / unknown / missing -> FULL_REINSPECTION, REVIEW grade -> no automatic ACCEPT;
  - the operating envelope includes the effective input scale (X8 result within the validated training range, with the pre-registered 0.95
    tolerance that also absorbs the 32 px rounding of X8);
  - CRC / LTT thresholds treat equal trust values as one block and compute risk on the set actually accepted by trust >= lambda;
  - zero detections are always REVIEW (never an automatic "no foreign object"), whatever lambda is;
  - non-finite / missing inputs, signals, metadata, calibration fields and calibration artifacts that are incompatible with the current
    model / preprocessing / scale / trust definition are FULL_REINSPECTION instead of flowing into a normal state.
The XGD-1.0.0 rule functions are kept verbatim with the suffix _v1_0 only to reproduce the sealed X14 results (tools/regression_x14.py).
The detection, preprocessing, X8 and quality-signal functions are unchanged from 1.0.0 (bitwise-checked by tools/regression_x14.py and
src/verify.py). This module never imports Phase G code.

Usage (from this model folder):
  python -m src.xgd_analyzer --image sample.bmp --metadata metadata.json [--profile dfl_src12] [--seed 17] [--out result.json]
  python -m src.xgd_analyzer --folder images/ --metadata metadata.json --out results/          (batch)
  python -m src.xgd_analyzer calibrate --profile dfl_src12                                     (build deployment artifacts)
metadata.json: {"calibration_id": "<device calibration key>", "session": "<acquisition session/date>", "preprocessed": false}
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import collections
import datetime
import enum
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    from . import xgd_config as XC
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import xgd_config as XC

QS = ("Q2", "Q4")
COVERAGE = [round(x, 2) for x in np.arange(.1, 1.01, .1)]
STATES = ("ACCEPT", "REVIEW", "FULL_REINSPECTION")


class Decision(str, enum.Enum):
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    FULL = "FULL_REINSPECTION"


class Grade(str, enum.Enum):
    """Empirical device operating grade (fixed 2026-10-04 rule). Not a probability guarantee."""
    QUALIFIED = "QUALIFIED"
    REVIEW = "REVIEW"
    UNQUALIFIED = "UNQUALIFIED"


GRADE_TEXT = {Grade.QUALIFIED: "QUALIFIED within envelope (device-conditional)", Grade.REVIEW: "REVIEW (partial automation)",
              Grade.UNQUALIFIED: "UNQUALIFIED (full reinspection on this device)"}


def now():
    return datetime.datetime.now().astimezone().isoformat()


def _numerics():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def protocol(cfg):
    """Protocol operations (preprocessing, product mask, centre matcher) of this folder — never the Phase G-patched variants."""
    assert "phase_g_core" not in sys.modules, "XGD must not run in a process that imported phase_g_core (patched preprocessing config)"
    m = str(cfg["_root"])
    if m not in sys.path:
        sys.path.insert(0, m)
    from src.metrics import centre_matches
    from src.preprocess import clean_v3, product_mask
    return clean_v3, product_mask, centre_matches


def read_image(path):
    import cv2
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def _finite(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# ------------------------------------------------------------------------------------------------ X8 scale normalization
def scale_policy(w, h, scale_range, sp, mode="x8", fixed_scale=None):
    canvas, tol = sp["canvas"], sp["tolerance"]
    t = XC.letterbox_ratio(w, h, canvas)
    lo, hi = scale_range["min_ratio"], scale_range["max_ratio"]
    out = {"original_resolution": [int(w), int(h)], "acquisition_scale": t, "training_scale_range": [lo, hi], "mode": mode}
    if mode == "fixed":
        z = max(64, int(32 * round(fixed_scale * max(w, h) / 32)))
        out.update(normalized=None, imgsz=z, reason=f"평가용 고정 배율 {fixed_scale:.3f} (운영 정책 아님)")
    elif mode == "native":
        out.update(normalized=False, imgsz=canvas, reason="정규화 비활성(비교용 원래 해상도)")
    elif t < tol * lo:
        z = int(32 * round(lo * max(w, h) / 32))
        out.update(normalized=True, imgsz=z, reason=f"촬영 배율 {t:.3f}이 학습 배율 하한 {lo:.3f}보다 낮음 → 하한 배율로 확대({z}px)")
    else:
        out.update(normalized=False, imgsz=canvas,
                   reason=("촬영 배율이 학습 배율 범위 안 → 원래 해상도 유지" if t <= hi / tol else
                           f"촬영 배율 {t:.3f}이 학습 범위({lo:.3f}~{hi:.3f})보다 높음 → 축소하지 않고 원래 해상도 유지"))
    out["effective_scale"] = out["imgsz"] / max(w, h)
    return out


def scale_envelope(sc, sp):
    """XGD-1.1.0: the effective input scale actually used must lie in [tolerance * min, max / tolerance] of the profile's training
    letterbox ratios (same pre-registered tolerance 0.95 as X8; it absorbs the <= 16 px / max-side rounding of X8). Forced diagnostic
    scales (mode fixed / native) are evaluated with the same rule but are never the operating path."""
    lo, hi = sc["training_scale_range"]
    tol = sp["tolerance"]
    eff = sc["effective_scale"]
    ok = bool(_finite(eff) and lo * tol <= eff <= hi / tol)
    reason = (f"유효 입력 배율 {eff:.3f}이 검증 범위 {lo * tol:.3f}~{hi / tol:.3f} 안" if ok else
              f"유효 입력 배율 {eff:.3f}이 검증 범위 {lo * tol:.3f}~{hi / tol:.3f} 밖 → 자동 판정 중단")
    return ok, reason


# ------------------------------------------------------------------------------------------------ DFL detection (+ E0)
def detect_raw(model, bgr, imgsz, pc, device):
    h, w = bgr.shape[:2]
    r = model.predict(bgr, imgsz=imgsz, rect=pc["rect"], conf=pc["conf"], iou=pc["iou"], max_det=pc["max_det"], verbose=False,
                      device=device, half=pc["half"])[0]
    bb = np.c_[r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()]
    if len(bb):
        bb[:, :4] /= [w, h, w, h]
    return bb.tolist()


def e0(boxes, w, h, radius=4.0):
    keep = []
    for b in sorted(boxes, key=lambda v: -v[4]):
        x, y = (b[0] + b[2]) * w / 2, (b[1] + b[3]) * h / 2
        if all(math.hypot(x - (a[0] + a[2]) * w / 2, y - (a[1] + a[3]) * h / 2) > radius for a in keep):
            keep.append(b)
    return keep


def trust_of(boxes, thr):
    d = [b[4] for b in boxes if b[4] >= thr]
    return float(np.mean(d)) if d else 0.0


def read_truth(path):
    out = []
    p = Path(path)
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf8").splitlines():
        if line.strip():
            _, x, y, bw, bh = map(float, line.split())
            out.append([x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2])
    return out


def outcome(boxes, gt, w, h, centre_matches, thr):
    """3 px one-to-one score-order outcome of one image (= validated image_counts / assurance per_image failure)."""
    g = [((b[0] + b[2]) / 2 * w, (b[1] + b[3]) / 2 * h) for b in gt]
    rows = centre_matches({"x": boxes}, {"x": [[(c[0] - .5) / w, (c[1] - .5) / h, (c[0] + .5) / w, (c[1] + .5) / h] for c in g]},
                          {"x": (w, h)}, 3.0) if g else []
    tp = sum(1 for z in rows if z[1] and z[0] >= thr)
    fp = sum(1 for p in boxes if p[4] >= thr) - tp
    return {"tp": tp, "n": len(gt), "fp": fp, "fail": int(tp < len(gt) or fp > 0)}


# ------------------------------------------------------------------------------------------------ X12 quality gate
def quality_signals(gray, product_mask):
    import cv2
    g = gray.astype(np.float32)
    m = cv2.erode((product_mask(gray, empty=0) > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if m.sum() < 500:
        m = np.ones_like(m, bool)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    nz = cv2.filter2D(g, cv2.CV_32F, k)
    mx, my = m[:, :-1] & m[:, 1:], m[:-1, :] & m[1:, :]
    cx = np.corrcoef(hp[:, :-1][mx], hp[:, 1:][mx])[0, 1]
    cy = np.corrcoef(hp[:-1, :][my], hp[1:, :][my])[0, 1]
    return {"Q1": float(np.log(lap[m].var() + 1e-6)), "Q2": float(np.log(hp[m].var() / (g[m].var() + 1e-6) + 1e-9)),
            "Q3": float(np.log(np.abs(nz[m]).mean() * math.sqrt(math.pi / 2) / 6 + 1e-6)), "Q4": float((cx + cy) / 2)}


def build_quality_reference(stems, signals, session_of, exclude_session=None):
    """X12 multi-session static reference. `stems` in sorted order fixes the session iteration order of the validated implementation."""
    by = collections.OrderedDict()
    for s in stems:
        if session_of[s] != exclude_session:
            by.setdefault(session_of[s], []).append(s)
    ref = {"sessions": list(by), "n_images": sum(len(v) for v in by.values())}
    for q in QS:
        meds = {k: float(np.median([signals[s][q] for s in v])) for k, v in by.items()}
        resid = [signals[s][q] - meds[k] for k, v in by.items() for s in v]
        sw = 1.4826 * float(np.median(np.abs(resid)))
        mv = np.array(list(meds.values()))
        sb = float(np.std(mv, ddof=1)) if len(mv) > 1 else 0.
        ref[q] = {"loc": float(np.mean(mv)), "scale": max(float(np.sqrt(sw ** 2 + sb ** 2)), 1e-6)}
    return ref


def quality_state(sig, ref, qg):
    """XGD-1.1.0: identical to 1.0.0 for finite inputs; a missing / non-finite signal or reference value is FULL_REINSPECTION (1.0.0 let a NaN
    z-score fall through every comparison into NORMAL)."""
    try:
        z = {q: (sig[q] - ref[q]["loc"]) / ref[q]["scale"] for q in QS}
    except (KeyError, TypeError, ZeroDivisionError):
        return "FULL_REINSPECTION", None, None
    if not all(_finite(v) for v in z.values()) or not all(_finite(ref[q]["scale"]) and ref[q]["scale"] > 0 for q in QS):
        return "FULL_REINSPECTION", None, z
    m = max(abs(v) for v in z.values())
    state = "FULL_REINSPECTION" if m > qg["z_full"] else ("REVIEW" if m > qg["z_review"] else "NORMAL")
    return state, m, z


def quality_state_v1_0(sig, ref, qg):
    """XGD-1.0.0 (sealed X12 rule) — regression reproduction only."""
    z = {q: (sig[q] - ref[q]["loc"]) / ref[q]["scale"] for q in QS}
    m = max(abs(v) for v in z.values())
    state = "FULL_REINSPECTION" if m > qg["z_full"] else ("REVIEW" if m > qg["z_review"] else "NORMAL")
    return state, m, z


# ------------------------------------------------------------------------------------------------ CRC / LTT
def crc_lambda(trust, fail, alpha):
    """Conformal risk control (Angelopoulos et al.) on L(lambda) = 1{trust >= lambda and failure}, loss bound B = 1.
    lambda = the lowest trust value v whose accepted set {trust >= v} satisfies (failures + 1) / (n + 1) <= alpha. Equal trust values form one
    block (a threshold cannot split them), so the bound is computed on the set the threshold actually accepts. Returns (lambda, bound);
    (inf, None) = accept nothing. The guarantee is on the EXPECTED joint risk P(ACCEPT and failure) under exchangeability."""
    t, f = np.asarray(trust, float), np.asarray(fail, float)
    n = len(t)
    if n == 0 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(f)):
        return math.inf, None
    best = (math.inf, None)
    for v in np.unique(t)[::-1]:                  # distinct trust values, accepting more images at each step
        F = float(f[t >= v].sum())
        if F + 1 <= alpha * (n + 1):
            best = (float(v), (F + 1) / (n + 1))
        else:
            break                                 # failures only grow as more images are accepted
    return best


def ltt_lambda(trust, fail, alpha, delta):
    """Learn-then-Test (Bonferroni over the fixed 10-point coverage grid): reject H_c: P(failure | accepted) > alpha when the exact binomial
    p-value of the set actually accepted by the coverage-c threshold (ties included) is <= delta / 10; return the threshold of the largest
    rejected coverage (inf if none). Valid only if calibration and operation images are exchangeable."""
    from scipy import stats
    t, f = np.asarray(trust, float), np.asarray(fail, float)
    n = len(t)
    if n == 0 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(f)):
        return math.inf
    srt = np.sort(t)[::-1]
    best = math.inf
    for c in COVERAGE:
        k = max(1, int(round(c * n)))
        lam = float(srt[k - 1])
        acc = t >= lam
        if stats.binom.cdf(f[acc].sum(), int(acc.sum()), alpha) <= delta / len(COVERAGE):
            best = lam
    return best


def crc_lambda_v1_0(trust, fail, alpha):
    """XGD-1.0.0 (tie-unaware) — regression reproduction only."""
    t, f = np.asarray(trust, float), np.asarray(fail, float)
    n = len(t)
    order = np.argsort(-t, kind="stable")
    cf = np.cumsum(f[order])
    ok = np.nonzero(cf <= alpha * (n + 1) - 1)[0]
    if not len(ok):
        return math.inf, None
    k = ok[-1]
    return float(t[order][k]), float((cf[k] + 1) / (n + 1))


def ltt_lambda_v1_0(trust, fail, alpha, delta):
    """XGD-1.0.0 (tie-unaware) — regression reproduction only."""
    from scipy import stats
    t, f = np.asarray(trust, float), np.asarray(fail, float)
    n = len(t)
    order = np.argsort(-t, kind="stable")
    cf = np.cumsum(f[order])
    best = math.inf
    for c in COVERAGE:
        k = max(1, int(round(c * n)))
        if stats.binom.cdf(cf[k - 1], k, alpha) <= delta / len(COVERAGE):
            best = float(t[order][k - 1])
    return best


def _evaluate_threshold(trust, fail, lam):
    t, f = np.asarray(trust, float), np.asarray(fail, float)
    acc = t >= lam
    k = int(acc.sum())
    return {"coverage": k / len(t), "joint_false_safe_risk": float(f[acc].sum() / len(t)), "conditional_failure_risk": float(f[acc].mean()) if k else None}


def crc_split_assessment(trust, fail, stems, cluster_of, alphas=(.05, .10), splits=200, rng=20261004, delta=.10, crc_fn=None, ltt_fn=None):
    """Repeated date-cluster half splits within the calibration set (validated assurance procedure, same RNG stream).
    The splits are NOT independent samples: with D date clusters there are only C(D, D//2) distinct halves."""
    crc_fn, ltt_fn = crc_fn or crc_lambda, ltt_fn or ltt_lambda
    dates = collections.defaultdict(list)
    for s in stems:
        dates[cluster_of[s]].append(s)
    keys = sorted(dates)
    gen = np.random.default_rng(rng)
    out = {}
    for alpha in alphas:
        rows = {"CRC": [], "LTT": []}
        for _ in range(splits):
            perm = list(gen.permutation(keys))
            cal = [s for k in perm[:len(keys) // 2] for s in dates[k]]
            test = [s for k in perm[len(keys) // 2:] for s in dates[k]]
            tc, fc = [trust[s] for s in cal], [fail[s] for s in cal]
            tt, ft = [trust[s] for s in test], [fail[s] for s in test]
            rows["CRC"].append(_evaluate_threshold(tt, ft, crc_fn(tc, fc, alpha)[0]))
            rows["LTT"].append(_evaluate_threshold(tt, ft, ltt_fn(tc, fc, alpha, delta)))
        res = {}
        for m, rr in rows.items():
            cov = np.array([r["coverage"] for r in rr])
            if m == "CRC":
                realized = float(np.mean([r["joint_false_safe_risk"] for r in rr]))
                viol = float(np.mean([r["joint_false_safe_risk"] > alpha for r in rr]))
                ok = realized <= alpha
            else:
                cr = [r["conditional_failure_risk"] for r in rr if r["conditional_failure_risk"] is not None]
                viol = float(np.mean([c > alpha for c in cr])) if cr else 0.0
                realized = float(np.mean(cr)) if cr else None
                ok = viol <= delta
            status = "FULL_REINSPECTION" if np.median(cov) == 0 else ("QUALIFIED_DEVICE_CONDITIONAL" if ok else "GUARANTEE_NOT_MET")
            res[m] = {"median_coverage": float(np.median(cov)), "mean_realized_risk": realized, "violation_rate": viol, "status": status}
        out[str(alpha)] = res
    out["distinct_halves"] = math.comb(len(keys), len(keys) // 2)
    return out


def crc_split_assessment_v1_0(trust, fail, stems, cluster_of, alphas=(.05, .10), splits=200, rng=20261004, delta=.10):
    """XGD-1.0.0 split assessment (tie-unaware thresholds, no distinct-halves field) — regression reproduction only."""
    out = crc_split_assessment(trust, fail, stems, cluster_of, alphas, splits, rng, delta, crc_lambda_v1_0, ltt_lambda_v1_0)
    out.pop("distinct_halves")
    return out


def qualification_grade(worst_seed, fp, crc05_cov, crc05_status):
    """Fixed 2026-10-04 reporting rule (unchanged thresholds), returned as an enum."""
    if crc05_status != "QUALIFIED_DEVICE_CONDITIONAL" or crc05_cov == 0 or worst_seed < .90:
        return Grade.UNQUALIFIED
    if worst_seed >= .95 and fp <= .05 and crc05_cov >= .5:
        return Grade.QUALIFIED
    return Grade.REVIEW


def qualification_v1_0(worst_seed, fp, crc05_cov, crc05_status):
    """XGD-1.0.0 text form (same rule) — kept for artifact text and regression."""
    return GRADE_TEXT[qualification_grade(worst_seed, fp, crc05_cov, crc05_status)]


def parse_grade(value):
    """Structured comparison: only an exact enum value is accepted ('UNQUALIFIED' never matches 'QUALIFIED' by substring)."""
    try:
        return Grade(value)
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------------------------------------------ decision
def calibration_level(n, crc):
    return "UNCALIBRATED" if n < crc["min_calibration"] else ("LIMITED" if n < crc["stable_calibration"] else "STABLE")


def calibration_problems(calib, crc, compatible=True):
    """Problems that make a device calibration unusable (each -> FULL_REINSPECTION)."""
    if calib is None:
        return ["이 장비의 보정 정보 없음(미등록 장비) → 자동 판정 불가"]
    if not isinstance(calib, dict):
        return ["보정 정보 형식 오류 → 자동 판정 불가"]
    p = []
    if not compatible:
        p.append("보정 산출물이 현재 모델·전처리·배율·신뢰도 정의와 호환되지 않음 → 재보정 필요")
    n = calib.get("N")
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        p.append("보정 영상 수가 유효하지 않음")
    elif n < crc["min_calibration"]:
        p.append(f"라벨 있는 보정 영상 {n}장 < {crc['min_calibration']}장 → CRC 적용 불가")
    lam = calib.get("lambda")
    if not isinstance(lam, (int, float)) or isinstance(lam, bool) or math.isnan(lam) or lam < 0:
        p.append("CRC 임계값이 유효하지 않음")
    g = parse_grade(calib.get("grade"))
    if g is None:
        p.append("장비 운영 등급이 없거나 알 수 없음 → 미검증 장비로 처리")
    elif g is Grade.UNQUALIFIED:
        p.append("장비 운영 등급 UNQUALIFIED → 이 장비는 전수 재검사")
    return p


def decide(gate_state, calib, trust, n_det, crc, scale_ok=True, scale_reason=None, input_problems=(), compatible=True):
    """XGD-1.1.0 decision. Every triggered condition is reported; FULL conditions take precedence over REVIEW conditions.
    FULL_REINSPECTION: invalid input / metadata / signals, missing or incompatible or invalid calibration, N < min, grade UNQUALIFIED,
                       effective input scale outside the validated range, quality FULL.
    REVIEW:            quality REVIEW, grade REVIEW, zero detections, trust < lambda (including lambda = inf).
    ACCEPT:            none of the above (grade QUALIFIED, quality NORMAL, scale valid, N >= min, >= 1 detection, trust >= lambda).
    Returns (decision, reasons, envelope_valid)."""
    full, review = list(input_problems), []
    if gate_state not in ("NORMAL", "REVIEW", "FULL_REINSPECTION"):
        full.append("알 수 없는 품질 상태 → 자동 판정 중단")
    if not _finite(trust) or not 0 <= float(trust) <= 1:
        full.append("신뢰도가 유효한 확률 범위 밖 → 자동 판정 중단")
    if not isinstance(n_det, int) or isinstance(n_det, bool) or n_det < 0:
        full.append("검출 수 형식 오류 → 자동 판정 중단")
    full += calibration_problems(calib, crc, compatible)
    if not scale_ok:
        full.append(scale_reason or "유효 입력 배율이 검증 범위 밖 → 자동 판정 중단")
    if gate_state == "FULL_REINSPECTION":
        full.append("촬영 품질이 검증된 운영 범위(X12 정상 envelope) 밖이거나 계산 불가 → 자동 판정 중단")
    envelope = not full and gate_state == "NORMAL" and parse_grade((calib or {}).get("grade")) is Grade.QUALIFIED
    if full:
        return Decision.FULL.value, full, False
    if gate_state == "REVIEW":
        review.append("촬영 품질 경계 구간(X12 z 3~6) → 자동 확정 제한")
    if parse_grade(calib.get("grade")) is Grade.REVIEW:
        review.append("장비 운영 등급 REVIEW → 자동 확정 제한(부분 자동화)")
    if n_det == 0:
        review.append("운영 점수 이상 검출 0개 → '이물 없음'은 자동 확정하지 않음(정상 제품 보정 자료 없음)")
    lam = calib["lambda"]
    if not (_finite(trust) and trust >= lam):
        lam_s = "∞(수용 불가)" if math.isinf(lam) else f"{lam:.3f}"
        review.append(f"검출 신뢰도 {trust:.3f} < CRC 임계값 {lam_s} → CRC가 자동 확정을 거부")
    if review:
        return Decision.REVIEW.value, review, envelope
    return Decision.ACCEPT.value, [f"검출 신뢰도 {trust:.3f} ≥ CRC 임계값 {lam:.3f}, 등급 QUALIFIED, 품질·배율 정상"], envelope


def decide_v1_0(gate_state, calib, trust, crc):
    """XGD-1.0.0 (sealed X14 decision order; no grade / scale / zero-detection / input checks) — regression reproduction only."""
    reasons = []
    if gate_state == "FULL_REINSPECTION":
        return "FULL_REINSPECTION", ["촬영 품질이 검증된 운영 범위(X12 정상 envelope) 밖 → 자동 판정 중단"]
    if calib is None:
        return "FULL_REINSPECTION", ["이 장비의 보정 정보 없음 → 자동 판정 불가"]
    if calib["N"] < crc["min_calibration"]:
        return "FULL_REINSPECTION", [f"라벨 있는 보정 영상 {calib['N']}장 < {crc['min_calibration']}장 → CRC 적용 불가"]
    if gate_state == "REVIEW":
        return "REVIEW", ["촬영 품질 경계 구간(X12 z 3~6) → 자동 ACCEPT 제한"]
    if not trust >= calib["lambda"]:
        lam = "∞(수용 불가)" if math.isinf(calib["lambda"]) else f"{calib['lambda']:.3f}"
        return "REVIEW", [f"검출 신뢰도 {trust:.3f} < CRC 임계값 {lam} → CRC가 자동 판정을 거부"]
    reasons.append(f"검출 신뢰도 {trust:.3f} ≥ CRC 임계값 {calib['lambda']:.3f}")
    return "ACCEPT", reasons


def input_problems(image, calibration_id, sig=None, boxes=None, trust=None):
    """Malformed inputs must never reach a normal state."""
    p = []
    if image is None or not hasattr(image, "shape") or image.ndim != 3 or image.shape[2] != 3 or image.size == 0 or image.dtype != np.uint8:
        p.append("영상을 읽을 수 없거나 형식이 다름(8비트 3채널 아님) → 자동 판정 중단")
    if not isinstance(calibration_id, str) or not calibration_id.strip():
        p.append("메타데이터에 장비 식별(calibration_id)이 없음 → 자동 판정 중단")
    if sig is not None and not all(_finite(sig.get(q)) for q in QS):
        p.append("촬영 품질 신호(Q2/Q4) 계산 불가(NaN/Inf) → 자동 판정 중단")
    if boxes is not None and not all(isinstance(b, (list, tuple, np.ndarray)) and len(b) == 5 and
                                     all(_finite(v) for v in b) and 0 <= b[4] <= 1 and b[0] <= b[2] and b[1] <= b[3] for b in boxes):
        p.append("검출 결과에 비정상 값(NaN/Inf) → 자동 판정 중단")
    if trust is not None and not _finite(trust):
        p.append("검출 신뢰도 계산 불가 → 자동 판정 중단")
    return p


def compatibility_signature(cfg, checkpoint_sha256):
    """What a calibration artifact depends on: the detector weights and every function that produces trust / failures / quality signals."""
    d = {"rules_version": cfg["version"], "checkpoint_sha256": checkpoint_sha256, "predict": cfg["detector"]["predict"], "e0_radius_px": cfg["detector"]["e0_radius_px"],
         "decision_threshold": cfg["detector"]["decision_threshold"], "scale_policy": {k: cfg["scale_policy"][k] for k in ("canvas", "tolerance")},
         "preprocess": {k: cfg["preprocess"][k] for k in ("overlay_gain", "gain_shrink")}, "trust": cfg["crc"]["trust"],
         "quality_signals": list(cfg["quality_gate"]["signals"]),
         "profiles": cfg["profiles"],
         "split_sha256": sha256_file(cfg["_root"] / "data/validation_split.json"),
         "metadata_sha256": sha256_file(cfg["_root"] / cfg["paths"]["metadata"]),
         "source_sha256": {p: sha256_file(cfg["_root"] / p) for p in ("src/xgd_analyzer.py", "src/preprocess.py", "src/metrics.py")}}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def quality_signature(cfg):
    d = {"rules_version": cfg["version"], "preprocess": {k: cfg["preprocess"][k] for k in ("overlay_gain", "gain_shrink")}, "signals": list(cfg["quality_gate"]["signals"]),
         "reference": cfg["quality_gate"]["reference"],
         "source_sha256": {p: sha256_file(cfg["_root"] / p) for p in ("src/xgd_analyzer.py", "src/preprocess.py")}}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ analyzer
class XGDAnalyzer:
    def __init__(self, config=None, profile=None, seed=None, device=None, quality_reference=None, crc_calibration=None, load_artifacts=True):
        import torch
        from ultralytics import YOLO
        self.cfg = config if isinstance(config, dict) else XC.load(config)
        self.profile = profile or self.cfg["detector"]["default_profile"]
        self.seed = int(seed or self.cfg["detector"]["default_seed"])
        self.clean_v3, self.product_mask, self.centre_matches = protocol(self.cfg)
        _numerics()
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.ckpt = XC.checkpoint(self.cfg, self.profile, self.seed)
        self.ckpt_sha256 = sha256_file(self.ckpt)
        self.signature = compatibility_signature(self.cfg, self.ckpt_sha256)
        self.qsignature = quality_signature(self.cfg)
        self.model = YOLO(str(self.ckpt))
        self.scale_range = XC.profile_scale_range(self.cfg, self.profile)
        art = self.cfg["_artifacts"] / "calibration"
        self.quality_reference = quality_reference
        self.crc_calibration = crc_calibration
        if load_artifacts and self.quality_reference is None and (art / "quality_reference.json").exists():
            self.quality_reference = json.loads((art / "quality_reference.json").read_text(encoding="utf8"))
        if load_artifacts and self.crc_calibration is None and (art / f"crc_{self.profile}.json").exists():
            self.crc_calibration = json.loads((art / f"crc_{self.profile}.json").read_text(encoding="utf8"))

    def _calibration(self, calibration_id):
        if not self.crc_calibration or calibration_id is None:
            return None
        dev = self.crc_calibration.get("devices", {}).get(calibration_id)
        return (dev or {}).get("seeds", {}).get(str(self.seed))

    def _reference(self, calibration_id):
        q = self.quality_reference or {}
        if q.get("signature") != self.qsignature or not str(q.get("version", "")).startswith(self.cfg["version"] + "-"):
            return None, False
        return q.get("devices", {}).get(calibration_id), True

    def analyze(self, image_path=None, image=None, metadata=None, calibration_id=None, preprocessed=None, scale_mode="x8", fixed_scale=None,
                signals=None, reference=None, calibration=None, raw_boxes=False):
        """Analyze one image. Overrides (reference, calibration, signals, scale_mode) exist for the validation harnesses."""
        import cv2
        if metadata is not None and not isinstance(metadata, dict):
            return self._invalid(["메타데이터가 객체 형식이 아님"], calibration_id)
        md = dict(metadata or {})
        calibration_id = calibration_id or md.get("calibration_id")
        pre = md.get("preprocessed", False) if preprocessed is None else preprocessed
        if not isinstance(pre, bool):
            return self._invalid(["preprocessed는 true/false이어야 함"], calibration_id)
        src = read_image(image_path) if image is None else image
        bad = input_problems(src, calibration_id)
        if input_problems(src, "validation-placeholder"):
            return self._invalid(bad, calibration_id)
        clean = src if pre else self.clean_v3(src)
        h, w = clean.shape[:2]
        dc, crc, qg = self.cfg["detector"], self.cfg["crc"], self.cfg["quality_gate"]
        sc = scale_policy(w, h, self.scale_range, self.cfg["scale_policy"], scale_mode, fixed_scale)
        scale_ok, scale_reason = scale_envelope(sc, self.cfg["scale_policy"])
        boxes = e0(detect_raw(self.model, clean, sc["imgsz"], dc["predict"], self.device), w, h, dc["e0_radius_px"])
        trust = trust_of(boxes, dc["decision_threshold"])
        thr = dc["decision_threshold"]
        n_det = sum(1 for b in boxes if b[4] >= thr)
        sig = signals if signals is not None else quality_signals(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY), self.product_mask)
        problems = input_problems(src, calibration_id, sig, boxes, trust)
        if reference is not None:
            ref, qcompat = reference, True
        else:
            ref, qcompat = self._reference(calibration_id)
        if ref is None:
            gate_state, zmax, z = "FULL_REINSPECTION", None, None
        else:
            gate_state, zmax, z = quality_state(sig, ref, qg)
        calib = calibration if calibration is not None else self._calibration(calibration_id)
        compatible = calibration is not None or calib is None or (isinstance(calib, dict) and calib.get("compatibility") == self.signature)
        if not qcompat:
            problems.append("품질 기준 산출물이 현재 전처리·신호 정의와 호환되지 않음 → 재보정 필요")
        decision, why, envelope = decide(gate_state, calib, trust, n_det, crc, scale_ok, scale_reason, problems, compatible)
        reasons = [sc["reason"], scale_reason]
        reasons.append("품질 기준 없음(미등록 장비)" if ref is None else
                       (f"촬영 품질 {gate_state} (Q2/Q4 최대 |z| = {zmax:.2f}; REVIEW > {qg['z_review']:g}, FULL > {qg['z_full']:g})" if zmax is not None
                        else "촬영 품질 신호 계산 불가"))
        if calib is not None and isinstance(calib.get("N"), int):
            level = calibration_level(calib["N"], crc)
            reasons.append(f"보정 영상 {calib['N']}장 ({level})" + (f", 보정 데이터 분할 추정 전체 위험 {calib['estimated_risk']:.1%}"
                                                                if _finite(calib.get("estimated_risk")) else ""))
        reasons += [r for r in why if r not in reasons]
        dets = [{"bbox_xyxy": [b[0] * w, b[1] * h, b[2] * w, b[3] * h], "center": [(b[0] + b[2]) / 2 * w, (b[1] + b[3]) / 2 * h], "confidence": b[4]}
                for b in boxes if b[4] >= thr]
        out = {"decision": decision, "detections": dets, "object_count": len(dets), "detections_confirmed": decision == Decision.ACCEPT.value,
               "scale": {**{k: sc[k] for k in ("original_resolution", "acquisition_scale", "training_scale_range", "normalized", "imgsz", "effective_scale", "mode")},
                         "operating_path": sc["mode"] == "x8", "envelope_ok": scale_ok},
               "quality": {"state": gate_state, "max_abs_z": zmax, "z": z, "signals": {q: sig[q] for q in sig}},
               "assurance": None if calib is None else {
                   "calibration_id": calibration_id, "calibration_n": calib.get("N"), "level": calibration_level(calib["N"], crc) if isinstance(calib.get("N"), int) else None,
                   "alpha": crc["alpha"], "crc_lambda": calib.get("lambda"), "crc_joint_risk_bound_at_lambda": calib.get("bound"),
                   "estimated_joint_risk": calib.get("estimated_risk"), "operating_grade": calib.get("grade"),
                   "operating_grade_text": calib.get("qualification"), "probability_guarantee": calib.get("probability_guarantee", calib.get("ltt")),
                   "compatible": compatible},
               "trust": trust, "operating_envelope_valid": bool(envelope), "input_problems": problems,
               "reason": reasons,
               "model": {"profile": self.profile, "seed": self.seed, "checkpoint": XC.rel(self.cfg, self.ckpt), "checkpoint_sha256": self.ckpt_sha256},
               "rules_version": self.cfg["version"], "claim": self.cfg["claim"]}
        if raw_boxes:
            out["_boxes_normalized_e0"] = boxes
        return out

    def _invalid(self, problems, calibration_id):
        return {"decision": Decision.FULL.value, "detections": [], "object_count": 0, "detections_confirmed": False, "scale": None, "quality": None,
                "assurance": None, "trust": None, "operating_envelope_valid": False, "input_problems": problems, "reason": problems,
                "model": {"profile": self.profile, "seed": self.seed, "checkpoint": XC.rel(self.cfg, self.ckpt), "checkpoint_sha256": self.ckpt_sha256},
                "rules_version": self.cfg["version"], "claim": self.cfg["claim"]}

    def analyze_folder(self, folder, metadata=None, out_dir=None, pattern=("*.bmp", "*.png", "*.jpg")):
        folder = Path(folder)
        paths = sorted(p for pat in pattern for p in folder.glob(pat))
        res = {}
        for p in paths:
            r = self.analyze(image_path=p, metadata=metadata)
            res[p.name] = r
            if out_dir:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                (Path(out_dir) / f"{p.stem}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf8")
        summary = collections.Counter(r["decision"] for r in res.values())
        if out_dir:
            (Path(out_dir) / "SUMMARY.json").write_text(json.dumps({"n": len(res), "decisions": dict(summary)}, ensure_ascii=False, indent=1), encoding="utf8")
        return res, dict(summary)


# ------------------------------------------------------------------------------------------------ deployment calibration artifacts
def calibrate(profile=None, config=None, seeds=None):
    """Builds versioned artifacts: quality_reference.json (all dev sessions per device, profile-independent) and crc_<profile>.json (per seed and
    device: CRC lambda on the device's labeled images NOT used to train the profile, LTT test status, the empirical operating grade computed
    over the profile's seeds with the date-split assessment and LOSO-session gate, and a compatibility signature of the model / preprocessing /
    scale / trust definition that produced them)."""
    import cv2
    cfg = config if isinstance(config, dict) else XC.load(config)
    profile = profile or cfg["detector"]["default_profile"]
    seeds = seeds or sorted(int(s) for s in cfg["profiles"][profile]["checkpoints"])
    _, product_mask, centre_matches = protocol(cfg)
    meta = XC.provider_metadata(cfg)
    dev = XC.dev_stems(cfg)
    sig = {s: quality_signals(cv2.cvtColor(read_image(XC.clean_path(cfg, s)), cv2.COLOR_BGR2GRAY), product_mask) for s in dev}
    session = {s: meta[s]["date"] for s in dev}
    devices = sorted({meta[s]["device"] for s in dev})
    art = cfg["_artifacts"] / "calibration"
    art.mkdir(parents=True, exist_ok=True)
    qref = {"version": cfg["version"] + "-qref1", "created": now(), "source": "dev356 clean sessions (no hold65 / test79)",
            "signature": quality_signature(cfg), "update_policy": cfg["quality_gate"]["update_policy"], "devices": {}}
    loso = {}
    for d in devices:
        ss = [s for s in dev if meta[s]["device"] == d]
        qref["devices"][d] = build_quality_reference(ss, sig, session)
        for sess in sorted({session[s] for s in ss}):
            r = build_quality_reference(ss, sig, session, exclude_session=sess)
            for s in ss:
                if session[s] == sess:
                    loso[s] = r
    (art / "quality_reference.json").write_text(json.dumps(qref, ensure_ascii=False, indent=1), encoding="utf8")
    training = set(XC.profile_training_stems(cfg, profile))
    crc_cfg, dc = cfg["crc"], cfg["detector"]
    out = {"version": cfg["version"] + f"-{profile}", "created": now(), "profile": profile, "scale_range": XC.profile_scale_range(cfg, profile),
           "calibration_policy": "labeled device images not used to train the profile; trust at the X8 scale; alpha 0.05",
           "risk_definitions": {"crc": "expected joint risk P(ACCEPT and failure); denominator = all images; valid under exchangeability",
                                "ltt": "P(failure | ACCEPT) <= alpha with probability >= 1 - delta; valid only if calibration and operation images are exchangeable",
                                "operating_grade": "empirical reporting grade (fixed 2026-10-04 rule); not a probability guarantee"},
           "devices": {}}
    per = collections.defaultdict(dict)
    sha = {}
    for sd in seeds:
        an = XGDAnalyzer(cfg, profile, sd, load_artifacts=False)
        sha[sd] = (an.ckpt_sha256, an.signature)
        for d in devices:
            cal = [s for s in dev if meta[s]["device"] == d and s not in training]
            if not cal:
                continue
            for s in cal:
                r = an.analyze(image=read_image(XC.clean_path(cfg, s)), preprocessed=True, signals=sig[s], reference=qref["devices"][d],
                               calibration={"N": 0, "lambda": math.inf, "grade": None}, raw_boxes=True, calibration_id=d)
                w, h = meta[s]["resolution"]
                o = outcome(r["_boxes_normalized_e0"], read_truth(XC.label_path(cfg, s)), w, h, centre_matches, dc["decision_threshold"])
                per[d].setdefault(sd, {})[s] = {"trust": r["trust"], **o}
    for d, by_seed in per.items():
        cal = sorted(next(iter(by_seed.values())))
        cluster = {s: meta[s]["date"] + meta[s]["device"] for s in cal}
        gate_clean = {s: quality_state(sig[s], loso[s], cfg["quality_gate"])[0] for s in cal}
        seeds_out, succ, fps, cov, st = {}, [], [], [], []
        for sd, rec in by_seed.items():
            trust = [rec[s]["trust"] for s in cal]
            fail = [rec[s]["fail"] for s in cal]
            lam, bound = crc_lambda(trust, fail, crc_cfg["alpha"])
            assess = crc_split_assessment({s: rec[s]["trust"] - (1. if gate_clean[s] != "NORMAL" else 0.) for s in cal}, {s: rec[s]["fail"] for s in cal}, cal, cluster,
                                          delta=crc_cfg["ltt_delta"])
            ltt = {str(a): (lambda v: None if math.isinf(v) else v)(ltt_lambda(trust, fail, a, crc_cfg["ltt_delta"])) for a in (.05, .10)}
            ltt_txt = {"alpha_" + a: ("ESTABLISHED (lambda %.3f)" % v if v is not None else "NOT_ESTABLISHED") for a, v in ltt.items()}
            seeds_out[str(sd)] = {"N": len(cal), "lambda": lam, "bound": bound, "coverage_in_calibration": float(np.mean(np.asarray(trust) >= lam)),
                                  "estimated_risk": assess["0.05"]["CRC"]["mean_realized_risk"], "crc_split_assessment": assess, "ltt": ltt_txt,
                                  "probability_guarantee": {"ltt_test": ltt_txt, "preconditions_verified": False,
                                                            "claim": "none (exchangeability of calibration and operation images is not established)"},
                                  "checkpoint_sha256": sha[sd][0], "compatibility": sha[sd][1]}
            succ.append(sum(rec[s]["tp"] for s in cal) / sum(rec[s]["n"] for s in cal))
            fps.append(float(np.mean([rec[s]["fp"] for s in cal])))
            cov.append(assess["0.05"]["CRC"]["median_coverage"])
            st.append(assess["0.05"]["CRC"]["status"])
        g = qualification_grade(min(succ), float(np.mean(fps)), float(np.median(cov)), collections.Counter(st).most_common(1)[0][0])
        for v in seeds_out.values():
            v["grade"], v["qualification"] = g.value, GRADE_TEXT[g]
        out["devices"][d] = {"N": len(cal), "level": calibration_level(len(cal), crc_cfg), "grade": g.value, "qualification": GRADE_TEXT[g],
                             "worst_seed_3px": min(succ), "FP_per_image": float(np.mean(fps)), "seeds": seeds_out}
    (art / f"crc_{profile}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf8")
    return qref, out


def _cli():
    ap = argparse.ArgumentParser(description="XGD Generalization Assurance Analyzer")
    ap.add_argument("command", nargs="?", default="analyze", choices=["analyze", "calibrate"])
    ap.add_argument("--image")
    ap.add_argument("--folder")
    ap.add_argument("--metadata")
    ap.add_argument("--calibration-id")
    ap.add_argument("--profile")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--config")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.command == "calibrate":
        _, crc = calibrate(a.profile, a.config)
        print(json.dumps({d: {k: v[k] for k in ("N", "level", "grade")} for d, v in crc["devices"].items()}, ensure_ascii=False, indent=1))
        return
    md = json.loads(Path(a.metadata).read_text(encoding="utf8")) if a.metadata else {}
    an = XGDAnalyzer(a.config, a.profile, a.seed)
    if a.folder:
        res, summary = an.analyze_folder(a.folder, md, a.out)
        print(json.dumps(summary, ensure_ascii=False))
    else:
        r = an.analyze(image_path=a.image, metadata=md, calibration_id=a.calibration_id)
        txt = json.dumps(r, ensure_ascii=False, indent=1, default=float)
        if a.out:
            Path(a.out).write_text(txt, encoding="utf8")
        print(txt)


if __name__ == "__main__":
    _cli()
