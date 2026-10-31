"""Shared helpers for the XGD-1.1 validation experiments (docs/PREREGISTRATION_XGD-1.1.md).

Predictions are cached per (model key, condition) under outputs/experiments/cache/ so every experiment can be resumed and re-scored
without the GPU. Metrics follow the pre-registered definitions; decisions use the XGD-1.1.0 rules under the P-IND / P-TIME / P-X14
calibration designs.
"""
import collections
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src import xgd_analyzer as X      # noqa: E402
from src import xgd_config as XC       # noqa: E402
from src.metrics import centre_matches  # noqa: E402
from tools import x14_sealed as XS     # noqa: E402  (verbatim sealed froc_rows / partial_froc_auc)
from tools.regression_x14 import degrade as degrade_x14, phase_rng, halves  # noqa: E402

CFG = XC.load()
META = XC.provider_metadata(CFG)
DEV = XC.dev_stems(CFG)
SEEDS = [17, 29, 43]
THR = CFG["detector"]["decision_threshold"]
OUT = ROOT / "outputs" / "experiments"
CACHE = OUT / "cache"
FAMILIES = {"3호기": "dfl_src12", "1·2호기": "dfl_src3_2"}
X14_HD = "3호기"            # held-device string used by the sealed X14 degradation RNG
_PRODUCT_MASK = None


def jdump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    def finite(o):
        if isinstance(o, dict):
            return {str(k): finite(v) for k, v in o.items()}
        if isinstance(o, (tuple, list)):
            return [finite(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return float(o) if math.isfinite(o) else None
        if isinstance(o, np.integer):
            return int(o)
        return o
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(finite(obj), ensure_ascii=False, indent=1, allow_nan=False), encoding="utf8")
    tmp.replace(p)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    raise TypeError(type(o))


def targets(profile):
    ts = CFG["profiles"][profile]["training_split"]
    a = XC.split(CFG)["schemes"][ts["scheme"]]["assignment"]
    return sorted(s for s, f in a.items() if f == ts["held_fold"])


def gt(stem):
    if stem not in DEV:
        raise RuntimeError("Evaluation labels must not be read by development experiments")
    return X.read_truth(XC.label_path(CFG, stem))


def size_of(stem):
    return tuple(META[stem]["resolution"])


def clean_img(stem):
    return X.read_image(XC.clean_path(CFG, stem))


def product_mask():
    global _PRODUCT_MASK
    if _PRODUCT_MASK is None:
        from src.preprocess import product_mask as pm
        _PRODUCT_MASK = pm
    return _PRODUCT_MASK


def rng_for(*parts):
    h = hashlib.sha256("/".join(map(str, ("XGD11",) + parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


# ------------------------------------------------------------------------------------------------ models
def cfg_with_checkpoints(mapping):
    """Copy of the config whose profile checkpoints are replaced: mapping {profile: {seed: relative path}}."""
    c = copy.deepcopy(CFG)
    for prof, ck in mapping.items():
        c["profiles"][prof]["checkpoints"] = {int(k): v for k, v in ck.items()}
    return c


class Model:
    def __init__(self, profile, seed, cfg=None, key=None):
        self.cfg = cfg or CFG
        self.an = X.XGDAnalyzer(self.cfg, profile, seed, load_artifacts=False)
        self.profile, self.seed = profile, seed
        self.key = key or f"{profile}_s{seed}"
        self.rng = self.an.scale_range

    def scale(self, w, h, mode="x8", fixed=None, factor=1.0):
        sc = X.scale_policy(w, h, self.rng, CFG["scale_policy"], mode, fixed)
        if factor != 1.0:
            sc = dict(sc)
            sc["imgsz"] = int(32 * round(sc["imgsz"] * factor / 32))
            sc["effective_scale"] = sc["imgsz"] / max(w, h)
            sc["mode"] = f"{mode}x{factor:g}"
        return sc

    def predict(self, bgr, imgsz):
        h, w = bgr.shape[:2]
        return X.e0(X.detect_raw(self.an.model, bgr, imgsz, CFG["detector"]["predict"], self.an.device), w, h, CFG["detector"]["e0_radius_px"])


def cached(model_key, cond, stems, fn):
    """fn(stem) -> (boxes, imgsz). Returns {stem: {"boxes", "imgsz"}} and caches to disk."""
    p = CACHE / model_key / f"{cond}.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf8"))
        if all(s in d for s in stems):
            return d
    else:
        d = {}
    for s in stems:
        if s not in d:
            boxes, z = fn(s)
            d[s] = {"boxes": boxes, "imgsz": z}
    jdump(p, d)
    return d


# ------------------------------------------------------------------------------------------------ matching / metrics
def centers_px(boxes, w, h, thr=THR):
    return [((b[0] + b[2]) / 2 * w, (b[1] + b[3]) / 2 * h, b[4]) for b in boxes if b[4] >= thr]


def iou(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.


def iou_counts(boxes, g, w, h, thr=THR, t_iou=0.5):
    P = sorted([b for b in boxes if b[4] >= thr], key=lambda b: -b[4])
    used, tp = set(), 0
    for b in P:
        best, bj = t_iou, None
        for j, q in enumerate(g):
            if j not in used:
                v = iou([b[0] * w, b[1] * h, b[2] * w, b[3] * h], [q[0] * w, q[1] * h, q[2] * w, q[3] * h])
                if v >= best:
                    best, bj = v, j
        if bj is not None:
            used.add(bj)
            tp += 1
    return tp, len(P) - tp


def ap50(boxes, gts):
    """101-point interpolated AP at IoU .5, same E0/top30 predictions. Not the COCO API metric."""
    rows=[]
    for s,bx in boxes.items():
        used=set()
        for b in sorted(bx,key=lambda b:-b[4])[:30]:
            choices=[(iou(b,q),j) for j,q in enumerate(gts[s]) if j not in used and iou(b,q)>=.5]
            j=max(choices)[1] if choices else None
            if j is not None: used.add(j)
            rows.append((b[4],j is not None))
    rows.sort(key=lambda x:-x[0]); n=sum(len(gts[s]) for s in boxes)
    if not n: return None
    if not rows: return 0.0
    tp=np.cumsum([int(x[1]) for x in rows]); fp=np.cumsum([not x[1] for x in rows])
    rec=tp/n; prec=tp/(tp+fp)
    return float(np.mean([max(prec[rec>=t],default=0) for t in np.linspace(0,1,101)]))


def pair_errors_8px(boxes, g, w, h, thr=THR, radius=8.0, signed=False):
    """Same rule as the protocol pair_errors(mode='score'): score-ordered, top 30, nearest unused GT within radius."""
    gs = [((q[0] + q[2]) / 2 * w, (q[1] + q[3]) / 2 * h) for q in g]
    ps = sorted([p for p in boxes if p[4] >= thr], key=lambda p: -p[4])[:30]
    used, errs = set(), []
    for p in ps:
        x, y = (p[0] + p[2]) / 2 * w, (p[1] + p[3]) / 2 * h
        near = [(math.hypot(x - gx, y - gy), k) for k, (gx, gy) in enumerate(gs) if k not in used and math.hypot(x - gx, y - gy) <= radius]
        if near:
            e, k = min(near)
            used.add(k)
            errs.append((e, x - gs[k][0], y - gs[k][1]) if signed else e)
    return errs


def image_eval(boxes, g, w, h):
    o3 = X.outcome(boxes, g, w, h, centre_matches, THR)
    gc = [((q[0] + q[2]) / 2 * w, (q[1] + q[3]) / 2 * h) for q in g]
    tiny = {"x": [[(c[0] - .5) / w, (c[1] - .5) / h, (c[0] + .5) / w, (c[1] + .5) / h] for c in gc]}
    out = {"n": o3["n"], "tp3": o3["tp"], "fp3": o3["fp"], "fail": o3["fail"], "fail_fn": int(o3["tp"] < o3["n"]), "fail_fp": int(o3["fp"] > 0),
           "trust": X.trust_of(boxes, THR), "n_det": sum(1 for b in boxes if b[4] >= THR)}
    for r in (1.0, 2.0, 5.0):
        rows = centre_matches({"x": boxes}, tiny, {"x": (w, h)}, r) if gc else []
        tp = sum(1 for z in rows if z[1] and z[0] >= THR)
        out[f"tp{int(r)}"] = tp
        out[f"fp{int(r)}"] = out["n_det"] - tp
    out["tp_iou"], out["fp_iou"] = iou_counts(boxes, g, w, h)
    out["err8"] = pair_errors_8px(boxes, g, w, h)
    out["signed8"] = pair_errors_8px(boxes, g, w, h, signed=True)
    return out


def detection_metrics(recs, boxes_by_stem=None, gts=None, resolutions=None):
    """recs: {stem: image_eval}. Micro P/R/F1 at 3px, 5px and IoU 0.5, FP/image, centre errors with matching rate, miss-inclusive
    tolerance exceedance, image failure rate; pFROC3 when the raw boxes are given."""
    n = sum(r["n"] for r in recs.values())
    k = len(recs)
    out = {"images": k, "objects": n}
    for tag in ("3", "5", "_iou"):
        tp = sum(r[f"tp{tag}"] for r in recs.values())
        fp = sum(r[f"fp{tag}"] for r in recs.values())
        p = tp / (tp + fp) if tp + fp else None
        rcl = tp / n if n else None
        out[f"P{tag}"], out[f"R{tag}"] = p, rcl
        out[f"F1{tag}"] = (2 * p * rcl / (p + rcl)) if p is not None and rcl and (p + rcl) else (0.0 if rcl == 0 else None)
        out[f"FP_per_image{tag}"] = fp / k if k else None
    errs = [e for r in recs.values() for e in r["err8"]]
    out["matched8_rate"] = len(errs) / n if n else None
    out["err_P50_P90_P95"] = [float(v) for v in np.percentile(errs, [50, 90, 95])] if errs else None
    out["miss_or_err_gt2px"] = 1 - sum(r["tp2"] for r in recs.values()) / n if n else None
    out["miss_or_err_gt3px"] = 1 - sum(r["tp3"] for r in recs.values()) / n if n else None
    out["image_failure_rate"] = sum(r["fail"] for r in recs.values()) / k if k else None
    out["FN"] = n - sum(r["tp3"] for r in recs.values())
    out["FP"] = sum(r["fp3"] for r in recs.values())
    out["success1px"] = sum(r["tp1"] for r in recs.values()) / n if n else None
    out["success2px"] = sum(r["tp2"] for r in recs.values()) / n if n else None
    signs = [p for r in recs.values() for p in r.get("signed8", [])]
    out["signed_dx_dy"] = np.mean([p[1:] for p in signs], axis=0).tolist() if signs else None
    if boxes_by_stem is not None:
        out["AP50_101pt_E0_top30"] = ap50({s:boxes_by_stem[s] for s in recs},gts)
        rows = []
        for s in recs:
            w, h = (resolutions or {}).get(s, size_of(s))
            rows += centre_matches({s: boxes_by_stem[s]}, {s: gts[s]}, {s: (w, h)}, 3.0)
        rows.sort(key=lambda r: -r[0])
        out["pFROC3"] = XS.partial_froc_auc(XS.froc_rows(rows, n, k)) if n and k else None
    return out


def clopper_pearson(x, n, a=0.05):
    from scipy import stats
    if n == 0:
        return None
    lo = 0.0 if x == 0 else float(stats.beta.ppf(a / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(stats.beta.ppf(1 - a / 2, x + 1, n - x))
    return [lo, hi]


def decision_metrics(decisions, recs):
    """decisions: {stem: decision}, recs: {stem: image_eval}. Every rate carries its denominator."""
    st = list(recs)
    n = len(st)
    acc = [s for s in st if decisions[s] == "ACCEPT"]
    fails = [s for s in st if recs[s]["fail"]]
    ok = [s for s in st if not recs[s]["fail"]]
    af = [s for s in acc if recs[s]["fail"]]
    out = {"n": n, "accept": len(acc), "review": sum(decisions[s] == "REVIEW" for s in st), "full": sum(decisions[s] == "FULL_REINSPECTION" for s in st),
           "failures": len(fails), "accepted_failures": len(af),
           "coverage": len(acc) / n if n else None,
           "joint_risk": len(af) / n if n else None,
           "conditional_risk": (len(af) / len(acc)) if acc else "N/A",
           "conditional_risk_CI95": clopper_pearson(len(af), len(acc)) if acc else "N/A",
           "failure_capture": (1 - len(af) / len(fails)) if fails else "N/A",
           "correct_hold_rate": (sum(decisions[s] != "ACCEPT" for s in ok) / len(ok)) if ok else "N/A",
           "joint_risk_FN": sum(1 for s in acc if recs[s]["fail_fn"]) / n if n else None,
           "joint_risk_FP": sum(1 for s in acc if recs[s]["fail_fp"]) / n if n else None}
    out["review_rate"] = out["review"] / n if n else None
    out["full_rate"] = out["full"] / n if n else None
    ar = {s: recs[s] for s in acc}
    out["accepted_detection"] = detection_metrics(ar) if acc else "N/A"
    return out


def summarize_seeds(per_seed, keys_min=(), keys_max=()):
    """per_seed: {seed: metrics}. Mean over seeds, operating seed 17, and the true worst over seeds for the given keys."""
    out = {"seed17": per_seed.get(17, per_seed.get("17")), "per_seed": per_seed}
    num = collections.defaultdict(list)
    for m in per_seed.values():
        for k2, v in m.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None and math.isfinite(v):
                num[k2].append(v)
    out["mean"] = {k2: float(np.mean(v)) for k2, v in num.items()}
    out["worst"] = {**{k2: float(min(num[k2])) for k2 in keys_min if num.get(k2)}, **{k2: float(max(num[k2])) for k2 in keys_max if num.get(k2)}}
    return out


# ------------------------------------------------------------------------------------------------ calibration designs
def signals_of(img):
    import cv2
    return X.quality_signals(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), product_mask())


def date_halves(stems):
    return halves(stems, META)


def time_split(stems):
    dates = sorted({META[s]["date"] for s in stems})
    k = math.ceil(len(dates) / 2)
    early = set(dates[:k])
    return sorted(s for s in stems if META[s]["date"] in early), sorted(s for s in stems if META[s]["date"] not in early)


def calibration_from(cal_stems, clean_recs_by_seed, sig_clean):
    """Device calibration built only from cal_stems: multi-session quality reference, per-seed CRC lambda, and the empirical grade (fixed
    rule over seeds, date-split assessment within cal_stems with LOSO gate inside cal_stems)."""
    crc = CFG["crc"]
    session = {s: META[s]["date"] for s in cal_stems}
    ref = X.build_quality_reference(cal_stems, sig_clean, session)
    loso = {}
    for sess in sorted(set(session.values())):
        r = X.build_quality_reference(cal_stems, sig_clean, session, exclude_session=sess)
        for s in cal_stems:
            if session[s] == sess:
                loso[s] = r
    gate = {s: X.quality_state(sig_clean[s], loso[s], CFG["quality_gate"])[0] if len(set(session.values())) > 1 else "NORMAL" for s in cal_stems}
    cluster = {s: META[s]["date"] + META[s]["device"] for s in cal_stems}
    per_seed, succ, fps, cov, st = {}, [], [], [], []
    for sd, rec in clean_recs_by_seed.items():
        trust = [rec[s]["trust"] for s in cal_stems]
        fail = [rec[s]["fail"] for s in cal_stems]
        lam, bound = X.crc_lambda(trust, fail, crc["alpha"])
        if len(set(cluster.values())) >= 2:
            assess = X.crc_split_assessment({s: rec[s]["trust"] - (1. if gate[s] != "NORMAL" else 0.) for s in cal_stems},
                                            {s: rec[s]["fail"] for s in cal_stems}, cal_stems, cluster, delta=crc["ltt_delta"])
            cov.append(assess["0.05"]["CRC"]["median_coverage"])
            st.append(assess["0.05"]["CRC"]["status"])
        else:
            cov.append(0.0)
            st.append("FULL_REINSPECTION")
        per_seed[sd] = {"N": len(cal_stems), "lambda": lam, "bound": bound}
        succ.append(sum(rec[s]["tp3"] for s in cal_stems) / max(1, sum(rec[s]["n"] for s in cal_stems)))
        fps.append(float(np.mean([rec[s]["fp3"] for s in cal_stems])))
    g = X.qualification_grade(min(succ), float(np.mean(fps)), float(np.median(cov)), collections.Counter(st).most_common(1)[0][0])
    for v in per_seed.values():
        v["grade"] = g.value
    return {"reference": ref, "seeds": per_seed, "grade": g.value, "n_cal": len(cal_stems), "cal_dates": sorted(set(session.values()))}


def design(profile, clean_recs_by_seed, sig_clean, kind="IND", grade_mode="strict"):
    """{stem: calibration package} for the evaluation images of `profile`'s held fold, per device, under P-IND (other date half) or P-TIME
    (earlier dates calibrate later dates; earlier-date images are not evaluated).
    grade_mode 'strict': the empirical grade is computed from the calibration part only (fully independent);
    grade_mode 'deployment': the grade is taken from the deployment calibration (all labeled device images, i.e. in-sample for the grade
    only) while the quality reference and CRC lambda still come from the calibration part — isolates the lambda / gate behaviour."""
    tg = targets(profile)
    dep = deployment_grades(profile) if grade_mode == "deployment" else None
    out = {}
    for d in sorted({META[s]["device"] for s in tg}):
        ss = [s for s in tg if META[s]["device"] == d]
        parts = []
        if kind == "IND":
            hv = date_halves(ss)
            parts = [(hv[1 - h], hv[h]) for h in (0, 1)]
        elif kind == "TIME":
            early, late = time_split(ss)
            parts = [(early, late)]
        for cal, ev in parts:
            pkg = calibration_from(cal, clean_recs_by_seed, sig_clean)
            if dep is not None:
                pkg = {**pkg, "grade": dep[d], "seeds": {sd: {**v, "grade": dep[d]} for sd, v in pkg["seeds"].items()}}
            for s in ev:
                out[s] = pkg
    return out


def deployment_grades(profile):
    """Device grades of the deployment calibration artifact (outputs/calibration/crc_<profile>.json; 1.0.0 artifacts carry the text form)."""
    p = ROOT / "outputs" / "calibration" / f"crc_{profile}.json"
    d = json.loads(p.read_text(encoding="utf8"))["devices"]
    out = {}
    for dev, v in d.items():
        g = v.get("grade") or next(k.value for k, t in X.GRADE_TEXT.items() if t == v.get("qualification"))
        out[dev] = g
    return out


def loso_references(stems_all, sig_all):
    """X14 design: per device, the quality reference of each session = all OTHER development sessions of that device (all dev images,
    including images that trained the profile), as in the sealed X14 / regression harness."""
    session = {s: META[s]["date"] for s in stems_all}
    out = {}
    for d in sorted({META[s]["device"] for s in stems_all}):
        ss = [s for s in stems_all if META[s]["device"] == d]
        for sess in sorted({session[s] for s in ss}):
            r = X.build_quality_reference(ss, sig_all, session, exclude_session=sess)
            for s in ss:
                if session[s] == sess:
                    out[s] = r
    return out


def decide_image(pkg, seed, sig, rec, sc):
    gate_state, zmax, z = X.quality_state(sig, pkg["reference"], CFG["quality_gate"])
    ok, why = X.scale_envelope(sc, CFG["scale_policy"])
    probs = X.input_problems(np.zeros((1, 1, 3), np.uint8), "device", sig, None, rec["trust"])
    d, reasons, env = X.decide(gate_state, pkg["seeds"][seed], rec["trust"], rec["n_det"], CFG["crc"], ok, why, probs, True)
    return d, gate_state, zmax


# ------------------------------------------------------------------------------------------------ degradations
def degrade_new(im, cond, rng):
    import cv2
    h, w = im.shape[:2]
    f = im.astype(np.float32)
    if cond.startswith("cnoise"):
        sc = float(cond[6:])
        n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sc)
        n = n / (n.std() + 1e-9) * 4.0
        return np.clip(f + n[..., None], 0, 255).astype(np.uint8)
    if cond.startswith("disk"):
        r = int(cond[4:])
        k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
        cv2.circle(k, (r, r), r, 1, -1)
        return np.clip(cv2.filter2D(f, -1, k / k.sum(), borderType=cv2.BORDER_REFLECT_101), 0, 255).astype(np.uint8)
    if cond == "motion3":
        return np.clip(cv2.filter2D(f, -1, np.ones((1, 3), np.float32) / 3, borderType=cv2.BORDER_REFLECT_101), 0, 255).astype(np.uint8)
    if cond == "aniso":
        return np.clip(cv2.GaussianBlur(f, (0, 0), sigmaX=1.0, sigmaY=0.3, borderType=cv2.BORDER_REFLECT_101), 0, 255).astype(np.uint8)
    if cond in ("blur16", "resample05", "noise8"):
        return degrade_x14(im, cond, rng)
    raise ValueError(cond)


def degrade_any(im, cond, stem):
    if cond == "clean":
        return im
    if cond in ("blur04", "blur08", "resample085", "resample07", "noise2", "noise4", "composition"):
        return degrade_x14(im, cond, phase_rng(20261002, X14_HD, 0, stem, cond))
    if "+" in cond:                      # combination, e.g. blur08+noise4 (applied in order)
        for i, c in enumerate(cond.split("+")):
            im = degrade_any(im, c, stem + f"#{i}")
        return im
    return degrade_new(im, cond, rng_for("degrade", cond, stem))


# ------------------------------------------------------------------------------------------------ synthesis (R2)
def shift_image(im, dx, dy):
    import cv2
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)


def gray(im):
    import cv2
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def bgr(g):
    import cv2
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def box_px(b, w, h):
    return [b[0] * w, b[1] * h, b[2] * w, b[3] * h]


def object_patch(g8, bx, margin=5):
    """Attenuation ratio window of one object: ratio = image / local background (closing), 1 outside the GT box."""
    import cv2
    x0, y0, x1, y1 = bx
    bw, bh = x1 - x0, y1 - y0
    k = int(2 * max(bw, bh) + 1) | 1
    X0, Y0 = int(max(0, math.floor(x0 - margin))), int(max(0, math.floor(y0 - margin)))
    X1, Y1 = int(min(g8.shape[1], math.ceil(x1 + margin))), int(min(g8.shape[0], math.ceil(y1 + margin)))
    big = g8[max(0, Y0 - k):Y1 + k, max(0, X0 - k):X1 + k].astype(np.float32)
    bg = cv2.morphologyEx(big, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    oy, ox = Y0 - max(0, Y0 - k), X0 - max(0, X0 - k)
    win = g8[Y0:Y1, X0:X1].astype(np.float32)
    bgw = bg[oy:oy + (Y1 - Y0), ox:ox + (X1 - X0)]
    ratio = np.clip(win / np.maximum(bgw, 1.0), 0, 1.0)
    inbox = np.zeros_like(ratio, bool)
    inbox[int(max(0, y0 - 1 - Y0)):int(min(Y1, y1 + 1) - Y0), int(max(0, x0 - 1 - X0)):int(min(X1, x1 + 1) - X0)] = True
    ratio[~inbox] = 1.0
    return {"ratio": ratio, "win": win.astype(np.uint8), "mask": (ratio < 0.985).astype(np.uint8), "off": (x0 - X0, y0 - Y0), "size": (bw, bh)}


def remove_object(g8, bx, pad=3):
    import cv2
    m = np.zeros_like(g8)
    x0, y0, x1, y1 = bx
    m[int(max(0, y0 - pad)):int(min(g8.shape[0], y1 + pad + 1)), int(max(0, x0 - pad)):int(min(g8.shape[1], x1 + pad + 1))] = 255
    return cv2.inpaint(g8, m, 3, cv2.INPAINT_TELEA)


def insert_object(g8, patch, cx, cy, method):
    """Place the object so that its GT-box centre lands on (cx, cy). Returns (image, new normalized box) or (None, None) if it does not fit."""
    import cv2
    h, w = g8.shape
    ratio, win = patch["ratio"], patch["win"]
    ph, pw = ratio.shape
    bw, bh = patch["size"]
    offx, offy = patch["off"]
    X0 = int(round(cx - bw / 2 - offx))
    Y0 = int(round(cy - bh / 2 - offy))
    if X0 < 1 or Y0 < 1 or X0 + pw >= w - 1 or Y0 + ph >= h - 1:
        return None, None
    out = g8.copy()
    if method == "M1":
        reg = out[Y0:Y0 + ph, X0:X0 + pw].astype(np.float32) * ratio
        out[Y0:Y0 + ph, X0:X0 + pw] = np.clip(np.round(reg), 0, 255).astype(np.uint8)
    else:
        m = cv2.dilate(patch["mask"] * 255, np.ones((3, 3), np.uint8))
        if m.sum() == 0:
            return None, None
        src = np.zeros_like(out)
        src[Y0:Y0 + ph, X0:X0 + pw] = win
        full = np.zeros_like(out)
        full[Y0:Y0 + ph, X0:X0 + pw] = m
        ys, xs = np.nonzero(full)
        center = (int((xs.min() + xs.max()) // 2), int((ys.min() + ys.max()) // 2))
        res = cv2.seamlessClone(bgr(src), bgr(out), full, center, cv2.NORMAL_CLONE)
        out = gray(res)
    nx0, ny0 = X0 + offx, Y0 + offy
    return out, [nx0 / w, ny0 / h, (nx0 + bw) / w, (ny0 + bh) / h]


def free_spot(g8, avoid, rng, min_dist=20, edge=12, near=None, dist_range=None, tries=600):
    """Random product-interior spot (distance transform > edge) away from `avoid` points; optional distance range from `near`."""
    import cv2
    dist = cv2.distanceTransform(product_mask()(g8, empty=0), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(dist > edge)
    if not len(ys):
        return None
    for _ in range(tries):
        i = rng.integers(len(ys))
        x, y = float(xs[i]), float(ys[i])
        if near is not None:
            dd = math.hypot(x - near[0], y - near[1])
            if not (dist_range[0] <= dd <= dist_range[1]):
                continue
        if all(math.hypot(x - a, y - b) > min_dist for a, b in avoid):
            return x, y
    return None


def detected_at(boxes, w, h, pt, r=3.0, thr=THR):
    return any(math.hypot((b[0] + b[2]) / 2 * w - pt[0], (b[1] + b[3]) / 2 * h - pt[1]) <= r for b in boxes if b[4] >= thr)
