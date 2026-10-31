"""E2. GPU prediction stage for the pre-registered robustness / comparison experiments (docs/PREREGISTRATION_XGD-1.1.md §3-6).

Predictions are cached in outputs/experiments/cache/<model key>/<condition>.json (boxes normalized to the input image, after E0; for
synthetic conditions also the updated ground truth and the special points). Scoring is done separately (e3_score.py, CPU).

  python -m experiments.e2_predict --groups clean r1 r3 grid c2
  python -m experiments.e2_predict --groups r2 r2b
  python -m experiments.e2_predict --groups baseline native
  python -m experiments.e2_predict --groups v3only
"""
import argparse
import math
import hashlib
import json
import importlib.metadata as package_metadata
from pathlib import Path

import numpy as np

from experiments import lib as L
from src import preprocess as P

SHIFTS = [(1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2), (3, 0), (-3, 0), (0, 3), (0, -3), (1, 1), (-1, -1)]
R3_CONDS = ["blur16", "resample05", "noise8", "cnoise1", "cnoise2", "disk1", "disk2", "motion3", "aniso", "blur08+noise4"]
GRID = [(f, q) for f in (0.9, 1.1) for q in ("clean", "blur08", "noise4", "blur08+noise4")]
METHODS = ("M1", "M2")


def deploy_models():
    return [(prof, sd) for prof in ("dfl_src12", "dfl_src3_2") for sd in L.SEEDS]


def run_condition(m, cond, stems, make, scale_factor=1.0, mode="x8"):
    def fn(s):
        img, extra = make(s)
        if img is None:
            return None, None
        h, w = img.shape[:2]
        sc = m.scale(w, h, mode=mode, factor=scale_factor)
        return {"boxes": m.predict(img, sc["imgsz"]), "resolution": [w, h], "signals": L.signals_of(img), **extra}, sc["imgsz"]
    p = L.CACHE / m.key / f"{cond}.json"
    seal = p.with_suffix(".manifest.json")
    signature = {"checkpoint": m.an.ckpt_sha256, "source": {str(f.relative_to(L.ROOT)): L.X.sha256_file(f)
                 for f in (Path(__file__), L.ROOT / "experiments/lib.py", L.ROOT / "src/preprocess.py", L.ROOT / "src/xgd_analyzer.py")},
                 "config": L.X.sha256_file(L.ROOT / "config.yaml"), "condition": cond,
                 "stems": hashlib.sha256("\n".join(stems).encode()).hexdigest(),
                 "inputs": {f: L.X.sha256_file(L.ROOT / f) for f in ("data/split.csv", "data/metadata.json", "data/v3_pixel_sha256.json")},
                 "packages": {k: package_metadata.version(k) for k in ("torch", "ultralytics", "numpy", "opencv-python")},
                 "scale_factor": scale_factor, "mode": mode}
    if p.exists() and (not seal.exists() or json.loads(seal.read_text(encoding="utf8")) != signature):
        raise RuntimeError(f"Cache provenance mismatch: {p}. Existing result preserved; no silent reuse.")
    if not seal.exists():
        L.jdump(seal, signature)
    d = json.loads(p.read_text(encoding="utf8")) if p.exists() else {}
    todo = [s for s in stems if s not in d]
    for i, s in enumerate(todo):
        res, z = fn(s)
        d[s] = None if res is None else {**res, "imgsz": z}
        if (i + 1) % 20 == 0:
            L.jdump(p, d)
    if todo:
        L.jdump(p, d)
        print("prediction", m.key, cond, len(d), flush=True)
    return d


# ------------------------------------------------------------------------------------------------ synthetic variants (R2)
class Synth:
    def __init__(self, profile):
        self.tg = L.targets(profile)
        self.g = {}

    def base(self, s):
        if s not in self.g:
            img = L.clean_img(s)
            w, h = L.size_of(s)
            g8 = L.gray(img)
            gt = L.gt(s)
            bxs = [L.box_px(b, w, h) for b in gt]
            self.g[s] = {"g8": g8, "gt": gt, "bx": bxs, "w": w, "h": h, "marks": [tuple(map(float, c)) for c in P.mark_centres(s)],
                         "centres": [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in bxs]}
        return self.g[s]

    def donor(self, s):
        """Object from another image of the same device and a different date (deterministic)."""
        dev, date = L.META[s]["device"], L.META[s]["date"]
        pool = [t for t in self.tg if L.META[t]["device"] == dev and L.META[t]["date"] != date]
        i = self.tg.index(s)
        t = min(pool, key=lambda x: (self.tg.index(x) - i) % len(self.tg))
        B = self.base(t)
        j = int(L.rng_for("donor", s).integers(len(B["bx"])))
        return t, j, L.object_patch(B["g8"], B["bx"][j])

    def move(self, s, j, method, same=False):
        B = self.base(s)
        if j >= len(B["bx"]):
            return None, {"skip_reason": "registered object index does not exist"}
        patch = L.object_patch(B["g8"], B["bx"][j])
        base = L.remove_object(B["g8"], B["bx"][j])
        old = B["centres"][j]
        if same:
            spot = old
        else:
            avoid = [c for k, c in enumerate(B["centres"]) if k != j] + B["marks"]
            spot = L.free_spot(base, avoid, L.rng_for("R2a", s, j), min_dist=20, edge=12, near=old, dist_range=(15, 60))
            if spot is None:
                return None, {}
        img, nb = L.insert_object(base, patch, spot[0], spot[1], method)
        if img is None:
            return None, {}
        gt = [nb if k == j else b for k, b in enumerate(B["gt"])]
        actual = [(nb[0] + nb[2]) / 2 * B["w"], (nb[1] + nb[3]) / 2 * B["h"]]
        return L.bgr(img), {"gt": gt, "points": {"new": actual, "requested": list(spot), "old": list(old), "object_index": j}}

    def sweep(self, s, k, method):
        import cv2
        B = self.base(s)
        j = 0
        patch = L.object_patch(B["g8"], B["bx"][j])
        base = L.remove_object(B["g8"], B["bx"][j])
        pm = L.product_mask()(base, empty=0)
        ys, xs = np.nonzero(pm)
        x = float(xs.min() + (k + 0.5) / 10 * (xs.max() - xs.min()))
        dist = cv2.distanceTransform(pm, cv2.DIST_L2, 5)
        avoid = [c for kk, c in enumerate(B["centres"]) if kk != j] + B["marks"]
        y0 = B["centres"][j][1]
        for dy in [0] + [d * sgn for d in range(4, 44, 4) for sgn in (1, -1)]:
            y = y0 + dy
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < dist.shape[0] and 0 <= xi < dist.shape[1] and dist[yi, xi] > 12 and all(math.hypot(x - a, y - b) > 20 for a, b in avoid):
                img, nb = L.insert_object(base, patch, x, y, method)
                if img is None:
                    continue
                gt = [nb if kk == j else b for kk, b in enumerate(B["gt"])]
                return L.bgr(img), {"gt": gt, "points": {"new": [x, y], "bin": k, "x_rel": (x - xs.min()) / max(1, xs.max() - xs.min())}}
        return None, {}

    def surround(self, s):
        import cv2
        B = self.base(s)
        out = cv2.GaussianBlur(B["g8"], (0, 0), 1.2, borderType=cv2.BORDER_REFLECT_101)
        for bx in B["bx"]:
            x0, y0 = int(max(0, math.floor(bx[0] - 8))), int(max(0, math.floor(bx[1] - 8)))
            x1, y1 = int(min(B["w"], math.ceil(bx[2] + 8))), int(min(B["h"], math.ceil(bx[3] + 8)))
            out[y0:y1, x0:x1] = B["g8"][y0:y1, x0:x1]
        return L.bgr(out), {"gt": B["gt"], "points": {}}

    def context(self, s, method):
        """Same object's signal on another date's product. Recipient GT retained; label updates are explicit."""
        B = self.base(s)
        pool = [t for t in self.tg if L.META[t]["device"] == L.META[s]["device"] and L.META[t]["date"] != L.META[s]["date"]]
        t = pool[int(L.rng_for("R2c", s).integers(len(pool)))]
        C = self.base(t)
        patch = L.object_patch(B["g8"], B["bx"][0])
        spot = L.free_spot(C["g8"], C["centres"] + C["marks"], L.rng_for("R2c-spot", s), min_dist=25, edge=12)
        if spot is None:
            return None, {}
        img, nb = L.insert_object(C["g8"], patch, *spot, method)
        if img is None:
            return None, {}
        return L.bgr(img), {"gt": C["gt"] + [nb], "points": {"new": [(nb[0]+nb[2])*C["w"]/2, (nb[1]+nb[3])*C["h"]/2],
                            "recipient": t, "donor": [s, 0]}}

    def trace(self, s):
        B = self.base(s)
        rng = L.rng_for("R2e", s)
        g = B["g8"].copy()
        spots = []
        for _ in range(2):
            sp = L.free_spot(g, B["centres"] + B["marks"] + spots, rng, min_dist=30, edge=18)
            if sp is None:
                break
            spots.append(sp)
            g = P._gray(P.clean_v3(P.draw_mark(g, int(sp[0]), int(sp[1]), rng)))
        return L.bgr(g), {"gt": B["gt"], "points": {"spots": [list(p) for p in spots]}}

    def free(self, s, method, mark=False):
        B = self.base(s)
        t, j, patch = self.donor(s)
        rng = L.rng_for("R2f", s)
        spot = L.free_spot(B["g8"], B["centres"] + B["marks"], rng, min_dist=25, edge=12)
        if spot is None:
            return None, {}
        img, nb = L.insert_object(B["g8"], patch, spot[0], spot[1], method)
        if img is None:
            return None, {}
        if mark:
            c = ((nb[0] + nb[2]) / 2 * B["w"], (nb[1] + nb[3]) / 2 * B["h"])
            img = P._gray(P.clean_v3(P.draw_mark(img, int(round(c[0])), int(round(c[1])), L.rng_for("R2g", s))))
        return L.bgr(img), {"gt": B["gt"] + [nb], "points": {"new": [(nb[0]+nb[2])*B["w"]/2, (nb[1]+nb[3])*B["h"]/2], "donor": [t, j]}}


def r2_variants(syn, include_sweep=False):
    v = {}
    for meth in METHODS:
        for j in range(3):
            v[f"R2a_move_{meth}_{j}"] = (lambda s, j=j, meth=meth: syn.move(s, j, meth))
            v[f"R2a_same_{meth}_{j}"] = (lambda s, j=j, meth=meth: syn.move(s, j, meth, same=True))
        v[f"R2f_free_{meth}"] = (lambda s, meth=meth: syn.free(s, meth))
        v[f"R2g_mark_{meth}"] = (lambda s, meth=meth: syn.free(s, meth, mark=True))
        v[f"R2c_context_{meth}"] = (lambda s, meth=meth: syn.context(s, meth))
        if include_sweep:
            for k in range(10):
                v[f"R2b_sweep_{meth}_{k}"] = (lambda s, k=k, meth=meth: syn.sweep(s, k, meth))
    v["R2d_surround"] = syn.surround
    v["R2e_trace"] = syn.trace
    return v


def plain(s, cond):
    return L.degrade_any(L.clean_img(s), cond, s), {}


def shifted(s, dx, dy):
    return L.shift_image(L.clean_img(s), dx, dy), {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=L.SEEDS)
    a = ap.parse_args()
    g = set(a.groups)
    if g & {"clean", "r1", "r3", "grid", "c2", "r2", "r2b", "native"}:
        for prof, sd in deploy_models():
            if sd not in a.seeds:
                continue
            m = L.Model(prof, sd)
            tg = L.targets(prof)
            if "clean" in g:
                run_condition(m, "clean", tg, lambda s: plain(s, "clean"))
            if "native" in g:
                run_condition(m, "native", tg, lambda s: plain(s, "clean"), mode="native")
            if "r1" in g:
                for dx, dy in SHIFTS:
                    run_condition(m, f"R1_{dx}_{dy}", tg, lambda s, dx=dx, dy=dy: shifted(s, dx, dy))
            if "r3" in g:
                for c in R3_CONDS:
                    run_condition(m, f"R3_{c}", tg, lambda s, c=c: plain(s, c))
            if "grid" in g and prof == "dfl_src12":
                for f, q in GRID:
                    run_condition(m, f"GRID_{f:g}_{q}", tg, lambda s, q=q: plain(s, q), scale_factor=f)
            if "c2" in g:
                run_condition(m, "GRID_1.1_clean", tg, lambda s: plain(s, "clean"), scale_factor=1.1)
            if "r2" in g or "r2b" in g:
                syn = Synth(prof)
                for name, fn in r2_variants(syn, include_sweep="r2b" in g).items():
                    if name.startswith("R2b") and "r2b" not in g:
                        continue
                    if not name.startswith("R2b") and "r2" not in g:
                        continue
                    run_condition(m, name, tg, fn)
            print("E2", prof, sd, sorted(g), flush=True)
            del m
    if "baseline" in g:
        for prof in ("yolo_src12", "yolo_src3_2"):
            for sd in a.seeds:
                m = L.Model(prof, sd)
                tg = L.targets(prof)
                run_condition(m, "clean", tg, lambda s: plain(s, "clean"))
                run_condition(m, "native", tg, lambda s: plain(s, "clean"), mode="native")
                print("E2 baseline", prof, sd, flush=True)
                del m
    if "v3only" in g:
        for prof in ("dfl_src12", "dfl_src3_2"):
            c = L.cfg_with_checkpoints({prof: {sd: f"outputs/controls/v3only/retrain/{prof}_s{sd}/weights/last.pt" for sd in L.SEEDS}})
            for sd in a.seeds:
                m = L.Model(prof, sd, cfg=c, key=f"v3only_{prof}_s{sd}")
                tg = L.targets(prof)
                run_condition(m, "clean", tg, lambda s: plain(s, "clean"))
                syn = Synth(prof)
                v = r2_variants(syn)
                for name in ("R2e_trace", "R2f_free_M1", "R2f_free_M2", "R2g_mark_M1", "R2g_mark_M2"):
                    run_condition(m, name, tg, v[name])
                print("E2 v3only", prof, sd, flush=True)
                del m


if __name__ == "__main__":
    main()
