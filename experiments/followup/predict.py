"""Raw prediction collection for the XGD-1.2 follow-up (GPU). Boxes are stored before E0 with the deployment predict settings so that
score thresholds, E0 radius, view fusion and completeness checks can be re-applied offline.

  python -m experiments.followup.predict --family dfl --views base hflip vflip hvflip sm32 sp32 --conds dev_clean
  python -m experiments.followup.predict --family dfl --views base hflip vflip --conds synth
  python -m experiments.followup.predict --family v3only --views base --conds dev_clean synth

Condition 'dev_clean' = all 356 development images (held fold and training images); synthetic conditions = the profile's held fold.
"""
import argparse
import time

from experiments import lib as L
from experiments.followup import common as F
from experiments.followup import synth as S


def inputs_for(profile, cond):
    if cond == "dev_clean":
        stems = sorted(L.DEV)
        digest = F.sha_bytes("\n".join(f"{s}:{F.sha_file(L.XC.clean_path(L.CFG, s))}" for s in stems).encode())
        return stems, (lambda s: L.clean_img(s)), digest
    meta = F.read_done(F.SYNTH / profile / cond / "meta.json")
    if meta is None:
        raise RuntimeError(f"synthetic set missing or incomplete: {profile}/{cond}")
    stems = [s for s in L.targets(profile) if meta.get(s)]
    digest = F.sha_file(F.SYNTH / profile / cond / "meta.json")
    return stems, (lambda s: S.load_image(profile, cond, s)), digest


def run(family, profiles, seeds, views, conds):
    for prof in profiles:
        for sd in seeds:
            m = F.load_model(family, prof, sd)
            for cond in conds:
                stems, load, digest = inputs_for(prof, cond)
                for view in views:
                    path = F.PRED / m.key / view / f"{cond}.json"
                    sig = F.signature(m, view, cond, stems, digest)
                    if F.read_done(path, sig) is not None:
                        continue
                    t0 = time.perf_counter()
                    partial=path.with_name(path.stem+'.partial.json')
                    res={}
                    if partial.exists():
                        old=__import__('json').loads(partial.read_text(encoding='utf8'))
                        if old['signature']!=sig:raise RuntimeError('Partial checkpoint signature changed; preserved')
                        res=old['records']
                    for i,s in enumerate(stems):
                        if s in res:continue
                        img = load(s)
                        boxes, z = F.predict_view(m, img, view)
                        h, w = img.shape[:2]
                        res[s] = {"boxes": boxes, "imgsz": z, "resolution": [w, h]}
                        if (i+1)%20==0:
                            # Partial checkpoints are separate from immutable completed predictions.
                            F.atomic_json(partial,{'signature':sig,'records':res})
                    F.publish(path, res, sig)
                    dt = time.perf_counter() - t0
                    F.status(stage="predict", family=family, model=m.key, view=view, condition=cond, images=len(stems),
                             seconds_per_image=dt / max(1, len(stems)))
                    print("pred", m.key, view, cond, len(stems), f"{dt / max(1, len(stems)):.3f}s/img", flush=True)
            del m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["dfl", "v3only", "trace05"])
    ap.add_argument("--profiles", nargs="+", default=["dfl_src12", "dfl_src3_2"])
    ap.add_argument("--seeds", type=int, nargs="+", default=L.SEEDS)
    ap.add_argument("--views", nargs="+", default=["base"])
    ap.add_argument("--conds", nargs="+", default=["dev_clean"])
    a = ap.parse_args()
    conds = [c for x in a.conds for c in (S.CONDS if x == "synth" else [x])]
    run(a.family, a.profiles, a.seeds, a.views, conds)


if __name__ == "__main__":
    main()
