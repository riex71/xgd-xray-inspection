"""Example figures for XGD-1.2 (read-only over completed follow-up artifacts; no selection, no new values).

For each profile, from the fixed post-selection validation file (SELECTED_VALIDATION_<profile>_<family>_<policy>.json, seed 17):
  A. missed object caught: baseline ACCEPT with an actually missing GT (no prediction within 8 px) and XGD-1.2 not ACCEPT;
  B. remaining risk: XGD-1.2 ACCEPT with a 3 px failure (any condition);
  C. normal case: clean development image, XGD-1.2 ACCEPT, no failure.
Green = TXT GT, red = confirmed detections (score >= operating score), orange = lower-score proposals (0.05 <= score < operating score).
Images are enlarged 3x. Writes outputs/followup/examples/*.png and docs/FAILURE_EXAMPLES_XGD-1.2.md.

python -m experiments.followup.examples12
"""
import json

import cv2
import numpy as np

from experiments import lib as L
from experiments.followup import analyze as A
from experiments.followup import synth as S

OUT = L.ROOT / "outputs" / "followup"
EX = OUT / "examples"
FAM = {"dfl_src12": "3호기", "dfl_src3_2": "1·2호기"}


def image_of(prof, cond, stem):
    return L.clean_img(stem) if cond == "dev_clean" else S.load_image(prof, cond, stem)


def draw(img, gt, boxes, score, scale=3):
    h, w = img.shape[:2]
    big = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    def rect(b, color, thick):
        x0, y0, x1, y1 = (int(round(v * s * scale)) for v, s in zip(b[:4], (w, h, w, h)))
        cv2.rectangle(big, (x0, y0), (x1, y1), color, thick)
    for b in boxes:
        if 0.05 <= b[4] < score:
            rect(b, (0, 165, 255), 1)
    for g in gt:
        rect(g, (0, 200, 0), 2)
    for b in boxes:
        if b[4] >= score:
            rect(b, (0, 0, 255), 2)
    return big


def crop_to_product(big, gt, w, h, scale=3, margin=60):
    xs = [v * w * scale for g in gt for v in (g[0], g[2])]
    ys = [v * h * scale for g in gt for v in (g[1], g[3])]
    if not xs:
        return big
    x0, x1 = max(0, int(min(xs)) - margin), min(big.shape[1], int(max(xs)) + margin)
    y0, y1 = max(0, int(min(ys)) - margin), min(big.shape[0], int(max(ys)) + margin)
    return big[y0:y1, x0:x1]


def main():
    sel = json.loads((OUT / "FINAL_SELECTION.json").read_text(encoding="utf8"))
    A.initialize()
    EX.mkdir(parents=True, exist_ok=True)
    lines = ["# XGD-1.2 사례 그림", "",
             "초록 = TXT 정답, 빨강 = 확정 검출(운영 점수 이상), 주황 = 낮은 점수 후보(0.05 이상 운영 점수 미만). 영상은 3배 확대 후 정답 주변을 잘랐다. "
             "seed 17, 고정 검증(선택 뒤) 결과에서 골랐고 정답은 개발 자료 TXT다. 합성 조건 영상은 실제 현장 영상이 아니다.", ""]
    for prof in ("dfl_src12", "dfl_src3_2"):
        hist = sel["validation_history"][prof]
        item = [h for h in hist if h["passed"]][-1]["item"]
        fam, cfg = item["family"], item["policy"]
        val = json.loads((OUT / f"SELECTED_VALIDATION_{prof}_{fam}_{cfg['id']}.json").read_text(encoding="utf8"))
        cand, base = val["candidate_results"], val["baseline_results"]
        picks = []
        for cond in ("R2f_free_M2", "R2f_free_M2v2", "R2a_move_M2_0", "R2c_context_M2", "R2f_free_M1"):
            if cond not in cand:
                continue
            brows = {r["stem"]: r for r in base[cond]["per_seed"]["17"]["rows"]}
            for r in cand[cond]["per_seed"]["17"]["rows"]:
                b = brows.get(r["stem"])
                if b and b["decision"] == "ACCEPT" and b["actual_missing8"] > 0 and r["decision"] != "ACCEPT":
                    picks.append(("A", cond, r, b))
                    break
            if any(p[0] == "A" for p in picks):
                break
        for cond, C in cand.items():
            rows = [r for r in C["per_seed"]["17"]["rows"] if r["decision"] == "ACCEPT" and r["fail"]]
            if rows:
                picks.append(("B", cond, rows[0], None))
                break
        rows = [r for r in cand["dev_clean"]["per_seed"]["17"]["rows"] if r["decision"] == "ACCEPT" and not r["fail"]]
        if rows:
            picks.append(("C", "dev_clean", rows[0], None))
        title = {"A": "놓친 이물을 자동 확정하지 않음(1.1 정책은 자동 확정)", "B": "남은 위험: XGD-1.2도 자동 확정한 실패", "C": "정상 사례: 자동 확정, 실패 없음"}
        lines += [f"## {FAM[prof]} — 최종 {fam} / {cfg['id']}", ""]
        for kind, cond, r, b in picks:
            stem = r["stem"]
            img = image_of(prof, cond, stem)
            gt, (w, h), _ = A.data(prof, stem, cond)
            boxes = A.processed(fam, prof, 17, cond, cfg)[0][stem]
            raw = A.raw(fam, prof, 17, "base", cond)[stem]["boxes"]
            low = [x for x in raw if 0.05 <= x[4] < cfg["score"]]
            fig = crop_to_product(draw(img, gt, list(boxes) + low, cfg["score"]), gt, w, h)
            name = f"{prof}_{kind}_{cond}_{stem}.png"
            ok, buf = cv2.imencode(".png", fig)          # cv2.imwrite cannot open non-ASCII Windows paths
            if not ok:
                raise RuntimeError(f"PNG encoding failed: {name}")
            (EX / name).write_bytes(buf.tobytes())
            why = " / ".join(r["reasons"])[:300]
            lines += [f"### {title[kind]}", "", f"![{stem} {cond}](../outputs/followup/examples/{name})", "",
                      f"- 조건 `{cond}`, 영상 `{stem}`: 정답 {r['n']}개, 확정 검출 {r['n_det']}개, 3px 놓침 {r['FN3']}, 오탐 {r['FP3']}, 8px 안 예측 없는 놓침 {r['actual_missing8']}.",
                      f"- XGD-1.2 판정 {r['decision']} (사유: {why})" + (f"; 1.1 정책 판정 {b['decision']}" if b else ""), ""]
    (L.ROOT / "docs" / "FAILURE_EXAMPLES_XGD-1.2.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("examples written", flush=True)


if __name__ == "__main__":
    main()
