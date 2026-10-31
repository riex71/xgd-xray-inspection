"""E9 (exploratory, post-hoc; docs/PREREG_E9_MISSED_OBJECT_BLINDSPOT.md recorded 2026-10-06 01:49 KST before the C3 part ran).

1. Accepted failures in the synthetic-object conditions split into FN risk P(ACCEPT and miss) and FP risk P(ACCEPT and false alarm), from the
   E3 decision metrics (no new decisions).
2. Pre-registered candidate C3 (seed disagreement at 3 px) recomputed on the synthetic conditions from the sealed E2 caches: how often it
   flags failed vs correct images. No adoption, no threshold or rule change.

python -m experiments.e9_missed_object_blindspot   -> outputs/experiments/E9_missed_object_blindspot/RESULTS.json, SUMMARY.md
"""
import json

from experiments import lib as L
from experiments.e3_score import condition_records, load_cache, pairs

CONDS = ["clean", "R2f_free_M1", "R2f_free_M2", "R2a_move_M1_0", "R2a_move_M2_0", "R2a_same_M2_0", "R2c_context_M2", "R2g_mark_M2"]
DESIGNS = ("P-IND-strict", "P-IND-deployment-grade")


def main():
    e3 = json.loads((L.OUT / "E3_scoring/RESULTS.json").read_text(encoding="utf8"))
    out = {"prereg": "docs/PREREG_E9_MISSED_OBJECT_BLINDSPOT.md (2026-10-06 01:49 KST)", "exploratory": True, "adopted": False, "profiles": {}}
    lines = ["# E9. 놓친 이물의 자동 확정(탐색, 사후 동기)", "", f"- 사전 기록: {out['prereg']}. 규칙·임계값 변경 없음, C3 채택 없음.", ""]
    for prof in ("dfl_src12", "dfl_src3_2"):
        P = {}
        lines += [f"## {prof}", "", "| 조건 | 설계 | ACCEPT(3 seed 합) | 그중 실패 | FN 위험(평균) | FP 위험(평균) | C3 표시: 실패 영상 | C3 표시: 정상 영상 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for cond in CONDS:
            C = e3["profiles"][prof]["conditions"].get(cond)
            if not C:
                continue
            ps = C["per_seed"]
            dec = {}
            for d in DESIGNS:
                acc = sum(m[d]["accept"] for m in ps.values())
                af = sum(m[d]["accepted_failures"] for m in ps.values())
                fn = sum(m[d].get("joint_risk_FN") or 0 for m in ps.values()) / len(ps)
                fp = sum(m[d].get("joint_risk_FP") or 0 for m in ps.values()) / len(ps)
                dec[d] = {"accept": acc, "accepted_failures": af, "FN_risk_mean": fn, "FP_risk_mean": fp}
            caches = {sd: load_cache(f"{prof}_s{sd}", cond) for sd in L.SEEDS}
            recs = {sd: condition_records(c, cond)[0] for sd, c in caches.items()}
            flag = {"fail": [0, 0], "ok": [0, 0]}
            for sd in L.SEEDS:
                for s, r in recs[sd].items():
                    e = caches[sd].get(s)
                    if e is None:
                        continue
                    w, h = e.get("resolution", L.size_of(s))
                    others = [caches[t].get(s) for t in L.SEEDS if t != sd]
                    c3 = any(o is not None and pairs(e["boxes"], o["boxes"], w, h, 3)[1:] != (0, 0) for o in others)
                    k = "fail" if r["fail"] else "ok"
                    flag[k][0] += int(c3)
                    flag[k][1] += 1
            P[cond] = {"decisions": dec, "C3_flag": {k: {"flagged": v[0], "images": v[1], "rate": v[0] / v[1] if v[1] else None} for k, v in flag.items()}}
            for d in DESIGNS:
                x = dec[d]
                fr, orr = P[cond]["C3_flag"]["fail"], P[cond]["C3_flag"]["ok"]
                lines.append(f"| {cond} | {d} | {x['accept']} | {x['accepted_failures']} | {x['FN_risk_mean']:.3f} | {x['FP_risk_mean']:.3f} | "
                             f"{fr['flagged']}/{fr['images']} | {orr['flagged']}/{orr['images']} |")
        out["profiles"][prof] = P
        lines.append("")
    lines += ["## 해석 제한", "", "- 합성 이물(특히 M2)은 실제 새 이물 분포의 증거가 아니다. C3는 같은 학습 자료의 seed 간 불일치이며 독립 검출기가 아니다.",
              "- 이 결과로 운영 규칙을 바꾸지 않았다."]
    L.jdump(L.OUT / "E9_missed_object_blindspot" / "RESULTS.json", out)
    (L.OUT / "E9_missed_object_blindspot" / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("E9 written", flush=True)


if __name__ == "__main__":
    main()
