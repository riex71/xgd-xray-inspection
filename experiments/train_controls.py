"""비교·대조 검출기 학습 (최종 모델 교체용이 아님).

  baseline  일반 YOLO26s (reg_max 1 = 직접 박스 회귀). DFL 과 같은 원본 TXT·v3tr 학습 영상·조리법·t2 분할·seed, 모델 구조(reg_max)만 다름.
            seed 17 은 원 protocol 의 b1_sc4_orig 가중치(같은 조리법)를 쓰고, seed 29·43 만 새로 학습한다.
  v3only    DFL 을 흔적 무력화(가짜 표시) 없이 v3 정제 영상만으로 학습. 흔적 무력화 학습의 순수 효과를 분리하는 대조.

  python -m experiments.train_controls baseline --seeds 29 43
  python -m experiments.train_controls v3only --seeds 17 29 43
결과: outputs/controls/<name>/config.yaml (이 config 로 학습한 기록), outputs/controls/<name>/retrain/<profile>_s<seed>/weights/last.pt
"""
import argparse
import copy
import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import train_dfl as T          # noqa: E402
from src import xgd_config as XC        # noqa: E402

CONTROLS = {
    "baseline": {"train": {"model": "yolo26s.yaml"},
                 "note": "YOLO26s 기본 구조(reg_max 1). data/models/yolo26.yaml 이 ultralytics 8.4.157 내장 정의의 사본(scale s)."},
    "v3only": {"train": {"train_input": "clean_v3"},
               "note": "DFL(reg_max 16), 학습 영상 = v3 정제(가짜 표시 없음). 나머지는 최종 DFL 과 같음."},
}


def control_config(name):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf8"))
    cfg = copy.deepcopy(cfg)
    cfg["train"].update(CONTROLS[name]["train"])
    cfg["paths"]["artifacts"] = f"outputs/controls/{name}"
    cfg["control_note"] = CONTROLS[name]["note"]
    out = ROOT / "outputs" / "controls" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf8")
    return XC.load(out / "config.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=list(CONTROLS))
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--profiles", nargs="+", default=["dfl_src12", "dfl_src3_2"])
    a = ap.parse_args()
    cfg = control_config(a.name)
    for prof in a.profiles:
        for sd in a.seeds:
            folder = cfg["_artifacts"] / "retrain" / f"{prof}_s{sd}"
            if (folder / "weights" / "last.pt").exists():
                with (folder / "results.csv").open(encoding="utf8") as f:
                    rows = list(csv.DictReader(f))
                if not rows or int(float(rows[-1]["epoch"])) != cfg["train"]["epochs"]:
                    raise RuntimeError(f"Incomplete checkpoint preserved: {folder}. Not a completed control; no silent skip/overwrite.")
                print("건너뜀(이미 학습됨)", a.name, prof, sd, flush=True)
                continue
            T.train(cfg, prof, sd)


if __name__ == "__main__":
    main()
