"""과제④ X-ray 이물 위치 탐지·재검사 판정 — XGD-1.2.0 분석모델 실행 진입점.

  python run_all.py              prepare → predict12  (기본, 약 5분)
                                 원본 BMP 435장 전처리(표시 제거·흔적 무력화 학습 영상 생성, 픽셀 해시 검증) 후
                                 고정 XGD-1.2 설정·보정으로 평가 79장을 판정하고 제출 예측과 비교(다르면 0이 아닌 종료 코드)
  python run_all.py full         prepare → train → calibrate → predict  (전처리부터 검출기 재학습·보정·예측까지 자동, 약 1시간)
                                 결과는 outputs/full_run/ 에 따로 둔다(제출 가중치·결과는 그대로)
  python run_all.py controls     XGD-1.2 완결성 확인용 대조 검출기(v3only) 6개 재학습 → outputs/full_run/controls/ (약 50분)
  python run_all.py prepare      원본 BMP → v3 표시 제거 영상(435장)·흔적 무력화 학습 영상(356장), 픽셀 해시 검증
  python run_all.py predict12    XGD-1.2 평가 79장 판정 재실행·제출 예측과 비교 (outputs/followup/rerun/)
  python run_all.py check        DFL 학습 재현 계약 확인(학습 목록·인자·초기값·학습 영상; 학습 없음)
  python run_all.py train        DFL 검출기 6개 재학습 → outputs/retrain (제출 가중치는 그대로)

  제출 예측 결과: outputs/followup/deployment/ (TEST_PREDICTIONS.json, test_predictions_*.csv, test_predictions_yolo/)
  calibrate·verify·predict·regression 은 이전 버전(XGD-1.1) 판정 경로이며 기록 보존용이다.
"""
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf8"))
PROFILES = [p for p, v in CONFIG["profiles"].items() if v.get("role", "deployment") == "deployment"]   # 비교 기준 profile 은 운영 보정 안 함
FULL = "outputs/full_run"
STEPS = {
    "prepare": [["-m", "src.preprocess"]],
    "check": [["-m", "src.train_dfl", "--check"]],
    "calibrate": [["-m", "src.xgd_analyzer", "calibrate", "--profile", p] for p in PROFILES],
    "verify": [["-m", "src.verify"]],
    "predict": [["-m", "src.predict_test"]],
    "predict12": [["-m", "experiments.followup.rerun12"]],
    "controls": [["-m", "experiments.retrain_controls_full"]],
    "train": [["-m", "src.train_dfl", "--all"]],
    "regression": [["tools/regression_x14.py", "run"]],
    "full": None,                                               # full_steps() 가 만든다
}
DEFAULT = ["prepare", "predict12"]
NEEDS_PREPARED = {"check", "calibrate", "verify", "train", "regression", "full", "controls"}   # data/clean_v3·v3tr_orig 를 읽는 단계


def full_steps():
    """재학습 가중치로 보정·예측하는 설정을 outputs/full_run/config_full.yaml 에 만들고, 학습 → 보정 → 예측 명령을 돌려준다."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf8"))
    cfg["paths"]["artifacts"] = FULL
    for prof in PROFILES:
        p = cfg["profiles"][prof]
        p["checkpoints"] = {sd: f"{FULL}/retrain/{prof}_s{sd}/weights/last.pt" for sd in p["checkpoints"]}
    (ROOT / FULL).mkdir(parents=True, exist_ok=True)
    path = f"{FULL}/config_full.yaml"
    (ROOT / path).write_text("# run_all.py full 이 만든 설정: config.yaml 과 같고 가중치·산출물 위치만 outputs/full_run 으로 바꿈\n"
                             + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf8")
    return ([["-m", "src.train_dfl", "--all", "--config", path]]
            + [["-m", "src.xgd_analyzer", "calibrate", "--profile", p, "--config", path] for p in PROFILES]
            + [["-m", "src.predict_test", "--config", path]])


def main():
    steps = sys.argv[1:] or DEFAULT
    for step in steps:
        if step not in STEPS:
            raise SystemExit(f"알 수 없는 단계: {step} (가능: {', '.join(STEPS)})")
    prepared = (ROOT / "data" / "clean_v3" / "images").is_dir() and (ROOT / "data" / "v3tr_orig" / "images").is_dir()
    if "prepare" not in steps and NEEDS_PREPARED & set(steps) and not prepared:
        print("전처리 영상이 없어 prepare 를 먼저 실행한다", flush=True)
        steps = ["prepare", *steps]
    for step in steps:
        print(f"== {step}", flush=True)
        for args in (full_steps() if step == "full" else STEPS[step]):
            subprocess.run([sys.executable, "-u", *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
