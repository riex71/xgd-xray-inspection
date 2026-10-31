"""XGD-1.2 완결성 확인에 쓰는 대조 검출기(v3only, 흔적 증강 없음 DFL) 6개를 처음부터 다시 학습한다(재현용).

제출 가중치(outputs/controls/v3only/retrain/*)를 덮어쓰지 않고 outputs/full_run/controls/v3only/retrain/ 에 만든다.
조리법은 experiments/train_controls.py 의 v3only 와 같다(학습 영상만 clean_v3, 나머지는 최종 DFL 과 동일).

python -m experiments.retrain_controls_full            # 6개 모두(12GB GPU 에서 약 50분)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import train_controls as TC   # noqa: E402
from src import train_dfl as T                 # noqa: E402


def main():
    cfg = TC.control_config("v3only")
    cfg["_artifacts"] = ROOT / "outputs" / "full_run" / "controls" / "v3only"
    for prof in ("dfl_src12", "dfl_src3_2"):
        for sd in (17, 29, 43):
            T.train(cfg, prof, sd)


if __name__ == "__main__":
    main()
