# 과제④ X-ray 이물 위치 탐지·재검사 판정 분석기 — XGD-1.2.0

완제품 X-ray 영상에서 작은 이물(대부분 폭 5~21px)의 **위치를 찾아 표시**하고, 영상마다 다음 세 판정 중 하나와 사유를 붙인다.

| 판정 | 뜻 | 현장 조치 |
| --- | --- | --- |
| **ACCEPT** | 검출한 이물 위치를 자동 확인 | 표시된 위치의 이물을 제거·기록 |
| **REVIEW** | 예측 사이 불일치 등 확인 필요 | 위치 후보를 보고 작업자가 확인 |
| **FULL_REINSPECTION** | 검증된 촬영 범위 밖 | 전수 재검사, 장비·촬영 조건 점검 |

ACCEPT는 '검출한 위치를 자동 확인'한다는 뜻이며 제품 합격이나 '놓친 이물 없음'의 보장이 아니다. 개발 자료에 정상 제품 영상이 없어 검출 0개 영상은 항상 REVIEW다.

## 1. 빠른 실행

검증 환경: Windows 11, **Python 3.13.15**, NVIDIA GPU 12GB(CUDA 13.0), `requirements.txt`의 고정 버전.

```bash
pip install -r requirements.txt
python run_all.py                 # 전처리 → XGD-1.2 평가 79장 판정 → 제출 예측과 비교 (약 1분)
python -m pytest -q tests -p no:cacheprovider     # 회귀 테스트 84개
```

- `python run_all.py`가 하는 일:
  1. 원본 BMP 435장에서 검사기 색 표시를 지운 영상(v3)과 흔적 무력화 학습 영상(v3tr)을 만들고 픽셀 해시를 기록과 대조한다.
  2. 고정된 XGD-1.2 설정·보정으로 평가 79장을 판정해 `outputs/followup/rerun/TEST_PREDICTIONS_RERUN.json`에 쓴다.
  3. 제출 예측과 판정·검출 수·중심 좌표를 비교한다. 하나라도 다르면 0이 아닌 종료 코드로 멈춘다.
- 운영 코드는 Python 버전·라이브러리 버전·가중치·보정 파일까지 서명해 확인한다. 다른 버전에서 실행하면 서명이 맞지 않아 모든 영상이 FULL로 판정된다. 검증 범위 밖 환경에서 자동 확정을 하지 않기 위한 안전장치다.

**영상 1장 판정:**

```bash
python -m src.followup_analyzer --image data/raw/<파일명>.bmp --metadata meta.json --profile dfl_src3_2 --out result.json
```

- `meta.json` 예: `{"calibration_id": "2호기", "preprocessed": false}`
- profile은 1·2호기 영상이면 `dfl_src3_2`, 3호기 영상이면 `dfl_src12`를 쓴다.

## 2. 제출 예측 결과 (평가 79장)

`test_predictions/`에 있다. `outputs/followup/deployment/`와 같은 파일이다.

| 파일 | 내용 |
| --- | --- |
| `test_predictions_images.csv` | 영상별 판정·검출 수·사유 |
| `test_predictions_boxes.csv` | 검출 박스(원본 픽셀 좌표)·점수 |
| `test_predictions_yolo/*.txt` | YOLO 형식(클래스, 중심 x, 중심 y, 폭, 높이, 점수; 정규화) |
| `TEST_PREDICTIONS.json` | 위 내용 전체와 판정 사유·서명 |

- 79장 모두 이물 위치를 출력한다(검출 79개).
- 판정은 FULL 76, REVIEW 3이다. 평가 영상은 학습·보정에 없던 9월 촬영이고, 1호기는 새 해상도(316×332)다. 그래서 위치는 표시하되 자동 확정은 하지 않았다.
- 평가 79장의 정답 라벨은 사용하지 않았다.

## 3. 전처리부터 학습·추론까지 다시 만들기

```bash
python run_all.py full        # 전처리 → DFL 검출기 6개 재학습 → 보정 → 평가 79장 예측 (약 1시간, outputs/full_run/)
python run_all.py controls    # XGD-1.2 완결성 확인용 대조 검출기 6개 재학습 (약 50분, outputs/full_run/controls/)
python run_all.py check       # 학습 재현 계약 확인(학습 목록·인자·초기값·학습 영상 해시; 학습 없음)
```

- 재학습 결과는 따로 저장되며, 제출 가중치와 결과는 바뀌지 않는다.
- 같은 조리법으로 재학습했을 때 3px 검출률:
  - 3호기 0.988 → 0.989
  - 1·2호기 0.998 → 0.998
- 학습 초기값은 COCO 사전학습 YOLO26s(`data/initial/coco_yolo26s.pt`)다.

## 4. 분석 흐름 (`src/followup_analyzer.py`)

원본 BMP → ① 색 표시 제거(v3) → ② 입력 배율 정규화(X8) → ③ DFL 검출(YOLO26s, reg_max 16) + 4px 중복 제거 → ④ 검출 완결성 확인 → ⑤ 촬영 품질 검사(X12) → ⑥ 장비별 위험 통제 임계값(CRC, α 0.05) → ⑦ 해상도 등록부·서명 확인 → ACCEPT / REVIEW / FULL

- **④ 검출 완결성 확인:** 아래 8회 예측의 확정 검출이 6px 안에서 하나씩 짝지어지지 않으면 REVIEW로 보낸다.
  - 학습 seed 3개
  - 좌우·상하 뒤집기와 배율 ±32 view
  - 흔적 증강 없이 학습한 대조 모델
- **⑤ 촬영 품질 검사:** 잡음·선명도 신호가 보정 영상 분포에서 크게 벗어나면 REVIEW 또는 FULL로 보낸다.

## 5. 핵심 결과 (개발 자료 356장, 장비 단위 검증, seed 17·29·43 평균)

| 항목 | 3호기(학습에 없던 장비) | 1·2호기 |
| --- | --- | --- |
| 3px F1, 기준 모델 YOLO26s → XGD | 0.923 → **0.988** | 0.996 → **0.997** |
| 정상 화질 3px 재현율 | 0.9878 | 0.9975 |
| 놓친 이물이 자동 확정될 위험(14조건 평균), 1.1 정책 → 1.2 | 0.0698 → **0.0014** | 0.0287 → **0.0070** |
| 처리 시간 | 장당 평균 0.151초(모델 호출 8회) | 같음 |

**한계:** 같은 공급처가 준 개발 자료 안의 내부 검증이며, 정상 제품과 외부 장비 자료는 없다. 자세한 내용은 `docs/LIMITATIONS.md`에 있다.

## 6. 폴더 구성

| 경로 | 내용 |
| --- | --- |
| `run_all.py` | 단일 실행 진입점 |
| `src/` | 운영 코드: 1.2 분석기, 안전 규칙, 전처리, 검출기 학습, 지표 |
| `experiments/` | 검증·비교 실험 코드(E1~E11, 1.2 후속 실험) |
| `config.yaml`, `outputs/followup/deployment/` | 공통 설정, 1.2 운영 설정·보정·평가 79장 예측 |
| `weights/` | DFL 검출기 6개(장비 전이 방향 2 × seed 3)와 MANIFEST |
| `outputs/controls/v3only/retrain/` | 완결성 확인용 대조 검출기 6개 |
| `data/raw/`, `data/labels_orig/` | 학습용 데이터: 제공 원본 BMP 435장, 개발 정답 TXT 356개 |
| `data/clean_v3/`, `data/v3tr_orig/` | 전처리 영상(`python run_all.py`가 다시 만듦) |
| `tests/` | 회귀 테스트 84개 |
| `docs/` | 결과·검증 문서(요약, 선택 근거, 위험–자동확정 분석, 한계, 참고문헌) |
| `licenses/` | Ultralytics AGPL-3.0 |

## 7. 블라인드 처리

- 가중치 12개 안의 학습 당시 절대 경로를 패키지 상대 경로로 바꿨다.
  - 텐서는 원래 학습 결과와 비트 단위로 같다.
  - 바뀐 해시로 운영 설정과 보정 파일의 서명만 다시 했고, 보정값과 판정 결과는 그대로다(`BLIND_EXPORT.json`).
- 연구 기록 파일에 남은 원래 해시는 `BLIND_EXPORT.json`에서 새 해시와 대응시킬 수 있다.

## 8. 데이터와 라이선스

- KAMP 「X-ray 검사장비 AI 데이터셋」(대회 제공)을 사용했다. 외부 데이터는 쓰지 않았다.
- 정답은 제공 TXT만 쓴다. 영상의 색 표시는 정답이나 특징으로 쓰지 않고 지운다.
- 검출기 구현은 Ultralytics 8.4.157(AGPL-3.0)을 따른다.
