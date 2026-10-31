# 참고한 기존 방법

아래 논문은 2026-10-06에 arXiv 초록 페이지에서 제목·저자·제출일을 직접 확인했다. 이 분석모델은 아래 방법을 새로 발명한 것이 아니라, X-ray 데이터의 실패 요인(검사기 표시 흔적, 장비별 입력 배율, 촬영 품질 변화)에 맞춰 결합하고 검증했다.

| 방법 | 원 논문 | 이 분석모델에서 쓰는 방식 |
| --- | --- | --- |
| 분포형 박스 회귀(DFL) | Li et al., "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection", arXiv:2006.04388 (2020) | YOLO26s의 `reg_max`를 16으로 둔 검출기(`data/models/yolo26s-dfl.yaml`). 같은 조건의 일반 YOLO26s(`reg_max` 1)와 비교했다. |
| Conformal Risk Control (CRC) | Angelopoulos, Bates, Fisch, Lei, Schuster, "Conformal Risk Control", arXiv:2208.02814 (2022) | 손실 = 1{ACCEPT ∧ 실패}. 교환가능성 가정에서 이 **기대 전체 위험**을 α = 0.05로 통제하는 신뢰도 임계값 λ를 장비별로 정한다. ACCEPT 안의 조건부 실패율 보장이 아니다. |
| Learn then Test (LTT) | Angelopoulos, Bates, Candès, Jordan, Lei, "Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control", arXiv:2110.01052 (2021) | 고정된 coverage 격자 10개에 Bonferroni 다중검정을 적용해 조건부 위험 검정을 기록한다. 교환가능성 전제가 확인되지 않아 확률 보장으로 주장하지 않는다. |

- **구현 도구:** Ultralytics 8.4.157(YOLO26, AGPL-3.0, `licenses/ULTRALYTICS_AGPL-3.0.txt`), OpenCV 5.0.0의 `cv2.inpaint`(INPAINT_TELEA)·`cv2.seamlessClone`·형태학 연산, SciPy의 이항분포·베타분포 함수.
- **위치 지표:** 중심 거리 매칭은 프로젝트 자체 구현이다(`src/metrics.py`). 이 지표를 외부 논문의 공식 지표라고 부르지 않는다.
