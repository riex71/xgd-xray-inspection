# XGD-1.2 변경 기록 (완료)

| 구분 | XGD-1.1.0 | XGD-1.2.0 |
| --- | --- | --- |
| 3호기 검출기 | DFL(흔적 증강 1.0) | DFL(흔적 증강 1.0) |
| 3호기 정책 | P0 | P2_bundle_6 seed·view·대조 불일치 묶음(6px) |
| 3호기 safety | none | bundle |
| 3호기 radius | 3.0 | 6.0 |
| 1·2호기 검출기 | DFL(흔적 증강 1.0) | DFL(흔적 증강 1.0) |
| 1·2호기 정책 | P0 | P2_bundle_6 seed·view·대조 불일치 묶음(6px) |
| 1·2호기 safety | none | bundle |
| 1·2호기 radius | 3.0 | 6.0 |
| 품질 기준(X12) 자료 범위 | 장비별 개발 영상 전체. 2호기는 1·2호기용 검출기 학습에 쓴 8/31·9/1 촬영 32장 포함(132장) | CRC와 같은 보류 개발 영상만(2호기 6월 100장; 1호기 124장·3호기 100장은 범위 동일). CRC는 1.1에서도 보류 영상만 썼다. 03:41 봉인 계약에서 정했고 평가 79장 분포를 보고 바꾸지 않았다 |
| 해상도 | 규칙 없음 | 등록부에 없는 native 해상도 → FULL |
| 보정 서명 | 가중치·전처리·규칙·소스 해시 | 여기에 정책·추가 모델 가중치·의존성 버전까지 서명하고, 불일치 시 FULL |
| 출력 | decision, detections_confirmed | 추가 필드: confirmation_scope(검출 위치만), completeness_status(NOT_CERTIFIED/INCONSISTENT), product_pass_certified=false, 불일치 신호 |
| 운영 코드 | src/xgd_analyzer.py | src/followup_analyzer.py(1.1 코드는 그대로 보존) |

## 평가 79장 판정 변화

- 1.2 판정 분포: {'FULL_REINSPECTION': 76, 'REVIEW': 3}
- 1.1 → 1.2 전이: ACCEPT -> FULL_REINSPECTION 36, ACCEPT -> REVIEW 3, FULL_REINSPECTION -> FULL_REINSPECTION 32, REVIEW -> FULL_REINSPECTION 8
- 정답이 없으므로 정확도 비교가 아니다. 분포를 보고 규칙을 바꾸지 않았다.

## 보존

- 1.1 코드·설정·보정 산출물·평가 79장 예측·문서: `versions/XGD-1.1.0/`(SNAPSHOT.json에 해시).
- 원본 BMP·TXT·동결 가중치 6개는 바꾸지 않았다. 흔적 0.5배 가중치는 `outputs/followup/train/trace05/`에 따로 있다.
