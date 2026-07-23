📚 [팀노션](https://app.notion.com/p/31d7635a991d83b995cb01378ede55c7?source=copy_link)


## requirement

<Python 3.12.10>

* Python 3.13은 사용하지 말 것!(해당 버전은 mediapipe를 사용할 수 없음)

<가상환경 생성>

py -3.12 -m venv .venv

.\.venv\Scripts\Activate.ps1

<라이브러리 설치>

python -m pip install mediapipe==0.10.20

python -m pip install opencv-python==4.11.0.86

python -m pip install opencv-contrib-python==4.11.0.86

python -m pip install numpy==1.26.4

python -m pip install pillow

python -m pip install jamo

<릿지>

pip install scikit-learn

<백본>

pip install torch torchvision onnx onnxruntime huggingface_hub

## 폴더 구조

```
Look-Talk/
├── main.py                          # 앱 진입점. 웹캠 캡처→자동 밝기 보정(auto_brightness)→MediaPipe FaceMesh
│                                     #   →캘리브레이션→시선 파이프라인→dwell/입벌림 클릭→키보드 입력의 메인 루프.
│                                     #   run_gaze_accuracy_test(9점 테스트)와 종료 시 결과 팝업(show_session_popup)도 포함
├── README.md                        # 개발 환경 세팅 가이드 (Python 3.12 venv, mediapipe/opencv 등 설치 명령)
├── assets/
│   └── cursor.png                   # 시선 커서로 그리는 PNG 이미지 (ui.draw_gaze_cursor에서 사용)
├── calibration_results/
│   └── baseline.json                # 입벌림 캘리브레이션 결과 저장 파일 (mar_baseline, threshold 등)
│                                     #   — baseline_manager.save_baseline()의 출력물
│
└── src/
    ├── config.py                    # 전역 상수: 화면 해상도 자동감지, PX_PER_CM(모니터 인치→px/cm 환산),
    │                                 #   캘리브레이션 16점 좌표, 안정화 시간, dwell/fixation 임계값 등
    ├── hangul.py                    # 한글 자모 조합 엔진 (초성/중성/종성 상태머신, 겹받침·이중모음 처리, jamo_buffer)
    ├── keyboard.py                  # 가상 키보드: 레이아웃 정의(한/영×기본/Shift), 버튼 생성·배치, 키 입력 처리(process_key)
    ├── ui.py                        # OpenCV+PIL 렌더링: 카운트다운, 캘리브레이션 화면, 키보드 그리기,
    │                                 #   시선 커서, 상태바, 테스트 완료 오버레이, 입벌림 캘리브레이션 화면
    │
    ├── calibrations/
    │   ├── baseline_manager.py      # 입벌림 캘리브레이션 결과를 JSON으로 저장/로드
    │   ├── mouth_calibration.py     # 입벌림 캘리브레이션 상태머신 (rest_collect→trial_ready→trial_wait→trial_active→done)
    │   │                            # MAR 기준값·활성화 임계값 산출, 성공률/일관성/false trigger율 등 지표 계산
    │   ├── blink_calibration.py     # 빈 파일 (미구현 스텁)
    │   └── gaze_calibration.py      # 빈 파일 (미구현 스텁)
    │
    ├── tracking/
    │   ├── calibration.py           # 16점 홍채→화면 매핑 학습(Homography),
    │   │                            #   head pose 기반 iris 좌표 보정(compensate_iris_by_head_pose),
    │   │                            #   STB-11(재투영오차)/STB-12(입력안정성)/STB-13(수집품질) 계산 및 CSV 내보내기
    │   ├── dwell.py                 # 시선 dwell(응시 유지) 클릭 판정 (hover lock, dwell_ratio)
    │   ├── eye_tracking.py          # MediaPipe 홍채 좌표 계산, 눈 깜빡임(EAR) 검출, 홍채 추적 신뢰도(iris_confidence), 눈/홍채 시각화
    │   ├── gaze_pipeline.py         # Kalman 필터 스무딩 + fixation(응시 고정) 감지로 최종 시선 좌표 산출
    │   ├── head_pose.py             # solvePnP/SQPnP 기반 head pose(yaw/pitch/roll/face_scale) 추정
    │   └── mouth.py                 # 입벌림 비율(MAR) 계산, MouthClickDetector(입벌림 클릭 판정)
    │
    ├── metrics/
    │   ├── collector.py             # MetricsCollector: 9점 테스트 지표 수집 엔진. 타깃별 오차(ACC-06)·
    │   │                             #   표준편차·STB-01~04(FPS/랜드마크율/얼굴검출실패율/dropout) 계산,
    │   │                             #   sessions/gaze_accuracy CSV로 export (SCHEMA_VERSION 관리)
    │   └── csv_export.py            # 여러 모듈이 공용으로 쓰는 CSV append 유틸 (append_rows)
    │
    ├── recommendation/               # 전부 빈 파일 (미구현 스텁: models/recommender/scorer)
    │
    └── viz/
        ├── viz.py                    # 9점 테스트 결과 시각화 공통 모듈. 데이터 로딩/병합/시간순 정렬,
        │                             #   최근 N세션 선택, 세션 요약, 오차 지도·오차 막대·세션 추세 그래프
        ├── calib_viz.py              # 캘리브레이션 품질(STB-11/12/13) 시각화. 재투영오차 지도, 입력 안정성
        │                             #   막대, 수집 품질 막대, calib_id별 RMSE 추세 그래프
        └── make_report.py            # CLI 리포트 생성 스크립트. 세션/캘리브레이션 그래프를 PNG로 저장 +
                                      #   콘솔 요약 출력 (--session, --calib, --calib-trend 옵션 지원)

tests/
├── test_runner.py                    # TestRunner: 문장 입력 테스트 진행 상태 관리 (키입력 수, 백스페이스, 반응시간, 완료 판정)
└── test_sentences.py                 # 테스트용 문장 목록 (현재 2개: "안녕하세요", "감사합니다")
```
