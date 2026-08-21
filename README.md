📚 [팀노션](https://app.notion.com/p/31d7635a991d83b995cb01378ede55c7?source=copy_link)

## 프로젝트 소개

Look Talk은 웹캠만으로 시선을 추적해 한글을 입력하는 AAC(보완대체의사소통) 키보드입니다. ALS 등 중증 운동장애로 손을 쓰기 어려운 사용자가 별도의 고가 장비 없이 일반 노트북/웹캠 환경에서 의사소통할 수 있게 하는 것이 목표입니다. dwell(응시 유지) 클릭과 입벌림(MAR) 클릭, 두 가지 입력 방식을 지원합니다. 동시에 입력 과정에서 시선 정확도·안정성·타건(tap) 로그 등 평가 지표를 함께 수집해, 입력 방식과 파라미터를 정량적으로 비교·개선하는 것도 이 프로젝트의 핵심 목적입니다.

## Tech Stack

- **Python 3.12**
- **MediaPipe** — 얼굴/홍채 랜드마크 추출
- **OpenCV** (opencv-contrib-python) — 영상 캡처 및 화면 렌더링
- **NumPy / Pandas** — 수치 계산 및 CSV 처리
- **Pillow** — 한글 폰트 렌더링
- **Matplotlib** — 지표 시각화
- **jamo** — 한글 자모 조합
- **scikit-learn** (선택) — 릿지 회귀 기반 하이브리드(`ridge_hybrid`) 매핑에 사용. 미설치 시 동일한 수식의 numpy 폐형해(closed-form)로 자동 대체되어 동작 자체는 유지됩니다.
- **pytest** — 테스트 (개발용)

## Getting Started

### Python 버전

**Python 3.12 필수.**
Python 3.13은 사용 불가 — MediaPipe가 3.13을 지원하지 않습니다.

### 1. 가상환경 생성 (Windows PowerShell 기준)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. 의존성 설치

```powershell
pip install -r requirements.txt
```

(선택) 릿지 회귀 매핑 사용 시:

```powershell
pip install scikit-learn
```

## 실행 방법

```powershell
python main.py [--gaze-mode calibrated|no_calibration] [--strategy 이름] [--keyboard-layout qwerty|cheonjiin] [--user-id 이름] [--condition-label 라벨]
```

| 옵션 | 기본값 | 선택지 | 설명 |
|---|---|---|---|
| `--gaze-mode` | `calibrated` | `calibrated`, `no_calibration` | 시선 매핑 모드. `calibrated`는 16점 캘리브레이션을 거치고, `no_calibration`은 캘리브레이션 화면 없이 바로 키보드로 진입한다. |
| `--strategy` | `head_pose_relative_iris` | 등록된 strategy 이름(현재 `head_pose_relative_iris` 1개뿐) | `--gaze-mode no_calibration`일 때만 사용하는 좌표 추정 전략. |
| `--keyboard-layout` | `qwerty` | `qwerty`, `cheonjiin` | 키보드 배열. |
| `--user-id` | `yejin` | 자유 문자열 | 결과 CSV(`sessions.csv` 등)에 기록될 참가자 식별자. 지정하지 않으면 모든 로그가 `yejin`으로 기록된다. |
| `--condition-label` | (빈 문자열) | 자유 문자열 | `sessions.csv`에 기록될 실험 조건 라벨(예: `baseline`, `new-smoothing`). 코드를 바꿔가며 실험할 때 `git_commit`(자동 기록)만으로 부족하면 직접 지정한다. |

## 키보드 단축키

`main.py`의 키 입력 분기를 기준으로 정리했습니다. 화면 상태에 따라 동작하는 키가 다릅니다.

### 메인 화면 (일반 사용 중)

| 키 | 동작 |
|---|---|
| `q` | 종료 |
| `r` | 재캘리브레이션/리셋 |
| `p` | pose_corrected 매핑 토글 (`calibrated` 모드 전용) |
| `h` | sqpnp_corrected 매핑 토글 (`calibrated` 모드 전용) |
| `g` | ridge_hybrid 매핑 토글 (`calibrated` 모드 전용) |
| `o` | raw 매핑으로 전환 (`calibrated` 모드 전용) |
| `b` | 후보 좌표 마커 토글 |
| `d` | 디버그 오버레이 토글 |
| `k` | 시선 커서 표시/숨김 토글 |
| `m` | 입벌림 입력 모드 진입 (개인별 MAR 캘리브레이션 시작) |
| `y` | 10회 원형 타겟팅 테스트 시작 |
| `t` | 9점 시선 정확도 테스트 (`calibrated` 모드에서 캘리브레이션 완료 후에만 동작) |

### 캘리브레이션 / 입벌림 캘리브레이션 화면

| 키 | 동작 |
|---|---|
| `q` | 종료 |
| `r` | 진행 중인 해당 캘리브레이션만 리셋 |

그 외 키는 이 화면에서 동작하지 않습니다.

### 타겟팅 테스트 화면 (`y`로 진입 후)

| 키 | 동작 |
|---|---|
| `q` | 종료 |
| `ESC` | 타겟팅 테스트 종료, 키보드 화면으로 복귀 |
| `y` | 10회 타겟팅 테스트를 처음부터 재시작 |

이 화면에서는 일반 키보드 입력이 비활성화됩니다.

## 폴더 구조

```
Look-Talk/
├── main.py                                   # 앱 진입점, 메인 루프 (약 1,500줄 — 4단계 리팩토링으로 축소)
├── requirements.txt                          # 런타임 의존성
├── requirements-dev.txt                      # 테스트(pytest) 의존성
├── assets/
│   └── cursor.png                            # 시선 커서로 그리는 PNG 이미지
├── docs/                                     # 내부 진단/조사 문서 (git-ignored)
├── scripts/
│   ├── verify_stage1.py                      # 1단계(식별자 통합) CSV 정합성 검사
│   └── gate_stage3.py                        # 3단계(원시 이벤트→파생 지표) 관문 검증
├── calibration_results/                      # 입벌림 캘리브레이션 산출물 (실행 시 생성, git-ignored)
├── gaze_accuracy_results/                    # 시선/입력 지표 CSV 산출물 (실행 시 생성, git-ignored)
├── report/                                   # make_report.py 산출 그래프 (수동 실행 시 생성, git-ignored)
└── src/
    ├── config.py                             # 전역 상수 (화면 해상도, 캘리브레이션 16점 좌표, dwell/fixation 임계값 등)
    ├── keyboard.py                           # 가상 키보드 레이아웃·버튼 생성·키 입력 처리(process_key)
    ├── hangul.py                             # 한글 자모 조합 엔진 (초성/중성/종성 상태머신)
    ├── cheonjiin.py                          # 천지인 입력 처리 (자음 연타, 모음 조합)
    ├── ui.py                                 # OpenCV+PIL 렌더링 (캘리브레이션 화면, 키보드, 커서, 오버레이 등)
    │
    ├── app/
    │   └── cli.py                            # CLI 인자 파싱 (parse_args)
    │
    ├── vision/
    │   └── preprocessing.py                  # 자동 밝기 보정 (auto_brightness)
    │
    ├── calibrations/
    │   ├── baseline_manager.py               # 입벌림 캘리브레이션 결과 JSON 저장/로드
    │   └── mouth_calibration.py              # 입벌림 캘리브레이션 상태머신
    │
    ├── tracking/
    │   ├── calibration.py                    # 16점 홈그래피 학습 + 릿지 회귀 하이브리드 매핑
    │   ├── dwell.py                          # 시선 dwell 클릭 판정 (키·추천 공용 update_target, 고정 확장 적용 지점)
    │   ├── eye_tracking.py                   # 홍채 좌표 계산, 눈 깜빡임(EAR) 검출
    │   ├── fixation.py                       # 고정 감지형 히트박스 확장 (I-VT 속도 + I-DT 분산)
    │   │                                     #   고정된 키캡·추천 슬롯의 판정 영역을 넓히고 화면에서도 확대
    │   │                                     #   이웃을 최대 1/3까지 덮고, 겹치는 구간에서는 확장된 쪽이 우선
    │   │                                     #   주변 대상의 위치·크기는 정적으로 유지(반응형 재배치 없음)
    │   ├── gaze_pipeline.py                  # Kalman 스무딩 + fixation 감지
    │   ├── head_pose.py                      # solvePnP/SQPnP 기반 head pose 추정
    │   ├── mouth.py                          # 입벌림 비율(MAR) 계산, 입벌림 클릭 판정
    │   ├── feature_builder.py                # 릿지 회귀용 특징 벡터 조립
    │   ├── gaze_mapper.py                    # 매퍼 공통 인터페이스(GazeMapper, MappingResult)
    │   └── mappers/
    │       ├── factory.py                    # 매퍼 생성 팩토리 (calibrated / no_calibration)
    │       ├── calibrated_mapper.py          # 16점 캘리브레이션 매퍼
    │       ├── no_calibration_mapper.py      # 캘리브레이션 없이 동작하는 매퍼
    │       └── strategies/                   # no_calibration 모드 좌표 추정 전략 모음
    │
    ├── testing/
    │   ├── gaze_accuracy.py                  # 9점 시선 정확도 테스트, 종료 시 결과 팝업
    │   └── targeting_export.py               # 원형 타겟팅 테스트 결과 CSV 저장
    │
    ├── metrics/
    │   ├── collector.py                      # MetricsCollector: 9점 테스트 지표 수집·CSV export
    │   ├── session_logger.py                 # 매 프레임 로그 (SessionLogger)
    │   ├── input_event_logger.py             # 키 탭 단위 원시 이벤트 로그
    │   ├── baseline_history.py               # 입벌림 캘리브레이션 이력 CSV append
    │   ├── tap_logging.py                    # 탭 커밋 이벤트를 InputEventLogger로 전달
    │   ├── derive_input.py                   # 원시 입력 이벤트에서 지표 사후 파생(replay)
    │   ├── csv_export.py                     # 공용 CSV append 유틸
    │   └── pose_agg.py                       # head pose 집계 유틸리티 (테스트에서만 쓰이는 미배선 상태 — 확인 완료)
    │
    ├── common/
    │   ├── clock.py                          # 프로세스 전역 단조 시계 (모든 로그 타임스탬프의 단일 출처)
    │   ├── ids.py                            # run_id/calib_id/test_id 발급
    │   └── config_snapshot.py                # 실험 조건(파라미터) 스냅샷 + 해시
    │
    ├── viz/
    │   ├── viz.py                            # 9점 테스트 결과 시각화 공통 모듈
    │   ├── calib_viz.py                      # 캘리브레이션 품질 시각화
    │   └── make_report.py                    # CLI 리포트 생성 스크립트 (PNG 저장)
    │
    ├── analysis/                             # 세션/캘리브레이션 진단 배터리, Notion 업로드 등 수동 실행 진단 도구
    │
    └── recommendation/                       # 병원 특화 초성 자동완성 (사전/Trie/추천 상태)
        └── selection.py                      # 추천 슬롯과 키 중 프레임당 입력 대상 1개 결정
                                              #   고정 확장을 넘기면 추천·키 양쪽에 같은 규칙으로 적용

tests/
├── test_runner.py                            # TestRunner: 문장 입력 테스트 진행 상태 관리
├── targeting_test_runner.py                  # 10회 원형 타겟팅 테스트 관리
└── test_sentences.py                         # 테스트용 목표 문장 목록
```

## 산출 데이터

`main.py` 실행 중/종료 시 자동으로 생성되는 CSV/JSON입니다. 실제 쓰기 경로를 코드에서 확인해 작성했습니다.

| 파일 | 위치 | 내용 | 생성 시점 |
|---|---|---|---|
| `sessions_v1.9.csv` | `gaze_accuracy_results/` | 실행(run) 1회당 메타데이터 1행 | 앱 종료 시 항상 (9점 테스트를 안 했어도 저장) |
| `gaze_accuracy_v1.9.csv` | `gaze_accuracy_results/` | 9점 테스트 타깃별 오차·STB 지표 | `t` 키로 9점 테스트 실행 시 |
| `gaze_accuracy_{mode}_%Y%m%d_%H%M%S.csv` | `gaze_accuracy_results/` | 9점 테스트 원시 결과(매핑 모드별) | `t` 키로 9점 테스트 실행 시 |
| `mapper_session_log_v1.1.csv` | `gaze_accuracy_results/` | 매 프레임 로그 | 상시 (60프레임마다 flush) |
| `input_events_v1.0.csv` | `gaze_accuracy_results/` | 키 탭 단위 원시 이벤트 | 상시 (30탭마다 flush) |
| `targeting_results_v1.0.csv` | `gaze_accuracy_results/` | 원형 타겟팅 테스트 결과 (타깃당 1행) | `y` 테스트 완료 또는 `ESC` 중단 시 |
| `calibration_quality_v1.4.csv` | `gaze_accuracy_results/` | 캘리브레이션 포인트별 재투영오차 등 | 캘리브레이션(`r`) 완료 시 |
| `baseline.json` | `calibration_results/` | 입벌림 캘리브레이션 최신 결과 (덮어쓰기) | 입벌림 캘리브레이션 완료 시 |
| `mouth_baseline_history_v2.0.csv` | `calibration_results/` | 입벌림 캘리브레이션 이력 append | 입벌림 캘리브레이션 완료 시 |

### 그 외 산출물

다음은 `main.py` 실행이 아닌 별도 명령의 산출물이며, 실험 결과 제출 대상이 아닙니다.

| 파일 | 위치 | 생성 명령 |
|---|---|---|
| `{run_id 앞자리}_overview.png`, `{calib_id 앞자리}_calib_overview.png` 등 | `report/` | `python -m src.viz.make_report` |

## 실험 참여 가이드

분산 테스트를 위해 각자 컴퓨터에서 실행할 때 아래 설정을 확인해 주세요.

### 1. 실행 전 설정

- **모니터 크기**: `src/config.py`의 `MONITOR_DIAGONAL_INCH`를 자기 모니터의 대각선 인치로 수정합니다. px→cm 환산(`PX_PER_CM`)이 이 값에서 파생되므로, 안 맞추면 정확도 지표(cm 단위)가 왜곡됩니다.
- **참가자 식별자(`--user-id`)**: 기본값은 `yejin`으로 고정되어 있습니다. 다른 사람이 실행한 결과를 구분하려면 실행 시 반드시 지정하세요.
  ```powershell
  python main.py --user-id <자기이름>
  ```
  지정하지 않으면 `sessions.csv` 등 모든 로그에 `yejin`으로 기록되어, 팀원별 결과를 CSV 하나에서 구분할 수 없게 됩니다.

### 2. 실행 순서

```powershell
python main.py [--keyboard-layout qwerty|cheonjiin] [--user-id <이름>]
```

1. 캘리브레이션 화면에서 16점을 순서대로 응시 (자동 진행)
2. 캘리브레이션 완료 후 나오는 가상 키보드로 화면 상단에 표시된 목표 문장을 끝까지 입력 (dwell 또는 입벌림 클릭으로 키 선택, 백스페이스 포함해도 무방 — 오히려 오타율 지표 수집에 도움이 됨)
3. `t`: 9점 시선 정확도 테스트 실행 (`calibrated` 모드에서만 동작)
4. `y`: (선택) 10회 원형 타겟팅 테스트 실행/재시작. `ESC`로 중도 종료 가능
5. `q`: 앱 종료 — 이 시점에 `sessions.csv` 등 세션 메타데이터가 항상 저장됩니다 (9점 테스트를 아예 안 했어도, 문장을 끝까지 안 쳤어도 저장됨)

### 3. 결과 제출

`gaze_accuracy_results/`와 `calibration_results/` 폴더 전체를 CSV/JSON 그대로 제출합니다. 개별 파일을 골라내지 말고 폴더째 보내는 편이 안전합니다 (append 누적 파일이라 실행을 여러 번 섞어도 `run_id`로 나중에 구분 가능).
