# 현행 테스트/수집 로직 현황 보고서

> 본 보고서 작성 이후 리팩토링으로 일부 파일이 제거되었음 (blink_calibration_test.py 등).

> 이 문서는 코드 품질 리뷰가 아니라 현황 지도다. 이번 세션에서는 코드를 수정하지 않았다.
> 모든 항목에 파일 경로 + 함수/클래스명 + 라인 번호를 명시했다. 추측인 부분은 "(추정)"으로 표시했다.

---

## 1절. 테스트/수집 로직 인벤토리

### 1-1. 전체 표

| 이름 | 진입점 | 측정 대상 | 출력 파일 / 스키마 버전 | 담당 클래스 |
|---|---|---|---|---|
| 9점 정확도 테스트 | `main.py`의 `t` 키 (`MODE_CALIBRATED` 전용, `mapper.ready` 필요) → `run_gaze_accuracy_test()` (`main.py:135`) | 9개 화면 지점 응시 오차(px/cm), STB-01~04(FPS/랜드마크율/얼굴검출실패율/dropout), gaze_std, iris_std | `gaze_accuracy_results/sessions_v{SCHEMA_VERSION}.csv` + `gaze_accuracy_v{SCHEMA_VERSION}.csv` (SCHEMA_VERSION="1.5", `src/metrics/collector.py:23`) **+ 별도 원시 CSV** `gaze_accuracy_results/gaze_accuracy_{mode}_%Y%m%d_%H%M%S.csv` (`main.py:415-443`, `append_rows` 미경유 — 아래 4절 참고) | `MetricsCollector` (`src/metrics/collector.py`) |
| 16점 캘리브레이션 진단 | 앱 시작 시 자동 / `r` 키로 재시작. `Calibrator.update()` (`src/tracking/calibration.py:227`)가 `mapper.update_initialization()` 경유로 메인 루프에서 매 프레임 호출 | 포인트별 재투영오차(STB-11), 입력신호 안정성(STB-12), 수집품질(STB-13), 릿지 재투영오차 | `gaze_accuracy_results/calibration_quality_v{CALIB_SCHEMA_VERSION}.csv` (CALIB_SCHEMA_VERSION="1.3", `src/tracking/calibration.py:170`) | `Calibrator` (`src/tracking/calibration.py`) |
| 입벌림 캘리브레이션 | `main.py`의 `m` 키 → `mouth_calibrator.update()` (`src/calibrations/mouth_calibration.py`) | mar_baseline, threshold, `mouth_success_rate`/`mouth_consistency`/`mouth_false_trigger_rate` 등 | `calibration_results/baseline.json` — **CSV가 아니라 JSON 단일 파일이며 매번 덮어쓰기(append 아님)**. `session_id`/타임스탬프 이력 없음, `saved_at` 하나만 최신값으로 남음 (`src/calibrations/baseline_manager.py:12-39`) | `MouthCalibration` (`src/calibrations/mouth_calibration.py`) — 사용자 영역 아님(참고용) |
| 일반 사용 세션 로그 | `main()` 진입 시 자동 생성, 메인 루프 매 프레임 `session_logger.log_frame()` (`main.py:1148`) | 프레임 단위 raw iris/mapped 좌표, dwell_ratio, hover/click key, mapping_valid, mapper metadata | `gaze_accuracy_results/mapper_session_log_v1.0.csv` — **버전이 파일명에 하드코딩 문자열로만 있고 클래스 버전 상수가 없음** (`src/metrics/session_logger.py:55`) | `SessionLogger` (`src/metrics/session_logger.py`) |
| 문장 입력(타이핑) 테스트 | `main()` 시작 시 자동 생성(`tests/test_runner.py`), 키 입력마다 `on_key_press()` 호출, 목표 문장 완성 시 자동 완료 | keystrokes, backspace_count, reaction_times, cursor_travel_distance_px, input_duration_sec | 자체 CSV 없음. 완료 시 `get_input_metrics()` → `MetricsCollector.set_input_metrics()`로 전달되어 `sessions.csv`의 일부 컬럼에 반영 (`main.py:1073-1119`). `collector`가 `None`이면(=아직 `t` 테스트 안 함) 콘솔 출력만 되고 저장 안 됨 | `TestRunner` (`tests/test_runner.py`) |
| **10회 원형 타겟팅 테스트 (신규)** | `main()` 루프의 `y` 키 (`main.py:1487-1500`), 진행 중/결과 화면에서 ESC(종료) 또는 `y`(재시작) | 타겟별 성공/실패(dwell 또는 입벌림 판정), 반응시간, 명중률(`hit_rate_percent`) | **없음** — `get_results()`는 `src/ui.py`의 `draw_targeting_result_screen()`(`ui.py:1096`) 렌더링에만 쓰임. CSV export, `append_rows`, `session_id` 연결 코드를 리포에서 찾지 못함(grep 전수 확인) | `TargetingTestRunner` (`tests/targeting_test_runner.py`) — **팀원 신규 추가 로직**, 커밋 `2c7c9e6`(add independent targeting accuracy test) + `daec2b3`(add preparation time), PR #30 `feature/cheonjiin-targeting` |
| (참고) blink 보정 테스트 스크립트 | `blink_calibration_test.py` 단독 실행 (`python blink_calibration_test.py`) | 개인화된 눈감음/눈뜸 EAR 임계값(`close_threshold`/`open_threshold`) 캘리브레이션 + 판정 이력 | **파일은 쓴다.** `blink_calib.json`에 임계값을 덮어쓰기 저장(`blink_calibration_test.py:257-258`, CSV 아님). **다만 main.py와 상태를 공유하지 않는다** — main.py는 이 파일을 읽는 코드가 없고(`main.py` 전체 검색 결과 0건), `BlinkDetector`를 기본 임계값으로 별도 생성한다(`main.py:559-562`) | 독립 스크립트, 팀원 영역(눈 깜빡임) — 이번 조사에서 내부 판정 로직 상세 분석은 생략함 |

> **천지인 레이아웃**: 이 표에는 "천지인"이 별도 행으로 없다 — 측정/수집 로직이 아니라 입력 레이아웃이기 때문이다. 병합 상태·활성 여부·기존 confirm 경로와의 관계는 3절 앞부분에서 확인한다.

### 1-2. 팀원 신규 테스트 로직 — `TargetingTestRunner` 상세

- **측정 내용**: 화면에 무작위 배치된 10개 원형 타겟을 순서대로 드웰(1초 응시) 또는 입벌림으로 선택. 성공/실패, 반응시간(`reaction_time_sec`), 명중률을 계산.
- **기존 수집기와의 중복 여부**: `MetricsCollector`(9점 테스트)와 대상이 다르다 — 9점 테스트는 "고정 좌표 응시 시 예측 오차"를, 타겟팅 테스트는 "원형 타겟 명중/반응시간"을 잰다. 좌표 오차(px) 자체를 저장하지 않으므로 완전한 상위집합/중복은 아니지만, 같은 성격의 "정확도" 지표를 별도 클래스·별도 로직으로 병행 측정하고 있다.
- **별도 CSV 생성 여부**: 아니다 — 어떤 CSV도 생성하지 않는다. 결과는 화면 렌더링(`ui.py`) 후 앱 종료와 함께 메모리에서 소실된다(콘솔 print만 남음, `main.py:926-933`).
- **session_id 기록 여부**: 없다. `TargetingTestRunner` 자체에는 id 필드가 없고, `MetricsCollector`/`Calibrator`/`SessionLogger`의 어떤 session_id와도 연결되지 않는다.

### 1-3. 9점 테스트 / 캘리브레이션 / 일반 사용 상태별 수집기 on/off 표

| 상태 | Calibrator (진단) | SessionLogger | MetricsCollector(9점) | TestRunner | TargetingTestRunner |
|---|---|---|---|---|---|
| 캘리브레이션 진행 중 (`mapper.ready=False`) | 갱신 중 (`update()` 호출) | **정지** — `continue`(`main.py:772`)로 `session_logger.log_frame()` 라인(`1148`)에 도달하지 않음 | 없음(`None`) | 존재하나 미시작 | 비활성 |
| 일반 키보드 사용 중 (`mapper.ready=True`, `targeting_mode=False`) | 갱신 없음 (`r`로만 재시작) | 매 프레임 기록 | 이전 `t` 테스트 인스턴스가 남아있을 수 있으나 갱신 없음(문장 완료 시 1회만 export) | 매 키 입력마다 갱신 | 비활성 |
| 9점 테스트 중 (`t`, `run_gaze_accuracy_test()` 내부 자체 while 루프) | `map_to_screen()`만 호출(읽기), 학습 갱신 없음 | **정지** — 별도 루프라 `main()`의 프레임 처리 코드 전체를 타지 않음 | `add_sample()`/`add_frame()`으로 채워짐 | **정지** | 비활성 |
| 타겟팅 테스트 중 (`y`, `targeting_mode=True`) | 갱신 없음 | **정지** — `continue`(`main.py:997`)로 이후 코드(dwell/mouth 키보드 처리, `session_logger.log_frame()`) 전부 스킵 | 갱신 없음(진행 중이면 정지 상태로 유지) | **정지** | 갱신 중 |

---

---

## 2절. 이식 지점(훅 포인트) 맵

각 훅은 **[위치 / 함수 시그니처 / 통과하는 모드 / 로깅 삽입 가능 여부 / 장애 요인]** 순으로 정리한다.

### 2-1. 키 확정 지점

**Hook A — Dwell 확정**
- 위치: `main.py:1021-1030`
- 함수 시그니처: 인라인 블록. 호출 대상은 `process_key(key, is_korean, is_shift, buttonList, keyboard_layout=KEYBOARD_LAYOUT_QWERTY)` (`src/keyboard.py:519`)
- 통과하는 모드: QWERTY·천지인 공통, `MODE_CALIBRATED`·`MODE_NO_CALIBRATION` 공통 (dwell이 `tracking_valid`일 때만 도는 공통 경로라 매핑 모드와 무관)
- 로깅 삽입 가능 여부: 가능 — `tester.on_key_press(clicked_key)`가 이미 같은 자리에 후킹돼 있어 동일 패턴으로 삽입 가능
- 장애 요인: 이 지점의 `clicked_key`는 "탭"이지 확정 문자가 아니다(천지인에서 특히). `process_key()`가 확정 문자/확정 여부를 반환하지 않으므로(아래 703-707행), 탭과 확정을 구분하려면 반환 시그니처 확장이 선행되어야 한다.
```python
if clicked_key:
    tester.on_key_press(clicked_key)
    (is_korean, is_shift, buttonList) = process_key(
        clicked_key, is_korean, is_shift, buttonList, keyboard_layout
    )
```

**Hook B — Mouth(입벌림) 확정**
- 위치: `main.py:1033-1043`
- 함수 시그니처: Hook A와 동일한 `process_key()` 호출
- 통과하는 모드: QWERTY·천지인 공통, calibrated·no_calibration 공통. Dwell과는 독립된 `if` 조건(`mouth_click and hovered_key`)이지만 같은 `process_key()`를 통과한다.
- 로깅 삽입 가능 여부: 가능, Hook A와 동일 구조
- 장애 요인: Hook A와 동일(탭≠확정). 추가로 Dwell·Mouth 두 조건문이 서로 독립적이라 이론상 같은 프레임에서 둘 다 성립하면 `process_key()`가 한 프레임에 두 번 불릴 수 있다 — 로깅 삽입 시 프레임당 최대 2회 호출 가능성을 전제해야 한다.
- **Blink 확인**: `blink_detector`(`BlinkDetector`)는 `gaze.update()`에 `blink` 플래그로만 전달되어 "깜빡임 중이면 시선 좌표 무효화"에 쓰일 뿐(`main.py:726-727,341`; `gaze_pipeline.py:129`), 확정 트리거로 쓰는 코드는 `main.py` 전체에서 찾지 못했다. 즉 현재는 "Dwell/Blink/Mouth 3모드"가 아니라 **Dwell·Mouth 2모드만 confirm 경로를 가지며, Blink는 트래킹 유효성 게이트로만 존재**한다.

**Hook C — 천지인 자모 emit(실제 조합 지점)**
- 위치: `src/keyboard.py:519-546`(진입), 실제 emit 호출은 `keyboard.py:530`
- 함수 시그니처: `process_key(key, is_korean, is_shift, buttonList, keyboard_layout)` 내부의 `cheonjiin_composer.input_key(self, key: str) -> Optional[str]` (`src/cheonjiin.py:142`)
- 통과하는 모드: 천지인 + 한글 입력 상태에서만(`is_cheonjiin and is_korean and key in CHEONJIIN_CHARACTER_KEYS`, `keyboard.py:525-529`). QWERTY는 이 분기를 타지 않는다.
- 로깅 삽입 가능 여부: 가능하나, "확정 문자 카운트"를 만들려면 `emitted_jamo`가 `None`이 아닐 때만 세야 한다.
- 장애 요인: `emitted_jamo`는 현재 `print()`로만 소비되고 `process_key()` 밖으로 반환되지 않는다(`keyboard.py:532-539`, 함수 반환값은 703-707행에서 `(is_korean, is_shift, buttonList)` 고정 3-tuple). Hook A/B 쪽에서 확정 여부를 알 방법이 없다.
```python
emitted_jamo = cheonjiin_composer.input_key(key)
if emitted_jamo is None:
    print(f"[천지인] {key} → 모음 조합 대기")
else:
    print(f"[천지인] {key} → {emitted_jamo}")
return (is_korean, is_shift, buttonList)
```

### 2-2. 한글 조합 처리 위치

- `hangul.add_jamo(j)` (`src/hangul.py:92-194`) — 초성/중성/종성 상태머신. QWERTY·천지인 공통 최종 조합 엔진.
- `hangul.flush_buffer()` (`src/hangul.py:84-90`) — `jamo_buffer` → `finalText` 커밋. "글자 확정"이 아니라 스페이스/한영전환/Enter 등 여러 트리거에서 호출됨(`keyboard.py:590,601,617`).
- 천지인 고유 조합은 `CheonjiinComposer`가 `hangul.jamo_buffer`/`hangul.finalText`를 **직접 참조·수정**하는 방식으로 이루어진다(`cheonjiin.py:263-313`, 예: `hangul.jamo_buffer[2] == consonant`, `hangul.finalText.endswith(previous)`). 별도 엔진이 아니라 `hangul.py`의 전역 상태를 가로채 조작하는 구조라, `hangul.py` 쪽 상태머신을 바꾸면 `cheonjiin.py`도 같이 깨질 수 있다.

### 2-3. 시계 소스 (혼용 확인됨 — 전부 나열)

| 소스 | 종류 | 사용처 |
|---|---|---|
| `time.time()` | wall-clock, tz 없음 | `main.py:189,191,345,374`(9점테스트), `tests/test_runner.py:59,94,129`, `src/ui.py:62,76,345,733,773`, `src/calibrations/mouth_calibration.py:82,83,119,168,189,190,459`, `src/tracking/calibration.py:236,373,405`, `src/tracking/dwell.py:41`, `src/tracking/mouth.py:93`, `src/metrics/session_logger.py:92`(csv `timestamp` 필드) |
| `time.monotonic()` | 단조 시계, wall-clock과 무관(절대시각 아님, 재시작 시 리셋) | `src/cheonjiin.py:159`(연타 타임아웃), `tests/targeting_test_runner.py:168,252,308,340,365,430`, `src/tracking/blink.py:119`, `blink_calibration_test.py` 다수 |
| `datetime.now(timezone.utc).isoformat()` | tz-aware UTC ISO 문자열 | `src/metrics/collector.py:69,231`(`start_timestamp`/`end_timestamp`) — **유일하게 tz 정보를 가진 시계** |
| `datetime.now()` (naive, local) | tz 없는 로컬 시각 | `main.py:415`(9점 테스트 원시 CSV 파일명), `src/calibrations/baseline_manager.py:23`(`saved_at`) |

혼용 문제: `session_logger.py`의 `timestamp`(`time.time()` epoch, tz 없음)와 `collector.py`의 UTC ISO 문자열은 같은 세션 내에서도 형식이 달라 직접 조인/정렬하려면 변환이 필요하다. `time.monotonic()` 계열(천지인·타겟팅·blink)은 애초에 절대시각이 아니므로 다른 로그와 시각 자체를 비교할 수 없다 — 오직 같은 프로세스 실행 내 상대적 경과시간 계산에만 유효하다.

### 2-4. session_id 생성·전달 경로

세 개의 독립된 uuid 공간이 존재하며, 그중 두 개만 연결된다.

- `MetricsCollector.session_id = str(uuid.uuid4())` (`collector.py:43`) — `t` 키를 눌러 `MetricsCollector`가 생성될 때만 발급. `show_session_popup()`(`main.py:1573,1581-1582`)에 전달되고 `sessions.csv`/`gaze_accuracy.csv` 행에 기록됨.
- `Calibrator.calib_id = str(uuid.uuid4())` (`calibration.py:202`, `reset()`에서 매번 재발급) — `MetricsCollector` 생성 시 `calib_id` 인자로 전달되어(`main.py:1530-1535`) `sessions.csv`에 FK로 기록되고, `calibration_quality.csv`에도 자체 기록됨(`calibration.py:1155`). **이 연결 하나만 실제로 존재한다.**
- `SessionLogger.session_id = str(uuid.uuid4())` (`session_logger.py:59`) — `main()` 시작 시 1회 발급, 앱 실행 전체에 걸쳐 고정. `collector.session_id`나 `calibrator.calib_id`와 **연결되는 코드가 없다.** 즉 일반 사용 중 프레임 로그를 특정 캘리브레이션이나 특정 9점 테스트 세션과 이어 붙일 방법이 현재는 없다.
- `TestRunner`, `TargetingTestRunner`: 자체 id 필드 없음.

### 2-5. 파라미터 정의 위치 (config_hash 가능성 판단용)

| 파라미터 | 위치 | 비고 |
|---|---|---|
| `DWELL_SEC` | `config.py:63` → `dwell.py:120` | config 경유 |
| dwell `assist_radius`(35), `lock_radius`(40/60), cooldown(0.4) | `dwell.py:55,58,60,125` | **하드코딩**, config에 없음 |
| Kalman `processNoiseCov`(0.003)/`measurementNoiseCov`(0.3) | `gaze_pipeline.py:47-53` | **하드코딩** |
| EMA `alpha`(0.35) | `gaze_pipeline.py:187` | **하드코딩** — 아래 참고 |
| `SMOOTH_ALPHA`(0.20) | `config.py:60`, `gaze_pipeline.py:8`에서 import만 되고 **본문에서 미사용(죽은 값)** | config에 값이 있어도 실제 동작은 위 하드코딩 alpha=0.35가 지배 — config_hash를 config.py 기준으로 만들면 이 상태를 놓친다 |
| dead zone 임계값(10/15/25px), `max_step_px`(50.0) | `gaze_pipeline.py:207-212,225` | **하드코딩** |
| `FIXATION_RADIUS`(40), `FIXATION_FRAMES`(6) | `config.py:75-76` → `gaze_pipeline.py` | config 경유 |
| `CALIB_STD_X/Y`(0.008), `CALIB_STABILIZE_SEC`(1.0), `CALIB_COLLECT_SEC`(2.0) | `config.py:66-71` → `calibration.py` | config 경유 |
| `RIDGE_ALPHA`(1.0), `RIDGE_DEGREE`(2) | `config.py`에 있으면 사용, 없으면 `calibration.py:24-32`에서 자체 fallback 상수 사용 | 조건부 config |
| `max_calib_rmse_px`(150.0) | `calibration.py:615` | **하드코딩**, config에 없음 |
| mouth `threshold`(0.30), `hold_time`(0.3) | `mouth.py:64-65` (함수 기본 인자) | **하드코딩** |
| 타겟팅 `dwell_sec`(1.0), `timeout_sec`(5.0), `prepare_sec`(2.0) | 호출부 `main.py:548-555` (클래스 기본값도 `targeting_test_runner.py:39-42`에 중복 정의) | **하드코딩**, 두 곳에 값이 중복 |
| 천지인 `repeat_timeout_sec`(2.5) | `cheonjiin.py:93` (함수 기본 인자) | **하드코딩** |

판단: 파라미터가 `config.py` 상수 / 각 모듈 내 하드코딩 리터럴 / 호출부 하드코딩 인자 세 곳에 흩어져 있고, 심지어 `config.py`에 값이 있어도 실제로는 안 쓰이는 죽은 값(`SMOOTH_ALPHA`)까지 있다. **`config.py` 파일 하나만 해시해서 config_hash로 쓰면 실제 동작에 영향을 주는 파라미터 변경(dwell 반경, Kalman 계수, dead zone 등)을 감지하지 못한다.** config_hash를 만들려면 최소한 `gaze_pipeline.py`/`dwell.py`/`calibration.py`/`mouth.py`/`cheonjiin.py`의 하드코딩 상수까지 포함하는 범위를 다시 정의해야 한다.

### 2-6. 평활화 전 좌표 확보 가능성

- `main()` 루프: `mapping_result.x/y`(`main.py:825`, `mapper.map()`의 출력 = Homography/릿지 등 원시 매핑 좌표, `gaze.update()`의 Kalman/EMA/dead-zone 스무딩을 거치기 **전** 값)이 이미 `SessionLogger.log_frame()`의 `mapped_sx`/`mapped_sy`로 기록되고 있다(`main.py:1154-1155`). 단 SessionLogger가 정지 상태인 구간(캘리·9점·타겟팅 중, 1-3절 상태표 참고)에서는 이 값도 기록되지 않는다.
- `run_gaze_accuracy_test()` 루프: `raw_sx/raw_sy`(`main.py:264-267`), `corrected_sx/sy`(279-282), `sqpnp_corrected_sx/sy`(292-295), `ridge_sx/sy`(318-320)가 개별 지역 변수로 이미 계산돼 있지만, `collector.add_sample()`에는 `gaze.update()` 이후의 최종 스무딩 좌표(`gaze_x`, `gaze_y`)만 전달된다(362-367). **후킹 지점 자체는 이미 존재하고, 단지 CSV로 배선되어 있지 않을 뿐이다.**

### 2-7. (추가) 9점·타겟팅·캘리 루프가 main() 프레임 처리 경로를 공유하지 않는 구조 — 공유 함수로 뽑을 수 있는가

확인된 사실:
- **캘리브레이션 중 / 입벌림 캘리브레이션 중**: `main()` 루프 안에서 각각 `continue`로 빠진다(`main.py:772,812`) — 이후의 dwell/mouth/`session_logger.log_frame()` 코드에 도달하지 않는다.
- **타겟팅 모드**: 원시 iris→`mapper.map()`→`gaze.update()`까지의 프레임 단위 매핑 계산은 키보드 모드와 **이미 공유**한다(`main.py:703-870` 블록이 targeting_mode 분기보다 먼저 실행됨, 872행). 다만 `if targeting_mode:` 블록이 `continue`로 끝나므로(997행), 그 뒤에 있는 `session_logger.log_frame()`(1148행)은 건너뛴다.
- **9점 테스트(`run_gaze_accuracy_test()`)**: `main()`과 **완전히 별도인 자체 `while` 루프**(135-494행)이며, `cap.read()`부터 다시 한다. 결정적으로 `mapper.map()`을 쓰지 않고 `calibrator.map_to_screen()`/`compensate_iris_by_head_pose()` 등을 **직접 호출**한다(264-332행) — `main()`이 쓰는 추상화 계층(`mapper`)을 우회한다.

판단:
- **타겟팅 모드**는 공유 함수 추출이 필요 없다 — 매핑 계산 자체는 이미 공유되어 있으므로, `targeting_mode` 분기의 `continue` 이전에 `session_logger.log_frame()` 호출 한 줄만 추가(또는 이동)하면 된다. 저위험·개별 삽입 권장.
- **9점 테스트**는 사정이 다르다. `main()`의 프레임 처리와 형태만 비슷할 뿐, `mapper.map()` 추상화를 거치지 않고 `calibrator`를 직접 호출하는 **구조적으로 다른 코드 경로**다. 이걸 공유 함수로 묶으려면 먼저 `run_gaze_accuracy_test()`가 `mapper.map()`을 쓰도록 리팩터링해야 하는데, 이는 `src/tracking/mappers/`(팀원 영역)와 맞닿는 변경이라 **이번 조사 범위에서 임의로 판단할 수 없다.** 따라서 지금 단계에서는 9점 테스트 루프에도 개별적으로 로깅 호출을 삽입하는 쪽이 안전하며, `mapper.map()` 통합은 별도 상의 후 진행할 별개의 리팩터링 과제로 분리하는 것을 제안한다.

### 2-8. (추가) `collector`가 `None`일 때 `TestRunner` 지표가 저장되지 않는 경로

정확한 조건을 추적한 결과:

1. `tester = TestRunner()`는 `main()` 시작 시 **한 번만** 생성된다(`main.py:546`). `collector = None`으로 시작하고(558행), `t` 키를 눌러 `mode == MODE_CALIBRATED and mapper.ready`일 때만 `collector`가 새로 생성된다(1502-1558행).
2. 사용자가 **`t`를 한 번도 누르기 전에** 목표 문장을 다 입력하면 `tester.check_complete(current_text)`가 `True`를 반환하고(1073행), `input_metrics`가 계산·`print()`까지는 되지만(1074-1086행), `if collector is not None:` 블록(1088행) 전체가 스킵되어 `set_input_metrics()`/`end_session()`/`export_csv()`가 **호출되지 않는다.** 이 시점의 `input_metrics`는 지역 변수라 다음 프레임에서 사라진다.
3. 이 손실은 **일회성이며 되돌릴 수 없다.** `TestRunner.check_complete()`는 성공 시 `self.active = False`, `self.saved = True`를 영구히 설정하고(`tests/test_runner.py:93-95`), 이후 호출은 맨 앞 가드에서 즉시 `False`를 반환한다(77-84행: `if not self.active or self.saved or ...: return False`). `tester`는 앱 실행 중 재생성되지 않으므로, **해당 실행에서 문장 입력 테스트는 사실상 단 한 번만 유효하고, 그 한 번이 `t` 테스트 이전에 끝나면 입력 지표는 그 실행 동안 영구히 복구 불가능하다.**
4. 반대로 `collector`가 존재하는 경우에도 문제가 있다: `tester.session_start`는 **앱 실행 중 최초 키 입력 시점**에 한 번만 설정되고(`test_runner.py:61-63`) 리셋되지 않는다. `cursor_travel_distance_px`도 그 시점부터 계속 누적된다. 반면 `collector`는 `t`를 누를 때마다 새로 생성된다. 즉 문장 입력 완료 시점에 **우연히 참조되고 있는 `collector`**(=가장 최근 `t` 테스트)에 `input_duration_sec`/`cursor_travel_distance_px`가 기록되는데, 이 값들의 실제 측정 구간(최초 키 입력~문장 완료)은 그 `collector`가 만들어진 시점(가장 최근 `t` 키 입력)과 무관하다. **`sessions.csv`의 해당 컬럼이 그 세션에 속하지 않는 값을 담을 수 있다.**

---

## 3절. 천지인 레이아웃 영향 분석

### 3-0. 병합 상태 확인

- `git merge-base --is-ancestor 5a0b888 HEAD` / `daec2b3 HEAD` 모두 성공 — **천지인 관련 커밋(레이아웃 5a0b888, 조합 4b95557)은 현재 `develop` HEAD의 조상이며, 별도 브랜치에 고립되어 있지 않고 실제로 병합되어 활성 상태다.**
- 같은 PR 여부: 병합 커밋 `a229117`(PR #30, `feature/cheonjiin-targeting`)의 두 번째 부모 범위(`a229117^1..a229117^2`)에 다음 6개 커밋이 모두 포함된다 — `5a0b888`(천지인 레이아웃), `4b95557`(천지인 조합), `2c7c9e6`(타겟팅 테스트 추가), `2ffed61`(fix: restore keyboard dwell processing after targeting mode), `53a37a8`(문서 갱신), `daec2b3`(타겟팅 준비시간 추가). **천지인과 타겟팅 테스트는 같은 브랜치·같은 PR에서 함께 병합됐다.**
  - 참고로 `2ffed61`의 커밋 메시지("타겟팅 모드 이후 키보드 dwell 처리 복구") 자체가 2-7절에서 확인한 "`targeting_mode` 분기가 `continue`로 빠지면서 이후 코드를 건너뛴다"는 구조와 정확히 맞닿아 있다 — 과거에도 이 지점에서 실제 버그가 있었다는 방증이다.
- 활성화 방법: `--keyboard-layout cheonjiin` CLI 인자(`main.py:1613-1625`)로 즉시 활성화된다. 기본값은 `KEYBOARD_LAYOUT_QWERTY`.

### 3-1. 기존 키보드와 별도 모듈인가 / 같은 confirm 경로를 쓰는가

별도 모듈(`src/cheonjiin.py`)이지만, **confirm 경로는 QWERTY와 완전히 동일**하다 — 2-1절 Hook A/B(`main.py:1021-1043`)를 그대로 공유하고, `process_key()`(`keyboard.py:519`) 내부의 `is_cheonjiin` 플래그로만 분기한다.

### 3-2. 탭 1회와 문자 확정 1회의 관계

**현재 "키 입력 카운트"는 탭 기준이다.** 근거: dwell/mouth 클릭이 발생하면 `process_key()` 호출 여부와 무관하게 `tester.on_key_press(clicked_key)`가 **먼저** 호출된다(`main.py:1021-1022`, `1033-1035`). 천지인 모음 조합 중(예: `ㆍ` 단독 입력으로 아직 모음이 완성되지 않은 상태) `cheonjiin_composer.input_key()`가 `None`을 반환해도(`keyboard.py:532-535`) `tester.on_key_press()`는 이미 그 앞에서 실행된 뒤이므로 keystroke 카운트에는 포함된다. 즉 `TestRunner.keystrokes`(`test_runner.py:65`)는 "확정된 문자 수"가 아니라 "물리 탭 수"다.

### 3-3. 백스페이스가 천지인에서 탭 취소인가 글자 삭제인가

**두 동작이 조건에 따라 혼재한다.**
1. 아직 실제 모음으로 완성되지 않은 요소(`ㆍ`, `ㆍㆍ` 등)가 있으면 `cancel_uncommitted_vowel()`이 그 요소만 취소한다(`keyboard.py:548-560`, `cheonjiin.py:111-122`) — **탭 취소**에 가깝다.
2. 그 외 모든 경우(자음 연타 중이거나 조합 중인 것이 없을 때)는 `cheonjiin_composer.reset()`으로 조합 상태만 지우고, 실제 삭제는 QWERTY와 동일한 `hangul.jamo_buffer`/`finalText` 기반 **자모 단위 삭제**로 처리된다(`keyboard.py:564-586`). 자음 연타로 쌓은 순환(예: ㄱ→ㅋ까지 두 번 순환)을 한 단계씩 되돌리는 기능은 없다 — Del을 누르면 현재 초성/종성이 통째로 지워진다.

### 3-4. 현재 활성 레이아웃이 기록되는가

**없음.** `keyboard_layout`은 `main()` 함수 인자로만 존재하는 지역 변수(`main.py:518,593,1024-1029,1037-1042`)이며, `MetricsCollector`의 세션 필드(`collector.py:251-273`)에도 `SessionLogger.FIELDNAMES`(`session_logger.py:28-47`)에도 `keyboard_layout` 컬럼이 없다. 현재 CSV만으로는 특정 세션이 QWERTY로 측정됐는지 천지인으로 측정됐는지 **구분할 수 없다.**

### 3-5. `input_events`에 `keyboard_layout` 컬럼 및 탭/문자 이벤트 구분이 필요한가

**필요하다고 판단한다.** 근거:
- (a) 3-4에서 확인했듯 현재 어떤 CSV도 레이아웃을 구분하지 않는다. 천지인·QWERTY 세션이 한 CSV에 섞이면 "키 입력 수" 같은 지표를 레이아웃 간 비교/합산하는 순간 의미가 왜곡된다 — 천지인은 구조적으로 같은 글자를 만드는 데 필요한 탭 수가 QWERTY와 다르다(자음 연타, 모음 조합).
- (b) 3-2에서 확인했듯 현재 `keystrokes`는 탭 기준이라, 천지인 입력 효율의 핵심 지표인 "문자당 탭 수"를 계산할 원재료 자체가 없다. 탭 이벤트와 확정 문자 이벤트를 구분 기록해야 이 지표를 낼 수 있다.
- (c) 구현 난이도는 낮다 — 2-1절 Hook A/B와 `process_key()`의 반환 시그니처 확장(`keyboard.py`, 사용자 영역)만으로 가능하며, 팀원 영역 침범이 없다.

---

## 4절. 리스크·불일치 목록 (심각도 순)

`append_rows()` 우회 CSV(9점 테스트 원시 파일), 스키마 버전 하드코딩(`src/analysis/run_export.py`), `baseline.json` 덮어쓰기는 **1절에 이미 기록**했으므로 재서술하지 않고 아래에서 포인터로만 참조한다.

### 상 (High)

1. **`TestRunner` 입력 지표의 소실·오귀속** (2-8절) — `t` 테스트 이전에 문장을 완성하면 입력 지표가 영구 소실되고, `t`를 여러 번 눌렀으면 엉뚱한 세션에 지표가 귀속될 수 있다.
   → *이걸 안 고치면*: 개편 후에도 `sessions.csv`의 `input_duration_sec`/`cursor_travel_distance_px` 계열 컬럼을 그 세션의 실측치로 신뢰할 수 없다 — 세션 단위 비교 분석 자체가 불가능해진다.

2. **세 개의 독립된 session_id/uuid 공간이 서로 연결되지 않음** (2-4절) — `collector.session_id` / `calibrator.calib_id` / `session_logger.session_id`가 각자 따로 존재하고, `calib_id↔session_id` 하나만 연결된다.
   → *이걸 안 고치면*: "일반 사용 중 프레임 로그"를 특정 캘리브레이션·특정 9점 테스트 세션과 이어 붙이는 개편(공용 식별자 체계)이 지금 상태 위에서는 불가능하다 — 식별자 통합이 선행되지 않으면 이식할 지점 자체가 없다.

3. **시계 소스 3종 혼용** (2-3절) — `time.time()`(wall-clock) / `time.monotonic()`(비교 불가) / `datetime.now(timezone.utc)`(tz-aware) / `datetime.now()`(naive)가 모듈마다 다르게 쓰인다.
   → *이걸 안 고치면*: 서로 다른 로그(예: `session_logger`의 프레임 타임스탬프와 천지인/타겟팅의 `monotonic` 경과시간)를 같은 타임라인 위에 정렬해서 재구성하는 작업이 원천적으로 불가능하다.

4. **`TargetingTestRunner` 결과가 어디에도 저장되지 않음** (1절 1-2 참조) — 신규 기능인데 CSV·session_id 연결이 전혀 없다.
   → *이걸 안 고치면*: 천지인과 함께 새로 들어온 이 지표를 개편안에 편입할 방법이 없다 — 지금은 이식할 데이터가 존재하지 않는 상태다.

5. **"각도 오차(RMSE °)" 주장에 대한 코드상 근거 없음** — 저장소 전체에서 `arctan`/시야각 변환 로직을 찾지 못했다(`math.atan`, `distance_cm` 기반 각도 환산 등 0건). 현재 정확도 지표는 px 오차를 `px_per_cm`(`config.py:29`)으로 나눈 **cm 환산치뿐**이며(`collector.py:378-381`), 각도 단위 오차 계산 코드는 어디에도 없다. 이번에 확인한 사용자 본인의 미커밋 작업물(`src/metrics/pose_agg.py`, `tests/test_pose_agg.py`)도 head pose(yaw/pitch/roll)의 각도 통계이지 시선 정확도의 각도 오차가 아니다.
   → *이걸 안 고치면*: "RMSE 5.79°" 같은 각도 표현이 보고서 어딘가에 있다면 그 수치는 이 코드베이스가 계산한 값이 아니다 — 근거 없는 수치를 개편 자료의 기준값으로 쓰게 될 위험이 있다. (아래 질의 사항 참고)

### 중 (Medium)

6. **`keyboard_layout`이 어떤 CSV에도 기록되지 않음** (3-4절)
   → *이걸 안 고치면*: 천지인 도입 직후인 지금, QWERTY/천지인 세션이 한 CSV에 섞여 있어도 사후에 구분할 방법이 없다 — 레이아웃별 비교 분석이 불가능하다.

7. **"키 입력 카운트"가 탭 기준, 확정 문자 기준이 아님** (3-2, 3-5절)
   → *이걸 안 고치면*: 천지인 입력 효율의 핵심 지표("문자당 탭 수")를 낼 원재료가 없다 — 천지인 도입의 실효성 자체를 지표로 보여줄 수 없다.

8. **파라미터가 config.py / 모듈별 하드코딩 / 호출부 인자 세 곳에 산재, 일부는 죽은 config 값** (2-5절, `SMOOTH_ALPHA`)
   → *이걸 안 고치면*: `config.py`만 해시하는 `config_hash`는 실제 동작에 영향을 주는 파라미터 변경(dwell 반경, Kalman 계수 등)을 감지하지 못한다 — "이 세션이 어떤 설정으로 측정됐는지" 재현성 보장이 불가능하다.

9. **`append_rows()` 이외의 CSV 쓰기 경로** — 1절에 기록한 `main.py`의 9점 테스트 원시 CSV(`main.py:424-443`, `csv.writer` 직접 사용) 외에, 개인 분석 도구 `src/analysis/diagnostic.py:678-681`(`df.to_csv()`)도 `append_rows()`를 거치지 않는다. 이 도구는 git 미포함(개인용)이라 팀 공용 데이터에는 영향이 없지만, 스키마 변경 시 이 파일도 별도로 손봐야 한다는 점은 동일하다.
   → *이걸 안 고치면*: CSV 스키마를 한 곳(`csv_export.append_rows`)만 고쳐서 전체에 반영된다고 가정하고 개편하면, 최소 이 두 경로가 누락된다.

### 하 (Low)

10. **`blink_calibration_test.py`가 파일을 쓰지만 main.py와 상태를 공유하지 않음** — 현재는 완전히 고립된 독립 도구다.
    → *이걸 안 고치면*: 당장은 문제없다. 다만 향후 Blink를 세 번째 confirm 모드로 실제 통합할 계획이 생기면, 그때는 이 스크립트의 캘리브레이션 결과(`blink_calib.json`)를 main.py가 읽어 쓰도록 연결하는 작업이 별도로 필요하다는 것만 미리 인지해 둘 것.

11. **9점·타겟팅·캘리 루프가 프레임 처리 함수를 공유하지 않음, 특히 9점 테스트는 `mapper.map()`을 우회** (2-7절)
    → *이걸 안 고치면*: 당장 데이터 무결성 문제는 아니지만, 9점 테스트 루프에 새 로깅을 추가할 때마다 `main()` 루프와 별개로 두 번 구현해야 하는 유지보수 비용이 계속 발생한다.

---

## 질의 사항 (판단이 갈리는 지점)

1. **"RMSE 5.79°" 등 각도 표현의 출처** — 이 리포지토리(git 추적 + 로컬 미추적 파일 포함) 안에서는 시선 정확도를 각도로 환산하는 코드를 찾지 못했다. 이런 수치가 실제로 쓰이고 있다면 Notion이나 개인 계산(엑셀 등)에서 나온 근사치로 추정되는데, 맞는지, 맞다면 어떤 환산식(예: 가정 시야거리)을 썼는지 알려주면 4절 5번 항목의 심각도 판단을 더 정확히 할 수 있다.
2. **2-7절의 9점 테스트 리팩터링 범위** — `run_gaze_accuracy_test()`가 `mapper.map()`을 쓰도록 통합하는 것은 `src/tracking/mappers/`(팀원 영역)에 인접한 변경이다. 이걸 이번 개편 범위에 포함할지, 아니면 "9점 테스트 루프에 로깅만 개별 삽입"으로 한정할지 방향을 정해줘야 다음 단계(이식 실제 작업)를 잡을 수 있다.
3. **`MouthCalibration`을 1절 인벤토리에 포함할지** — 이번 초안에서는 참고용으로 표에 넣었지만(사용자 영역 아님, "9점 테스트"류의 정확도 테스트는 아님), `baseline.json` 덮어쓰기 이력 손실이 개편 대상에 포함되는지는 판단이 필요하다.
4. **Blink confirm 모드 도입 여부** — CLAUDE.md와 1절 조사 결과를 보면 Blink는 현재 confirm 경로가 아니라 트래킹 게이트로만 쓰인다. "Dwell/Blink/Mouth 세 모드"라는 원래 전제가 아직 실현되지 않은 로드맵상 목표인지, 아니면 이미 다른 곳(팀원 작업 중)에서 진행 중인지 확인이 필요하다.
