import time
import random

from tests.test_sentences import TEST_SENTENCES


class TestRunner:

    def __init__(self, run_id=None):

        # run_id: 앱 실행 1회 식별자(main.py 주입). 이 클래스는 자체 CSV
        # 출력이 없어(지표는 MetricsCollector.set_input_metrics()로 전달됨)
        # 지금 당장은 저장에 쓰이지 않지만, 다섯 클래스에 동일하게 주입한다는
        # 1단계 방침에 맞춰 받아만 둔다.
        self.run_id = run_id

        self.target_text = random.choice(
            TEST_SENTENCES
        )

        self.session_start = None

        self.keystrokes = 0
        self.backspace_count = 0

        self.reaction_times = []
        self.last_key_time = None

        self.complete_time = None
        self.saved = False
        self.active = True
        self.cursor_travel_distance_px = 0.0
        self.previous_cursor_position = None
    def update_cursor_position(self, gaze_x, gaze_y):

        # 입력 테스트가 실제로 시작된 이후에만 측정
        if not self.active or self.session_start is None:
            return

        if gaze_x is None or gaze_y is None:
            return

        if gaze_x < 0 or gaze_y < 0:
            return

        current_position = (
            float(gaze_x),
            float(gaze_y)
        )

        if self.previous_cursor_position is not None:
            prev_x, prev_y = self.previous_cursor_position

            dx = current_position[0] - prev_x
            dy = current_position[1] - prev_y

            self.cursor_travel_distance_px += (
                dx ** 2 + dy ** 2
            ) ** 0.5

        self.previous_cursor_position = current_position

    def on_key_press(self, key):

        now = time.time()

        if self.session_start is None:
            self.session_start = now
            self.previous_cursor_position = None

        self.keystrokes += 1

        if key == "Del":
            self.backspace_count += 1

        if self.last_key_time is not None:
            self.reaction_times.append(
                now - self.last_key_time
            )

        self.last_key_time = now

    def check_complete(self, current_text):

        if (
            not self.active
            or self.saved
            or current_text.strip() != self.target_text
        ):
            return False

        print()
        print("===== 테스트 완료 =====")
        print("목표:", self.target_text)
        print("입력:", current_text)
        print("키 입력:", self.keystrokes)
        print("백스페이스:", self.backspace_count)

        self.saved = True
        self.complete_time = time.time()
        self.active = False

        return True

    def get_target_char(self, committed_length):
        """목표 문장에서 committed_length 다음에 와야 할 글자를 반환한다.

        committed_length는 호출자가 넘기는 "지금까지 확정된 글자 수"
        (main.py에서는 hangul.finalText 길이를 넘긴다). 조합 중인 자모
        버퍼/천지인 pending preview는 세지 않는다 - 화면에 보이는 조합 중
        음절까지 포함하면 위치가 어긋날 수 있다(음절 확정이 소급적으로
        재해석되는 경로가 있음). 그래서 이 값은 근사치이며, finalText가
        나중에 소급 변경되는 경우(천지인 모음 치환 등) 부정확할 수 있다.
        비활성 상태이거나 범위를 벗어나면(문장 완료, 오타로 목표보다
        길어짐 등) 빈 문자열을 반환한다.
        """

        if (
            not self.active
            or committed_length < 0
            or committed_length >= len(self.target_text)
        ):
            return ""

        return self.target_text[committed_length]

    def get_input_metrics(self):

        if (
            self.session_start is None
            or self.complete_time is None
        ):
            input_duration_sec = 0.0
        else:
            input_duration_sec = (
                self.complete_time - self.session_start
        )

        if input_duration_sec > 0:
            average_cursor_speed_px_sec = (
                self.cursor_travel_distance_px
                / input_duration_sec
            )
        else:
            average_cursor_speed_px_sec = 0.0

        return {
            "input_duration_sec": input_duration_sec,
            "cursor_travel_distance_px": self.cursor_travel_distance_px,
            "average_cursor_speed_px_sec": average_cursor_speed_px_sec,
        }

    def is_showing_complete(self):

        if self.complete_time is None:
            return False

        return time.time() - self.complete_time < 2