import time
import random

from tests.test_sentences import TEST_SENTENCES


class TestRunner:

    def __init__(self):

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