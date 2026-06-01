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

    def on_key_press(self, key):

        now = time.time()

        if self.session_start is None:
            self.session_start = now

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

    def is_showing_complete(self):

        if self.complete_time is None:
            return False

        return time.time() - self.complete_time < 2