from tests.test_runner import TestRunner


def test_get_target_char_returns_first_char_when_nothing_committed():
    runner = TestRunner()
    assert runner.get_target_char(0) == runner.target_text[0]


def test_get_target_char_advances_with_committed_length():
    runner = TestRunner()

    for i in range(len(runner.target_text)):
        assert runner.get_target_char(i) == runner.target_text[i]


def test_get_target_char_empty_when_sentence_finished():
    runner = TestRunner()
    assert runner.get_target_char(len(runner.target_text)) == ""
    assert runner.get_target_char(len(runner.target_text) + 5) == ""


def test_get_target_char_empty_when_inactive():
    runner = TestRunner()
    runner.active = False
    assert runner.get_target_char(0) == ""


def test_get_target_char_empty_for_negative_length():
    runner = TestRunner()
    assert runner.get_target_char(-1) == ""
