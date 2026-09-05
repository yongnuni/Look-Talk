"""TEST_SENTENCE_GROUPS 데이터 무결성 검증(R8)."""

import unicodedata

from tests.test_sentences import TEST_SENTENCES, TEST_SENTENCE_GROUPS


def _is_hangul_or_space(text):
    return all(char == " " or "가" <= char <= "힣" for char in text)


def test_legacy_group_matches_existing_test_sentences_order():
    assert TEST_SENTENCE_GROUPS["legacy"] == TEST_SENTENCES
    assert TEST_SENTENCE_GROUPS["legacy"] == ["안녕하세요", "감사합니다"]


def test_all_sentences_are_nfc_normalized():
    for group_name, sentences in TEST_SENTENCE_GROUPS.items():
        for sentence in sentences:
            assert unicodedata.normalize("NFC", sentence) == sentence, (
                group_name,
                sentence,
            )


def test_all_sentences_contain_only_hangul_and_spaces():
    for group_name, sentences in TEST_SENTENCE_GROUPS.items():
        for sentence in sentences:
            assert _is_hangul_or_space(sentence), (group_name, sentence)


def test_no_duplicate_sentences_across_groups():
    all_sentences = [
        sentence
        for sentences in TEST_SENTENCE_GROUPS.values()
        for sentence in sentences
    ]
    assert len(all_sentences) == len(set(all_sentences))
