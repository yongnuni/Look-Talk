"""후보 문장의 어절별 초성 자동완성 3슬롯 적중 여부를 판정하는 읽기 전용 도구.

문장을 공백 기준 어절로 나누고, 각 어절의 초성열을 1글자부터 어절 전체
길이까지 접두사로 늘려가며 기존 추천 엔진(build_recommender)의 공개 API
(ChosungTrie.query)에 넣어 3슬롯 결과 안에 그 어절로 시작하는 후보가
등장하는 최소 접두사 길이를 찾는다. 등장하지 않으면 '-'로 표시한다.

엔진 내부(trie.py/recommender.py/hospital.py)는 수정하지 않고 공개 API만
호출한다.

사용법:
    python -m scripts.check_sentence_hits
    python -m scripts.check_sentence_hits --sentences "물 주세요" "창문 닦아요"
"""

from __future__ import annotations

import argparse
import sys
import unicodedata

from src.recommendation.chosung import extract_chosung
from src.recommendation.models import HOSPITAL_SOURCE, WORD_SOURCE
from src.recommendation.recommender import build_recommender


SUGGESTION_SLOT_COUNT = 3

DEFAULT_CANDIDATES = [
    "물 주세요",
    "아파요 진통제",
    "간호사 불러주세요",
    "화장실 가고싶어요",
    "창문 닦아요",
    "노란 우산이요",
    "강아지 산책해요",
    "연필 두개요",
]


def _classify_sources(sources: set[str]) -> str:
    if HOSPITAL_SOURCE in sources:
        return "병원"
    if WORD_SOURCE in sources:
        return "일반"
    return "없음"


def check_word(trie, word: str) -> tuple[int | None, str]:
    """어절이 3슬롯에 등장하는 최소 접두사 길이와 사전 출처를 반환한다."""

    chosung = extract_chosung(word)
    if not chosung:
        return None, "없음"

    for length in range(1, len(chosung) + 1):
        prefix = chosung[:length]
        candidates = trie.query(prefix, SUGGESTION_SLOT_COUNT)
        matched_sources = {
            candidate.source
            for candidate in candidates
            if candidate.word.startswith(word)
        }
        if matched_sources:
            return length, _classify_sources(matched_sources)

    return None, "없음"


def build_rows(trie, sentences: list[str]) -> list[tuple[str, str, str, str]]:
    rows = []
    for raw_sentence in sentences:
        sentence = unicodedata.normalize("NFC", raw_sentence)
        for word in sentence.split(" "):
            if not word:
                continue
            hit_length, source = check_word(trie, word)
            rows.append(
                (
                    sentence,
                    word,
                    "-" if hit_length is None else str(hit_length),
                    source,
                )
            )
    return rows


def format_table(rows: list[tuple[str, str, str, str]]) -> str:
    header = ("문장", "어절", "최소 적중 접두사 길이", "사전 출처")
    widths = [len(h) for h in header]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row):
        return " | ".join(
            cell.ljust(widths[index]) for index, cell in enumerate(row)
        )

    lines = [format_row(header), "-+-".join("-" * w for w in widths)]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "후보 문장을 어절로 나눠 초성 자동완성 3슬롯 적중 최소 접두사 "
            "길이를 판정한다."
        )
    )
    parser.add_argument(
        "--sentences",
        nargs="+",
        default=None,
        help="판정할 문장 목록(생략하면 내장된 8개 후보를 사용)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sentences = args.sentences if args.sentences else DEFAULT_CANDIDATES

    engine = build_recommender()
    if engine.trie is None:
        print("엔진 초기화 실패 - 사전 파일을 확인하세요.", file=sys.stderr)
        return 1

    rows = build_rows(engine.trie, sentences)
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
