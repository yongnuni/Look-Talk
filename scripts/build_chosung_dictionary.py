"""wordfreq 한국어 단어를 초성 포함 결정적 gzip JSON으로 생성한다.

프로젝트 루트에서 다음과 같이 실행한다.

    python -m scripts.build_chosung_dictionary
    python -m scripts.build_chosung_dictionary --max-words 100

이 스크립트는 ``iter_wordlist()``와 ``get_frequency_dict()``만 사용하며,
한국어 tokenizer나 ``zipf_frequency()``를 호출하지 않는다.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import time
import unicodedata
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.recommendation.chosung import extract_chosung


SCHEMA_VERSION = 1
LANGUAGE = "ko"
SOURCE = "wordfreq"
WORDFREQ_VERSION = "3.1.1"
WORDLIST = "best"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "src"
    / "recommendation"
    / "data"
    / "chosung_words.json.gz"
)

FILTERING_RULES = (
    "Normalize each source token to Unicode NFC.",
    "Keep only non-empty words whose every character is a precomposed "
    "Hangul syllable in U+AC00..U+D7A3.",
    "Allow one-syllable Korean words.",
    "After NFC normalization, keep only the first occurrence of a duplicate.",
)
SORT_ORDER = (
    "Descending wordfreq order from iter_wordlist('ko', wordlist='best'); "
    "ties and normalized duplicates preserve first source occurrence."
)
LICENSE = "CC BY-SA 4.0"
ATTRIBUTION = (
    "Derived by Look-Talk from wordfreq 3.1.1 by Robyn Speer; "
    "see data/README.md and the upstream NOTICE.md."
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _is_precomposed_hangul_word(word: str) -> bool:
    return bool(word) and all("가" <= char <= "힣" for char in word)


def build_dictionary(
    source_words: Iterable[str],
    frequencies: Mapping[str, float],
    *,
    max_words: int | None = None,
) -> dict[str, Any]:
    """주입된 빈도순 단어와 빈도 mapping으로 사전 payload를 만든다.

    ``max_words``는 필터링 전 source 단어 소비 개수를 제한한다. 빈도값은
    원본 토큰을 key로 조회하므로 NFC 정규화로 표기가 달라져도 원본
    wordfreq 값이 보존된다.
    """

    if max_words is not None and max_words <= 0:
        raise ValueError("max_words must be a positive integer")

    entries: list[dict[str, Any]] = []
    seen_words: set[str] = set()
    input_word_count = 0

    for source_word in source_words:
        if max_words is not None and input_word_count >= max_words:
            break

        input_word_count += 1

        if not isinstance(source_word, str):
            continue

        normalized_word = unicodedata.normalize("NFC", source_word)
        if not _is_precomposed_hangul_word(normalized_word):
            continue

        if normalized_word in seen_words:
            continue

        try:
            frequency = frequencies[source_word]
        except KeyError as exc:
            raise KeyError(
                f"frequency missing for source word: {source_word!r}"
            ) from exc

        seen_words.add(normalized_word)
        entries.append(
            {
                "word": normalized_word,
                "chosung": extract_chosung(normalized_word),
                "freq": float(frequency),
                "rank": len(entries) + 1,
            }
        )

    word_count = len(entries)

    return {
        "schema_version": SCHEMA_VERSION,
        "language": LANGUAGE,
        "source": SOURCE,
        "wordfreq_version": WORDFREQ_VERSION,
        "wordlist": WORDLIST,
        "input_word_count": input_word_count,
        "word_count": word_count,
        "excluded_word_count": input_word_count - word_count,
        "filtering_rules": list(FILTERING_RULES),
        "sort_order": SORT_ORDER,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "words": entries,
    }


def serialize_dictionary(payload: Mapping[str, Any]) -> bytes:
    """payload를 key 정렬된 공백 없는 UTF-8 JSON bytes로 직렬화한다."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def compress_deterministically(json_bytes: bytes) -> bytes:
    """파일명·현재 시각이 gzip header에 들어가지 않는 bytes를 만든다."""

    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=buffer,
        mtime=0,
    ) as compressed:
        compressed.write(json_bytes)

    return buffer.getvalue()


def write_dictionary(
    payload: Mapping[str, Any],
    output_path: Path,
) -> dict[str, int | str]:
    """결정적으로 압축한 payload를 저장하고 크기·SHA-256을 반환한다."""

    json_bytes = serialize_dictionary(payload)
    gzip_bytes = compress_deterministically(json_bytes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(gzip_bytes)

    return {
        "uncompressed_bytes": len(json_bytes),
        "compressed_bytes": len(gzip_bytes),
        "sha256": hashlib.sha256(gzip_bytes).hexdigest(),
    }


def build_from_wordfreq(
    *,
    max_words: int | None = None,
) -> dict[str, Any]:
    """설치된 wordfreq 3.1.1에서 전체 한국어 사전 payload를 만든다."""

    installed_version = version(SOURCE)
    if installed_version != WORDFREQ_VERSION:
        raise RuntimeError(
            f"wordfreq {WORDFREQ_VERSION} required, found {installed_version}"
        )

    from wordfreq import get_frequency_dict, iter_wordlist

    frequencies = get_frequency_dict(LANGUAGE, wordlist=WORDLIST)
    source_words = iter_wordlist(LANGUAGE, wordlist=WORDLIST)

    return build_dictionary(
        source_words,
        frequencies,
        max_words=max_words,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic Korean chosung word data.",
    )
    parser.add_argument(
        "--max-words",
        type=_positive_int,
        default=None,
        help="Limit source words before filtering (default: all words).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    started_at = time.perf_counter()
    payload = build_from_wordfreq(max_words=args.max_words)
    file_stats = write_dictionary(payload, args.output)
    elapsed_seconds = time.perf_counter() - started_at

    summary = {
        "output": str(args.output.resolve()),
        "input_word_count": payload["input_word_count"],
        "word_count": payload["word_count"],
        "excluded_word_count": payload["excluded_word_count"],
        "elapsed_seconds": round(elapsed_seconds, 6),
        **file_stats,
    }
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
