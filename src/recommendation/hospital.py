"""UTF-8 CSV로 관리하는 병원 특화 NLP 표현 로더."""

from __future__ import annotations

import csv
from pathlib import Path
import unicodedata

from src.recommendation.chosung import extract_chosung
from src.recommendation.models import HOSPITAL_SOURCE, ChosungWord


DEFAULT_HOSPITAL_DICTIONARY_PATH = (
    Path(__file__).resolve().parent / "data" / "hospital_phrases.csv"
)
HOSPITAL_CSV_FIELDS = ("text", "priority", "category", "context")


class HospitalDictionaryError(RuntimeError):
    """병원 표현 CSV를 읽거나 검증할 수 없을 때 발생한다."""


def _hospital_error(path: Path, detail: str) -> HospitalDictionaryError:
    return HospitalDictionaryError(f"invalid hospital dictionary {path}: {detail}")


def load_hospital_phrases(
    path: str | Path | None = None,
) -> tuple[ChosungWord, ...]:
    """병원 표현 CSV를 정규화·검증해 Trie 후보로 반환한다."""

    dictionary_path = (
        DEFAULT_HOSPITAL_DICTIONARY_PATH
        if path is None
        else Path(path).expanduser().resolve()
    )

    try:
        source = dictionary_path.open("r", encoding="utf-8-sig", newline="")
    except FileNotFoundError as exc:
        raise HospitalDictionaryError(
            f"hospital dictionary file not found: {dictionary_path}"
        ) from exc
    except OSError as exc:
        raise HospitalDictionaryError(
            f"failed to read hospital dictionary {dictionary_path}: {exc}"
        ) from exc

    phrases: list[ChosungWord] = []
    seen_texts: set[str] = set()

    try:
        with source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise _hospital_error(dictionary_path, "CSV header is missing")
            if tuple(reader.fieldnames) != HOSPITAL_CSV_FIELDS:
                raise _hospital_error(
                    dictionary_path,
                    "CSV header must be text,priority,category,context",
                )

            for row_number, row in enumerate(reader, start=2):
                raw_text = row.get("text")
                if not isinstance(raw_text, str) or not raw_text:
                    raise _hospital_error(
                        dictionary_path,
                        f"row {row_number} text must be non-empty",
                    )
                if raw_text != raw_text.strip() or "  " in raw_text:
                    raise _hospital_error(
                        dictionary_path,
                        f"row {row_number} text has invalid spacing",
                    )

                text = unicodedata.normalize("NFC", raw_text)
                if not all(char == " " or "가" <= char <= "힣" for char in text):
                    raise _hospital_error(
                        dictionary_path,
                        f"row {row_number} text contains unsupported characters",
                    )
                if text in seen_texts:
                    raise _hospital_error(
                        dictionary_path,
                        f"row {row_number} duplicates {text!r}",
                    )

                raw_priority = row.get("priority")
                try:
                    priority = int(raw_priority) if raw_priority is not None else -1
                except ValueError as exc:
                    raise _hospital_error(
                        dictionary_path,
                        f"row {row_number} priority must be an integer",
                    ) from exc
                if priority < 0 or str(priority) != raw_priority:
                    raise _hospital_error(
                        dictionary_path,
                        f"row {row_number} priority must be a non-negative integer",
                    )

                category = row.get("category")
                context = row.get("context")
                try:
                    phrase = ChosungWord(
                        word=text,
                        chosung=extract_chosung(text),
                        freq=0.0,
                        rank=len(phrases) + 1,
                        source=HOSPITAL_SOURCE,
                        priority=priority,
                        category=category,
                        context=context,
                    )
                except (TypeError, ValueError) as exc:
                    raise _hospital_error(
                        dictionary_path,
                        f"row {row_number} {exc}",
                    ) from exc

                phrases.append(phrase)
                seen_texts.add(text)
    except UnicodeDecodeError as exc:
        raise HospitalDictionaryError(
            f"hospital dictionary is not valid UTF-8: {dictionary_path}"
        ) from exc
    except csv.Error as exc:
        raise _hospital_error(dictionary_path, f"invalid CSV: {exc}") from exc

    return tuple(phrases)
