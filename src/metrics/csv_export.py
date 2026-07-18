"""여러 지표 수집 모듈이 공용으로 쓰는 CSV append 유틸."""

import os
import csv


def append_rows(path, fieldnames, rows):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
