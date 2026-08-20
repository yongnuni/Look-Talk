"""git 커밋 정보 취득 모듈.

sessions 행에 기록할 git_commit을 앱 시작 시 1회 취득해 재사용한다(매 세션
저장마다 subprocess를 새로 띄우지 않기 위함). git이 없는 환경(배포판, CI 등)
이나 git 저장소가 아닌 곳에서 실행돼도 앱이 죽으면 안 되므로 실패는 전부
빈 문자열로 조용히 흡수한다 — sessions.git_commit 결측으로만 드러난다.
"""

import subprocess

_GIT_TIMEOUT_SEC = 5


def get_git_commit():
    """HEAD의 짧은 커밋 해시(7자)를 반환한다.

    작업 트리에 커밋되지 않은 변경이 있으면 뒤에 "-dirty"를 붙인다 — 실험 중
    미커밋 상태로 코드를 돌리는 일이 잦아, 이 표시가 없으면 같은 커밋
    해시인데 실제 코드는 다른 세션들이 섞인다.

    git 미설치, git 저장소 아님, 타임아웃 등 어떤 이유로든 실패하면 빈
    문자열을 반환한다(예외를 던져 앱을 죽이지 않는다).
    """
    commit_out = _run_git(["git", "rev-parse", "--short", "HEAD"])
    if not commit_out:
        return ""

    commit = commit_out.strip()

    status_out = _run_git(["git", "status", "--porcelain"])
    if status_out is not None and status_out.strip():
        commit += "-dirty"

    return commit


def _run_git(args):
    """git 명령을 실행하고 stdout(미가공)을 반환한다. 실패 시 None."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    return result.stdout
