"""src/common/git_info.py 단위 테스트.

get_git_commit()은 main() 시작 시 1회만 호출되는 함수라 subprocess를 직접
실행해도 테스트가 느려지지 않는다. tmp_path에 실제 git 저장소를 새로 만들어
clean/dirty 두 상태와 저장소 밖 상태를 모두 검증한다.
"""

import subprocess

from src.common import git_info


def _run(args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


def _init_repo(path):
    _run(["git", "init"], cwd=path)
    _run(["git", "config", "user.email", "test@example.com"], cwd=path)
    _run(["git", "config", "user.name", "Test"], cwd=path)
    (path / "a.txt").write_text("1")
    _run(["git", "add", "a.txt"], cwd=path)
    _run(["git", "commit", "-m", "initial"], cwd=path)


def test_returns_empty_string_outside_git_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert git_info.get_git_commit() == ""


def test_returns_short_hash_in_clean_repo(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    commit = git_info.get_git_commit()

    assert commit != ""
    assert not commit.endswith("-dirty")


def test_appends_dirty_suffix_when_uncommitted_changes(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("2")  # 미커밋 변경
    monkeypatch.chdir(tmp_path)

    commit = git_info.get_git_commit()

    assert commit.endswith("-dirty")


def test_returns_empty_string_when_git_not_installed(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(git_info.subprocess, "run", _raise)

    assert git_info.get_git_commit() == ""


def test_returns_empty_string_on_timeout(monkeypatch):
    def _raise(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=5)

    monkeypatch.setattr(git_info.subprocess, "run", _raise)

    assert git_info.get_git_commit() == ""
