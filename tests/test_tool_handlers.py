"""Tests for ``dark_factory.llm.tool_handlers`` — sandboxed tool execution."""

from pathlib import Path

import pytest

from dark_factory.llm.tool_handlers import (
    _resolve_safe,
    execute_tool,
    get_tool_schemas,
    handle_bash,
    handle_edit,
    handle_glob,
    handle_grep,
    handle_read,
    handle_write,
)


# ── Path safety ───────────────────────────────────────────────────────────────


def test_resolve_safe_relative_path(tmp_path: Path):
    result = _resolve_safe("foo/bar.txt", tmp_path)
    assert result == (tmp_path / "foo" / "bar.txt").resolve()


def test_resolve_safe_rejects_traversal(tmp_path: Path):
    with pytest.raises(ValueError, match="outside"):
        _resolve_safe("../../etc/passwd", tmp_path)


def test_resolve_safe_rejects_absolute_outside(tmp_path: Path):
    with pytest.raises(ValueError, match="outside"):
        _resolve_safe("/etc/passwd", tmp_path)


# ── Read ──────────────────────────────────────────────────────────────────────


def test_handle_read_success(tmp_path: Path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    result, is_error = handle_read("hello.txt", tmp_path)
    assert not is_error
    assert result == "hello world"


def test_handle_read_not_found(tmp_path: Path):
    result, is_error = handle_read("missing.txt", tmp_path)
    assert is_error
    assert "not found" in result


def test_handle_read_rejects_traversal(tmp_path: Path):
    result, is_error = handle_read("../../etc/passwd", tmp_path)
    assert is_error
    assert "outside" in result


# ── Write ─────────────────────────────────────────────────────────────────────


def test_handle_write_creates_file(tmp_path: Path):
    result, is_error = handle_write("sub/test.py", "print('hi')", tmp_path)
    assert not is_error
    assert (tmp_path / "sub" / "test.py").read_text() == "print('hi')"


def test_handle_write_rejects_traversal(tmp_path: Path):
    result, is_error = handle_write("../../evil.txt", "bad", tmp_path)
    assert is_error
    assert "outside" in result


# ── Edit ──────────────────────────────────────────────────────────────────────


def test_handle_edit_replaces_text(tmp_path: Path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\ny = 2\n")
    result, is_error = handle_edit("code.py", "x = 1", "x = 42", tmp_path)
    assert not is_error
    assert f.read_text() == "x = 42\ny = 2\n"


def test_handle_edit_old_string_not_found(tmp_path: Path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n")
    result, is_error = handle_edit("code.py", "NOPE", "replacement", tmp_path)
    assert is_error
    assert "not found" in result


# ── Glob ──────────────────────────────────────────────────────────────────────


def test_handle_glob_finds_files(tmp_path: Path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")
    result, is_error = handle_glob("*.py", None, tmp_path)
    assert not is_error
    assert "a.py" in result
    assert "b.py" in result
    assert "c.txt" not in result


def test_handle_glob_no_matches(tmp_path: Path):
    result, is_error = handle_glob("*.xyz", None, tmp_path)
    assert not is_error
    assert "No files" in result


# ── Grep ──────────────────────────────────────────────────────────────────────


def test_handle_grep_finds_matches(tmp_path: Path):
    (tmp_path / "hello.txt").write_text("hello world\ngoodbye world\n")
    result, is_error = handle_grep("hello", None, None, tmp_path)
    assert not is_error
    assert "hello world" in result


def test_handle_grep_no_matches(tmp_path: Path):
    (tmp_path / "hello.txt").write_text("hello world\n")
    result, is_error = handle_grep("ZZZZZ", None, None, tmp_path)
    assert not is_error
    assert "No matches" in result


# ── Bash ──────────────────────────────────────────────────────────────────────


def test_handle_bash_success(tmp_path: Path):
    result, is_error = handle_bash("echo hello", None, tmp_path)
    assert not is_error
    assert "hello" in result


def test_handle_bash_failure(tmp_path: Path):
    result, is_error = handle_bash("exit 1", None, tmp_path)
    assert is_error


def test_handle_bash_timeout(tmp_path: Path):
    result, is_error = handle_bash("sleep 10", 1, tmp_path)
    assert is_error
    assert "timed out" in result


# ── Dispatcher ────────────────────────────────────────────────────────────────


def test_execute_tool_routes_correctly(tmp_path: Path):
    (tmp_path / "f.txt").write_text("data")
    result, is_error = execute_tool("Read", {"file_path": "f.txt"}, tmp_path)
    assert not is_error
    assert result == "data"


def test_execute_tool_unknown(tmp_path: Path):
    result, is_error = execute_tool("DoesNotExist", {}, tmp_path)
    assert is_error
    assert "unknown" in result


# ── Schema helper ─────────────────────────────────────────────────────────────


def test_get_tool_schemas_filters():
    schemas = get_tool_schemas(["Read", "Write"])
    assert len(schemas) == 2
    names = {s["name"] for s in schemas}
    assert names == {"Read", "Write"}


def test_get_tool_schemas_ignores_unknown():
    schemas = get_tool_schemas(["Read", "FakeToolXYZ"])
    assert len(schemas) == 1
    assert schemas[0]["name"] == "Read"
