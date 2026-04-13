"""Sandboxed Python-side tool handlers for the agentic loop.

Provides Read/Write/Edit/Glob/Grep/Bash tool implementations that execute
within a sandbox root directory, replacing the Claude Agent SDK's built-in
tools with controlled equivalents.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import structlog

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Tool schemas (Anthropic API compatible)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: dict[str, dict] = {
    "Read": {
        "name": "Read",
        "description": "Read a file and return its contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to read.",
                },
            },
            "required": ["file_path"],
        },
    },
    "Write": {
        "name": "Write",
        "description": "Write content to a file, creating parent directories as needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
            },
            "required": ["file_path", "content"],
        },
    },
    "Edit": {
        "name": "Edit",
        "description": "Replace the first occurrence of old_string with new_string in a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to edit.",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find in the file.",
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace old_string with.",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    "Glob": {
        "name": "Glob",
        "description": "Find files matching a glob pattern within the sandbox.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (e.g. '**/*.py').",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in. Defaults to sandbox root.",
                },
            },
            "required": ["pattern"],
        },
    },
    "Grep": {
        "name": "Grep",
        "description": "Search file contents for a regex pattern using grep -rn.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in. Defaults to sandbox root.",
                },
                "include": {
                    "type": "string",
                    "description": "Glob filter for file names (e.g. '*.py').",
                },
            },
            "required": ["pattern"],
        },
    },
    "Bash": {
        "name": "Bash",
        "description": "Execute a shell command in the sandbox directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 120, max 300).",
                },
            },
            "required": ["command"],
        },
    },
}


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def _resolve_safe(path_str: str, sandbox_root: Path) -> Path:
    """Resolve *path_str* relative to *sandbox_root* and verify containment.

    Raises ``ValueError`` if the resolved path escapes the sandbox.
    """
    resolved = (sandbox_root / path_str).resolve()
    if not resolved.is_relative_to(sandbox_root.resolve()):
        raise ValueError(
            f"Path '{path_str}' resolves to '{resolved}' which is outside "
            f"the sandbox root '{sandbox_root}'."
        )
    return resolved


# ---------------------------------------------------------------------------
# Individual handlers — each returns (result_text, is_error)
# ---------------------------------------------------------------------------


def handle_read(file_path: str, sandbox_root: Path) -> tuple[str, bool]:
    """Read a file within the sandbox and return its contents."""
    try:
        target = _resolve_safe(file_path, sandbox_root)
    except ValueError as exc:
        return str(exc), True

    if not target.exists():
        return f"Error: file not found: {target}", True
    if not target.is_file():
        return f"Error: not a regular file: {target}", True

    try:
        return target.read_text(encoding="utf-8", errors="replace"), False
    except Exception as exc:
        return f"Error reading file: {exc}", True


def handle_write(
    file_path: str, content: str, sandbox_root: Path
) -> tuple[str, bool]:
    """Write content to a file within the sandbox."""
    try:
        target = _resolve_safe(file_path, sandbox_root)
    except ValueError as exc:
        return str(exc), True

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        log.info("tool_write", path=str(target), bytes=len(content.encode("utf-8")))
        return f"Successfully wrote {len(content)} chars to {target}", False
    except Exception as exc:
        return f"Error writing file: {exc}", True


def handle_edit(
    file_path: str,
    old_string: str,
    new_string: str,
    sandbox_root: Path,
) -> tuple[str, bool]:
    """Replace the first occurrence of *old_string* with *new_string*."""
    try:
        target = _resolve_safe(file_path, sandbox_root)
    except ValueError as exc:
        return str(exc), True

    if not target.exists():
        return f"Error: file not found: {target}", True

    try:
        text = target.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Error reading file: {exc}", True

    if old_string not in text:
        return (
            f"Error: old_string not found in {target}. "
            f"Make sure the string matches exactly (including whitespace)."
        ), True

    new_text = text.replace(old_string, new_string, 1)
    try:
        target.write_text(new_text, encoding="utf-8")
        return f"Successfully edited {target}", False
    except Exception as exc:
        return f"Error writing file: {exc}", True


def handle_glob(
    pattern: str, path: str | None, sandbox_root: Path
) -> tuple[str, bool]:
    """Glob for files within the sandbox and return relative paths."""
    if path:
        try:
            search_dir = _resolve_safe(path, sandbox_root)
        except ValueError as exc:
            return str(exc), True
    else:
        search_dir = sandbox_root.resolve()

    if not search_dir.is_dir():
        return f"Error: directory not found: {search_dir}", True

    try:
        matches = sorted(search_dir.glob(pattern))
        # Filter to only paths within sandbox
        sandbox_resolved = sandbox_root.resolve()
        safe_matches = [
            m for m in matches if m.resolve().is_relative_to(sandbox_resolved)
        ]
        if not safe_matches:
            return "No files matched the pattern.", False
        rel_paths = [
            str(m.resolve().relative_to(sandbox_resolved)) for m in safe_matches
        ]
        return "\n".join(rel_paths), False
    except Exception as exc:
        return f"Error during glob: {exc}", True


def handle_grep(
    pattern: str,
    path: str | None,
    include: str | None,
    sandbox_root: Path,
) -> tuple[str, bool]:
    """Run grep -rn within the sandbox."""
    if path:
        try:
            search_path = _resolve_safe(path, sandbox_root)
        except ValueError as exc:
            return str(exc), True
    else:
        search_path = sandbox_root.resolve()

    cmd = ["grep", "-rn", pattern, str(search_path)]
    if include:
        cmd = ["grep", "-rn", f"--include={include}", pattern, str(search_path)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(sandbox_root),
        )
        output = result.stdout
        if result.returncode == 1:
            # grep returns 1 when no matches found — not an error
            return "No matches found.", False
        if result.returncode > 1:
            return f"grep error (rc={result.returncode}): {result.stderr}", True
        return output or "No matches found.", False
    except subprocess.TimeoutExpired:
        return "Error: grep timed out after 30 seconds.", True
    except Exception as exc:
        return f"Error running grep: {exc}", True


def handle_bash(
    command: str, timeout: int | None, sandbox_root: Path
) -> tuple[str, bool]:
    """Execute a shell command within the sandbox directory."""
    effective_timeout = min(timeout or 120, 300)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            cwd=str(sandbox_root),
        )
        output = result.stdout
        if result.stderr:
            output = output + ("\n" if output else "") + result.stderr
        if not output:
            output = f"(exit code {result.returncode})"
        is_error = result.returncode != 0
        return output, is_error
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {effective_timeout} seconds.", True
    except Exception as exc:
        return f"Error running command: {exc}", True


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def execute_tool(
    name: str, tool_input: dict, sandbox_root: Path
) -> tuple[str, bool]:
    """Route a tool call to the correct handler.

    Returns ``(result_text, is_error)``.
    """
    log.debug("execute_tool", tool=name, sandbox=str(sandbox_root))

    if name == "Read":
        return handle_read(tool_input["file_path"], sandbox_root)
    if name == "Write":
        return handle_write(
            tool_input["file_path"], tool_input["content"], sandbox_root
        )
    if name == "Edit":
        return handle_edit(
            tool_input["file_path"],
            tool_input["old_string"],
            tool_input["new_string"],
            sandbox_root,
        )
    if name == "Glob":
        return handle_glob(
            tool_input["pattern"], tool_input.get("path"), sandbox_root
        )
    if name == "Grep":
        return handle_grep(
            tool_input["pattern"],
            tool_input.get("path"),
            tool_input.get("include"),
            sandbox_root,
        )
    if name == "Bash":
        return handle_bash(
            tool_input["command"], tool_input.get("timeout"), sandbox_root
        )

    return f"Error: unknown tool '{name}'.", True


# ---------------------------------------------------------------------------
# Schema helper
# ---------------------------------------------------------------------------


def get_tool_schemas(allowed_tools: list[str]) -> list[dict]:
    """Return Anthropic-compatible tool definitions for the named tools."""
    schemas = []
    for name in allowed_tools:
        if name in TOOL_SCHEMAS:
            schemas.append(TOOL_SCHEMAS[name])
        else:
            log.warning("get_tool_schemas_unknown", tool=name)
    return schemas
