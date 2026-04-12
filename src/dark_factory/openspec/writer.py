"""Write domain models into OpenSpec markdown artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

import jinja2
import structlog

log = structlog.get_logger()

from dark_factory.models.domain import Spec

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,  # M9: explicitly set — output is Markdown, not HTML
)


def init_openspec_dir(root: Path) -> None:
    """Create the OpenSpec directory skeleton."""
    (root / "specs").mkdir(parents=True, exist_ok=True)
    (root / "changes").mkdir(parents=True, exist_ok=True)


def write_spec_md(spec: Spec, root: Path) -> Path:
    """Render a single spec into ``specs/<capability>/spec.md``."""
    capability = spec.capability or spec.title.lower().replace(" ", "-")
    out_dir = root / "specs" / capability
    out_dir.mkdir(parents=True, exist_ok=True)

    template = _env.get_template("spec.md.j2")
    content = template.render(specs=[spec])

    out_path = out_dir / "spec.md"
    out_path.write_text(content)
    log.info("wrote_openspec", path=str(out_path), capability=capability)
    return out_path


def write_proposal(
    change: str, specs: list[Spec], description: str, root: Path
) -> Path:
    """Write ``changes/<change>/proposal.md``."""
    out_dir = _ensure_change_dir(root, change)
    template = _env.get_template("proposal.md.j2")
    out_path = out_dir / "proposal.md"
    out_path.write_text(template.render(specs=specs, description=description))
    return out_path


def write_design(change: str, specs: list[Spec], root: Path) -> Path:
    """Write ``changes/<change>/design.md``."""
    out_dir = _ensure_change_dir(root, change)
    template = _env.get_template("design.md.j2")
    out_path = out_dir / "design.md"
    out_path.write_text(template.render(specs=specs))
    return out_path


def write_tasks(change: str, specs: list[Spec], root: Path) -> Path:
    """Write ``changes/<change>/tasks.md`` with checkbox items."""
    out_dir = _ensure_change_dir(root, change)
    template = _env.get_template("tasks.md.j2")
    out_path = out_dir / "tasks.md"
    out_path.write_text(template.render(specs=specs))
    return out_path


def write_change_specs(change: str, specs: list[Spec], root: Path) -> Path:
    """Write delta spec files into ``changes/<change>/specs/``."""
    out_dir = _ensure_change_dir(root, change) / "specs"
    out_dir.mkdir(exist_ok=True)

    template = _env.get_template("spec.md.j2")
    for spec in specs:
        capability = spec.capability or spec.title.lower().replace(" ", "-")
        spec_dir = out_dir / capability
        spec_dir.mkdir(exist_ok=True)
        (spec_dir / "spec.md").write_text(template.render(specs=[spec]))

    return out_dir


def archive_change(change: str, root: Path) -> Path:
    """Move ``changes/<change>/`` to ``archive/<change>/``."""
    src = root / "changes" / change
    if not src.is_dir():
        raise FileNotFoundError(f"Change not found: {src}")

    archive_dir = root / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / change
    shutil.move(str(src), str(dest))
    return dest


def mark_task_complete(tasks_path: Path, task_id: str) -> None:
    """Check off a task in a tasks.md file (``- [ ] N.M`` → ``- [x] N.M``)."""
    text = tasks_path.read_text()
    text = text.replace(f"- [ ] {task_id} ", f"- [x] {task_id} ", 1)
    tasks_path.write_text(text)


# ── Private helpers ───────────────────────────────────────────────────


def _ensure_change_dir(root: Path, change: str) -> Path:
    out_dir = root / "changes" / change
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
