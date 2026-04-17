"""Task-description loading and schema validation for benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Iterable


def _is_localized_markdown(path: Path) -> bool:
    suffixes = path.suffixes
    if len(suffixes) < 2:
        return False
    return suffixes[-2] in {".zh", ".en"}


@dataclass(frozen=True)
class DescriptionSectionSpec:
    """One canonical section in a task-description schema."""

    kind: str
    title: str
    required: bool = False
    aliases: tuple[str, ...] = ()
    guidance: str = ""

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.kind, *self.aliases)


@dataclass(frozen=True)
class TaskDescriptionSchema:
    """Schema describing the markdown files expected for one benchmark task."""

    name: str
    sections: tuple[DescriptionSectionSpec, ...]
    allow_additional_sections: bool = True

    def canonical_kind(self, raw_name: str) -> str:
        normalized = raw_name.lower().replace("-", "_").strip()
        for section in self.sections:
            if normalized in section.all_names:
                return section.kind
        return normalized if self.allow_additional_sections else "notes"

    def title_for(self, kind: str) -> str:
        for section in self.sections:
            if section.kind == kind:
                return section.title
        return kind.replace("_", " ").title()

    def required_sections(self) -> tuple[str, ...]:
        return tuple(section.kind for section in self.sections if section.required)

    def expected_filenames(self) -> tuple[str, ...]:
        return tuple(f"{section.kind}.md" for section in self.sections)

    def sort_key(self, path: Path) -> tuple[int, str]:
        kind = self.canonical_kind(path.stem)
        order = {section.kind: index for index, section in enumerate(self.sections)}
        return (order.get(kind, len(order) + 1), path.name)

    def discover_files(self, directory: Path | str) -> dict[str, Path]:
        root = Path(directory)
        discovered: dict[str, Path] = {}
        for path in sorted(root.glob("*.md"), key=self.sort_key):
            if _is_localized_markdown(path):
                continue
            kind = self.canonical_kind(path.stem)
            discovered.setdefault(kind, path)
        return discovered

    def missing_sections(self, directory: Path | str) -> tuple[str, ...]:
        discovered = self.discover_files(directory)
        return tuple(kind for kind in self.required_sections() if kind not in discovered)


STANDARD_TASK_DESCRIPTION_SCHEMA = TaskDescriptionSchema(
    name="agentic_bbo_task",
    sections=(
        DescriptionSectionSpec(
            kind="background",
            title="Background",
            required=True,
            aliases=("overview", "context"),
            guidance="Describe the real system, workload, and why this optimization problem matters.",
        ),
        DescriptionSectionSpec(
            kind="goal",
            title="Goal",
            required=True,
            aliases=("objective",),
            guidance="State the optimization target, evaluation contract, and success criteria.",
        ),
        DescriptionSectionSpec(
            kind="constraints",
            title="Constraints",
            required=True,
            aliases=("rules",),
            guidance="List hard constraints, forbidden actions, budgets, and operational limits.",
        ),
        DescriptionSectionSpec(
            kind="prior_knowledge",
            title="Domain Prior Knowledge",
            required=True,
            aliases=("domain_knowledge", "priors", "notes"),
            guidance="Capture heuristics, invariants, and expert priors that a strong human optimizer would use.",
        ),
        DescriptionSectionSpec(
            kind="evaluation",
            title="Evaluation Protocol",
            required=False,
            aliases=("protocol", "metrics"),
            guidance="Clarify metrics, logging, noise, seeds, and how submissions are judged.",
        ),
        DescriptionSectionSpec(
            kind="submission",
            title="Submission Interface",
            required=False,
            aliases=("interface", "io"),
            guidance="Document the exact knobs, outputs, and expected artifact layout.",
        ),
        DescriptionSectionSpec(
            kind="environment",
            title="Environment Setup",
            required=False,
            aliases=("setup", "installation", "runtime_environment"),
            guidance="Explain how collaborators can provision the task environment, or point to the task-local Docker setup.",
        ),
        DescriptionSectionSpec(
            kind="notes",
            title="Additional Notes",
            required=False,
            guidance="Optional implementation notes, caveats, or benchmark history.",
        ),
        DescriptionSectionSpec(
            kind="history",
            title="History",
            required=False,
            aliases=("changelog",),
            guidance="Optional benchmark evolution log.",
        ),
    ),
)


@dataclass(frozen=True)
class TaskDescriptionRef:
    """Reference to one task's markdown description files."""

    task_id: str
    primary_path: Path
    extra_paths: tuple[Path, ...] = ()
    directory: Path | None = None
    schema: TaskDescriptionSchema = STANDARD_TASK_DESCRIPTION_SCHEMA

    @classmethod
    def from_directory(
        cls,
        task_id: str,
        directory: Path | str,
        *,
        schema: TaskDescriptionSchema = STANDARD_TASK_DESCRIPTION_SCHEMA,
    ) -> "TaskDescriptionRef":
        root = Path(directory)
        discovered = schema.discover_files(root)
        sorted_paths = sorted(discovered.values(), key=schema.sort_key)
        if sorted_paths:
            primary = sorted_paths[0]
            extras = tuple(path for path in sorted_paths if path != primary)
        else:
            primary = root / "background.md"
            extras = ()
        return cls(task_id=task_id, primary_path=primary, extra_paths=extras, directory=root, schema=schema)

    def missing_sections(self) -> tuple[str, ...]:
        if self.directory is None:
            return ()
        return self.schema.missing_sections(self.directory)


@dataclass(frozen=True)
class TaskDescriptionDoc:
    """Single markdown document for a task description."""

    path: Path
    content: str
    kind: str
    title: str


@dataclass(frozen=True)
class TaskDescriptionBundle:
    """Loaded task description ready to hand to an algorithm or an agent."""

    task_id: str
    primary: TaskDescriptionDoc | None = None
    extras: tuple[TaskDescriptionDoc, ...] = ()
    rendered_context: str = ""
    fingerprint: str = ""
    schema_name: str = STANDARD_TASK_DESCRIPTION_SCHEMA.name
    section_map: dict[str, str] = field(default_factory=dict)

    @property
    def all_docs(self) -> tuple[TaskDescriptionDoc, ...]:
        if self.primary is None:
            return self.extras
        return (self.primary, *self.extras)

    @property
    def is_empty(self) -> bool:
        return self.primary is None and not self.extras

    @classmethod
    def empty(
        cls,
        task_id: str,
        *,
        schema: TaskDescriptionSchema = STANDARD_TASK_DESCRIPTION_SCHEMA,
    ) -> "TaskDescriptionBundle":
        return cls(task_id=task_id, schema_name=schema.name)


class MarkdownDescriptionLoader:
    """Loads task descriptions from markdown files or directories."""

    def __init__(self, schema: TaskDescriptionSchema = STANDARD_TASK_DESCRIPTION_SCHEMA):
        self.schema = schema

    def load(self, ref: TaskDescriptionRef | Path | str, task_id: str | None = None) -> TaskDescriptionBundle:
        if isinstance(ref, TaskDescriptionRef):
            return self._load_from_ref(ref)

        path = Path(ref)
        inferred_task_id = task_id or path.stem
        if path.is_dir():
            ref = TaskDescriptionRef.from_directory(inferred_task_id, path, schema=self.schema)
        else:
            ref = TaskDescriptionRef(task_id=inferred_task_id, primary_path=path, schema=self.schema)
        return self._load_from_ref(ref)

    def _load_from_ref(self, ref: TaskDescriptionRef) -> TaskDescriptionBundle:
        if not ref.primary_path.exists():
            raise FileNotFoundError(f"Task description file not found: {ref.primary_path}")

        primary = self._load_doc(ref.primary_path, ref.schema)
        extras = tuple(
            self._load_doc(path, ref.schema)
            for path in sorted((path for path in ref.extra_paths if path.exists()), key=ref.schema.sort_key)
        )
        docs = (primary, *extras)
        rendered_context = self._render_docs(docs)
        fingerprint = self._fingerprint(docs)
        section_map = {doc.kind: doc.content for doc in docs}
        return TaskDescriptionBundle(
            task_id=ref.task_id,
            primary=primary,
            extras=extras,
            rendered_context=rendered_context,
            fingerprint=fingerprint,
            schema_name=ref.schema.name,
            section_map=section_map,
        )

    @staticmethod
    def _load_doc(path: Path, schema: TaskDescriptionSchema) -> TaskDescriptionDoc:
        content = path.read_text(encoding="utf-8")
        kind = schema.canonical_kind(path.stem)
        return TaskDescriptionDoc(path=path, content=content, kind=kind, title=schema.title_for(kind))

    @staticmethod
    def _render_docs(docs: Iterable[TaskDescriptionDoc]) -> str:
        sections: list[str] = []
        for doc in docs:
            stripped = doc.content.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                sections.append(stripped)
            else:
                sections.append(f"## {doc.title}\n\n{stripped}")
        return "\n\n".join(sections)

    @staticmethod
    def _fingerprint(docs: Iterable[TaskDescriptionDoc]) -> str:
        digest = sha256()
        for doc in docs:
            digest.update(str(doc.path).encode("utf-8"))
            digest.update(b"\0")
            digest.update(doc.content.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()[:16]


def write_task_description_template(
    directory: Path | str,
    *,
    schema: TaskDescriptionSchema = STANDARD_TASK_DESCRIPTION_SCHEMA,
    include_optional: bool = True,
) -> list[Path]:
    """Create a standardized task-description directory for collaborators."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for section in schema.sections:
        if not include_optional and not section.required:
            continue
        path = root / f"{section.kind}.md"
        if path.exists():
            continue
        heading = f"# {section.title}\n\n"
        bullet = f"- {section.guidance}\n" if section.guidance else "- Fill in this section.\n"
        path.write_text(heading + bullet, encoding="utf-8")
        written.append(path)
    return written


__all__ = [
    "DescriptionSectionSpec",
    "MarkdownDescriptionLoader",
    "STANDARD_TASK_DESCRIPTION_SCHEMA",
    "TaskDescriptionBundle",
    "TaskDescriptionDoc",
    "TaskDescriptionRef",
    "TaskDescriptionSchema",
    "write_task_description_template",
]
