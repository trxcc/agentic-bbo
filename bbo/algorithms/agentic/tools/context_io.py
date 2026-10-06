"""Bounded context queries and host-owned candidate submission for raw agents."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import uuid
from typing import Any, Callable

from ....core import TaskSpec, TrialObservation, search_space_to_schema
from ..serialization import stable_config_identity
from .base import BaseBBOTool
from .core_tools import agent_visible_payload


INSTRUCTIONS = """# Optimization protocol

Read task.md first. Query parameter definitions, task details and evaluated
history as needed; full data files need not be printed. Use only permitted task
context and real observations. Do not execute or inspect the hidden evaluator,
fabricate observations, install dependencies, or modify task or history files.
Do not load task datasets or train a replica of the objective, even from a public
dataset library. Surrogate modeling fitted only to visible evaluated history and
scratch work below scratch/ are allowed.

Submit directly with submit_candidate(config={...}), or generate a complete JSON
configuration below scratch/ using native tools and submit_candidate(path="scratch/candidate.json").
The file may contain the raw configuration or {"config": {...}}. It is read and
snapshotted by the host at submission. You may generate any legal configuration;
you do not have to modify an existing trial. For convenience, submit_candidate
also accepts base_trial_id plus changes, inheriting all other values exactly.
write_candidate is optional preparation/validation, returning a candidate_id
that submit_candidate can also accept. A separate write call is not required.
After acceptance, stop calling tools; a short acknowledgement suffices. Do not
repeat the configuration. On later rounds use the new host feedback; query
get_trial_history(after_trial_id=...) for additional new observations rather
than repeatedly printing the full history. Older observations remain queryable.
The host evaluates the committed candidate and updates history in the next round.
A submission receipt is not an objective value. Correct rejected candidates by
submitting a corrected configuration, file, or optional saved candidate ID.

History queries are paginated by both row count and character budget. mode="all"
selects all observations but does not return them in a single page. Follow the
exact next_cursor until it is null; never treat a trial_id as an array index.
For local analysis, use get_trial_history(mode="all", include_config=true,
output_path="scratch/observations.json") to export all selected observations
without printing them. The file contains items and total; no pagination is needed.
Do not combine output_path with cursor, limit or max_chars.

get_incumbent returns scores by default. For configuration queries, explicit
parameter_names take precedence over include_config. To read a full incumbent,
use include_config=true; max_chars defaults to 24000. For larger configurations,
add output_path="scratch/incumbent.json" and omit max_chars. The exported items
array contains only the best observed trial, or is empty if none exists.
"""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as handle:
        handle.write(_canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def context_documents(task: TaskSpec, full_markdown: str, prior: dict | None = None) -> dict:
    """Derive views from already policy-filtered context and the canonical schema."""
    titles = {"How to submit a candidate": "submission", "How scoring works": "scoring",
              "Task mechanisms": "mechanisms", "Domain knowledge and when it applies": "domain_knowledge",
              "Reference molecules": "references", "Additional definitions": "additional"}
    parts = re.split(r"^## (.+)\n", full_markdown, flags=re.MULTILINE)
    sections = {"overview": parts[0].strip()}
    parameter_table = ""
    for title, body in zip(parts[1::2], parts[2::2]):
        if title == "Parameter definitions":
            parameter_table = body.strip()
        else:
            sections[titles.get(title, title.lower().replace(" ", "_"))] = body.strip()
    semantic = {p["name"]: p for p in (prior or {}).get("parameters", [])}
    parameters = []
    for schema in search_space_to_schema(task.search_space):
        definition = semantic.get(schema["name"], {})
        parameters.append({**schema, **{k: definition[k] for k in
                           ("meaning", "physical_definition", "transform", "group") if k in definition}})
    lines = [sections["overview"]]
    for key in ("submission", "references"):
        if sections.get(key):
            lines += [f"## {key.capitalize()}", sections[key]]
    if len(parameters) <= 8 and parameter_table:
        lines += ["## Parameters", parameter_table]
    lines += ["## Read details as needed",
              f"There are {len(parameters)} active parameters. get_search_space supports names, query, optional annotated groups and paged index/details views.",
              "get_task_context sections: " + ", ".join(sections) + ". Read scoring rules and relevant mechanisms before choosing a candidate.",
              "get_trial_history and get_incumbent return scores first; request parameter_names to inspect selected values.",
              "Follow instructions.md: submit_candidate accepts a full config or workspace JSON file directly; write_candidate is optional."]
    return {"protocol_version": 2, "short_task": "\n\n".join(lines) + "\n", "sections": sections, "parameters": parameters}


class ContextIOSession:
    """One immutable observation snapshot; writes never call an evaluator."""

    def __init__(self, task: TaskSpec, history: list[TrialObservation], documents: dict,
                 state_dir: Path, workspace: Path, context_fingerprint: str):
        self.task = task
        self.documents = copy.deepcopy(documents)
        self.history = [{"trial_id": o.suggestion.trial_id, "config": dict(o.suggestion.config),
                         "status": o.status.value, "objectives": dict(o.objectives)} for o in history]
        self.version = _hash({"context": context_fingerprint, "history": self.history,
                              "space": search_space_to_schema(task.search_space), "budget": task.max_evaluations})[:24]
        self.directory = state_dir / "context_io" / self.version
        self.directory.mkdir(parents=True, exist_ok=True)
        self.workspace = workspace
        self.lock = threading.RLock()
        self._seen = {stable_config_identity(o["config"]) for o in self.history}

    @property
    def committed_payload(self) -> dict | None:
        path = self.directory / "submission.json"
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        candidate = self._candidate(record["candidate_id"])
        return {"candidates": [{"config": candidate["config"]}]}

    def _candidate(self, candidate_id: str) -> dict:
        if not isinstance(candidate_id, str) or not re.fullmatch(r"c_[a-f0-9]{32}", candidate_id):
            raise ValueError("Invalid candidate_id.")
        path = self.directory / (candidate_id + ".json")
        if not path.exists():
            raise ValueError("Unknown or stale candidate_id for this round.")
        record = json.loads(path.read_text())
        if record["context_version"] != self.version or "c_" + _hash(record)[:32] != candidate_id:
            raise ValueError("Candidate integrity check failed.")
        return record

    def _validate(self, config: dict) -> dict:
        if not isinstance(config, dict) or set(config) != set(self.task.search_space.names()):
            raise ValueError("Provide every active parameter exactly once; unknown or missing names are invalid.")
        for parameter in self.task.search_space:
            parameter.validate(config[parameter.name])
        return self.task.search_space.coerce_config(config, use_defaults=False)

    def _page(self, values: list, args: dict, signature: Any) -> dict:
        limit = args.get("limit", 10)
        max_chars = args.get("max_chars", 6000)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100.")
        if type(max_chars) is not int or not 1000 <= max_chars <= 24000:
            raise ValueError("max_chars must be an integer from 1000 to 24000.")
        prefix = _hash([self.version, signature])[:20]
        cursor = args.get("cursor")
        start = 0
        if cursor is not None:
            match = re.fullmatch(re.escape(prefix) + r":(\d+)", str(cursor))
            if match is None:
                raise ValueError("Cursor does not match this query and context version.")
            start = int(match[1])
        if start > len(values):
            raise ValueError("Cursor is outside this result set.")
        result = {"context_version": self.version, "total": len(values), "returned": 0,
                  "items": [], "next_cursor": None}
        for item in values[start:start + limit]:
            proposed = {**result, "items": result["items"] + [item],
                        "returned": result["returned"] + 1,
                        "next_cursor": f"{prefix}:{start + result['returned'] + 1}"}
            if len(_canonical(proposed)) > max_chars - 256:
                if not result["items"]:
                    raise ValueError("One complete item exceeds max_chars; request a larger limit or fewer parameter_names.")
                break
            result = proposed
        end = start + result["returned"]
        result["next_cursor"] = f"{prefix}:{end}" if end < len(values) else None
        result.update(has_more=end < len(values), page_start=start, page_end=end)
        if end < len(values):
            result["note"] = "Partial page. Reuse the exact next_cursor with the same query; limit is only an upper bound."
        return result

    def _projection(self, config: dict, args: dict) -> dict | None:
        names = args.get("parameter_names")
        if names is not None:
            if not isinstance(names, list) or any(not isinstance(n, str) or n not in config for n in names):
                raise ValueError("parameter_names must contain declared parameter names.")
            return agent_visible_payload({n: config[n] for n in names})
        return agent_visible_payload(config) if args.get("include_config") else None

    def _best_rows(self, rows: list) -> list:
        objective = self.task.primary_objective
        successful = [r for r in rows if r["status"] == "success" and objective.name in r["objectives"]]
        return sorted(successful, key=lambda r: r["objectives"][objective.name],
                      reverse=objective.direction.value == "maximize")

    def execute(self, name: str, args: dict) -> dict:
        with self.lock:
            if (self.directory / "submission.json").exists() and name != "submit_candidate":
                raise ValueError("This round has been submitted. End the turn.")
            result = getattr(self, name)(**args)
            return {"context_version": self.version, **result}

    def get_task_context(self, section: str = "overview", **page: Any) -> dict:
        sections = self.documents["sections"]
        if section not in sections:
            raise ValueError("Unknown section. Available: " + ", ".join(sections))
        # Each bullet is an independently retrievable unit even when a renderer
        # uses single newlines between domain rules.
        paragraphs = [s for s in re.split(r"\n\n|\n(?=- )", sections[section]) if s]
        return {"section": section, "available_sections": list(sections),
                **self._page(paragraphs, page, ["context", section])}

    def get_search_space(self, view: str = "index", names: list[str] | None = None,
                         query: str = "", group: str | None = None, **page: Any) -> dict:
        if view not in {"index", "details"}:
            raise ValueError("view must be index or details.")
        parameters = self.documents["parameters"]
        if names is not None and (not isinstance(names, list) or any(n not in self.task.search_space.names() for n in names)):
            raise ValueError("names must contain declared parameter names.")
        selected = [p for p in parameters if (names is None or p["name"] in names)
                    and (group is None or p.get("group") == group)
                    and query.lower() in (p["name"] + " " + p.get("meaning", "")).lower()]
        if view == "index":
            selected = [{k: p[k] for k in ("name", "type", "meaning", "group") if k in p} for p in selected]
        return {"dimension": len(parameters), **self._page(selected, page, ["space", view, names, query, group])}

    def get_trial_history(self, mode: str = "recent", trial_ids: list | None = None,
                          after_trial_id: str | int | None = None,
                          parameter_names: list | None = None, include_config: bool = False,
                          output_path: str | None = None, **page: Any) -> dict:
        if mode not in {"recent", "best", "all"}:
            raise ValueError("mode must be recent, best or all.")
        rows = list(self.history)
        if after_trial_id is not None:
            positions = [i for i, row in enumerate(rows) if row["trial_id"] == after_trial_id]
            if len(positions) != 1:
                raise ValueError("Unknown or ambiguous after_trial_id.")
            rows = rows[positions[0] + 1:]
        if trial_ids is not None:
            if any(i not in [r["trial_id"] for r in rows] for i in trial_ids):
                raise ValueError("Unknown trial_id.")
            rows = [r for r in rows if r["trial_id"] in trial_ids]
        if mode == "recent":
            rows.reverse()
        elif mode == "best":
            rows = self._best_rows(rows)
        items = []
        for row in rows:
            item = {k: agent_visible_payload(row[k]) for k in ("trial_id", "status", "objectives")}
            projection = self._projection(row["config"], dict(parameter_names=parameter_names, include_config=include_config))
            if projection is not None:
                item["config"] = projection
            items.append(item)
        if output_path is not None:
            if page:
                raise ValueError("output_path exports all selected rows; omit cursor, limit and max_chars.")
            return self._export_history(output_path, items)
        return {"observations": len(self.history), "evaluations_remaining": max(0, self.task.max_evaluations - len(self.history)),
                **self._page(items, {"limit": 5, **page}, ["history", mode, trial_ids, after_trial_id, parameter_names, include_config])}

    def _export_history(self, output_path: str, items: list) -> dict:
        """Write only under scratch/, without following agent-controlled symlinks."""
        if not isinstance(output_path, str):
            raise ValueError("output_path must be a relative scratch/*.json path")
        parts = output_path.split("/")
        if (len(parts) < 2 or parts[0] != "scratch" or any(p in {"", ".", ".."} for p in parts)
                or not parts[-1].endswith(".json")):
            raise ValueError("output_path must be a relative scratch/*.json path without traversal")
        self.workspace.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        temporary = ".history-" + uuid.uuid4().hex + ".tmp"
        try:
            for part in parts[:-1]:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=descriptor)
            with os.fdopen(fd, "w") as handle:
                handle.write(_canonical({"context_version": self.version, "total": len(items), "items": items}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, parts[-1], src_dir_fd=descriptor, dst_dir_fd=descriptor)
        finally:
            try:
                os.unlink(temporary, dir_fd=descriptor)
            except FileNotFoundError:
                pass
            os.close(descriptor)
        return {"context_version": self.version, "output_path": output_path, "format": "json",
                "total": len(items), "exported": len(items), "has_more": False,
                "note": "All selected rows are in items in the exported JSON file. Read it locally; do not print the full file."}

    def get_incumbent(self, parameter_names: list | None = None, include_config: bool = False,
                      max_chars: int | None = None, output_path: str | None = None) -> dict:
        if output_path is not None and max_chars is not None:
            raise ValueError("output_path exports the incumbent without printing it; omit max_chars.")
        budget = 24000 if max_chars is None else max_chars
        if type(budget) is not int or not 1000 <= budget <= 24000:
            raise ValueError("max_chars must be an integer from 1000 to 24000.")
        rows = self._best_rows(self.history)
        incumbent = None
        if rows:
            row = rows[0]
            incumbent = {k: agent_visible_payload(row[k]) for k in ("trial_id", "status", "objectives")}
            projection = self._projection(row["config"], dict(parameter_names=parameter_names, include_config=include_config))
            if projection is not None:
                incumbent["config"] = projection
        if output_path is not None:
            return self._export_history(output_path, [incumbent] if incumbent is not None else [])
        result = {"incumbent": incumbent}
        if len(_canonical({"context_version": self.version, **result})) > budget:
            raise ValueError("Incumbent exceeds max_chars; increase max_chars (up to 24000), select fewer "
                             "parameter_names, or use output_path='scratch/incumbent.json' without max_chars.")
        return result

    def _prepare_candidate(self, config: dict | None, base_trial_id: Any, changes: dict | None) -> tuple[dict, dict]:
        if config is not None:
            if base_trial_id is not None or changes is not None:
                raise ValueError("Use config OR base_trial_id plus changes.")
            before = {}
            candidate = self._validate(config)
        else:
            if base_trial_id is None or not isinstance(changes, dict):
                raise ValueError("Supply config OR explicit base_trial_id and changes.")
            matching = [r for r in self.history if r["trial_id"] == base_trial_id]
            if len(matching) != 1:
                raise ValueError("Unknown or ambiguous base_trial_id.")
            before = matching[0]["config"]
            candidate = self._validate({**before, **changes})
        if stable_config_identity(candidate) in self._seen:
            raise ValueError("Candidate duplicates an evaluated configuration.")
        record = {"context_version": self.version, "config": candidate, "base_trial_id": base_trial_id}
        changed = {k: v for k, v in candidate.items() if k not in before or v != before[k]}
        return record, changed

    def write_candidate(self, config: dict | None = None, base_trial_id: Any = None,
                        changes: dict | None = None) -> dict:
        record, changed = self._prepare_candidate(config, base_trial_id, changes)
        candidate = record["config"]
        candidate_id = "c_" + _hash(record)[:32]
        _write_json(self.directory / (candidate_id + ".json"), record)
        # Large writes never echo the complete configuration into the transcript.
        return {"candidate_id": candidate_id, "valid": True, "parameter_count": len(candidate),
                "changed_count": len(changed), "inherited_count": len(candidate) - len(changed),
                "changes": agent_visible_payload(changed) if len(_canonical(changed)) < 3000 else None,
                "submitted": False, "evaluated": False}

    def _read_candidate_file(self, path: str) -> dict:
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise ValueError("path must be a relative path to a JSON file inside the workspace.")
        location = (self.workspace / path).resolve(strict=True)
        if not location.is_relative_to(self.workspace.resolve()):
            raise ValueError("Candidate file must remain inside the workspace, including symlink targets.")
        if not stat.S_ISREG(location.stat().st_mode):
            raise ValueError("Candidate path must be a regular JSON file.")
        with location.open("rb") as handle:
            data = handle.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024:
            raise ValueError("Candidate JSON exceeds the 2 MiB limit.")

        def unique_object(pairs: list) -> dict:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON field: " + key)
                result[key] = value
            return result

        def invalid_constant(value: str) -> None:
            raise ValueError("Non-finite JSON number: " + value)

        value = json.loads(data, object_pairs_hook=unique_object, parse_constant=invalid_constant)
        if isinstance(value, dict) and set(value) == {"config"}:
            value = value["config"]
        return value

    def submit_candidate(self, candidate_id: str | None = None, config: dict | None = None,
                         path: str | None = None, base_trial_id: Any = None,
                         changes: dict | None = None) -> dict:
        modes = [candidate_id is not None, config is not None, path is not None,
                 base_trial_id is not None or changes is not None]
        if sum(modes) != 1:
            raise ValueError("Choose exactly one: config, path, candidate_id, or base_trial_id plus changes.")
        if candidate_id is not None:
            record = self._candidate(candidate_id)
        else:
            if path is not None:
                config = self._read_candidate_file(path)
            record, _ = self._prepare_candidate(config, base_trial_id, changes)
            candidate_id = "c_" + _hash(record)[:32]
        submission_path = self.directory / "submission.json"
        if submission_path.exists():
            receipt = json.loads(submission_path.read_text())
            if receipt["candidate_id"] != candidate_id:
                raise ValueError("This round already has a different submitted candidate.")
        else:
            if len(self.history) >= self.task.max_evaluations:
                raise ValueError("Evaluation budget exhausted.")
            config = self._validate(record["config"])
            if stable_config_identity(config) in self._seen:
                raise ValueError("Candidate duplicates an evaluated configuration.")
            _write_json(self.directory / (candidate_id + ".json"), record)
            receipt = {"candidate_id": candidate_id, "submission_id": "s_" + _hash([self.version, candidate_id])[:24],
                       "status": "accepted", "round_complete": True, "terminal": True,
                       "evaluation_status": "pending", "objective_value": None}
            _write_json(submission_path, receipt)
        # Recover this public artifact from the authoritative durable receipt too.
        _write_json(self.workspace / "final_candidate.json", {"candidates": [{"config": record["config"]}]})
        return receipt


def _schema(properties: dict, required: tuple = ()) -> dict:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


_PAGE = {"cursor": {"type": "string", "description": "Use the exact returned next_cursor, with the same query."},
         "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Upper bound on rows; max_chars may return fewer."},
         "max_chars": {"type": "integer", "minimum": 1000, "maximum": 24000, "description": "Page character budget, default 6000. Follow next_cursor until null."}}
_NAMES = {"type": "array", "items": {"type": "string"}}
_PROJECT = {
    "parameter_names": {**_NAMES, "description": "Return only these configuration columns; takes precedence over include_config."},
    "include_config": {"type": "boolean", "description": "Return the full configuration when parameter_names is omitted; default false."},
}
SPECS = {
    "get_task_context": ("Read a named section of task details with pagination.", _schema({"section": {"type": "string"}, **_PAGE})),
    "get_search_space": ("Find parameter names or retrieve exact definitions; default is a paged index.", _schema({"view": {"enum": ["index", "details"], "type": "string"}, "names": _NAMES, "query": {"type": "string"}, "group": {"type": "string"}, **_PAGE})),
    "get_trial_history": ("Read observed scores with pagination; mode=all still returns a page. Follow next_cursor, or export all selected rows to a scratch JSON file with output_path. Default is 5 recent trials; after_trial_id gives incremental feedback.", _schema({"mode": {"enum": ["recent", "best", "all"], "type": "string"}, "trial_ids": {"type": "array", "items": {"type": ["string", "integer"]}}, "after_trial_id": {"type": ["string", "integer"]}, "output_path": {"type": "string", "description": "Export all selected rows to scratch/*.json, without printing rows. Omit pagination arguments."}, **_PROJECT, **_PAGE})),
    "get_incumbent": ("Return the best observed trial; request configuration columns only as needed, or export one row to a scratch JSON file without printing it.", _schema({
        **_PROJECT,
        "max_chars": {"type": "integer", "minimum": 1000, "maximum": 24000,
                      "description": "Response character budget, default 24000. Use output_path for larger configurations."},
        "output_path": {"type": "string", "description": "Export only the best selected row to scratch/*.json (items array, empty if no incumbent). Omit max_chars."},
    })),
    "write_candidate": ("Optional: validate and save a draft, without submitting. Direct submit_candidate does not require this step.", _schema({"config": {"type": "object"}, "base_trial_id": {"type": ["string", "integer"]}, "changes": {"type": "object"}})),
    "submit_candidate": ("Validate, commit and end this round in one call. Choose config, workspace JSON path, base_trial_id plus changes, or a saved candidate_id. Idempotent; host evaluates afterward.", _schema({"config": {"type": "object"}, "path": {"type": "string"}, "base_trial_id": {"type": ["string", "integer"]}, "changes": {"type": "object"}, "candidate_id": {"type": "string"}})),
}


class ContextIOTool(BaseBBOTool):
    def __init__(self, name: str, session: Callable[[], ContextIOSession]):
        self.name = name
        self.description, self.parameters_schema = SPECS[name]
        self.session = session

    async def execute(self, context: Any, **arguments: Any) -> dict:
        extra = set(arguments) - set(self.parameters_schema["properties"])
        if extra:
            raise ValueError("Unknown arguments: " + ", ".join(sorted(extra)))
        return self.session().execute(self.name, arguments)


def create_context_io_tools(session: Callable[[], ContextIOSession]) -> list[BaseBBOTool]:
    return [ContextIOTool(name, session) for name in SPECS]


class SessionBoundOptimizerTool(BaseBBOTool):
    """Compose an optimizer with the same terminal submission boundary as IO tools."""

    def __init__(self, tool: BaseBBOTool, session: Callable[[], ContextIOSession | None]):
        self.tool = tool
        self.session = session
        self.name = tool.name
        self.description = tool.description
        self.parameters_schema = tool.parameters_schema

    async def execute(self, context, **kwargs):
        session = self.session()
        if session is None:
            raise ValueError("No active optimization round.")
        # Shared with submission: a concurrent commit cannot race a policy mutation.
        with session.lock:
            if session.committed_payload is not None:
                raise ValueError("This round has been submitted. End the turn.")
            return await self.tool.execute(context, **kwargs)
