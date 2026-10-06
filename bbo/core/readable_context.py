"""Load and render packaged, English task priors without evaluator access."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_PRIOR_ROOT = Path(__file__).resolve().parents[1] / "task_context_profiles" / "defaults"


def load_default_setting() -> dict[str, Any]:
    return json.loads((DEFAULT_PRIOR_ROOT / "setting.json").read_text(encoding="utf-8"))


def _prior_task_id(task_name: str, metadata: dict[str, Any] | None = None) -> str:
    alias = load_default_setting().get("aliases", {}).get(task_name)
    if alias is None:
        return task_name
    values = {**alias.get("defaults", {}), **(metadata or {})}
    return alias["template"].format_map(values)


def default_context_profile(task_name: str, *, metadata: dict[str, Any] | None = None) -> str:
    """Use anonymity for BBOB and public task context for other tasks."""
    setting = load_default_setting()
    if task_name.lower().startswith("bbob_"):
        return setting["default_for_bbob"]
    if _prior_task_id(task_name, metadata) in setting["tasks"]:
        return setting["default_for_reviewed_tasks"]
    return setting["default_for_other_tasks"]


def load_default_prior(task_name: str, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    setting = load_default_setting()
    prior_task_id = _prior_task_id(task_name, metadata)
    try:
        filename = setting["tasks"][prior_task_id]
    except KeyError as exc:
        raise ValueError(f"No reviewed default prior for task {task_name!r}.") from exc
    text = (DEFAULT_PRIOR_ROOT / filename).read_text(encoding="utf-8")
    if re.search(r"[\u3400-\u9fff]", text):
        raise ValueError(f"Runtime prior for {task_name!r} must be English.")
    prior = json.loads(text)
    if prior["task_id"] != prior_task_id:
        raise ValueError("Default prior task identity does not match the requested task.")
    return prior


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def render_readable_prior(
    visible: dict[str, Any], *, objective: str | None = None,
    total_evaluations: int | None = None,
) -> str:
    """Render task facts; runtime objective and budget override reference values."""
    budget = visible["evaluation_budget"]
    lines = [f"# {visible['task_name']}", "", f"Task ID: `{visible['task_id']}`.", "",
             visible["background"], "", f"Objective: {objective or visible['objective']}.", ""]
    if total_evaluations is None:
        lines += [f"Budget: {budget['initial']} initial evaluations and {budget['optimization']} optimization evaluations.", ""]
    else:
        lines += [f"Total evaluation budget for this run: {total_evaluations}. Follow the run's initialization protocol and visible evaluated history.", ""]
    # Source links are maintainer provenance, not optimization instructions.
    handled = {"task_name", "task_id", "family", "background", "objective", "evaluation_budget",
               "documentation", "mechanism_sources", "source"}
    for key, title in (("input_instructions", "How to submit a candidate"),
                       ("evaluation_details", "How scoring works"),
                       ("prior_facts", "Task mechanisms"),
                       ("domain_knowledge", "Domain knowledge and when it applies")):
        handled.add(key)
        if visible.get(key):
            lines += [f"## {title}", ""]
            lines += [f"{i}. {fact}" for i, fact in enumerate(visible[key], 1)]
            lines.append("")
    if "public_reference_smiles" in visible:
        lines += ["## Reference molecules", "", "| Name | SMILES |", "| --- | --- |"]
        lines += [f"| {name} | `{smiles}` |" for name, smiles in zip(
            visible["reference_names"], visible["public_reference_smiles"], strict=True)]
        lines.append("")
        handled.update({"reference_names", "public_reference_smiles"})
    parameters = visible["parameters"]
    lines += ["## Parameter definitions", ""]
    if all("physical_definition" in p for p in parameters):
        lines += ["Every submitted parameter is a float in [0,1]. Physical definitions below are decoded using the rules above; they are not direct submission values.", "",
                  "| Parameter | Meaning | Physical definition | Default |", "| --- | --- | --- | --- |"]
        for p in parameters:
            physical = p["physical_definition"]
            actual = ("Enumeration: `" + _json(physical["enum_values"]) + "`") if physical["type"] == "enum" else f"{physical['type'].capitalize()}: `{physical['min']}` to `{physical['max']}`"
            lines.append(f"| `{p['name']}` | {p['meaning']} | {actual} | `{_json(physical['default'])}` |")
    else:
        lines += ["| Parameter | Meaning | Type | Allowed values | Search transform |", "| --- | --- | --- | --- | --- |"]
        for p in parameters:
            bounds = f"[{p['low']}, {p['high']}]" if "low" in p else f"SMILES string; maximum length {p['max_length']}"
            lines.append(f"| `{p['name']}` | {p.get('meaning', 'Molecular structure encoded as SMILES.')} | {p['type']} | {bounds} | {p.get('transform', '—')} |")
    handled.add("parameters")
    labels = {
        "dataset": "Dataset", "estimator": "Estimator", "fixed_estimator_parameters": "Fixed estimator settings",
        "n_macro": "Number of macros", "grid": "Grid dimensions",
        "decoder": "Placement decoder", "evaluator_seed": "Evaluator seed", "initialization": "Reference initialization",
        "canvas_size": "Canvas dimensions", "workload": "Workload", "database": "Database",
        "units_policy": "Units",
        "fingerprint_types": "Fingerprint types", "aggregation": "Score aggregation",
    }
    lines += ["", "## Additional definitions", ""]
    for key, value in visible.items():
        if key in handled:
            continue
        lines += [f"**{labels.get(key, key.replace('_', ' ').capitalize())} (`{key}`)**", ""]
        if isinstance(value, dict):
            lines += ["| Field | Value |", "| --- | --- |"]
            lines += [f"| `{field}` | `{_json(item)}` |" for field, item in value.items()]
        elif isinstance(value, list) and all(isinstance(item, str) and item.startswith("https://") for item in value):
            lines += [f"- [Reference {i}]({item})" for i, item in enumerate(value, 1)]
        elif isinstance(value, str) and value.startswith("https://"):
            lines.append(f"[Reference]({value})")
        else:
            lines.append(value if isinstance(value, str) else "`" + _json(value) + "`")
        lines.append("")
    return "\n".join(lines)
