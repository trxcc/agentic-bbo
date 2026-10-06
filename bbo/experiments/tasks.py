"""Frozen public context with lazy, journaled host evaluation."""
from __future__ import annotations

import copy
from dataclasses import asdict, fields, replace
from pathlib import Path

from bbo.core import (Task, TaskSpec, TaskDescriptionRef, TaskDescriptionBundle,
    SearchSpace, FloatParam, IntParam, CategoricalParam, StringParam, ObjectiveSpec,
    ObjectiveDirection, EvaluationResult, TrialObservation, TrialSuggestion, TrialStatus)
from bbo.core.description import TaskDescriptionDoc
from .io import digest, read, save

ASSETS = Path(__file__).resolve().parent / "assets"
DIAGNOSTIC_TASKS = (
    "bbob_f02_d10", "bbob_f15_d10", "hpo_bayesmark_breast_svm",
    "hpo_bayesmark_diabetes_random_forest", "knob_http_surrogate_pg_5",
    "knob_http_surrogate_sysbench_5", "bboplace_adaptec1_n32", "bboplace_bigblue1_n32",
)


def schema_space(parameters):
    classes = {"float": FloatParam, "int": IntParam, "categorical": CategoricalParam, "string": StringParam}
    result = []
    for p in parameters:
        cls = classes[p["type"]]
        allowed = {f.name for f in fields(cls)}
        result.append(cls(**{k: v for k, v in p.items() if k in allowed}))
    return SearchSpace(result)


def task_rows(suite="main"):
    if suite == "controlled":
        return read(ASSETS / "controlled.json")
    rows = read(ASSETS / ("frontier.json" if suite == "frontier" else "main.json"))
    if suite == "diagnostic":
        rows = [dict(r, budget=50 if r["family"] != "HPO" else 25,
                     total=r["initial"] + (50 if r["family"] != "HPO" else 25))
                for r in rows if r["task"] in DIAGNOSTIC_TASKS]
    return rows


class PaperTask(Task):
    """Only the host owns raw identities, evaluators, initializations and journals."""

    def __init__(self, name, *, suite="frontier", seed=2, journal=None,
                 dbtune_url=None, placement_url=None, information="full"):
        choices = [r for r in task_rows(suite) if r["task"] == name and r["seed"] == seed]
        if len(choices) != 1:
            raise ValueError(f"No frozen {suite} case for {name}, seed {seed}")
        self.row = choices[0]
        self.suite = suite
        self.journal = None if journal is None else Path(journal)
        self.dbtune_url, self.placement_url = dbtune_url, placement_url
        self.raw = None
        self.codec = None
        self.anonymous = self.row["family"] == "BBOB" or information == "anonymous"
        self.instructions = None
        if suite == "frontier":
            folder = ASSETS / "frontier" / self.row["id"]
            description_path = folder / "task.md"
            parameters = read(folder / "space.json")["parameters"]
            objective = read(folder / "objective.json")
            history = read(folder / "shared_initial.json")
            self.documents = dict(protocol_version=2, short_task=(folder / "task.md").read_text(),
                sections=read(folder / "task_details.json"), parameters=read(folder / "parameter_catalog.json"))
            self.instructions = (folder / "instructions.md").read_text()
        else:
            description_path = ASSETS / "main" / f"{name}__s{seed}.json"
            shared = read(description_path)
            parameters, objective, history = shared["parameter_definitions"], shared["objective"], shared["initial_history"]
            self.documents = dict(protocol_version=2, short_task=shared["short_task"],
                                  sections=shared["task_facts"], parameters=parameters)
            if suite == "diagnostic":
                old = shared["row"]
                old_text = f"{old['budget']} new evaluations ({old['total']} total)"
                new_text = f"{self.row['budget']} new evaluations ({self.row['total']} total)"
                self.documents["short_task"] = self.documents["short_task"].replace(old_text, new_text)
                self.documents["sections"] = {k: v.replace(old_text, new_text)
                                               for k, v in self.documents["sections"].items()}
        self._spec = TaskSpec(name=self.row["visible_task_id"], search_space=schema_space(parameters),
            objectives=(ObjectiveSpec(objective["name"], ObjectiveDirection(objective["direction"])),),
            max_evaluations=self.row["total"], metadata={},
            description_ref=TaskDescriptionRef(task_id=self.row["visible_task_id"], primary_path=description_path))
        if suite == "diagnostic":
            self._spec.metadata['optimizer_tool_policy'] = dict(minimal_suggest=True, gp_acquisition='ei')
        self._raw_objective = self.spec.primary_objective
        if information == "semantic":
            self.documents = copy.deepcopy(self.documents)
            old_index = "get_task_context sections: " + ", ".join(self.documents["sections"]) + "."
            self.documents["sections"] = {key: value for key, value in self.documents["sections"].items()
                if key not in {"mechanisms", "domain_knowledge"}}
            new_index = "get_task_context sections: " + ", ".join(self.documents["sections"]) + "."
            if self.documents["short_task"].count(old_index) != 1:
                raise ValueError("Expected one task-context section index for the I1 intervention")
            self.documents["short_task"] = self.documents["short_task"].replace(old_index, new_index)
        if information == "anonymous" and self.row["family"] != "BBOB":
            from bbo.algorithms.agentic.anonymous_space import AnonymousUnitCodec
            from bbo.algorithms.agentic.tools.context_io import context_documents
            transforms = {p["name"]: p["transform"] for p in parameters if p.get("transform") in {"linear", "log", "logit"}}
            self.codec = AnonymousUnitCodec(self.spec.search_space, transforms=transforms, salt=name)
            self._spec = replace(self.spec, name="task_" + digest(dict(task=name))[:12],
                search_space=self.codec.space, objectives=(ObjectiveSpec("value", self._raw_objective.direction),))
            text = (f"# Anonymous optimization task\n\nOptimize an unknown scalar objective over {len(parameters)} numerical inputs in [0,1].\n\n"
                    f"Objective: {self.spec.primary_objective.direction.value} `value`.\n\n"
                    f"Budget: {self.row['initial']} shared initial observations and {self.row['budget']} new evaluations ({self.row['total']} total).\n\n"
                    "## How to submit a candidate\nSubmit one complete finite configuration within [0,1] using the declared anonymous coordinate names.\n")
            self.documents = context_documents(self.spec, text, None)
        self.prefix = []
        if len(history) != self.row["initial"]:
            raise ValueError("Frozen initialization count mismatch")
        for i, item in enumerate(history):
            if item["trial_id"] != i or item["status"] != "success":
                raise ValueError("Invalid initialization sequence")
            config = self.codec.encode(item["config"]) if self.codec else item["config"]
            self.spec.search_space.validate_config(config)
            self.prefix.append(TrialObservation.from_evaluation(TrialSuggestion(config=config, trial_id=i),
                EvaluationResult(objectives={self.spec.primary_objective.name: item["objectives"][objective["name"]]})))

    @property
    def spec(self):
        return self._spec

    def get_description(self):
        return TaskDescriptionBundle(task_id=self.spec.name,
            primary=TaskDescriptionDoc(path=self.spec.description_ref.primary_path,
                content=self.documents["short_task"], kind="background", title="Task context"),
            rendered_context=self.documents["short_task"], fingerprint=digest(self.documents))

    def _evaluator(self):
        if self.raw is None:
            from bbo.tasks import create_task
            kwargs = {}
            if self.row["family"] == "DBTune" and self.dbtune_url:
                kwargs["base_url"] = self.dbtune_url
            if self.row["family"] == "BBOPlace" and self.placement_url:
                kwargs["base_url"] = self.placement_url
            self.raw = create_task(self.row["task"], seed=self.row["seed"],
                                   max_evaluations=self.row["total"], **kwargs)
        return self.raw

    def _journal_call(self, kind, index, request, invoke):
        if self.journal is None:
            raise RuntimeError("Dry-run task must not evaluate an objective")
        req = self.journal / kind / f"request_{index}.json"
        resp = self.journal / kind / f"response_{index}.json"
        if req.exists():
            if read(req) != request or not resp.exists():
                raise RuntimeError("Unresolved or changed evaluator request; refusing duplicate evaluation")
            return read(resp)
        save(req, request)
        result = invoke()
        value = None if result is None else asdict(result)
        save(resp, value)
        return value

    def evaluate(self, suggestion):
        if suggestion.trial_id is None or not self.row["initial"] <= suggestion.trial_id < self.row["total"]:
            raise ValueError("Objective call outside the new-evaluation budget")
        config = self.codec.decode(suggestion.config) if self.codec else suggestion.config
        raw = self._evaluator()
        result = self._journal_call("evaluations", suggestion.trial_id,
            dict(trial_id=suggestion.trial_id, config=config),
            lambda: raw.evaluate(replace(suggestion, config=config)))
        if result["status"] != "success":
            raise RuntimeError("Host evaluator failed; inspect the preserved journal")
        key = raw.spec.primary_objective.name
        value = result["objectives"][key]
        if self.row["family"] == "HPO" and self._raw_objective.name == "loss" and key == "accuracy":
            value = 1 - value
        return EvaluationResult(objectives={self.spec.primary_objective.name: value},
                                elapsed_seconds=result.get("elapsed_seconds"))

    def evaluate_final(self, suggestion):
        if self.row["family"] != "HPO":
            return None
        config = self.codec.decode(suggestion.config) if self.codec else suggestion.config
        result = self._journal_call("holdout", "incumbent", dict(config=config),
            lambda: self._evaluator().evaluate_final(replace(suggestion, config=config)))
        if result is None:
            return None
        result["status"] = TrialStatus(result["status"])
        return EvaluationResult(**result)

    def cleanup(self):
        if self.raw is not None:
            self.raw.cleanup()
