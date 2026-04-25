"""Core abstractions for benchmark-oriented black-box optimization."""

from .algo import Algorithm, Incumbent
from .adapters import ExternalOptimizerAdapter, from_configspace, to_configspace
from .description import (
    DescriptionSectionSpec,
    MarkdownDescriptionLoader,
    STANDARD_TASK_DESCRIPTION_SCHEMA,
    TaskDescriptionBundle,
    TaskDescriptionDoc,
    TaskDescriptionRef,
    TaskDescriptionSchema,
    write_task_description_template,
)
from .experimenter import ExperimentConfig, Experimenter, RunSummary
from .logger import JsonlMetricLogger, MetricLogger, ResumeState
from .plotting import (
    CumulativeEvalTimeComparisonPlotter,
    CumulativeEvalTimePlotter,
    Landscape2DPlotter,
    ObjectiveDistributionPlotter,
    OptimizationTracePlotter,
    OptimizerComparisonPlotter,
    PerTrialEvalTimePlotter,
    PlotArtifact,
    RegretTracePlotter,
    ScalarBarPlotter,
    ScientificPlotter,
)
from .space import CategoricalParam, FloatParam, IntParam, ParameterSpec, SearchSpace
from .task import (
    ObjectiveDirection,
    ObjectiveSpec,
    SanityCheckIssue,
    SanityCheckReport,
    Task,
    TaskSpec,
)
from .trial import EvaluationResult, TrialObservation, TrialRecord, TrialStatus, TrialSuggestion

__all__ = [
    "Algorithm",
    "CategoricalParam",
    "DescriptionSectionSpec",
    "EvaluationResult",
    "ExperimentConfig",
    "Experimenter",
    "ExternalOptimizerAdapter",
    "FloatParam",
    "Incumbent",
    "IntParam",
    "CumulativeEvalTimeComparisonPlotter",
    "CumulativeEvalTimePlotter",
    "JsonlMetricLogger",
    "Landscape2DPlotter",
    "MarkdownDescriptionLoader",
    "MetricLogger",
    "ObjectiveDirection",
    "ObjectiveDistributionPlotter",
    "ObjectiveSpec",
    "OptimizationTracePlotter",
    "OptimizerComparisonPlotter",
    "ParameterSpec",
    "PerTrialEvalTimePlotter",
    "PlotArtifact",
    "RegretTracePlotter",
    "ScalarBarPlotter",
    "ResumeState",
    "RunSummary",
    "STANDARD_TASK_DESCRIPTION_SCHEMA",
    "SanityCheckIssue",
    "SanityCheckReport",
    "ScientificPlotter",
    "SearchSpace",
    "Task",
    "TaskDescriptionBundle",
    "TaskDescriptionDoc",
    "TaskDescriptionRef",
    "TaskDescriptionSchema",
    "TaskSpec",
    "TrialObservation",
    "TrialRecord",
    "TrialStatus",
    "TrialSuggestion",
    "from_configspace",
    "to_configspace",
    "write_task_description_template",
]
