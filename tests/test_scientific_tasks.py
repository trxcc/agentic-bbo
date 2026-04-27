from __future__ import annotations

import os
from pathlib import Path

import pytest

from bbo.core import ObjectiveDirection, TrialSuggestion
from bbo.run import run_single_experiment
from bbo.tasks import (
    ALL_TASK_NAMES,
    BH_TASK_NAME,
    GUACAMOL_QED_TASK_NAME,
    HEA_TASK_NAME,
    HER_FEATURES,
    HER_TASK_NAME,
    MOLECULE_TASK_NAME,
    OER_TASK_NAME,
    QED_SELFIES_TASK_NAME,
    create_bh_task,
    create_guacamol_qed_task,
    create_hea_task,
    create_her_task,
    create_molecule_qed_task,
    create_oer_task,
    create_qed_selfies_task,
)
from bbo.tasks.scientific import CACHE_ROOT_ENV, SOURCE_ROOT_ENV, VENDORED_SOURCE_ROOT


def _require_bo_tutorial_source() -> Path:
    source_root = Path(os.environ.get(SOURCE_ROOT_ENV, str(VENDORED_SOURCE_ROOT)))
    if not source_root.exists():
        pytest.skip("Bundled scientific task datasets are not available in the workspace.")
    return source_root


@pytest.fixture
def scientific_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    source_root = _require_bo_tutorial_source()
    monkeypatch.setenv(SOURCE_ROOT_ENV, str(source_root))
    monkeypatch.setenv(CACHE_ROOT_ENV, str(tmp_path / "dataset_cache"))
    return source_root


def test_scientific_registry_contains_all_tasks() -> None:
    assert HER_TASK_NAME in ALL_TASK_NAMES
    assert HEA_TASK_NAME in ALL_TASK_NAMES
    assert OER_TASK_NAME in ALL_TASK_NAMES
    assert BH_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_QED_TASK_NAME in ALL_TASK_NAMES
    assert MOLECULE_TASK_NAME in ALL_TASK_NAMES
    assert QED_SELFIES_TASK_NAME in ALL_TASK_NAMES


def test_her_task_spec_and_sanity(scientific_env: Path) -> None:
    task = create_her_task(max_evaluations=3, seed=19, source_root=scientific_env)
    report = task.sanity_check()

    assert report.ok
    assert task.spec.name == HER_TASK_NAME
    assert task.spec.primary_objective.name == "regret"
    assert task.spec.primary_objective.direction == ObjectiveDirection.MINIMIZE
    assert task.spec.search_space.names() == list(HER_FEATURES)
    assert report.metadata["row_count"] == 812
    assert report.metadata["column_count"] == 11

    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults()))
    assert result.success
    assert result.objectives["regret"] >= 0.0
    assert "predicted_target" in result.metrics


def test_hea_task_spec_and_transform(scientific_env: Path) -> None:
    pytest.importorskip("openpyxl")
    task = create_hea_task(max_evaluations=3, seed=13, source_root=scientific_env)
    report = task.sanity_check()

    assert report.ok
    assert task.spec.name == HEA_TASK_NAME
    assert task.spec.primary_objective.name == "regret"
    assert report.metadata["transform_residual_max"] < 1e-9

    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults()))
    assert result.success
    assert result.objectives["regret"] >= 0.0
    assert all(key.startswith("composition::") for key in result.metrics if key.startswith("composition::"))


def test_oer_task_spec_and_sanity(scientific_env: Path) -> None:
    task = create_oer_task(max_evaluations=3, seed=11, source_root=scientific_env)
    report = task.sanity_check()

    assert report.ok
    assert task.spec.name == OER_TASK_NAME
    assert task.spec.primary_objective.name == "overpotential_mv"
    assert "Metal_1" in report.metadata["categorical_choices"]

    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults()))
    assert result.success
    assert result.objectives["overpotential_mv"] > 0.0


def test_bh_task_feature_selection_and_sanity(scientific_env: Path) -> None:
    task = create_bh_task(max_evaluations=3, seed=7, source_root=scientific_env)
    report = task.sanity_check()

    assert report.ok
    assert task.spec.name == BH_TASK_NAME
    assert task.spec.primary_objective.name == "regret"
    assert report.metadata["selected_features"]

    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults()))
    assert result.success
    assert result.objectives["regret"] >= 0.0
    assert "predicted_yield" in result.metrics


def test_molecule_qed_task_sanity(scientific_env: Path) -> None:
    pytest.importorskip("rdkit")
    task = create_molecule_qed_task(max_evaluations=3, seed=5, source_root=scientific_env)
    report = task.sanity_check()

    assert report.ok
    assert task.spec.name == MOLECULE_TASK_NAME
    assert task.spec.primary_objective.name == "qed_loss"
    assert report.metadata["item_count"] > 0

    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults()))
    assert result.success
    assert 0.0 <= result.objectives["qed_loss"] <= 1.0
    assert 0.0 <= result.metrics["qed"] <= 1.0


def test_qed_selfies_task_sanity(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("selfies")
    source_root = _require_bo_tutorial_source()
    task = create_qed_selfies_task(
        max_evaluations=3,
        seed=5,
        source_root=source_root,
        cache_root=tmp_path / "dataset_cache",
        max_selfies_tokens=8,
        vocabulary_source_limit=64,
    )
    report = task.sanity_check()

    assert report.ok
    assert task.spec.name == QED_SELFIES_TASK_NAME
    assert task.spec.primary_objective.name == "qed_loss"
    assert report.metadata["selfies_vocabulary_size"] > 0
    assert task.spec.search_space.names()[0] == "selfies_token_00"

    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults()))
    assert result.success
    assert result.metadata["valid_smiles"]
    assert 0.0 <= result.objectives["qed_loss"] <= 1.0
    assert 0.0 <= result.metrics["qed"] <= 1.0

    ethanol = task.evaluate(TrialSuggestion(config=task.config_from_smiles("CCO")))
    assert ethanol.success
    assert ethanol.metadata["valid_smiles"]
    assert ethanol.metadata["smiles"]


def test_guacamol_qed_task_sanity() -> None:
    pytest.importorskip("rdkit")
    task = create_guacamol_qed_task(max_evaluations=3, seed=23)
    report = task.sanity_check()

    assert report.ok
    assert task.spec.name == GUACAMOL_QED_TASK_NAME
    assert task.spec.primary_objective.name == "guacamol_qed_loss"
    assert report.metadata["candidate_pool_size"] > 0
    assert report.metadata["valid_candidate_count"] > 0

    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults()))
    assert result.success
    assert 0.0 <= result.objectives["guacamol_qed_loss"] <= 1.0
    assert 0.0 <= result.metrics["guacamol_qed_score"] <= 1.0


@pytest.mark.parametrize(
    "task_name",
    [HER_TASK_NAME, HEA_TASK_NAME, OER_TASK_NAME, BH_TASK_NAME],
)
def test_scientific_random_search_smoke(
    task_name: str,
    scientific_env: Path,
    tmp_path: Path,
) -> None:
    summary = run_single_experiment(
        task_name=task_name,
        algorithm_name="random_search",
        seed=5,
        max_evaluations=3,
        results_root=tmp_path,
        resume=False,
    )

    assert summary["trial_count"] == 3
    assert summary["best_primary_objective"] is not None
    assert Path(summary["results_jsonl"]).exists()
    assert len(summary["plot_paths"]) == 4
    for plot_path in summary["plot_paths"]:
        path = Path(plot_path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_molecule_random_search_smoke(scientific_env: Path, tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    summary = run_single_experiment(
        task_name=MOLECULE_TASK_NAME,
        algorithm_name="random_search",
        seed=5,
        max_evaluations=3,
        results_root=tmp_path,
        resume=False,
    )

    assert summary["trial_count"] == 3
    assert summary["best_primary_objective"] is not None
    assert Path(summary["results_jsonl"]).exists()
    assert len(summary["plot_paths"]) == 4
    for plot_path in summary["plot_paths"]:
        path = Path(plot_path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_qed_selfies_optuna_smoke(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    pytest.importorskip("rdkit")
    pytest.importorskip("selfies")
    source_root = _require_bo_tutorial_source()
    summary = run_single_experiment(
        task_name=QED_SELFIES_TASK_NAME,
        algorithm_name="optuna_tpe",
        seed=5,
        max_evaluations=3,
        task_kwargs={
            "source_root": source_root,
            "cache_root": tmp_path / "dataset_cache",
            "max_selfies_tokens": 8,
            "vocabulary_source_limit": 64,
        },
        results_root=tmp_path,
        resume=False,
        generate_plots=False,
    )

    assert summary["trial_count"] == 3
    assert summary["best_primary_objective"] is not None
    assert Path(summary["results_jsonl"]).exists()


def test_qed_selfies_random_search_smoke(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("selfies")
    source_root = _require_bo_tutorial_source()
    summary = run_single_experiment(
        task_name=QED_SELFIES_TASK_NAME,
        algorithm_name="random_search",
        seed=5,
        max_evaluations=3,
        task_kwargs={
            "source_root": source_root,
            "cache_root": tmp_path / "dataset_cache",
            "max_selfies_tokens": 8,
            "vocabulary_source_limit": 64,
        },
        results_root=tmp_path,
        resume=False,
        generate_plots=False,
    )

    assert summary["trial_count"] == 3
    assert summary["best_primary_objective"] is not None
    assert Path(summary["results_jsonl"]).exists()


def test_guacamol_qed_random_search_smoke(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    summary = run_single_experiment(
        task_name=GUACAMOL_QED_TASK_NAME,
        algorithm_name="random_search",
        seed=5,
        max_evaluations=3,
        results_root=tmp_path,
        resume=False,
    )

    assert summary["trial_count"] == 3
    assert summary["best_primary_objective"] is not None
    assert Path(summary["results_jsonl"]).exists()
    assert len(summary["plot_paths"]) == 4
    for plot_path in summary["plot_paths"]:
        path = Path(plot_path)
        assert path.exists()
        assert path.stat().st_size > 0
