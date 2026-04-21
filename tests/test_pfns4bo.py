from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pfns4bo")

from bbo.algorithms import ALGORITHM_REGISTRY, Pfns4BoAlgorithm
from bbo.algorithms.model_based import pfns4bo as pfns_module
from bbo.algorithms.model_based.pfns4bo_encoding import (
    MOLECULE_DESCRIPTOR_NAMES,
    build_oer_candidate_pool,
    compute_molecule_descriptor_dataset,
    encode_oer_config,
    oer_feature_names,
)
from bbo.algorithms.model_based.pfns4bo_utils import PfnsModelInfo
from bbo.run import build_arg_parser, run_single_experiment
from bbo.tasks import create_molecule_qed_task, create_oer_task, create_task
from bbo.tasks.scientific import CACHE_ROOT_ENV, SOURCE_ROOT_ENV, VENDORED_SOURCE_ROOT
from bbo.tasks.scientific.oer import OER_CATEGORICAL_FEATURES, OER_NUMERICAL_FEATURES


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


@pytest.fixture
def stub_pfns_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PfnsModelInfo:
    model_path = tmp_path / "fake_pfns_model.pt"
    model_path.write_bytes(b"stub")
    model_info = PfnsModelInfo(
        model_name="hebo_plus",
        attribute_name="hebo_plus_model",
        model_path=model_path,
        existed_before=True,
        exists_after=True,
        auto_download_attempted=False,
    )
    monkeypatch.setattr(pfns_module, "resolve_pfns_model", lambda model_name: model_info)
    monkeypatch.setattr(pfns_module, "select_pfns_device", lambda requested: "cpu:0")
    monkeypatch.setattr(pfns_module, "load_torch_model", lambda model_path: object())
    return model_info


def test_pfns4bo_is_registered_and_cli_visible_without_entering_suite() -> None:
    parser = build_arg_parser()
    algorithm_action = next(action for action in parser._actions if action.dest == "algorithm")

    assert "pfns4bo" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["pfns4bo"].family == "model_based"
    assert ALGORITHM_REGISTRY["pfns4bo"].numeric_only is False
    assert "pfns4bo" in algorithm_action.choices
    assert parser.parse_args(["--algorithm", "pfns4bo"]).algorithm == "pfns4bo"


def test_pfns4bo_routes_numeric_and_pool_tasks(
    scientific_env: Path,
    stub_pfns_model: PfnsModelInfo,
) -> None:
    numeric_task = create_task("branin_demo", max_evaluations=3, seed=7)
    pool_task = create_oer_task(max_evaluations=3, seed=7, source_root=scientific_env)

    numeric_algorithm = Pfns4BoAlgorithm()
    numeric_algorithm.setup(numeric_task.spec, seed=7)
    assert numeric_algorithm.backend_name == "continuous"
    assert numeric_algorithm.candidate_pool is None

    pool_algorithm = Pfns4BoAlgorithm(pool_size=32)
    pool_algorithm.setup(pool_task.spec, seed=7)
    assert pool_algorithm.backend_name == "pool"
    assert pool_algorithm.candidate_pool is not None
    assert len(pool_algorithm.candidate_pool.configs) == 32


def test_oer_encoding_matches_fixed_column_order(scientific_env: Path) -> None:
    task = create_oer_task(max_evaluations=3, seed=11, source_root=scientific_env)
    feature_names = oer_feature_names(task.spec.search_space)
    encoded = encode_oer_config(task.spec.search_space.defaults(), task.spec.search_space)

    expected_names: list[str] = []
    for column in OER_CATEGORICAL_FEATURES:
        choices = tuple(task.spec.search_space[column].choices)
        expected_names.extend(f"{column}::{choice}" for choice in choices)
    expected_names.extend(OER_NUMERICAL_FEATURES)

    assert feature_names == tuple(expected_names)
    assert encoded.shape == (len(feature_names),)

    cursor = 0
    for column in OER_CATEGORICAL_FEATURES:
        width = len(task.spec.search_space[column].choices)
        block = encoded[cursor: cursor + width]
        assert block.sum() == pytest.approx(1.0)
        assert set(block.tolist()) <= {0.0, 1.0}
        cursor += width
    numeric_block = encoded[cursor:]
    assert np.all(numeric_block >= 0.0)
    assert np.all(numeric_block <= 1.0)


def test_molecule_descriptor_dataset_uses_fixed_normalized_descriptors(scientific_env: Path) -> None:
    pytest.importorskip("rdkit")
    task = create_molecule_qed_task(max_evaluations=3, seed=5, source_root=scientific_env)
    smiles_choices = tuple(task.spec.search_space["SMILES"].choices[:256])
    raw, normalized = compute_molecule_descriptor_dataset(smiles_choices)

    assert raw.shape[1] == len(MOLECULE_DESCRIPTOR_NAMES)
    assert normalized.shape == raw.shape
    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= 1.0)
    assert np.allclose(normalized.min(axis=0), 0.0)
    assert np.allclose(normalized.max(axis=0), 1.0)


def test_oer_pool_generation_is_reproducible(scientific_env: Path) -> None:
    task = create_oer_task(max_evaluations=3, seed=9, source_root=scientific_env)
    pool_a = build_oer_candidate_pool(task.spec.search_space, seed=17, pool_size=16)
    pool_b = build_oer_candidate_pool(task.spec.search_space, seed=17, pool_size=16)
    pool_c = build_oer_candidate_pool(task.spec.search_space, seed=18, pool_size=16)

    assert pool_a.configs == pool_b.configs
    assert np.allclose(pool_a.features, pool_b.features)
    assert pool_a.configs != pool_c.configs


def test_pfns4bo_branin_resume_preserves_append_only_history(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PFNs4BO continuous smoke is only exercised on CUDA-enabled test hosts.")

    summary = run_single_experiment(
        task_name="branin_demo",
        algorithm_name="pfns4bo",
        seed=7,
        max_evaluations=4,
        results_root=tmp_path,
        resume=False,
        pfns_device="cuda:0",
    )

    resumed = run_single_experiment(
        task_name="branin_demo",
        algorithm_name="pfns4bo",
        seed=7,
        max_evaluations=6,
        results_root=tmp_path,
        resume=True,
        pfns_device="cuda:0",
    )

    assert summary["trial_count"] == 4
    assert resumed["trial_count"] == 6
    assert resumed["best_primary_objective"] is not None
    assert Path(resumed["results_jsonl"]).exists()
