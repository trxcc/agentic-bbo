"""Scientific benchmark task families."""

from .bh import BH_TASK_NAME, BhTask, BhTaskConfig, create_bh_task
from .data_assets import CACHE_ROOT_ENV, SOURCE_ROOT_ENV, DEFAULT_CACHE_ROOT, VENDORED_SOURCE_ROOT
from .guacamol import (
    GUACAMOL_QED_TASK_NAME,
    GuacamolQEDTask,
    GuacamolQEDTaskConfig,
    create_guacamol_qed_task,
)
from .hea import HEA_COMPONENTS, HEA_DESIGN_FEATURES, HEA_TASK_NAME, HeaTask, HeaTaskConfig, create_hea_task
from .her import HER_FEATURES, HER_TASK_NAME, HerTask, HerTaskConfig, create_her_task
from .molecule import MOLECULE_TASK_NAME, MoleculeQEDTask, MoleculeTaskConfig, create_molecule_qed_task
from .oer import OER_TASK_NAME, OerTask, OerTaskConfig, create_oer_task
from .qed_selfies import QED_SELFIES_TASK_NAME, QedSelfiesTask, QedSelfiesTaskConfig, create_qed_selfies_task
from .registry import SCIENTIFIC_TASK_REGISTRY, create_scientific_task

__all__ = [
    "BH_TASK_NAME",
    "CACHE_ROOT_ENV",
    "DEFAULT_CACHE_ROOT",
    "VENDORED_SOURCE_ROOT",
    "GUACAMOL_QED_TASK_NAME",
    "GuacamolQEDTask",
    "GuacamolQEDTaskConfig",
    "HEA_COMPONENTS",
    "HEA_DESIGN_FEATURES",
    "HEA_TASK_NAME",
    "HER_FEATURES",
    "HER_TASK_NAME",
    "MOLECULE_TASK_NAME",
    "MoleculeQEDTask",
    "MoleculeTaskConfig",
    "OER_TASK_NAME",
    "OerTask",
    "OerTaskConfig",
    "QED_SELFIES_TASK_NAME",
    "QedSelfiesTask",
    "QedSelfiesTaskConfig",
    "SCIENTIFIC_TASK_REGISTRY",
    "SOURCE_ROOT_ENV",
    "BhTask",
    "BhTaskConfig",
    "HeaTask",
    "HeaTaskConfig",
    "HerTask",
    "HerTaskConfig",
    "create_bh_task",
    "create_guacamol_qed_task",
    "create_hea_task",
    "create_her_task",
    "create_molecule_qed_task",
    "create_oer_task",
    "create_qed_selfies_task",
    "create_scientific_task",
]
