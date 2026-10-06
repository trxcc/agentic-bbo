"""The released molecular benchmark inventory."""
from .guacamol_smiles import GUACAMOL_SMILES_TASK_NAMES, create_guacamol_smiles_task

SCIENTIFIC_TASK_REGISTRY = {name: "GuacaMol direct-SMILES objective" for name in GUACAMOL_SMILES_TASK_NAMES}


def create_scientific_task(name, **kwargs):
    if name not in SCIENTIFIC_TASK_REGISTRY:
        raise ValueError(f"Unknown molecular task {name!r}")
    return create_guacamol_smiles_task(name, **kwargs)
