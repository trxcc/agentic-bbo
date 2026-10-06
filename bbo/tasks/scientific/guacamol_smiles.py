"""GuacaMol goal-directed benchmarks over direct SMILES strings."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ...core import (
    EvaluationResult,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    StringParam,
    Task,
    TaskDescriptionRef,
    TaskSpec,
    TrialStatus,
    TrialSuggestion,
)
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TASK_DESCRIPTION_ROOT = PACKAGE_ROOT / "task_descriptions"
GUACAMOL_SMILES_DEFAULT_MAX_EVALUATIONS = 40
GUACAMOL_SMILES_DEFAULT_MAX_LENGTH = 512
GUACAMOL_SMILES_SCHEMA_DEFAULT = ""
GUACAMOL_SMILES_SOURCE = "guacamol.goal_directed_suite"
GUACAMOL_SOURCE_REPO_URL = "https://github.com/BenevolentAI/guacamol"

GUACAMOL_FEXOFENADINE_MPO_SMILES_TASK_NAME = "guacamol_fexofenadine_mpo_smiles_demo"
GUACAMOL_MEDIAN1_SMILES_TASK_NAME = "guacamol_median1_smiles_demo"
GUACAMOL_AMLODIPINE_MPO_SMILES_TASK_NAME = "guacamol_amlodipine_mpo_smiles_demo"
GUACAMOL_MEDIAN2_SMILES_TASK_NAME = "guacamol_median2_smiles_demo"
GUACAMOL_OSIMERTINIB_MPO_SMILES_TASK_NAME = "guacamol_osimertinib_mpo_smiles_demo"
GUACAMOL_PERINDOPRIL_MPO_SMILES_TASK_NAME = "guacamol_perindopril_mpo_smiles_demo"
GUACAMOL_RANOLAZINE_MPO_SMILES_TASK_NAME = "guacamol_ranolazine_mpo_smiles_demo"
GUACAMOL_SITAGLIPTIN_MPO_SMILES_TASK_NAME = "guacamol_sitagliptin_mpo_smiles_demo"
GUACAMOL_VALSARTAN_SMARTS_SMILES_TASK_NAME = "guacamol_valsartan_smarts_smiles_demo"
GUACAMOL_ZALEPLON_MPO_SMILES_TASK_NAME = "guacamol_zaleplon_mpo_smiles_demo"

FEXOFENADINE_SMILES = "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4"
CAMPHOR_SMILES = "CC1(C)C2CCC1(C)C(=O)C2"
MENTHOL_SMILES = "CC(C)C1CCC(C)CC1O"
AMLODIPINE_SMILES = r"Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC"
OSIMERTINIB_SMILES = "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34"
PERINDOPRIL_SMILES = "O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC"
RANOLAZINE_SMILES = "COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2"
SITAGLIPTIN_SMILES = "Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F"
SITAGLIPTIN_FORMULA = "C16H15F6N5O"
VALSARTAN_SMARTS = "CN(C=O)Cc1ccc(c2ccccc2)cc1"
VALSARTAN_PROPERTY_TARGET_SMILES = "NC(CC(=O)N1CCn2c(nnc2C(F)(F)F)C1)Cc1cc(F)c(F)cc1F"
ZALEPLON_SMILES = "O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1"
ZALEPLON_FORMULA = "C19H17N3O2"
TADALAFIL_SMILES = "O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C"
SILDENAFIL_SMILES = "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)([0-9]*)")


def _parse_molecular_formula(formula: str) -> tuple[tuple[str, int], ...]:
    parts: list[tuple[str, int]] = []
    position = 0
    for match in _FORMULA_TOKEN_RE.finditer(formula):
        if match.start() != position:
            raise ValueError(f"Unsupported molecular formula syntax: {formula!r}")
        element, count_text = match.groups()
        parts.append((element, int(count_text or "1")))
        position = match.end()
    if position != len(formula) or not parts:
        raise ValueError(f"Unsupported molecular formula syntax: {formula!r}")
    return tuple(parts)


def _require_rdkit():
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem, Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "The GuacaMol SMILES tasks require RDKit. Install with "
            "`uv sync --extra dev --extra bo-tutorial`."
        ) from exc
    return Chem, DataStructs, AllChem, Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors


@dataclass(frozen=True)
class GuacamolSmilesBenchmarkDefinition:
    """Static definition for one GuacaMol-derived SMILES task."""

    task_name: str
    display_name: str
    objective_name: str
    metric_name: str
    source_benchmark: str
    target_smiles: tuple[str, ...] = ()
    fingerprint_types: tuple[str, ...] = ()
    description: str = ""
    category: str = "guacamol"

    @property
    def description_dir(self) -> Path:
        return TASK_DESCRIPTION_ROOT / self.task_name


@dataclass
class GuacamolSmilesTaskConfig:
    """Configuration for one direct-SMILES GuacaMol benchmark task."""

    task_name: str
    max_evaluations: int | None = None
    seed: int = 0
    description_dir: Path | None = None
    max_smiles_length: int = GUACAMOL_SMILES_DEFAULT_MAX_LENGTH
    metadata: dict[str, Any] = field(default_factory=dict)


class GuacamolSmilesTask(Task):
    """GuacaMol-derived objective over an open SMILES string parameter.

    Scoring formulas follow the local GuacaMol standard_benchmarks reference.
    The empty default SMILES is schema-only and must not be used as an optimizer
    initial population or seed.
    """

    def __init__(self, config: GuacamolSmilesTaskConfig):
        if config.task_name not in GUACAMOL_SMILES_TASK_DEFINITIONS:
            available = ", ".join(sorted(GUACAMOL_SMILES_TASK_DEFINITIONS))
            raise ValueError(f"Unknown GuacaMol SMILES task `{config.task_name}`. Available: {available}")
        if config.max_smiles_length <= 0:
            raise ValueError("max_smiles_length must be positive.")

        self.config = config
        self.definition = GUACAMOL_SMILES_TASK_DEFINITIONS[config.task_name]

        Chem, DataStructs, AllChem, Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors = _require_rdkit()
        self._chem = Chem
        self._data_structs = DataStructs
        self._all_chem = AllChem
        self._crippen = Crippen
        self._descriptors = Descriptors
        self._rd_mol_descriptors = rdMolDescriptors
        self._atom_pair_generator = rdFingerprintGenerator.GetAtomPairGenerator(maxDistance=10)
        self._target_fingerprints = self._build_target_fingerprints()

        search_space = SearchSpace(
            [
                StringParam(
                    "smiles",
                    default=GUACAMOL_SMILES_SCHEMA_DEFAULT,
                    min_length=0,
                    max_length=int(config.max_smiles_length),
                )
            ]
        )
        description_dir = config.description_dir or self.definition.description_dir
        self._dataset_summary = {
            "smiles_max_length": int(config.max_smiles_length),
            "schema_default_smiles": GUACAMOL_SMILES_SCHEMA_DEFAULT,
            "schema_default_role": "schema_only_not_initial_population",
            "target_smiles": list(self.definition.target_smiles),
            "fingerprint_types": list(self.definition.fingerprint_types),
        }
        self._spec = TaskSpec(
            name=self.definition.task_name,
            search_space=search_space,
            objectives=(ObjectiveSpec(self.definition.objective_name, ObjectiveDirection.MINIMIZE),),
            max_evaluations=config.max_evaluations or GUACAMOL_SMILES_DEFAULT_MAX_EVALUATIONS,
            description_ref=TaskDescriptionRef.from_directory(self.definition.task_name, description_dir),
            metadata={
                "display_name": self.definition.display_name,
                "source_repo": GUACAMOL_SOURCE_REPO_URL,
                "source_benchmark": self.definition.source_benchmark,
                "source_suite": GUACAMOL_SMILES_SOURCE,
                "representation": "smiles_string",
                "dimension": 1,
                "category": self.definition.category,
                "schema_default_role": "schema_only_not_initial_population",
                **config.metadata,
            },
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    @property
    def dataset_summary(self) -> dict[str, Any]:
        return dict(self._dataset_summary)

    def config_from_smiles(self, smiles: str) -> dict[str, str]:
        """Return the canonical task config for one SMILES string."""

        return self.spec.search_space.coerce_config({"smiles": str(smiles)}, use_defaults=False)

    def _build_target_fingerprints(self) -> dict[tuple[str, str], Any]:
        fingerprints: dict[tuple[str, str], Any] = {}
        for smiles in self.definition.target_smiles:
            mol = self._chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid GuacaMol target SMILES: {smiles!r}")
            for fp_type in self.definition.fingerprint_types:
                fingerprints[(smiles, fp_type)] = self._fingerprint(mol, fp_type)
        return fingerprints

    def _fingerprint(self, mol: Any, fp_type: str) -> Any:
        if fp_type == "ECFP4":
            return self._all_chem.GetMorganFingerprint(mol, 2)
        if fp_type == "ECFP6":
            return self._all_chem.GetMorganFingerprint(mol, 3)
        if fp_type == "FCFP4":
            return self._all_chem.GetMorganFingerprint(mol, 2, useFeatures=True)
        if fp_type == "AP":
            return self._atom_pair_generator.GetSparseCountFingerprint(mol)
        raise ValueError(f"Unsupported fingerprint type for GuacaMol SMILES task: {fp_type}")

    def _tanimoto(self, mol: Any, target_smiles: str, fp_type: str) -> float:
        return float(
            self._data_structs.TanimotoSimilarity(
                self._fingerprint(mol, fp_type),
                self._target_fingerprints[(target_smiles, fp_type)],
            )
        )

    @staticmethod
    def _clipped_score(x: float, upper_x: float, lower_x: float = 0.0) -> float:
        y = (x - lower_x) / (upper_x - lower_x)
        return float(min(max(y, 0.0), 1.0))

    @staticmethod
    def _gaussian(x: float, mu: float, sigma: float) -> float:
        return float(math.exp(-0.5 * ((x - mu) / sigma) ** 2))

    @classmethod
    def _min_gaussian(cls, x: float, mu: float, sigma: float) -> float:
        return cls._gaussian(max(x, mu), mu, sigma)

    @classmethod
    def _max_gaussian(cls, x: float, mu: float, sigma: float) -> float:
        return cls._gaussian(min(x, mu), mu, sigma)

    @staticmethod
    def _geometric_mean(values: Iterable[float]) -> float:
        vals = [max(float(value), 0.0) for value in values]
        if not vals:
            return 0.0
        if any(value == 0.0 for value in vals):
            return 0.0
        return float(math.prod(vals) ** (1.0 / len(vals)))

    def _atom_count(self, mol: Any, element: str) -> int:
        counted_mol = self._chem.AddHs(mol) if element == "H" else mol
        return sum(1 for atom in counted_mol.GetAtoms() if atom.GetSymbol() == element)

    def _formula_score(self, mol: Any, formula: str) -> float:
        formula_parts = _parse_molecular_formula(formula)
        total_atoms = sum(count for _, count in formula_parts)
        scores = [
            self._gaussian(float(self._atom_count(mol, element)), float(count), 1.0)
            for element, count in formula_parts
        ]
        scores.append(self._gaussian(float(self._chem.AddHs(mol).GetNumAtoms()), float(total_atoms), 2.0))
        return self._geometric_mean(scores)

    def _smarts_score(self, mol: Any, smarts: str) -> float:
        target = self._chem.MolFromSmarts(smarts)
        if target is None:
            raise ValueError(f"Invalid SMARTS pattern in GuacaMol SMILES task: {smarts!r}")
        return 1.0 if mol.HasSubstructMatch(target) else 0.0

    def _logp(self, mol: Any) -> float:
        return float(self._crippen.MolLogP(mol))

    def _tpsa(self, mol: Any) -> float:
        return float(self._rd_mol_descriptors.CalcTPSA(mol))

    def _bertz(self, mol: Any) -> float:
        return float(self._descriptors.BertzCT(mol))

    def _score_mol(self, mol: Any) -> float:
        name = self.definition.task_name
        if name == GUACAMOL_FEXOFENADINE_MPO_SMILES_TASK_NAME:
            similarity = self._clipped_score(self._tanimoto(mol, FEXOFENADINE_SMILES, "AP"), 0.8)
            tpsa_over_90 = self._max_gaussian(float(self._rd_mol_descriptors.CalcTPSA(mol)), 90.0, 10.0)
            logp_under_4 = self._min_gaussian(float(self._crippen.MolLogP(mol)), 4.0, 1.0)
            return self._geometric_mean((similarity, tpsa_over_90, logp_under_4))
        if name == GUACAMOL_MEDIAN1_SMILES_TASK_NAME:
            return self._geometric_mean(
                (
                    self._tanimoto(mol, CAMPHOR_SMILES, "ECFP4"),
                    self._tanimoto(mol, MENTHOL_SMILES, "ECFP4"),
                )
            )
        if name == GUACAMOL_AMLODIPINE_MPO_SMILES_TASK_NAME:
            amlodipine = self._tanimoto(mol, AMLODIPINE_SMILES, "ECFP4")
            rings = self._gaussian(float(self._rd_mol_descriptors.CalcNumRings(mol)), 3.0, 0.5)
            return self._geometric_mean((amlodipine, rings))
        if name == GUACAMOL_MEDIAN2_SMILES_TASK_NAME:
            return self._geometric_mean(
                (
                    self._tanimoto(mol, TADALAFIL_SMILES, "ECFP6"),
                    self._tanimoto(mol, SILDENAFIL_SMILES, "ECFP6"),
                )
            )
        if name == GUACAMOL_OSIMERTINIB_MPO_SMILES_TASK_NAME:
            similar_to_osimertinib = self._clipped_score(self._tanimoto(mol, OSIMERTINIB_SMILES, "FCFP4"), 0.8)
            but_not_too_similar = self._min_gaussian(self._tanimoto(mol, OSIMERTINIB_SMILES, "ECFP6"), 0.85, 0.1)
            tpsa_over_100 = self._max_gaussian(self._tpsa(mol), 100.0, 10.0)
            logp_scoring = self._min_gaussian(self._logp(mol), 1.0, 1.0)
            return self._geometric_mean(
                (similar_to_osimertinib, but_not_too_similar, tpsa_over_100, logp_scoring)
            )
        if name == GUACAMOL_PERINDOPRIL_MPO_SMILES_TASK_NAME:
            perindopril = self._tanimoto(mol, PERINDOPRIL_SMILES, "ECFP4")
            aromatic_rings = self._gaussian(float(self._rd_mol_descriptors.CalcNumAromaticRings(mol)), 2.0, 0.5)
            return self._geometric_mean((perindopril, aromatic_rings))
        if name == GUACAMOL_RANOLAZINE_MPO_SMILES_TASK_NAME:
            similar_to_ranolazine = self._clipped_score(self._tanimoto(mol, RANOLAZINE_SMILES, "AP"), 0.7)
            logp_under_4 = self._max_gaussian(self._logp(mol), 7.0, 1.0)
            fluorine = self._gaussian(float(self._atom_count(mol, "F")), 1.0, 1.0)
            tpsa_f = self._max_gaussian(self._tpsa(mol), 95.0, 20.0)
            return self._geometric_mean((similar_to_ranolazine, logp_under_4, fluorine, tpsa_f))
        if name == GUACAMOL_SITAGLIPTIN_MPO_SMILES_TASK_NAME:
            sitagliptin = self._chem.MolFromSmiles(SITAGLIPTIN_SMILES)
            if sitagliptin is None:
                raise ValueError(f"Invalid GuacaMol target SMILES: {SITAGLIPTIN_SMILES!r}")
            target_logp = self._logp(sitagliptin)
            target_tpsa = self._tpsa(sitagliptin)
            similarity = self._gaussian(self._tanimoto(mol, SITAGLIPTIN_SMILES, "ECFP4"), 0.0, 0.1)
            logp_score = self._gaussian(self._logp(mol), target_logp, 0.2)
            tpsa_score = self._gaussian(self._tpsa(mol), target_tpsa, 5.0)
            isomers = self._formula_score(mol, SITAGLIPTIN_FORMULA)
            return self._geometric_mean((similarity, logp_score, tpsa_score, isomers))
        if name == GUACAMOL_VALSARTAN_SMARTS_SMILES_TASK_NAME:
            property_target = self._chem.MolFromSmiles(VALSARTAN_PROPERTY_TARGET_SMILES)
            if property_target is None:
                raise ValueError(f"Invalid GuacaMol target SMILES: {VALSARTAN_PROPERTY_TARGET_SMILES!r}")
            smarts_score = self._smarts_score(mol, VALSARTAN_SMARTS)
            logp_score = self._gaussian(self._logp(mol), self._logp(property_target), 0.2)
            tpsa_score = self._gaussian(self._tpsa(mol), self._tpsa(property_target), 5.0)
            bertz_score = self._gaussian(self._bertz(mol), self._bertz(property_target), 30.0)
            return self._geometric_mean((smarts_score, logp_score, tpsa_score, bertz_score))
        if name == GUACAMOL_ZALEPLON_MPO_SMILES_TASK_NAME:
            zaleplon = self._tanimoto(mol, ZALEPLON_SMILES, "ECFP4")
            formula = self._formula_score(mol, ZALEPLON_FORMULA)
            return self._geometric_mean((zaleplon, formula))
        raise RuntimeError(f"No scoring implementation for task `{name}`.")

    def _score_smiles(self, smiles: str) -> tuple[float, bool, str]:
        if not str(smiles).strip():
            return 0.0, False, ""
        mol = self._chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0, False, ""
        try:
            canonical_smiles = self._chem.MolToSmiles(mol, canonical=True)
        except Exception:
            canonical_smiles = str(smiles)
        score = self._score_mol(mol)
        if not math.isfinite(score):
            return 0.0, False, str(canonical_smiles)
        return float(min(max(score, 0.0), 1.0)), True, str(canonical_smiles)

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        start = time.perf_counter()
        config = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        smiles = str(config["smiles"])
        score, valid, canonical_smiles = self._score_smiles(smiles)
        loss = 1.0 - score
        elapsed = time.perf_counter() - start
        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={self.definition.objective_name: loss},
            metrics={
                self.definition.metric_name: score,
                "guacamol_score": score,
            },
            elapsed_seconds=elapsed,
            metadata={
                "smiles": smiles,
                "canonical_smiles": canonical_smiles,
                "valid_smiles": valid,
                "source_benchmark": self.definition.source_benchmark,
                "target_smiles": list(self.definition.target_smiles),
            },
        )

    def sanity_check(self):
        report = super().sanity_check()
        try:
            default_result = self.evaluate(TrialSuggestion(config=self.spec.search_space.defaults()))
            loss = float(default_result.objectives[self.definition.objective_name])
            score = float(default_result.metrics[self.definition.metric_name])
            if not math.isfinite(loss):
                report.add_error("non_finite_objective", "The GuacaMol SMILES task produced a non-finite loss.")
            if not (0.0 <= score <= 1.0):
                report.add_error("invalid_score_range", "GuacaMol score must be in [0.0, 1.0].")
            if default_result.metadata.get("valid_smiles"):
                report.add_error("schema_default_is_molecule", "The schema-only default must not be a valid molecule.")
        except Exception as exc:  # pragma: no cover - defensive guard.
            report.add_error("guacamol_smiles_failed", f"The task could not score the schema default: {exc}")
        report.metadata.update(self._dataset_summary)
        return report


def _definition(
    task_name: str,
    display_name: str,
    objective_stem: str,
    source_benchmark: str,
    *,
    target_smiles: tuple[str, ...] = (),
    fingerprint_types: tuple[str, ...] = (),
    category: str,
    description: str,
) -> GuacamolSmilesBenchmarkDefinition:
    return GuacamolSmilesBenchmarkDefinition(
        task_name=task_name,
        display_name=display_name,
        objective_name=f"{objective_stem}_loss",
        metric_name=f"{objective_stem}_score",
        source_benchmark=source_benchmark,
        target_smiles=target_smiles,
        fingerprint_types=fingerprint_types,
        category=category,
        description=description,
    )


GUACAMOL_SMILES_TASK_DEFINITIONS: dict[str, GuacamolSmilesBenchmarkDefinition] = {
    GUACAMOL_FEXOFENADINE_MPO_SMILES_TASK_NAME: _definition(
        GUACAMOL_FEXOFENADINE_MPO_SMILES_TASK_NAME,
        "GuacaMol Fexofenadine MPO SMILES",
        "fexofenadine_mpo",
        "guacamol.standard_benchmarks.hard_fexofenadine",
        target_smiles=(FEXOFENADINE_SMILES,),
        fingerprint_types=("AP",),
        category="mpo",
        description="Optimize Fexofenadine similarity together with TPSA and logP modifiers.",
    ),
    GUACAMOL_MEDIAN1_SMILES_TASK_NAME: _definition(
        GUACAMOL_MEDIAN1_SMILES_TASK_NAME,
        "GuacaMol Median Molecules 1 SMILES",
        "median1",
        "guacamol.standard_benchmarks.median_camphor_menthol",
        target_smiles=(CAMPHOR_SMILES, MENTHOL_SMILES),
        fingerprint_types=("ECFP4",),
        category="median",
        description="Generate molecules between camphor and menthol by geometric mean ECFP4 similarity.",
    ),
    GUACAMOL_AMLODIPINE_MPO_SMILES_TASK_NAME: _definition(
        GUACAMOL_AMLODIPINE_MPO_SMILES_TASK_NAME,
        "GuacaMol Amlodipine MPO SMILES",
        "amlodipine_mpo",
        "guacamol.standard_benchmarks.amlodipine_rings",
        target_smiles=(AMLODIPINE_SMILES,),
        fingerprint_types=("ECFP4",),
        category="mpo",
        description="Optimize Amlodipine similarity together with a three-ring count target.",
    ),
    GUACAMOL_MEDIAN2_SMILES_TASK_NAME: _definition(
        GUACAMOL_MEDIAN2_SMILES_TASK_NAME,
        "GuacaMol Median Molecules 2 SMILES",
        "median2",
        "guacamol.standard_benchmarks.median_tadalafil_sildenafil",
        target_smiles=(TADALAFIL_SMILES, SILDENAFIL_SMILES),
        fingerprint_types=("ECFP6",),
        category="median",
        description="Generate molecules between tadalafil and sildenafil by geometric mean ECFP6 similarity.",
    ),
    GUACAMOL_OSIMERTINIB_MPO_SMILES_TASK_NAME: _definition(
        GUACAMOL_OSIMERTINIB_MPO_SMILES_TASK_NAME,
        "GuacaMol Osimertinib MPO SMILES",
        "osimertinib_mpo",
        "guacamol.standard_benchmarks.hard_osimertinib",
        target_smiles=(OSIMERTINIB_SMILES,),
        fingerprint_types=("FCFP4", "ECFP6"),
        category="mpo",
        description="Optimize Osimertinib similarity together with dissimilarity, TPSA, and logP modifiers.",
    ),
    GUACAMOL_PERINDOPRIL_MPO_SMILES_TASK_NAME: _definition(
        GUACAMOL_PERINDOPRIL_MPO_SMILES_TASK_NAME,
        "GuacaMol Perindopril MPO SMILES",
        "perindopril_mpo",
        "guacamol.standard_benchmarks.perindopril_rings",
        target_smiles=(PERINDOPRIL_SMILES,),
        fingerprint_types=("ECFP4",),
        category="mpo",
        description="Optimize Perindopril similarity together with a two-aromatic-ring count target.",
    ),
    GUACAMOL_RANOLAZINE_MPO_SMILES_TASK_NAME: _definition(
        GUACAMOL_RANOLAZINE_MPO_SMILES_TASK_NAME,
        "GuacaMol Ranolazine MPO SMILES",
        "ranolazine_mpo",
        "guacamol.standard_benchmarks.ranolazine_mpo",
        target_smiles=(RANOLAZINE_SMILES,),
        fingerprint_types=("AP",),
        category="mpo",
        description="Optimize Ranolazine similarity together with logP, fluorine count, and TPSA modifiers.",
    ),
    GUACAMOL_SITAGLIPTIN_MPO_SMILES_TASK_NAME: _definition(
        GUACAMOL_SITAGLIPTIN_MPO_SMILES_TASK_NAME,
        "GuacaMol Sitagliptin MPO SMILES",
        "sitagliptin_mpo",
        "guacamol.standard_benchmarks.sitagliptin_replacement",
        target_smiles=(SITAGLIPTIN_SMILES,),
        fingerprint_types=("ECFP4",),
        category="mpo",
        description="Optimize for Sitagliptin-like properties and formula while discouraging ECFP4 similarity.",
    ),
    GUACAMOL_VALSARTAN_SMARTS_SMILES_TASK_NAME: _definition(
        GUACAMOL_VALSARTAN_SMARTS_SMILES_TASK_NAME,
        "GuacaMol Valsartan SMARTS SMILES",
        "valsartan_smarts",
        "guacamol.standard_benchmarks.valsartan_smarts",
        target_smiles=(VALSARTAN_PROPERTY_TARGET_SMILES,),
        fingerprint_types=(),
        category="smarts",
        description="Match the Valsartan SMARTS pattern together with Sitagliptin-derived property targets.",
    ),
    GUACAMOL_ZALEPLON_MPO_SMILES_TASK_NAME: _definition(
        GUACAMOL_ZALEPLON_MPO_SMILES_TASK_NAME,
        "GuacaMol Zaleplon MPO SMILES",
        "zaleplon_mpo",
        "guacamol.standard_benchmarks.zaleplon_with_other_formula",
        target_smiles=(ZALEPLON_SMILES,),
        fingerprint_types=("ECFP4",),
        category="mpo",
        description="Optimize Zaleplon similarity together with the C19H17N3O2 formula target.",
    ),
}


def create_guacamol_smiles_task(
    task_name: str,
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    description_dir: Path | None = None,
    max_smiles_length: int = GUACAMOL_SMILES_DEFAULT_MAX_LENGTH,
    metadata: dict[str, Any] | None = None,
) -> GuacamolSmilesTask:
    return GuacamolSmilesTask(
        GuacamolSmilesTaskConfig(
            task_name=task_name,
            max_evaluations=max_evaluations,
            seed=seed,
            description_dir=description_dir,
            max_smiles_length=max_smiles_length,
            metadata=dict(metadata or {}),
        )
    )


def _factory(task_name: str) -> Callable[..., GuacamolSmilesTask]:
    def create_task(**kwargs: Any) -> GuacamolSmilesTask:
        return create_guacamol_smiles_task(task_name, **kwargs)

    return create_task


create_guacamol_fexofenadine_mpo_smiles_task = _factory(GUACAMOL_FEXOFENADINE_MPO_SMILES_TASK_NAME)
create_guacamol_median1_smiles_task = _factory(GUACAMOL_MEDIAN1_SMILES_TASK_NAME)
create_guacamol_amlodipine_mpo_smiles_task = _factory(GUACAMOL_AMLODIPINE_MPO_SMILES_TASK_NAME)
create_guacamol_median2_smiles_task = _factory(GUACAMOL_MEDIAN2_SMILES_TASK_NAME)
create_guacamol_osimertinib_mpo_smiles_task = _factory(GUACAMOL_OSIMERTINIB_MPO_SMILES_TASK_NAME)
create_guacamol_perindopril_mpo_smiles_task = _factory(GUACAMOL_PERINDOPRIL_MPO_SMILES_TASK_NAME)
create_guacamol_ranolazine_mpo_smiles_task = _factory(GUACAMOL_RANOLAZINE_MPO_SMILES_TASK_NAME)
create_guacamol_sitagliptin_mpo_smiles_task = _factory(GUACAMOL_SITAGLIPTIN_MPO_SMILES_TASK_NAME)
create_guacamol_valsartan_smarts_smiles_task = _factory(GUACAMOL_VALSARTAN_SMARTS_SMILES_TASK_NAME)
create_guacamol_zaleplon_mpo_smiles_task = _factory(GUACAMOL_ZALEPLON_MPO_SMILES_TASK_NAME)

GUACAMOL_SMILES_TASK_NAMES: tuple[str, ...] = tuple(GUACAMOL_SMILES_TASK_DEFINITIONS)

__all__ = [
    "GUACAMOL_AMLODIPINE_MPO_SMILES_TASK_NAME",
    "GUACAMOL_FEXOFENADINE_MPO_SMILES_TASK_NAME",
    "GUACAMOL_MEDIAN1_SMILES_TASK_NAME",
    "GUACAMOL_MEDIAN2_SMILES_TASK_NAME",
    "GUACAMOL_OSIMERTINIB_MPO_SMILES_TASK_NAME",
    "GUACAMOL_PERINDOPRIL_MPO_SMILES_TASK_NAME",
    "GUACAMOL_RANOLAZINE_MPO_SMILES_TASK_NAME",
    "GUACAMOL_SITAGLIPTIN_MPO_SMILES_TASK_NAME",
    "GUACAMOL_SMILES_DEFAULT_MAX_EVALUATIONS",
    "GUACAMOL_SMILES_DEFAULT_MAX_LENGTH",
    "GUACAMOL_SMILES_SCHEMA_DEFAULT",
    "GUACAMOL_SMILES_TASK_DEFINITIONS",
    "GUACAMOL_SMILES_TASK_NAMES",
    "GUACAMOL_VALSARTAN_SMARTS_SMILES_TASK_NAME",
    "GUACAMOL_ZALEPLON_MPO_SMILES_TASK_NAME",
    "GuacamolSmilesBenchmarkDefinition",
    "GuacamolSmilesTask",
    "GuacamolSmilesTaskConfig",
    "create_guacamol_amlodipine_mpo_smiles_task",
    "create_guacamol_fexofenadine_mpo_smiles_task",
    "create_guacamol_median1_smiles_task",
    "create_guacamol_median2_smiles_task",
    "create_guacamol_osimertinib_mpo_smiles_task",
    "create_guacamol_perindopril_mpo_smiles_task",
    "create_guacamol_ranolazine_mpo_smiles_task",
    "create_guacamol_sitagliptin_mpo_smiles_task",
    "create_guacamol_smiles_task",
    "create_guacamol_valsartan_smarts_smiles_task",
    "create_guacamol_zaleplon_mpo_smiles_task",
]
