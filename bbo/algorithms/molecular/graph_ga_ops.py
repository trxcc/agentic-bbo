"""Reusable Graph GA molecular operators.

Adapted from local PMO sources:

- PMO mol_opt/main/gpbo/graph_ga/graph_ga.py
- PMO mol_opt/main/gpbo/graph_ga/crossover.py
- PMO mol_opt/main/gpbo/graph_ga/mutate.py
- PMO mol_opt/main/graph_ga/run.py

This module is an operator, not a task oracle. Callers provide a batch scoring
function; GPBO passes an acquisition function, while standalone Graph GA uses
only the offspring-generation pieces and receives true scores through ask/tell.
"""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

MINIMUM_SCORE_WEIGHT = 1e-10


def _require_rdkit():
    try:
        from rdkit import Chem, RDLogger, rdBase
        from rdkit.Chem import AllChem, rdMolDescriptors
        from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
    except ImportError as exc:  # pragma: no cover - depends on optional molecular deps.
        raise ImportError(
            "Graph GA molecular operators require RDKit. Install with "
            "`uv sync --extra bo-tutorial` or another RDKit-capable environment."
        ) from exc
    rdBase.DisableLog("rdApp.error")
    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    return Chem, AllChem, rdMolDescriptors, ConvertToNumpyArray


@contextlib.contextmanager
def _temporary_global_seed(seed: int | None):
    """Seed PMO-style global RNGs for one deterministic operator call."""

    if seed is None:
        yield
        return
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def load_smiles_file(path: str | Path) -> list[str]:
    """Load one SMILES per line, using the first whitespace-delimited token."""

    smiles: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        smiles.append(text.split()[0])
    return smiles


def canonicalize_smiles(smiles: str) -> str | None:
    Chem, _, _, _ = _require_rdkit()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        return str(Chem.MolToSmiles(mol))
    except ValueError:
        return None


def sanitize_smiles(population_smiles: Iterable[str]) -> list[str]:
    new_population: list[str] = []
    smile_set: set[str] = set()
    for smiles in population_smiles:
        canonical = canonicalize_smiles(str(smiles))
        if canonical is not None and canonical not in smile_set:
            new_population.append(canonical)
            smile_set.add(canonical)
    return new_population


def _cut(mol: Any):
    Chem, _, _, _ = _require_rdkit()
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[*]-;!@[*]")):
        return None

    bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts("[*]-;!@[*]")))
    bond_ids = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]
    fragments_mol = Chem.FragmentOnBonds(mol, bond_ids, addDummies=True, dummyLabels=[(1, 1)])
    try:
        return Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
    except ValueError:
        return None


def _cut_ring(mol: Any):
    Chem, _, _, _ = _require_rdkit()
    for _ in range(10):
        if random.random() < 0.5:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]@[R]@[R]@[R]")):
                return None
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R]@[R]@[R]")))
            bis = ((bis[0], bis[1]), (bis[2], bis[3]))
        else:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]@[R;!D2]@[R]")):
                return None
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R;!D2]@[R]")))
            bis = ((bis[0], bis[1]), (bis[1], bis[2]))

        bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bis]
        fragments_mol = Chem.FragmentOnBonds(
            mol,
            bond_ids,
            addDummies=True,
            dummyLabels=[(1, 1), (1, 1)],
        )
        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
            if len(fragments) == 2:
                return fragments
        except ValueError:
            return None
    return None


def ring_ok(mol: Any) -> bool:
    Chem, _, _, _ = _require_rdkit()
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]")):
        return True
    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts("[R]=[R]=[R]"))
    cycle_list = mol.GetRingInfo().AtomRings()
    max_cycle_length = max(len(cycle) for cycle in cycle_list)
    macro_cycle = max_cycle_length > 6
    double_bond_in_small_ring = mol.HasSubstructMatch(Chem.MolFromSmarts("[r3,r4]=[r3,r4]"))
    return not ring_allene and not macro_cycle and not double_bond_in_small_ring


def mol_ok(
    mol: Any,
    *,
    min_num_atoms: int = 5,
    mean_num_atoms: float = 40.0,
    std_num_atoms: float = 10.0,
) -> bool:
    Chem, _, _, _ = _require_rdkit()
    try:
        Chem.SanitizeMol(mol)
        target_size = std_num_atoms * np.random.randn() + mean_num_atoms
        return bool(mol.GetNumAtoms() > min_num_atoms and mol.GetNumAtoms() < target_size)
    except ValueError:
        return False


def _crossover_ring(parent_a: Any, parent_b: Any, **mol_ok_kwargs: Any):
    Chem, AllChem, _, _ = _require_rdkit()
    ring_smarts = Chem.MolFromSmarts("[R]")
    if not parent_a.HasSubstructMatch(ring_smarts) and not parent_b.HasSubstructMatch(ring_smarts):
        return None

    rxn_smarts1 = [
        "[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]",
        "[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]",
    ]
    rxn_smarts2 = [
        "([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]",
        "([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]",
    ]

    for _ in range(10):
        fragments_a = _cut_ring(parent_a)
        fragments_b = _cut_ring(parent_b)
        if fragments_a is None or fragments_b is None:
            return None

        new_mol_trial = []
        for smarts in rxn_smarts1:
            rxn1 = AllChem.ReactionFromSmarts(smarts)
            new_mol_trial = []
            for fa in fragments_a:
                for fb in fragments_b:
                    new_mol_trial.append(rxn1.RunReactants((fa, fb))[0])

        new_mols = []
        for smarts in rxn_smarts2:
            rxn2 = AllChem.ReactionFromSmarts(smarts)
            for mol_tuple in new_mol_trial:
                mol = mol_tuple[0]
                if mol_ok(mol, **mol_ok_kwargs):
                    new_mols += list(rxn2.RunReactants((mol,)))

        new_mols2 = []
        for mol_tuple in new_mols:
            mol = mol_tuple[0]
            if mol_ok(mol, **mol_ok_kwargs) and ring_ok(mol):
                new_mols2.append(mol)
        if new_mols2:
            return random.choice(new_mols2)
    return None


def _crossover_non_ring(parent_a: Any, parent_b: Any, **mol_ok_kwargs: Any):
    Chem, AllChem, _, _ = _require_rdkit()
    for _ in range(10):
        fragments_a = _cut(parent_a)
        fragments_b = _cut(parent_b)
        if fragments_a is None or fragments_b is None:
            return None
        rxn = AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]")
        new_mol_trial = []
        for fa in fragments_a:
            for fb in fragments_b:
                new_mol_trial.append(rxn.RunReactants((fa, fb))[0])

        new_mols = []
        for mol_tuple in new_mol_trial:
            mol = mol_tuple[0]
            if mol_ok(mol, **mol_ok_kwargs):
                new_mols.append(mol)
        if new_mols:
            return random.choice(new_mols)
    return None


def crossover(parent_a: Any, parent_b: Any, **mol_ok_kwargs: Any):
    Chem, _, _, _ = _require_rdkit()
    parent_smiles = [Chem.MolToSmiles(parent_a), Chem.MolToSmiles(parent_b)]
    try:
        Chem.Kekulize(parent_a, clearAromaticFlags=True)
        Chem.Kekulize(parent_b, clearAromaticFlags=True)
    except ValueError:
        pass

    for _ in range(10):
        if random.random() <= 0.5:
            new_mol = _crossover_non_ring(parent_a, parent_b, **mol_ok_kwargs)
        else:
            new_mol = _crossover_ring(parent_a, parent_b, **mol_ok_kwargs)
        if new_mol is not None:
            new_smiles = Chem.MolToSmiles(new_mol)
            if new_smiles is not None and new_smiles not in parent_smiles:
                return new_mol
    return None


def _delete_atom() -> str:
    choices = [
        "[*:1]~[D1:2]>>[*:1]",
        "[*:1]~[D2:2]~[*:3]>>[*:1]-[*:3]",
        "[*:1]~[D3:2](~[*;!H0:3])~[*:4]>>[*:1]-[*:3]-[*:4]",
        "[*:1]~[D4:2](~[*;!H0:3])(~[*;!H0:4])~[*:5]>>[*:1]-[*:3]-[*:4]-[*:5]",
        "[*:1]~[D4:2](~[*;!H0;!H1:3])(~[*:4])~[*:5]>>[*:1]-[*:3](-[*:4])-[*:5]",
    ]
    return str(np.random.choice(choices, p=[0.25, 0.25, 0.25, 0.1875, 0.0625]))


def _append_atom() -> str:
    choices = [
        ["single", ["C", "N", "O", "F", "S", "Cl", "Br"], 7 * [1.0 / 7.0]],
        ["double", ["C", "N", "O"], 3 * [1.0 / 3.0]],
        ["triple", ["C", "N"], 2 * [1.0 / 2.0]],
    ]
    index = int(np.random.choice(list(range(3)), p=[0.60, 0.35, 0.05]))
    bond_order, atom_list, probs = choices[index]
    new_atom = str(np.random.choice(atom_list, p=probs))
    if bond_order == "single":
        return "[*;!H0:1]>>[*:1]X".replace("X", "-" + new_atom)
    if bond_order == "double":
        return "[*;!H0;!H1:1]>>[*:1]X".replace("X", "=" + new_atom)
    return "[*;H3:1]>>[*:1]X".replace("X", "#" + new_atom)


def _insert_atom() -> str:
    choices = [
        ["single", ["C", "N", "O", "S"], 4 * [1.0 / 4.0]],
        ["double", ["C", "N"], 2 * [1.0 / 2.0]],
        ["triple", ["C"], [1.0]],
    ]
    index = int(np.random.choice(list(range(3)), p=[0.60, 0.35, 0.05]))
    bond_order, atom_list, probs = choices[index]
    new_atom = str(np.random.choice(atom_list, p=probs))
    if bond_order == "single":
        return "[*:1]~[*:2]>>[*:1]X[*:2]".replace("X", new_atom)
    if bond_order == "double":
        return "[*;!H0:1]~[*:2]>>[*:1]=X-[*:2]".replace("X", new_atom)
    return "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#X-[*:2]".replace("X", new_atom)


def _change_bond_order() -> str:
    choices = [
        "[*:1]!-[*:2]>>[*:1]-[*:2]",
        "[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]",
        "[*:1]#[*:2]>>[*:1]=[*:2]",
        "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]",
    ]
    return str(np.random.choice(choices, p=[0.45, 0.45, 0.05, 0.05]))


def _delete_cyclic_bond() -> str:
    return "[*:1]@[*:2]>>([*:1].[*:2])"


def _add_ring() -> str:
    choices = [
        "[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1",
        "[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1",
    ]
    return str(np.random.choice(choices, p=[0.05, 0.05, 0.45, 0.45]))


def _change_atom(mol: Any) -> str:
    Chem, _, _, _ = _require_rdkit()
    choices = ["#6", "#7", "#8", "#9", "#16", "#17", "#35"]
    probs = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]
    source = str(np.random.choice(choices, p=probs))
    while not mol.HasSubstructMatch(Chem.MolFromSmarts("[" + source + "]")):
        source = str(np.random.choice(choices, p=probs))
    target = str(np.random.choice(choices, p=probs))
    while target == source:
        target = str(np.random.choice(choices, p=probs))
    return "[X:1]>>[Y:1]".replace("X", source).replace("Y", target)


def mutate(mol: Any, mutation_rate: float, **mol_ok_kwargs: Any):
    if random.random() > mutation_rate:
        return mol

    Chem, AllChem, _, _ = _require_rdkit()
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except ValueError:
        return mol

    for _ in range(10):
        rxn_smarts_list = [
            _insert_atom(),
            _change_bond_order(),
            _delete_cyclic_bond(),
            _add_ring(),
            _delete_atom(),
            _change_atom(mol),
            _append_atom(),
        ]
        rxn_smarts = str(np.random.choice(rxn_smarts_list, p=[0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15]))
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
        new_mol_trial = rxn.RunReactants((mol,))

        new_mols = []
        for mol_tuple in new_mol_trial:
            candidate = mol_tuple[0]
            if mol_ok(candidate, **mol_ok_kwargs) and ring_ok(candidate):
                new_mols.append(candidate)
        if new_mols:
            return random.choice(new_mols)
    return None


def make_mating_pool(population: Sequence[Any], population_scores: Sequence[float], offspring_size: int) -> np.ndarray:
    scores = np.asarray(population_scores, dtype=float)
    if np.any(scores < 0.0):
        scores = scores - float(np.min(scores))
    scores = scores + MINIMUM_SCORE_WEIGHT
    population_probs = scores / float(np.sum(scores))
    return np.random.choice(list(population), p=population_probs, size=int(offspring_size), replace=True)


def reproduce(mating_pool: Sequence[str], mutation_rate: float, crossover_kwargs: dict[str, Any] | None = None):
    Chem, _, _, _ = _require_rdkit()
    kwargs = dict(crossover_kwargs or {})
    parent_a = Chem.MolFromSmiles(str(random.choice(mating_pool)))
    parent_b = Chem.MolFromSmiles(str(random.choice(mating_pool)))
    if parent_a is None or parent_b is None:
        return None
    child = crossover(parent_a, parent_b, **kwargs)
    if child is not None:
        child = mutate(child, mutation_rate, **kwargs)
    return child


def generate_offspring_smiles(
    *,
    population_smiles: Sequence[str],
    population_scores: Sequence[float],
    offspring_size: int,
    mutation_rate: float,
    mating_pool_size: int | None = None,
    min_func_val: float | None = 0.0,
    crossover_kwargs: dict[str, Any] | None = None,
    seed: int | None = None,
) -> list[str]:
    """Generate Graph GA offspring without evaluating them with an oracle."""

    Chem, _, _, _ = _require_rdkit()
    if len(population_smiles) != len(population_scores):
        raise ValueError("population_smiles and population_scores must have the same length.")
    if not population_smiles:
        raise ValueError("Graph GA requires a non-empty population.")

    with _temporary_global_seed(seed):
        scores = np.asarray(population_scores, dtype=float)
        bottom_score = float(np.min(scores))
        if min_func_val is not None:
            bottom_score = min(float(min_func_val), bottom_score)
        mating_pool = make_mating_pool(
            list(population_smiles),
            scores - bottom_score,
            int(mating_pool_size or len(population_smiles)),
        )
        offspring: list[str] = []
        seen: set[str] = set()
        for _ in range(int(offspring_size)):
            mol = reproduce(mating_pool, float(mutation_rate), crossover_kwargs=crossover_kwargs)
            if mol is None:
                continue
            try:
                smiles = Chem.MolToSmiles(mol)
            except ValueError:
                continue
            canonical = canonicalize_smiles(smiles)
            if canonical is not None and canonical not in seen:
                offspring.append(canonical)
                seen.add(canonical)
        return offspring


class CachedBatchScoringFunction:
    """PMO-style cached scoring wrapper adapted from gpbo/function_utils.py."""

    def __init__(
        self,
        function: Callable[[list[str]], list[float]],
        *,
        cache: dict[str, float] | None = None,
        transform: Callable[[float], float] | None = None,
    ) -> None:
        self._function = function
        self._cache = dict(cache or {})
        self.transform = transform

    @property
    def cache(self) -> dict[str, float]:
        return self._cache

    def __call__(self, inputs: str | Sequence[str], *, batch: bool = False) -> float | list[float]:
        if batch:
            input_list = [str(item) for item in inputs] if not isinstance(inputs, str) else [inputs]
        else:
            input_list = [str(inputs)]
        missing = [item for item in input_list if item not in self._cache]
        if missing:
            outputs = self._function(missing)
            for item, output in zip(missing, outputs, strict=True):
                self._cache[item] = float(output)
        values = [self._cache[item] for item in input_list]
        if self.transform is not None:
            values = [float(self.transform(value)) for value in values]
        if not batch:
            return values[0]
        return values


@dataclass(frozen=True)
class GraphGAOptimizationResult:
    queried_smiles: list[str]
    scores_by_smiles: dict[str, float]
    generation_info: list[dict[str, Any]] = field(default_factory=list)
    early_stop: bool = False
    reached_budget: bool = False


@dataclass
class GraphGACandidateOptimizer:
    """Graph GA maximizer reusable by molecular algorithms.

    The loop follows PMO
    PMO mol_opt/main/gpbo/graph_ga/graph_ga.py::run_ga_maximization.
    """

    max_generations: int
    population_size: int
    offspring_size: int
    mutation_rate: float
    patience: int | None = None
    max_func_calls: int | None = None
    min_func_val: float | None = 0.0
    crossover_kwargs: dict[str, Any] = field(default_factory=dict)

    def maximize(
        self,
        *,
        starting_population_smiles: Sequence[str],
        scoring_function: Callable[[list[str]], list[float]] | CachedBatchScoringFunction,
        seed: int | None = None,
    ) -> GraphGAOptimizationResult:
        Chem, _, _, _ = _require_rdkit()
        if self.population_size <= 0 or self.offspring_size <= 0 or self.max_generations <= 0:
            raise ValueError("Graph GA sizes and max_generations must be positive.")
        if self.mutation_rate < 0:
            raise ValueError("mutation_rate must be non-negative.")

        with _temporary_global_seed(seed):
            cached = (
                scoring_function
                if isinstance(scoring_function, CachedBatchScoringFunction)
                else CachedBatchScoringFunction(scoring_function)
            )
            start_cache_size = len(cached.cache)
            max_cache_size = None if self.max_func_calls is None else int(self.max_func_calls) + start_cache_size
            population_smiles = sanitize_smiles(starting_population_smiles)
            if not population_smiles:
                raise ValueError("Graph GA requires at least one valid starting SMILES.")

            population_scores = list(cached(population_smiles, batch=True))
            queried_smiles = list(population_smiles)
            generation_info: list[dict[str, Any]] = []
            early_stop = False
            reached_budget = False
            num_no_change_gen = 0

            for generation in range(int(self.max_generations)):
                bottom_score = float(np.min(population_scores))
                if self.min_func_val is not None:
                    bottom_score = min(float(self.min_func_val), bottom_score)
                mating_pool = make_mating_pool(
                    population_smiles,
                    np.asarray(population_scores, dtype=float) - bottom_score,
                    int(self.population_size),
                )
                offspring_smiles: list[str] = []
                for _ in range(int(self.offspring_size)):
                    mol = reproduce(mating_pool, float(self.mutation_rate), crossover_kwargs=self.crossover_kwargs)
                    if mol is None:
                        continue
                    try:
                        offspring_smiles.append(Chem.MolToSmiles(mol))
                    except ValueError:
                        pass

                population_and_offspring = list(set(population_smiles + sanitize_smiles(offspring_smiles)))
                old_scores = population_scores
                next_population: list[str] = []
                planned_cache = set(cached.cache)
                for smiles in population_and_offspring:
                    if max_cache_size is None or smiles in cached.cache or len(planned_cache) < max_cache_size:
                        next_population.append(smiles)
                        planned_cache.add(smiles)
                population_scores = list(cached(next_population, batch=True))
                queried_smiles += next_population

                order = np.argsort(-np.asarray(population_scores, dtype=float))[: int(self.population_size)]
                population_smiles = [next_population[int(index)] for index in order]
                population_scores = [population_scores[int(index)] for index in order]
                generation_info.append(
                    {
                        "generation": generation,
                        "max": float(np.max(population_scores)),
                        "avg": float(np.mean(population_scores)),
                        "median": float(np.median(population_scores)),
                        "min": float(np.min(population_scores)),
                        "std": float(np.std(population_scores)),
                        "size": int(len(population_scores)),
                        "num_func_eval": int(len(cached.cache) - start_cache_size),
                    }
                )

                if len(population_scores) == len(old_scores) and np.allclose(population_scores, old_scores):
                    num_no_change_gen += 1
                    if self.patience is not None and num_no_change_gen > int(self.patience):
                        early_stop = True
                        break
                else:
                    num_no_change_gen = 0

                if max_cache_size is not None and len(cached.cache) >= max_cache_size:
                    reached_budget = True
                    break

            unique_queried: list[str] = []
            seen: set[str] = set()
            for smiles in queried_smiles:
                if smiles not in seen:
                    unique_queried.append(smiles)
                    seen.add(smiles)

            return GraphGAOptimizationResult(
                queried_smiles=unique_queried,
                scores_by_smiles=dict(cached.cache),
                generation_info=generation_info,
                early_stop=early_stop,
                reached_budget=reached_budget,
            )


def morgan_fingerprint_array(smiles: str, *, radius: int = 2, n_bits: int = 4096) -> np.ndarray:
    """PMO Morgan bit-vector fingerprint from gpbo/fingerprints.py."""

    Chem, _, rdMolDescriptors, ConvertToNumpyArray = _require_rdkit()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Cannot fingerprint invalid SMILES: {smiles!r}")
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=int(radius), nBits=int(n_bits))
    fp_arr = np.zeros((1,), dtype=np.int8)
    ConvertToNumpyArray(fp, fp_arr)
    return fp_arr.flatten()


__all__ = [
    "CachedBatchScoringFunction",
    "GraphGACandidateOptimizer",
    "GraphGAOptimizationResult",
    "canonicalize_smiles",
    "generate_offspring_smiles",
    "load_smiles_file",
    "morgan_fingerprint_array",
    "sanitize_smiles",
]
