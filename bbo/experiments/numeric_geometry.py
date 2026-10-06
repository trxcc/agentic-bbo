"""Respect declared warps while preserving native integer TPE/Random domains."""
import copy
from dataclasses import replace

from bbo.core import ExternalOptimizerAdapter, FloatParam, SearchSpace, UnitCubeSearchSpaceConverter
from bbo.algorithms.traditional.pycma import PyCmaAlgorithm


class UnitCubeCma(PyCmaAlgorithm):
    def __init__(self):
        super().__init__(sigma_fraction=0.18, popsize=None)

    def setup(self, task_spec, seed=0, **kwargs):
        super().setup(task_spec, seed=seed, **kwargs)
        self._continuous_converter = UnitCubeSearchSpaceConverter(
            task_spec.search_space, transforms=task_spec.metadata.get('parameter_transforms'))
        if any(p.low == p.high for p in task_spec.search_space):
            raise ValueError('Remove fixed dimensions before using unit-cube CMA')

    def ask(self):
        suggestion = super().ask()
        suggestion.metadata.update(numeric_revision='numeric_v2',
            parameter_transforms=dict(self._continuous_converter.transforms),
            cma_coordinate_system='transformed_unit_cube', cma_initial_sigma=0.18,
            cma_population_size=None if self._strategy is None else int(self._strategy.popsize))
        return suggestion


class WarpedNative(ExternalOptimizerAdapter):
    """Native Random/TPE log domains plus unit-scaled logit float dimensions.

    Linear and log integer parameters retain their discrete distributions. Unit
    scaling is affine in the warped coordinate, not a change to its geometry.
    """
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    @property
    def name(self):
        return self.backend.name

    def setup(self, task_spec, seed=0, **kwargs):
        self.bind_task_spec(task_spec)
        self.converter = UnitCubeSearchSpaceConverter(task_spec.search_space,
            transforms=task_spec.metadata.get('parameter_transforms'))
        self.names = task_spec.search_space.names()
        self.warped = set()
        params = []
        for p in task_spec.search_space:
            transform = self.converter.transforms[p.name]
            if transform == 'logit':
                if not isinstance(p, FloatParam):
                    raise ValueError('Logit requires a continuous parameter')
                self.warped.add(p.name)
                params.append(FloatParam(p.name, low=0., high=1., default=0.5))
            else:
                params.append(replace(p, log=transform == 'log'))
        metadata = copy.deepcopy(task_spec.metadata)
        metadata['parameter_transforms'] = {
            p.name: ('log' if p.log else 'linear') for p in params}
        initialization = metadata.get('benchmark_protocol', {}).get('initialization', {})
        self.prefix = copy.deepcopy(initialization.get('configurations', []))
        if self.prefix:
            initialization['configurations'] = [self.encode(c) for c in self.prefix]
        self.backend.setup(replace(task_spec, search_space=SearchSpace(params), metadata=metadata), seed=seed)
        self._asked = 0

    def encode(self, config):
        unit = self.converter.encode_vector(config)
        return {k: float(unit[i]) if k in self.warped else config[k]
                for i, k in enumerate(self.names)}

    def decode(self, config):
        unit = [config[k] if k in self.warped else 0.5 for k in self.names]
        decoded = self.converter.decode_vector(unit)
        return {k: decoded[k] if k in self.warped else config[k] for k in self.names}

    def ask(self):
        suggestion = self.backend.ask()
        # Return the exact shared physical prefix, avoiding a roundtrip alteration.
        config = (copy.deepcopy(self.prefix[self._asked]) if self._asked < len(self.prefix)
                  else self.decode(suggestion.config))
        self._asked += 1
        return replace(suggestion, config=config, metadata={**suggestion.metadata,
            'numeric_revision': 'numeric_v2', 'parameter_transforms': dict(self.converter.transforms),
            'native_internal_config': dict(suggestion.config)})

    def tell(self, observation):
        internal = observation.suggestion.metadata.get('native_internal_config')
        if internal is None:
            raise ValueError('Missing native internal suggestion; restore history with replay()')
        self.assert_matching_config(self.decode(internal), observation.suggestion.config)
        self.backend.tell(replace(observation, suggestion=replace(observation.suggestion, config=dict(internal))))
        self.update_best_incumbent(observation)
