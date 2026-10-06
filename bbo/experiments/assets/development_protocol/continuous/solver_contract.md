# Generated optimizer interface

This contract is the proposed interface for the development experiment. The host
must supply and validate the curated solver_support implementation before a run.
The supplied version-0 strategy is not a standalone runnable program without it.

Public API: solver_support exports Algorithm, Incumbent, TrialSuggestion,
TrialObservation, TaskSpec, SearchSpace, FloatParam, IntParam, ObjectiveDirection,
GpEiAlgorithm and create_reference_gp. These are the frozen benchmark-agnostic
types and GP algorithm. create_reference_gp() returns the exact registered WF147
GP-EI defaults. Curated source for these public types and GP internals is readable;
task loaders, evaluator implementations, datasets and repository files are absent.

Your only submitted file is strategy.py, defining create_optimizer() -> Algorithm.
The host uses setup(task_spec, seed, task_description=visible_description), then
replay(shared_initial_observations), followed by ask() -> TrialSuggestion(config),
host evaluation and tell(TrialObservation). incumbents() returns list[Incumbent].
The host assigns monotonically increasing trial IDs and enforces the budget.
On resume, setup followed by replay(all_recorded_observations) must reproduce the
same next proposal. Handle MINIMIZE and MAXIMIZE according to the supplied task.
Full legal configurations must include every declared parameter, use finite values
and satisfy declared types and bounds. No hidden coordinate coercion is promised.

TaskSpec contains the visible alias, declared space and objective, total budget,
and public protocol information only. The visible description is the same revised
task context available to the online comparator. It contains no evaluator handle,
filesystem path to private data, optimum, function identity for anonymous tasks,
or result from another run. The program runs with fresh state per task and seed.

Allowed libraries: the frozen Python standard library and NumPy, SciPy,
scikit-learn, torch, BoTorch and GPyTorch. No network, subprocess escape, LLM,
dynamic package install, task-data loading, objective callback or self-modification.
Code can fit surrogates to supplied observations and change numerical search logic.
Only its configuration proposals and algorithm metadata return to the host.

Invalid configurations and program failures remain visible as failed evaluations
or failed runs according to the host protocol. The host does not replace them by
random search. A version with an incomplete development run is ineligible for
selection. Final evaluation never edits or repairs a frozen program.
