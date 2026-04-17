# Constraints

- Parameter proposals must stay inside declared bounds or enumerated choices.
- The evaluator must be deterministic or explicitly document its noise model.
- Runs should be resumable from append-only JSONL logs without hidden optimizer checkpoints.
- If the real task has safety-critical regions, spell them out here rather than assuming the optimizer will infer them.
