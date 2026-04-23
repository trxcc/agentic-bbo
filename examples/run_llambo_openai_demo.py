"""Run the LLAMBO demo with a real LLM backend (OpenAI-compatible API).

Setup:
    export OPENAI_API_KEY="sk-..."
    # Optional: override endpoint for compatible providers
    export OPENAI_BASE_URL="https://api.openai.com/v1"

Run:
    python examples/run_llambo_openai_demo.py

If your endpoint does NOT support json_schema structured outputs, add:
    --llambo-openai-no-use-structured-outputs
"""

from __future__ import annotations

import json
import os

from bbo.run import run_single_experiment


if __name__ == "__main__":
    # Verify API key is present so the failure is early and obvious.
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Please set OPENAI_API_KEY in your environment before running this demo.\n"
            "Example: export OPENAI_API_KEY='sk-...'"
        )

    summary = run_single_experiment(
        task_name="branin_demo",
        algorithm_name="llambo",
        seed=7,
        max_evaluations=12,
        llambo_backend="openai",
        llambo_model="gpt-4o-mini",
        llambo_initial_samples=4,
        llambo_candidates=8,
        llambo_templates=2,
        llambo_predictions=6,
        llambo_alpha=-0.1,
        # The following are read from environment variables by default:
        #   OPENAI_API_KEY
        #   OPENAI_BASE_URL (optional)
        #   OPENAI_ORGANIZATION (optional)
        #   OPENAI_PROJECT (optional)
        # You can also pass them explicitly via CLI args when using `python -m bbo.run`.
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
