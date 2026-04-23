"""Run the OPRO demo with a real LLM backend (OpenAI-compatible API).

Setup:
    export OPENAI_API_KEY="sk-..."
    # Optional: override endpoint for compatible providers
    export OPENAI_BASE_URL="https://api.openai.com/v1"

Run:
    python examples/run_opro_openai_demo.py
"""

from __future__ import annotations

import json
import os

from bbo.run import run_single_experiment


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Please set OPENAI_API_KEY in your environment before running this demo.\n"
            "Example: export OPENAI_API_KEY='sk-...'"
        )

    summary = run_single_experiment(
        task_name="branin_demo",
        algorithm_name="opro",
        seed=7,
        max_evaluations=12,
        opro_backend="openai",
        opro_model="gpt-4o-mini",
        opro_initial_samples=4,
        opro_candidates=8,
        opro_prompt_pairs=10,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
