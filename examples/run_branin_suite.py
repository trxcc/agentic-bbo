"""Run the full Branin demo suite and write plots under `artifacts/demo_runs/`."""

from __future__ import annotations

import json

from bbo.run import run_demo_suite


if __name__ == "__main__":
    summary = run_demo_suite()
    print(json.dumps(summary, indent=2, sort_keys=True))
