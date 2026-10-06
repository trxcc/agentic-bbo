# Environment

The task uses the official COCO Python experiment package and is installed with the project environment.

Run uv sync --extra dev --extra hpo from the project root.

The pinned evaluator dependency is coco-experiment 2.8.2, imported as cocoex. SciPy supplies the reproducible scrambled Sobol initialization. No external service or dataset is required.
