FROM python:3.11-slim-bookworm
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash procps util-linux coreutils \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.17.1 scikit-learn==1.8.0 joblib==1.5.3 rdkit==2026.3.1
RUN python3 - <<'PY'
from pathlib import Path
import shutil
import site
for directory in site.getsitepackages():
    root = Path(directory)
    for name in ('sklearn/datasets', 'rdkit/Contrib', 'rdkit/Data', 'rdkit/Chem/Scaffolds/data'):
        shutil.rmtree(root / name, ignore_errors=True)
    for path in list(root.rglob('tests')):
        if path.is_dir():
            shutil.rmtree(path)
PY
COPY docker/agent_process_control/process-control.md /usr/local/share/agent-process-control.md
WORKDIR /workspace
