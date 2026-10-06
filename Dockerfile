FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /opt/benchmark
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY bbo ./bbo
RUN pip install --no-cache-dir '.[hpo,molecular]'
ENTRYPOINT ["python", "-m", "bbo.run"]
