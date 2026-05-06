# Build from ``bbo/tasks/dbtune`` (context = this package root):
#   cd bbo/tasks/dbtune && docker build -f docker_surrogate/Dockerfile -t agentbbo-surrogate-http-py37:v1 .
# Includes ``assets/`` (joblib + knobs json); the server reads them from /app/assets.
FROM python:3.7-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8090 \
    SURROGATE_ASSETS_DIR=/app/assets

WORKDIR /app

COPY docker_surrogate/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY docker_surrogate/server.py /app/server.py
COPY knob_space.py /app/knob_space.py
COPY assets /app/assets

EXPOSE 8090

CMD ["python", "/app/server.py"]
