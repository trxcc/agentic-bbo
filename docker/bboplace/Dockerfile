FROM python:3.11-slim-bookworm

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md /workspace/
COPY bbo /workspace/bbo

RUN pip install --no-cache-dir .

EXPOSE 8070
CMD ["python", "-m", "bbo.tasks.bboplace.local_service", "--host", "0.0.0.0", "--port", "8070"]
