# Environment

Run the published BBOPlace-Bench container and expose the HTTP API (default port **8080**):

```bash
sudo docker run --rm -p 8080:8080 bboplace-bench
```

The benchmark task defaults to `http://127.0.0.1:8080`. Override the base URL when needed:

```bash
export BBOPLACE_BASE_URL=http://127.0.0.1:8080
uv run python -m bbo.run --algorithm random_search --task bboplace_bench --max-evaluations 3
```

Ensure the container is reachable before starting long runs.
