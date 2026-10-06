# Surrogate evaluator service (Python 3.7)

Offline knob surrogates are sklearn models stored as `.joblib`. **Training and unpickling** are pinned to a Python 3.7 + compatible `numpy` / `scikit-learn` stack; the main AgentBBO code targets Python 3.11+. This image isolates that stack and exposes a small JSON API similar in spirit to `bbo/tasks/dbtune/docker_mariadb/` (MariaDB + sysbench), but for **in-memory prediction only**.

## Build

From **`bbo/tasks/dbtune`** (repository root, then `cd bbo/tasks/dbtune`):

```bash
docker build -f docker_surrogate/Dockerfile -t agentbbo-surrogate-http-py37:v1 .
```

Download the required `.joblib` files from the **Google Drive** link in `../assets/README.md`, place them under `bbo/tasks/dbtune/assets/`, then build; or **mount** that folder at run time (see below) so the container sees the same files.

**Unpickling / `scikit-learn` version:** the image pins `scikit-learn==0.21.3` in `docker/requirements.txt` because many old RF models reference `sklearn.ensemble.forest`, which is incompatible with scikit-learn 0.22+ in the way joblib was serialized. If `joblib.load` still fails, align `scikit-learn` and `numpy` in `requirements.txt` to the same versions as the environment where the model was **trained** (`pip show scikit-learn`), then rebuild the image (no cache: `docker build --no-cache`).

**`pandas`:** some checkouts include a `No module named 'pandas'` error during `joblib.load` (indirect import). The image includes `pandas` in `requirements.txt` for that case; if another missing module appears, add it the same way and rebuild.

## Run

```bash
docker rm -f agentbbo_surrogate_http 2>/dev/null
docker run -d --name agentbbo_surrogate_http -p 8090:8090 \
  -e AGENTIC_BBO_SYSBENCH5_SURROGATE=/app/assets/RF_SYSBENCH_5knob.joblib \
  agentbbo-surrogate-http-py37:v1
```

Default port **8090** (distinct from the MariaDB evaluator on **8080**). Override with `-e PORT=...`.

**Bind-mount assets** (no rebuild) example:

```bash
docker run -d --name agentbbo_surrogate_http -p 8090:8090 \
  -v /path/to/your/assets:/app/assets:ro \
  agentbbo-surrogate-http-py37:v1
```

## API (matches `HttpSurrogateKnobTask`)

- `GET /health` returns `{"status":"ok"}`.
- `GET /task/<canonical_task_id>` (e.g. `knob_surrogate_sysbench_5`) returns feature names, objective name, optimization direction, and the input contract.
- `POST /evaluate` accepts `{"task_id": "<canonical_id>", "x": [u1,...,ud]}`. Each coordinate is normalized to `[0,1]`, matching the BBO search space. The container decodes it with `assets/knobs_*.json` and runs prediction.
- A successful evaluation returns `{"status":"success", "y": <float>, <objective_name>: <float>}`.

For legacy clients and debugging, a request containing `features` instead of `x`
passes already decoded physical features directly to the model. Prefer `x` for
benchmark runs. Canonical IDs use `knob_surrogate_*`; the host client maps its
`knob_http_surrogate_*` IDs to these server IDs.

## Host-side (Python 3.11) tasks

Run BBO with e.g. the legacy task id `--task knob_http_surrogate_sysbench_5` and set:

- `AGENTBBO_HTTP_SURROGATE_BASE_URL` (default `http://127.0.0.1:8090`)
- `AGENTBBO_HTTP_SURROGATE_TIMEOUT_SEC` (default `120`)

The host decodes normalized knobs using local `bbo/tasks/dbtune/assets/knobs_*.json` (must stay consistent with what you used offline).

## Keeping server metadata in sync

`docker/server.py` lists joblib file names and env overrides. When you change `bbo/tasks/dbtune/catalog.py`, update the `TASK_DEFS` block in `server.py` or add a test that compares them.
