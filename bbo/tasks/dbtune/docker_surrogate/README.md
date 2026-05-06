# Surrogate evaluator service (Python 3.7)

Offline knob surrogates are sklearn models stored as `.joblib`. **Training and unpickling** are pinned to a Python 3.7 + compatible `numpy` / `scikit-learn` stack; the main AgentBBO code targets Python 3.11+. This image isolates that stack and exposes a small JSON API similar in spirit to `bbo/tasks/dbtune/docker_mariadb/` (MariaDB + sysbench), but for **in-memory prediction only**.

## Reusable image

The compose stack and docs default to the reusable Docker Hub tag:

```bash
docker pull fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1
docker rm -f agentbbo_surrogate_http 2>/dev/null
docker run -d --name agentbbo_surrogate_http -p 8090:8090 \
  fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1
```

Default port **8090** (distinct from the MariaDB evaluator on **8080**). Override with `-e PORT=...`.

Override the tag used by `docker-compose.task-services.yml` with:

```bash
export AGENTBBO_DBTUNE_SURROGATE_IMAGE=<your-dockerhub-user>/agentbbo-dbtune-surrogate-http-py37:v1
```

The published image includes the full `.joblib` files under `/app/assets`. A bind mount is only needed when you intentionally want to replace those bundled assets.

## Build/export for publishing

From the repository root, after staging the full `.joblib` files in `bbo/tasks/dbtune/assets/`, build both dbtune evaluator images and export them as tar packages:

```bash
scripts/package_dbtune_images.sh --tag v1
```

The script writes two tarballs under `dist/docker-images/` and prints the `docker load` / `docker push` commands for the upload machine. It fails if any full surrogate checkpoint is missing unless `--allow-missing-surrogate-assets` is passed.

## Local build

From **`bbo/tasks/dbtune`** (repository root, then `cd bbo/tasks/dbtune`):

```bash
docker build -f docker_surrogate/Dockerfile -t agentbbo-surrogate-http-py37:v1 .
```

For a custom local build, stage the required `.joblib` files under `bbo/tasks/dbtune/assets/` before building, or mount an alternate assets folder at run time (see below).

**Unpickling / `scikit-learn` version:** the image pins `scikit-learn==0.21.3` in `docker_surrogate/requirements.txt` because many old RF models reference `sklearn.ensemble.forest`, which is incompatible with scikit-learn 0.22+ in the way joblib was serialized. If `joblib.load` still fails, align `scikit-learn` and `numpy` in `requirements.txt` to the same versions as the environment where the model was **trained** (`pip show scikit-learn`), then rebuild the image (no cache: `docker build --no-cache`).

**`pandas`:** some checkouts include a `No module named 'pandas'` error during `joblib.load` (indirect import). The image includes `pandas` in `requirements.txt` for that case; if another missing module appears, add it the same way and rebuild.

## Run a local build

```bash
docker rm -f agentbbo_surrogate_http 2>/dev/null
docker run -d --name agentbbo_surrogate_http -p 8090:8090 \
  agentbbo-surrogate-http-py37:v1
```

**Bind-mount assets** (no rebuild) example:

```bash
docker run -d --name agentbbo_surrogate_http -p 8090:8090 \
  -v /path/to/your/assets:/app/assets:ro \
  agentbbo-surrogate-http-py37:v1
```

## API（与主仓库 ``HttpSurrogateKnobTask`` 一致）

- `GET /health` → `{"status":"ok"}`
- `GET /task/<canonical_task_id>` (e.g. `knob_surrogate_sysbench_5`) → `feature_names`, `objective_name`, `maximize`, `input_contract` hint, …
- **`POST /evaluate`（主路径）** → `{"task_id": "<canonical_id>", "x": [u1,...,ud]}`，其中每个 `u_i` 为 **\[0,1\]** 上的归一化坐标（与 BBO 搜索空间一致）。容器内用 `assets/knobs_*.json` 解码为物理量，再 `predict`。
- **响应** → `{"status":"success", "y": <float>, <objective_name>: <float>}`

**兼容**：若请求体为 `{"task_id", "features": [...]}` 且**没有** `x`，则把 `features` 当作**已解码的物理**特征向量直送模型（旧行为 / 调试用）。主路径应使用 `x`。
`canonical_task_id` 是仓库里注册的 surrogate 名（`knob_surrogate_sysbench_5` 等），**不是**宿主机 BBO 的 `knob_http_surrogate_*`（后者由客户端映射到 canonical `task_id`）。

## Host-side (Python 3.11) tasks

Run BBO with e.g. the legacy task id `--task knob_http_surrogate_sysbench_5` and set:

- `AGENTBBO_HTTP_SURROGATE_BASE_URL` (default `http://127.0.0.1:8090`)
- `AGENTBBO_HTTP_SURROGATE_TIMEOUT_SEC` (default `120`)

The host asks the service for feature metadata and sends normalized vectors; `.joblib` loading and knob decoding happen inside the Python 3.7 service.

## Keeping server metadata in sync

`docker/server.py` lists joblib file names and env overrides. When you change `bbo/tasks/dbtune/catalog.py`, update the `TASK_DEFS` block in `server.py` or add a test that compares them.
