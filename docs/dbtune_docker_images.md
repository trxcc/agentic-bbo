# dbtune Docker Images

The dbtune HTTP tasks use two reusable evaluator images:

| Service | Default image | Host port |
|---------|---------------|-----------|
| MariaDB/sysbench evaluator | `fakerstrawberry/agentbbo-dbtune-mariadb-eval:v1` | `8080` |
| Python 3.7 surrogate evaluator | `fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1` | `8090` |

Start both, plus BBOPlace, from the repository root:

```bash
docker compose -f docker-compose.task-services.yml up -d
```

If the images live under a different Docker Hub namespace or tag:

```bash
export AGENTBBO_DBTUNE_MARIADB_IMAGE=<user>/agentbbo-dbtune-mariadb-eval:v1
export AGENTBBO_DBTUNE_SURROGATE_IMAGE=<user>/agentbbo-dbtune-surrogate-http-py37:v1
docker compose -f docker-compose.task-services.yml up -d
```

## Build tar packages for upload

The published `v1` images already bundle the surrogate checkpoints, so normal users do not need local `.joblib` files.
Only when rebuilding a new surrogate image tag should maintainers stage the full `.joblib` files listed in `bbo/tasks/dbtune/assets/README.md` under `bbo/tasks/dbtune/assets/`.
The packaging script checks for all six full checkpoints by default to prevent publishing an incomplete default image.

```bash
scripts/package_dbtune_images.sh --tag v1
```

Outputs:

```text
dist/docker-images/agentbbo-dbtune-mariadb-eval_v1.tar
dist/docker-images/agentbbo-dbtune-surrogate-http-py37_v1.tar
dist/docker-images/SHA256SUMS
```

On the machine that has stable Docker Hub access:

```bash
docker load -i dist/docker-images/agentbbo-dbtune-mariadb-eval_v1.tar
docker load -i dist/docker-images/agentbbo-dbtune-surrogate-http-py37_v1.tar
docker push fakerstrawberry/agentbbo-dbtune-mariadb-eval:v1
docker push fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1
```

For a smoke-test-only surrogate image without all full checkpoints:

```bash
scripts/package_dbtune_images.sh --tag smoke \
  --allow-missing-surrogate-assets
```

Do not publish a smoke-test-only surrogate tag as the default benchmark image.
