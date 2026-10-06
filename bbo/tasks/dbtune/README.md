# DBTune Evaluator

The six benchmark tasks are HTTP surrogate evaluations: Sysbench-5, Sysbench-196,
JOB-5, JOB-196, PostgreSQL-5 and PostgreSQL-20. No live database setup or placeholder
model is part of the paper evaluator.

The host sends a normalized vector in the declared feature order. The Python 3.7
service decodes it using the matching knob JSON and evaluates the released sklearn
checkpoint. The Python 3.11 HPO environment must not deserialize these old models.

Download the checkpoints listed in `assets/README.md`. Use a directory containing
both the checkpoint files and their matching `knobs_*.json` files:

```bash
export BBO_DBTUNE_ASSETS=/absolute/path/to/complete/assets
docker compose --profile dbtune up --build dbtune
```

The service listens on host loopback port 8090. The task constructor checks feature
metadata and health before accepting a run. Missing checkpoints and invalid
responses are errors, not zero scores or random surrogate replacements.
