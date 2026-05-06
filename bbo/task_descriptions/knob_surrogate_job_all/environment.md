# Environment Setup

For the registered HTTP wrapper `knob_http_surrogate_job_all`, prefer the reusable Python 3.7 sidecar image:

```bash
docker pull fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1
docker rm -f agentbbo_surrogate_http 2>/dev/null
docker run -d --name agentbbo_surrogate_http -p 8090:8090 fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1
```

No local `.joblib` file is required for the HTTP task. Local checkpoint files are only needed if you intentionally bypass the HTTP sidecar and call the unregistered in-process surrogate helpers.
