# Background

This benchmark optimizes a **surrogate model** of database throughput under Sysbench-style workloads.
The registered HTTP wrapper evaluates through the reusable Python 3.7 sidecar image `fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1`, which bundles the RF checkpoint and performs knob JSON decoding inside the container.

The optimizer proposes normalized knob coordinates in `[0, 1]^d`; the task decodes them to physical MySQL knob values and returns the surrogate's predicted objective (higher is better).
