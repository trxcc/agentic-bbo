# Prior Knowledge

- Five Sysbench-related knobs are active (see task metadata `feature_order`).
- The surrogate predicts throughput (TPS) from physical knob vectors; the registered HTTP service keeps the model and legacy sklearn runtime inside Docker.
- This task is intended for optimizer comparison; keep the Docker image tag fixed across methods.
