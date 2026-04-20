# Scientific Task Data

This directory vendors the datasets required by the scientific benchmark tasks.
The benchmark resolves data from this workspace-local directory by default, so the repo can be packaged or uploaded together with its scientific task assets.

## Layout

The files under `examples/` preserve the original BO tutorial relative paths:

- `examples/HER/HER_virtual_data.csv`
- `examples/HEA/data/oracle_data.xlsx`
- `examples/OER/OER.csv`
- `examples/OER/OER_clean.csv`
- `examples/BH/BH_dataset.csv`
- `examples/Molecule/zinc.txt.gz`

Code refers to these files by their original `examples/...` relative paths.
At runtime the benchmark stages them into `artifacts/dataset_cache/bo_tutorial/` unless `BBO_BO_TUTORIAL_CACHE_ROOT` is set.

## Overriding the bundled data

If you want to replace the bundled files with another uploaded dataset bundle, set:

```bash
BBO_BO_TUTORIAL_SOURCE_ROOT=/path/to/source_root
```

That override root must keep the same `examples/...` directory layout as this vendored directory.
