# Third-party notices and benchmark assets

The project source is released under the [MIT License](LICENSE). Dependencies,
adapted code, benchmark data, and external model weights retain their respective
upstream terms. The project license does not replace those terms.

| Component | Source and treatment |
| --- | --- |
| BoTorch tutorial code | Vendored adaptations under `bbo/algorithms/model_based/_vendor/`; the upstream copyright and license are in [`BOTORCH_LICENSE.txt`](bbo/algorithms/model_based/_vendor/BOTORCH_LICENSE.txt). |
| Molecular Graph GA and GPBO routines | Adapted from [PMO / mol_opt](https://github.com/wenhao-gao/mol_opt), released under MIT. The upstream notice is in [`PMO_LICENSE.txt`](bbo/algorithms/molecular/PMO_LICENSE.txt); source attribution is retained in the relevant modules. |
| COCO / BBOB | Installed as the external `coco-experiment` package. Its evaluator is not distributed inside the agent image. |
| HPO arrays | Frozen Bayesmark/LLAMBO-derived splits in `bbo/tasks/hpo/assets/`; file provenance is in [`manifest.json`](bbo/tasks/hpo/assets/manifest.json). Their redistribution in this release was authorized by the project maintainers. |
| BBOPlace repair bundles | Derived geometry and netlist data from BBOPlace ISPD/ICCAD instances in `bbo/tasks/bboplace/assets/repair_bundles/`. Their redistribution in this release was authorized by the project maintainers. |
| DBTune surrogate checkpoints | Downloaded separately according to [`bbo/tasks/dbtune/assets/README.md`](bbo/tasks/dbtune/assets/README.md); checkpoint files are not in this repository. |
| TabPFN / GIT-BO weights | Optional upstream dependency and weights; they are not bundled here. Check upstream terms before downloading and using them. |
| Codex model catalog | A frozen historical catalog containing model-provider instructions is excluded from the public repository. Its checksum remains in `bbo/experiments/assets/provenance.json`; a user-supplied catalog can be passed with `BBO_CODEX_MODEL_CATALOG`. |

Frozen reference hashes are recorded in `bbo/experiments/assets/provenance.json`.
The verifier checks every bundled reference and explicitly skips the excluded
catalog entry.
