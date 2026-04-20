# Domain Prior Knowledge

- Categorical metal identities and process settings interact strongly, so the best local numeric tweak often depends on which metal triplet is chosen first.
- The cleaned dataset includes an explicit `None` level for missing secondary or tertiary metals; absence is therefore modeled as a real categorical state rather than dropped as missing data.
- The benchmark search space intentionally leaves composition ratios independent; do not assume they are normalized to sum to `100` unless that rule is added explicitly by a future task revision.
- Lower overpotential is always better, and there is no noise model beyond the deterministic fitted oracle.
