# Domain Prior Knowledge

The geometric mean rewards balancing both similarities. For example, s1=0.9 and s2=0.1 give score 0.3, whereas two similarities of 0.5 give score 0.5. These are formula examples, not evaluated molecules.

If either similarity is 0, the combined score is 0. Improving similarity to just one reference does not guarantee a higher combined score.

Changes to atoms, bonds, and rings alter local fingerprint features and may affect both similarities. A small string edit need not imply a small structural or score change.

This task uses only ECFP4. Candidate validity, objective evaluation, and evaluation budgets follow the shared task instructions.
