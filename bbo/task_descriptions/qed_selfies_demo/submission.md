# Submission Interface

- Optimizers must submit a full configuration for `selfies_token_00` through the final token slot.
- Each token value must be one of the task-declared categorical choices.
- `__EOS__` ends the molecule sequence; `__PAD__` is ignored.
- The canonical logged configuration remains the token dictionary, while decoded SELFIES and SMILES are recorded in metadata.
- The default evaluation budget is `40`; smoke tests usually use smaller budgets such as `3` or `6`.

