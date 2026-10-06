# GuacaMol Median Molecules 1 SMILES

Task ID: `guacamol_median1_smiles_demo`.

Find a molecule whose structure is simultaneously similar to camphor and menthol.

Objective: minimize `median1_loss`.

Budget: 50 shared initial observations and 200 new evaluations (250 total).

## Submission

1. Submit one molecule per candidate as a SMILES string in the smiles field, with a maximum length of 512.
2. SMILES represents atoms and connections on one line: C and O denote carbon and oxygen, parentheses denote branches, paired digits mark ring closures, and = denotes a double bond.
3. Scoring compares the molecular structures represented by the strings. Different SMILES strings can represent the same molecule.

## References

| Name | SMILES |
| --- | --- |
| Camphor | `CC1(C)C2CCC1(C)C(=O)C2` |
| Menthol | `CC(C)C1CCC(C)CC1O` |

## Parameters

| Parameter | Meaning | Type | Allowed values | Search transform |
| --- | --- | --- | --- | --- |
| `smiles` | Molecular structure encoded as SMILES. | string | SMILES string; maximum length 512 | — |

## Read details as needed

There are 1 active parameters. get_search_space supports names, query, optional annotated groups and paged index/details views.

get_task_context sections: overview, submission, scoring, mechanisms, references, additional. Read scoring rules and relevant mechanisms before choosing a candidate.

get_trial_history and get_incumbent return scores first; request parameter_names to inspect selected values.

Follow instructions.md: submit_candidate accepts a full config or workspace JSON file directly; write_candidate is optional.
