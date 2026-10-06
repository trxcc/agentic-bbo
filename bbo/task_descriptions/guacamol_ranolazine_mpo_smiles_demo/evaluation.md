# Evaluation Protocol

The evaluator combines 4 component scores in [0,1] as score = (s1 * ... * s4)**(1/4). It returns loss = 1-score; lower loss is better. Only the combined objective is returned, not individual components.

s1 = min(AP similarity to ranolazine / 0.7, 1).

s2 = G(min(logP, 7); 7, 1), so logP at least 7 is unpenalized in this evaluator.

s3 = G(fluorine atom count; 1, 1).

s4 = G(min(TPSA, 95); 95, 20), so TPSA at least 95 is unpenalized.

Empty or unparseable SMILES receive score 0 and loss 1. Only the listed scoring components are evaluated; this score does not measure clinical efficacy, safety, or synthetic feasibility.
