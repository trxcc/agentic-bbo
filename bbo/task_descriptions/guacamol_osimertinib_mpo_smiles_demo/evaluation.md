# Evaluation Protocol

The evaluator combines 4 component scores in [0,1] as score = (s1 * ... * s4)**(1/4). It returns loss = 1-score; lower loss is better. Only the combined objective is returned, not individual components.

s1 = min(FCFP4 similarity to osimertinib / 0.8, 1).

s2 = G(max(ECFP6 similarity to osimertinib, 0.85); 0.85, 0.1), so similarity at or below 0.85 is unpenalized.

s3 = G(min(TPSA, 100); 100, 10), so TPSA at least 100 is unpenalized.

s4 = G(max(logP, 1); 1, 1), so logP at most 1 is unpenalized.

Empty or unparseable SMILES receive score 0 and loss 1. Only the listed scoring components are evaluated; this score does not measure clinical efficacy, safety, or synthetic feasibility.
