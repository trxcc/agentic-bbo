# Evaluation Protocol

The evaluator combines 3 component scores in [0,1] as score = (s1 * ... * s3)**(1/3). It returns loss = 1-score; lower loss is better. Only the combined objective is returned, not individual components.

s1 = min(AP similarity to fexofenadine / 0.8, 1).

s2 = G(min(TPSA, 90); 90, 10), so TPSA at least 90 is unpenalized.

s3 = G(max(logP, 4); 4, 1), so logP at most 4 is unpenalized.

Empty or unparseable SMILES receive score 0 and loss 1. Only the listed scoring components are evaluated; this score does not measure clinical efficacy, safety, or synthetic feasibility.
