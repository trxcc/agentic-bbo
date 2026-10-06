# Evaluation Protocol

The evaluator combines 2 component scores in [0,1] as score = (s1 * ... * s2)**(1/2). It returns loss = 1-score; lower loss is better. Only the combined objective is returned, not individual components.

s1 and s2 are ECFP6 Tanimoto similarities to tadalafil and sildenafil, respectively.

Empty or unparseable SMILES receive score 0 and loss 1. Only the listed scoring components are evaluated; this score does not measure clinical efficacy, safety, or synthetic feasibility.
