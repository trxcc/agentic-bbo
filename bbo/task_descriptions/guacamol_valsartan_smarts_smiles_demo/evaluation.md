# Evaluation Protocol

The evaluator combines 4 component scores in [0,1] as score = (s1 * ... * s4)**(1/4). It returns loss = 1-score; lower loss is better. Only the combined objective is returned, not individual components.

s1 is 1 if the molecule matches SMARTS CN(C=O)Cc1ccc(c2ccccc2)cc1, otherwise 0.

s2 = G(logP; property-reference logP, 0.2).

s3 = G(TPSA; property-reference TPSA, 5).

s4 = G(Bertz complexity; property-reference Bertz complexity, 30).

Empty or unparseable SMILES receive score 0 and loss 1. Only the listed scoring components are evaluated; this score does not measure clinical efficacy, safety, or synthetic feasibility.
