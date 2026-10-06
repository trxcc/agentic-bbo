# Evaluation Protocol

The evaluator combines 4 component scores in [0,1] as score = (s1 * ... * s4)**(1/4). It returns loss = 1-score; lower loss is better. Only the combined objective is returned, not individual components.

s1 = G(ECFP4 similarity to sitagliptin; 0, 0.1): lower structural similarity is preferred here.

s2 = G(logP; reference logP, 0.2), with the target computed from the supplied sitagliptin structure.

s3 = G(TPSA; reference TPSA, 5), with the target computed from the same reference.

s4 is the molecular-formula score for C16H15F6N5O.

Empty or unparseable SMILES receive score 0 and loss 1. Only the listed scoring components are evaluated; this score does not measure clinical efficacy, safety, or synthetic feasibility.
