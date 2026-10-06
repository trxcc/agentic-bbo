# Evaluation Protocol

An ECFP4 molecular fingerprint describes local environments around atoms. This implementation uses a Morgan count fingerprint with radius 2, covering neighborhoods up to two bonds away.

Compute Tanimoto fingerprint similarities to the two reference molecules, s1 and s2, each in [0,1]. Larger values indicate more similar fingerprints.

score = sqrt(s1*s2), and loss = 1-score. The returned objective is median1_loss, which is minimized. Only the combined objective is returned, not s1 and s2 separately.

Empty or unparseable SMILES receive score 0, hence loss 1. There are no additional drug-efficacy, toxicity, or synthesizability scores.
