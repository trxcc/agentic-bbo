# Domain Prior Knowledge

A molecular fingerprint summarizes structural features. Tanimoto similarity compares two fingerprints, from 0 for no overlap to 1 for identical fingerprint features; it does not compare SMILES text.

AP uses atom-pair count fingerprints, describing pairs of atoms and their topological distances.

G(x; mu, sigma) = exp(-0.5*((x-mu)/sigma)**2). It scores 1 at the target mu and decreases smoothly away from it; sigma controls tolerance. min/max inside G makes one side unpenalized.

logP estimates preference for an oil-like phase over water. TPSA is topological polar surface area, a structural measure of polar atom contributions. Both are computed molecular descriptors, not laboratory measurements here.

A low component limits the geometric mean; any zero component makes the combined score zero. Changing atoms, bonds, or rings can alter several components at once.

Different SMILES can represent the same molecule. A small text edit need not imply a small change in structure or score.
