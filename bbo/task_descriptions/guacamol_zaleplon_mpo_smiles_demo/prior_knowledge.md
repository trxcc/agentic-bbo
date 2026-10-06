# Domain Prior Knowledge

A molecular fingerprint summarizes structural features. Tanimoto similarity compares two fingerprints, from 0 for no overlap to 1 for identical fingerprint features; it does not compare SMILES text.

ECFP4 uses Morgan count fingerprints for atomic environments within two bonds (radius 2).

A molecular formula counts each element; it does not specify bonds. Its score is the geometric mean of G(element count; target count, 1) for target elements and G(total atom count including hydrogens; target total, 2).

A low component limits the geometric mean; any zero component makes the combined score zero. Changing atoms, bonds, or rings can alter several components at once.

Different SMILES can represent the same molecule. A small text edit need not imply a small change in structure or score.
