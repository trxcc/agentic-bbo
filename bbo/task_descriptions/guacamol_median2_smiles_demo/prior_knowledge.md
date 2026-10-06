# Domain Prior Knowledge

A molecular fingerprint summarizes structural features. Tanimoto similarity compares two fingerprints, from 0 for no overlap to 1 for identical fingerprint features; it does not compare SMILES text.

ECFP6 uses Morgan count fingerprints for atomic environments within three bonds (radius 3).

A low component limits the geometric mean; any zero component makes the combined score zero. Changing atoms, bonds, or rings can alter several components at once.

Different SMILES can represent the same molecule. A small text edit need not imply a small change in structure or score.
