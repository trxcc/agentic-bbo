# Domain Prior Knowledge

G(x; mu, sigma) = exp(-0.5*((x-mu)/sigma)**2). It scores 1 at the target mu and decreases smoothly away from it; sigma controls tolerance. min/max inside G makes one side unpenalized.

logP estimates preference for an oil-like phase over water. TPSA is topological polar surface area, a structural measure of polar atom contributions. Both are computed molecular descriptors, not laboratory measurements here.

SMARTS is a structural pattern language used to test whether a molecule contains a required arrangement of atoms and bonds. Bertz complexity summarizes molecular graph complexity. This task does not use fingerprint similarity.

A low component limits the geometric mean; any zero component makes the combined score zero. Changing atoms, bonds, or rings can alter several components at once.

Different SMILES can represent the same molecule. A small text edit need not imply a small change in structure or score.
