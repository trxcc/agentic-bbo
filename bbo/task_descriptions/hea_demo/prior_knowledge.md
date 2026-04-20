# Domain Prior Knowledge

- This is a bounded simplex problem in disguise: increasing one alloy component necessarily reduces the remaining mass available to the others.
- The exposed `x1..x4` coordinates are not raw compositions; they are decoder-friendly latent variables, so equal steps in design space do not correspond to equal steps in physical composition space.
- Feasible compositions are symmetric only through the decoder and per-component bounds; there is no free permutation symmetry because the components keep distinct identities.
- No metallurgy prior beyond the staged data, the simplex constraint, and the tutorial transform should be assumed.
