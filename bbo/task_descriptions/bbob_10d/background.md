# Background

This task is one deterministic black-box problem drawn from the official COCO BBOB suite. The optimizer receives only evaluated configurations and their scalar losses. The hidden objective identity, transformation, and reference solution are not part of the optimization context.

The search space has ten continuous variables, x1 through x10. Every variable is bounded to [-5, 5]. Evaluations are deterministic for a fixed task instance.
