# Domain Prior Knowledge

hidden_layer_sizes is one integer: the width of one hidden layer, not a list of layer sizes.

The solver is SGD with inverse-scaling learning rate, early stopping, and Nesterov momentum. random_state=0 and max_iter=40 are fixed.

The effective learning rate follows learning_rate_init / t**power_t; t is the training time counter, not simply an epoch index.

batch_size controls mini-batches, and momentum controls the contribution of accumulated update direction. Nesterov momentum is active when momentum is positive.

Each fitting fold reserves validation_fraction for internal early stopping. tol is the required validation-score improvement; n_iter_no_change=10 consecutive epochs without it trigger stopping.

max_iter limits epochs, not individual gradient updates. The L2 penalty controlled by alpha is divided by the sample count when added to the loss.

The fixed activation is relu. The internal early-stopping metric is validation accuracy.

## Domain experience and conditions

Learning rate, its decay, momentum, and batch size jointly affect training within the 40-epoch limit. A larger validation fraction leaves fewer samples for fitting inside each fold.
