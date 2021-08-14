###################
# Data generation #
###################
data_dir = "predictions/"  # Where generated data is saved to
repetitions = 10           # How many times pattern is repeated during single period
signal_length = 501        # Data signal length (number of samples)
step_size = 6.2832 / signal_length  # Sampling step. (2PI / L, for one cycle).
noise = 0.01               # Adds noise to training data. 0.01 = 1% of data max amplitude.

harmonics = {              # Defines signal shape.
    1 : 5.5,               # First harmonic order and its max amplitude.
    2 : 1.3,               # Second order
    6 : 4.0                # ...
}

###################
#      Model      #
###################
device = "cuda"            # Use cpu / cuda for training
hidden = 32                # A base number for hidden neurons.
predict_n = 1              # Sets how many steps into future tries to predict.

# Optimizer
max_iter = 20              # Maximum allowed number of iterations
history_size = 80          # Maximum allowed size for history
learning_rate = 0.10       # Optimizer lr
