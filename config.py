###################
# Data generation #
###################
signal_length = 500       # Signal length (starts from zero)
step_size = 6.2832 / 500  # Sample step. (2PI / L, for one cycle).
repetitions = 15          # How many times pattern is repeated during single period
datafile = "traindata"    # Where generated data is saved

harmonics = {             # Signal shape.
    1 : 5.5,
    2 : 1.3,
    6 : 4.0
}

noise = 0.04              # Makes the signal jagged. 0.01 = 1% error.

###################
#      Model      #
###################
device = "cuda"           # cpu / cuda
hidden_layers = 32        # Amount of neurons in hidden conv layers (do not change)
data_length   = signal_length - 1 # Signal length

# Optimizer
max_iter = 20
history_size = 80
learning_rate = 0.10      # Optimizer lr


###################
#    Plotting     #
###################
color = 'b'
dpi = 100
