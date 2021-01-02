###################
# Data generation #
###################
predict_n = 3
repetitions = 15          # How many times pattern is repeated during single period
signal_length = 351      # Signal length (starts from zero)
step_size = 6.2832 / signal_length  # Sample step. (2PI / L, for one cycle).
datafile = "traindata"    # Where generated data is saved

harmonics = {             # Signal shape.
    1 : 5.5,
    2 : 1.3,
    6 : 4.0
}

noise = 0.01              # Makes the signal jagged. 0.01 = 1% error.

###################
#      Model      #
###################
device = "cuda"           # cpu / cuda
hidden = 64              # a base number of hidden neurons. Actual number may be different. 

# Optimizer
max_iter = 20
history_size = 80
learning_rate = 0.10      # Optimizer lr


###################
#    Plotting     #
###################
color = 'b'
dpi = 100
