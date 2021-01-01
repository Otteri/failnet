import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config as cfg
from enum import IntEnum

# Brief:
# The model tries to learn provided periodical signal from input data.
# Learned signal pattern can be leveraged for doing something useful.
# Expects data to have form: [B, S, L], where B is batch size,
# S is number of signals and L is the signal length.


# The model tries to learn correlation between these data values
# Could be e.g. time and speed v(t) or angle and magnetic field f(theta)
class Channel(IntEnum): # Channels in data block
    BASE = 0 # Base measure that represents advancement. E.g. time/angle/distance.
    SIG1 = 1 # First measurement channel. These values are sampled with respect to BASE.

class Sequence(nn.Module):
    def __init__(self, hidden=32):
        super(Sequence, self).__init__()
        self.linear = nn.Linear(1312, cfg.data_length)
        self.conv1 = nn.Conv1d(2, hidden, 5, stride=3, padding=2)
        self.avg_pool = nn.AvgPool1d(6, stride=4)
        self.flatten = nn.Flatten()
        self.batchnorm = nn.BatchNorm1d(32)

        print("Using %d hidden layers..." % hidden)

    def forward(self, input):
        x = self.avg_pool(F.relu(self.conv1(input)))
        x = self.batchnorm(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

class Model(object):
    def __init__(self, device="cpu"):
        self.device = device
        self.seq = Sequence(cfg.hidden_layers).double().to(device) #config.hidden_layers = N or repetitions
        self.criterion = nn.MSELoss().to(device)
        # use LBFGS as optimizer since we can load the whole data to train
        # LBFGS is very memory intensive, so use history and max iter to adjust memory usage!
        self.optimizer = optim.LBFGS(self.seq.parameters(), lr=cfg.learning_rate,
            max_iter=cfg.max_iter, history_size=cfg.history_size)

    # One step forward shift for signals
    # Replace old input values with shifted signal data; old_tensor cannot be overwritten directly!
    def shift(self, new_tensor, old_tensor):
        tensor = old_tensor.clone() # keep graph
        tensor[:, Channel.SIG1, :] = new_tensor[:, :]
        tensor[:, Channel.BASE, :-1] = old_tensor[:, Channel.BASE, 1:] # shift one forward
        return tensor

    def computeLoss(self, filtered_input_data, filtered_target_data):
        y = self.seq(filtered_input_data)
        if filtered_target_data is not None:
            shift = filtered_target_data.size(2)
            filtered_target_signal = filtered_target_data[:, Channel.SIG1, :shift]
            loss = self.criterion(y[:, :shift], filtered_target_signal) # Easier to compare input
            return loss, y
        return y

    # In prediction, do not update NN-weights
    def predict(self, test_input, test_target):
        with torch.no_grad(): # Do not update network when predicting
            loss, out = self.computeLoss(test_input.to(self.device), test_target.to(self.device))
            print("prediction loss:", loss.item())
            out = self.shift(out, test_input) # Combine angle and signal again; use original input data
            y = out.detach().numpy()
        return y #[:, 0] # return the 'new' prediction value

    def train(self, train_input, train_target):
        def closure(): # LBFGS requires closure
            self.optimizer.zero_grad()
            loss, out = self.computeLoss(train_input.to(self.device), train_target.to(self.device))
            print("loss:", loss.item())
            loss.backward()
            return loss # this comment may prevent learning
        self.optimizer.step(closure)
