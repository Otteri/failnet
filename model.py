import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config as cfg
from enum import IntEnum
from math import floor

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
    """
    Model can learn repeating patterns from provided sequential data.
    It is also possible to train this model to predict n-steps into future.
    Layer sizes are proportional, so input data size can be freely adjusted.
    Hence, the prediction step and/or sample number can change and the same
    model can be still used without any tweaking.

    This kind of implementation requires that layer input and output sizes are
    calculated during initialization. The layer sizes are proportional to data length
    and base number of hidden neurons. Layer sizes can be calculated using formulas from here:    
    https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html
    
    Args:
        hidden (int, optional): Base number of neurons. Can be used to adjust size and
        learning capabilities of the NN. (Actual number neurons is likely to be different).

    """
    def __init__(self, hidden=32):
        super(Sequence, self).__init__()
        channels  = len(Channel) # Number of data channels
        kernel1   = 5  # Conv kernel size
        stride1   = 3  # Conv stride size
        padding1  = 2  # Conv padding size
        groups1   = 1  # Conv groups (default)
        dilation1 = 1  # Conv dilation (default)
        kernel2   = 6  # AvgPool kernel size
        stride2   = 4  # AvgPool stride size
        padding2  = 0  # AvgPoll padding (default)

        # Calculate layer sizes n1 and n2
        L_in = cfg.signal_length - cfg.predict_n
        n1 = floor((hidden / 500.0) * L_in)  # Get actual number of neurons from base.
        L_conv_out = floor((L_in + 2 * padding1 - dilation1 * (kernel1 - 1) - 1) / stride1 + 1)
        L_batch_out = floor((L_conv_out + 2 * padding2 - kernel2) / stride2 + 1)
        n2 = floor(n1 * L_batch_out)        
        
        # Define NN build blocks using calulated sizes
        self.linear = nn.Linear(n2, cfg.signal_length - 1)
        self.conv = nn.Conv1d(channels, n1, kernel1, stride=stride1, padding=padding1, groups=groups1)
        self.avg_pool = nn.AvgPool1d(kernel2, stride=stride2)
        self.flatten = nn.Flatten()
        self.batchnorm = nn.BatchNorm1d(n1)

        print(f"Using {n1} hidden neurons...")

    def forward(self, input_data):
        x = self.avg_pool(F.relu(self.conv(input_data)))
        x = self.batchnorm(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

class Model(object):
    def __init__(self, device="cpu"):
        self.device = device
        self.seq = Sequence(cfg.hidden).double().to(device)
        self.criterion = nn.MSELoss().to(device)
        # use LBFGS as optimizer since we can load the whole data to train
        # LBFGS is very memory intensive, so use history and max iter to adjust memory usage!
        self.optimizer = optim.LBFGS(self.seq.parameters(), lr=cfg.learning_rate,
            max_iter=cfg.max_iter, history_size=cfg.history_size)

    # One step forward shift for signals
    # Replace old input values with shifted signal data; old_tensor cannot be overwritten directly!
    # Advance only one step at time (hence 1), but can predict further
    def shift(self, new_tensor, old_tensor):
        N = cfg.predict_n - 1 # indices start from zero
        tensor = old_tensor.clone() # keep graph
        tensor[:, Channel.SIG1, :] = new_tensor[:, N:]
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
