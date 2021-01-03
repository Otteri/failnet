import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config as cfg
from enum import IntEnum
from math import floor

# The model tries to learn correlation between features. Features could be
# e.g. time and speed v(t) or angle and magnetic field f(theta). This enum
# allows to reference feature vectors in an abstract manner.
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
        hidden (int): Base number of neurons. Can be used to adjust size and
        learning capabilities of the NN. (Actual number of neurons is likely to be different).

        signal_length (int): Length of the data fed to model.
        
        predict_n (int): Sets how much in future model should predict.
        Defaults to 1, which means that does not try to predict future.
    """
    def __init__(self, signal_length=500, hidden=32, predict_n=1):
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
        L_in = signal_length - predict_n
        n1 = floor((hidden / 500.0) * L_in)  # Get actual number of neurons from base.
        L_conv_out = floor((L_in + 2 * padding1 - dilation1 * (kernel1 - 1) - 1) / stride1 + 1)
        L_batch_out = floor((L_conv_out + 2 * padding2 - kernel2) / stride2 + 1)
        n2 = floor(n1 * L_batch_out)        
        
        # Define NN build blocks using calulated sizes
        self.linear = nn.Linear(n2, signal_length - 1)
        self.conv = nn.Conv1d(channels, n1, kernel1, stride=stride1, padding=padding1, groups=groups1)
        self.avg_pool = nn.AvgPool1d(kernel2, stride=stride2)
        self.flatten = nn.Flatten()
        self.batchnorm = nn.BatchNorm1d(n1)

        print(f"Using {n1} hidden neurons...")

    def forward(self, input_data):
        """
        Forwards input data through neural net.

        Args:
            input_data (tensor): sequential data.

        Returns:
            tensor: NN output.
        """
        x = self.avg_pool(F.relu(self.conv(input_data)))
        x = self.batchnorm(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

class Model(object):
    """
    Full model capable of learning and predicting patterns in data sequence.
    Uses Lightweight Sequence network defined above and LBFGS optimizer.

    Args:
        device (str): device used for training. cpu / cuda.
    """
    def __init__(self, device="cpu"):
        self.device = device
        self.seq = Sequence(cfg.signal_length, cfg.hidden, cfg.predict_n).double().to(device)
        self.criterion = nn.MSELoss().to(device)
        # use LBFGS as optimizer since we can load full data batch to train
        # LBFGS is very memory intensive, so use history and max iter to adjust memory usage!
        self.optimizer = optim.LBFGS(self.seq.parameters(), lr=cfg.learning_rate,
            max_iter=cfg.max_iter, history_size=cfg.history_size)

    def _forwardShift(self, new_tensor, old_tensor):
        """
        Forwards data tensor one step by shifting data.

        Args:
            new_tensor (tensor): tensor obtained from NN.
            old_tensor (tensor): original data tensor.

        Returns:
            [tensor]: shifted data tensor.
        """
        N = cfg.predict_n - 1 # indices start from zero
        tensor = old_tensor.clone() # keep graph
        tensor[:, Channel.SIG1, :] = new_tensor[:, N:]
        tensor[:, Channel.BASE, :-1] = old_tensor[:, Channel.BASE, 1:]
        return tensor

    def _computeLoss(self, input_data, target_data):
        """
        Passes data through NN and then computes loss.
        When predicting outside of training, target_data can be same as input_data.

        Args:
            input_data (tensor): data used for learning
            target_data (tensor): data values that NN should obtain

        Returns:
           tensor: predictions. What NN thinks the values should be.
           tensor: loss value, which indicates NN performance.
        """
        assert input_data.size(2) > 0, "No input data provided. It is required!"
        assert input_data.shape == target_data.shape, "Target data size must match with input data."

        y = self.seq(input_data)
        shift = target_data.size(2)
        target_signal = target_data[:, Channel.SIG1, :shift]
        loss = self.criterion(y[:, :shift], target_signal) # Easier to compare input
        return y, loss

    def predict(self, test_input, test_target, verbose=True):
        """
        Predicts values in sequence. Does not update NN-weights.

        Args:
            test_input (array): input data for NN.
            test_target (array): values that should be obtained.

        Returns:
            tensor: predictions. What NN thinks the values should be.
        """
        with torch.no_grad(): # Do not update network when predicting
            out, loss = self._computeLoss(test_input.to(self.device), test_target.to(self.device))
            if verbose:
                print("prediction loss:", loss.item())
            out = self._forwardShift(out, test_input) # Combine angle and signal again; use original input data
            y = out.detach().numpy()
        return y # [:, 0] # return the 'new' prediction value

    def train(self, train_input, train_target, verbose=True):
        """
        Predicts values in sequence. Updates NN-weigths.
        LBFGS optimizer requires closure.
        
        Args:
            train_input (array): input data for NN.
            train_target (array): values that NN should obtain.
        """
        def closure():
            self.optimizer.zero_grad()
            out, loss = self._computeLoss(train_input.to(self.device), train_target.to(self.device))
            if verbose:
                print("loss:", loss.item())            
            loss.backward()
            return loss
        self.optimizer.step(closure)
