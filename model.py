import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config as cfg
from enum import IntEnum
from math import floor

# This enum allows to reference feature vectors in an abstract manner.
class Channel(IntEnum): # Channels in data block
    SIG1 = 0 # First measurement channel.

# The aim of this datastructure is to recude user errors by
# defining clear ways to handle data while also doing checks.
# Data is stored as a 3d array: [B, S, L], where B is batch
# S is recorded signal and L is recorded data value index.
class Batch(object):
    """
    Shape [B, S, L]
    Data is stored as a torch tensor, so model can use data directly.
    """
    def __init__(self, b, s, l): # Use config to set these
        self.data = torch.zeros(b, s, l, dtype=torch.float64)
        self.n = 0
        self.end = b # size for 1st dim
        self.shape = self.data.shape

        self.consider_axis = 0

    def __len__(self):
        return self.end

    def __get__(self):
        return self.data

    def __set__(self, obj):
        self.data = torch.from_numpy(obj)

    def __setitem__(self, indices, signal_data):
        i, j = indices
        self.data[i, j, :] = torch.from_numpy(np.asarray(signal_data))

    def __getitem__(self, indices):
        i, j = indices
        return self.data[i, j, :]

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.end:
            signals = self.data[self.n, :, :]
            self.n += 1
            return signals
        else:
            raise StopIteration

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
    def __init__(self, signal_length=500, hidden=32, predict_n=1) -> None:
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

        print(f"[INFO] using {n1} hidden neurons.")

    def forward(self, input_data) -> torch.tensor:
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
        training (bool): Sets if the model is in training or eval mode.
        device (str): device used for training and predicting. cpu / cuda.
        load_path (str): Path to a model save file, which will be loaded.
    """
    def __init__(self, training=True, device="cpu", load_path=None) -> None:
        self.training = training
        self.device = device
        self.seq = Sequence(cfg.signal_length, cfg.hidden, cfg.predict_n).double().to(device)
        self.criterion = nn.MSELoss().to(device)
        # Use LBFGS as optimizer since we can load full data batch to train
        # LBFGS is very memory intensive, so use history and max iter to adjust memory usage!
        self.optimizer = optim.LBFGS(self.seq.parameters(), lr=cfg.learning_rate,
            max_iter=cfg.max_iter, history_size=cfg.history_size)
        
        if load_path:
            checkpoint = torch.load(load_path)
            self.seq.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("[INFO] model has been loaded succesfully!")

        # Needed, so we can put model into eval mode with ONNX generation
        if not self.training:
            self.seq.eval()
            print("[INFO] evaluation mode has been enabled.")

    def _forwardShift(self, new_tensor, old_tensor) -> torch.tensor:
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
        return tensor

    def _computeLoss(self, input_data, target_data) -> (torch.tensor, torch.tensor):
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

    def predict(self, test_input, test_target, verbose=True) -> np.array:
        """
        Predicts values in sequence. Does not update NN-weights.

        Args:
            test_input (array): input data for NN.
            test_target (array): values that should be obtained.

        Returns:
            array: predictions. What NN thinks the values should be.
        """
        with torch.no_grad(): # Do not update network -> reduced memory usage
            out, loss = self._computeLoss(test_input.to(self.device), test_target.to(self.device))
            if verbose:
                print("prediction loss:", loss.item())
            out = self._forwardShift(out, test_input) # Combine angle and signal again; use original input data
            y = out.cpu().detach().numpy()
        return y # [:, 0] # return the 'new' prediction value

    def train(self, train_input, train_target, verbose=True) -> None:
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

    def save_model(self, model_path, epoch=None, loss=None) -> None:
        """
        Saves current state of the model.

        Args:
            model_path (str): path + filename + extension
            epoch (optional)
            loss (optional)
        """
        torch.save({
                    'model_state_dict': self.seq.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': loss,
                    }, model_path)
