import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from math import floor
from statistics import mean

class Batch(object):
    """
    The aim of this datastructure is to recude user errors by defining clear
    ways to handle data. Some correctness asserts can be done as well.
    Internally, data is stored to 3d tensor: [B, S, L], where B is batch
    S is signal channel and L is data value index. Because data is converted
    to tensor, the NN model can use class data directly for computation.
    It is beneficial to keep data structure very simple, so this class
    is not necessary for running the model (ONNX). Essentially, you
    could just call: np.zeros((b, s, l)) and manage indices and access
    to array yourself, but this is more error prone.

    Default zero initialization values allow premature initialization and
    make assigning possible. Notice, that none of the b,s,l values should
    stay zero when storing real data. You should always have at least one
    batch with certain length signal(s).
    
    Args:
        b (int) : number of batches.
        s (int) : number of channels (different signals)
        l (int) : signal length.
    """
    def __init__(self, b=0, s=0, l=0): # Use config to set these

        assert b >= 0, "Batch number cannot be negative or zero"
        assert s >= 0, "Number of signals cannot be negative or zero"
        assert l >= 0, "Signal length cannot be negative or zero"

        self.data = torch.zeros(b, s, l, dtype=torch.float64)
        self.n = 0
        self.end = b # size for 1st dim
        self.shape = self.data.shape

    def __len__(self):
        return self.end

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
    def __init__(self, channels=1, signal_length=500, hidden=32, predict_n=1) -> None:
        super(Sequence, self).__init__()
        channels  = channels # Number of data channels (in batches)
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
        self.conv = nn.Conv1d(channels, n1, kernel1, stride=stride1, padding=padding1, groups=groups1)
        self.avg_pool = nn.AvgPool1d(kernel2, stride=stride2)
        self.flatten = nn.Flatten()
        self.batchnorm = nn.BatchNorm1d(n1)
        self.linear = nn.Linear(n2, signal_length - 1)

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
    def __init__(self, training=True, device="cpu", load_path=None, signal_length=500, hidden=32, predict_n=1, lr=0.10, max_iter=20, history_size=80) -> None:
        self.training = training
        self.device = device
        self.seq = Sequence(1, signal_length, hidden, predict_n).double().to(device)
        self.criterion = nn.MSELoss().to(device)
        # Use LBFGS as optimizer since we can load full data batch to train
        # LBFGS is very memory intensive, so use history and max iter to adjust memory usage!
        self.optimizer = optim.LBFGS(self.seq.parameters(), lr=lr,
            max_iter=max_iter, history_size=history_size)

        self.predict_n = predict_n
        self.signal_length = signal_length
        
        if load_path:
            checkpoint = torch.load(load_path)
            self.seq.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("[INFO] model has been loaded succesfully!")

        # Needed, so we can put model into eval mode with ONNX generation
        if not self.training:
            self.seq.eval()
            print("[INFO] evaluation mode has been enabled.")

    def _get_prediction(self, input_data):
        """
        Forwards data through neural network in order to obtain a
        prediction. Then output is packed to a 3D Batch.

        Args:
            input_data (Batch): data for the NN.

        Returns:
            out (tensor): model prediction
        """
        y = self.seq(input_data)
        out = input_data.clone()
        out[:, 0, :] = y[:, 0:] # 2d -> 3d tensor
        return out

    def _compute_loss(self, prediction, target_data) -> (torch.tensor):
        """
        Computes loss between model prediction and target data batch.
        Currently computes loss only using signal no. 0.

        Args:
            prediction (tensor 3d): model prediction (NN output)
            target_data (tensor 3d): data values that NN should obtain

        Returns:
           tensor (double): loss value, which indicates NN performance.
        """
        n = self.signal_length - 1
        assert prediction.size(2) >= n, "Input tensor signal < config signal length."
        assert prediction.shape == target_data.shape, "Target data size must match with input data."

        target = target_data[:, 0, :n]
        loss = self.criterion(prediction[:, 0, :n], target)
        return loss

    def predict(self, test_input, test_target=None) -> np.array:
        """
        Predicts values in sequence. Does not update NN-weights.
        test_target is optional. When given, we can compute loss
        with test data, otherwise we simply just predict using model.

        Args:
            test_input (Batch): input data for the NN model.
            test_target (Batch): values that should be obtained.

        Returns:
            np.array: predictions. What NN thinks the values should be.
        """
        with torch.no_grad(): # Do not update network -> reduced memory usage
            prediction = self._get_prediction(test_input.data.to(self.device))
            if test_target:
                loss = self._compute_loss(prediction, test_target.data.to(self.device))
                print("prediction loss:", loss.item())

        return prediction.cpu().detach().numpy()

    def train(self, train_input, train_target, verbose=False) -> float:
        """
        Predicts values in a sequence. Updates NN-weigths.
        LBFGS optimizer requires closure.
        
        Args:
            train_input (Batch): input data for NN.
            train_target (Batch): values that NN should obtain.

        Returns:
            Mean loss of closure iterations
        """
        losses = [] # losses in closure

        def closure():
            self.optimizer.zero_grad()
            prediction = self._get_prediction(train_input.data.to(self.device))
            loss = self._compute_loss(prediction, train_target.data.to(self.device))
            if verbose:
                print("loss:", loss.item())
            losses.append(loss.item())
            loss.backward()
            return loss
        self.optimizer.step(closure)
        return mean(losses)

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
