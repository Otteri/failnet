import pytest
import torch
import numpy as np
from os import path, remove
from failnet.model import Model, Sequence, Batch

SIGNAL_LENGTH = 100

# Test helper class
class BatchData:
    def __init__(self):
        self.n = None
        self.signal_length = None
        self.input = None
        self.target = None
        self.input = None
        self.target = None
        self.create_batch(SIGNAL_LENGTH)

    # A helper method, which allows to create
    # input and target data batches with desired dimensions
    # Returns these as a tuple that can be unpacked an passed for training
    def create_batch(self, length):
        self.n = 1
        self.signal_length = length
        self.input = Batch(1, 1, length-self.n)
        self.target = Batch(1, 1, length-self.n)
        signal = np.arange(self.signal_length-self.n+1)
        self.input[0, 0] = signal[:-self.n]
        self.target[0, 0] = signal[self.n:]
        return (self.input, self.target)

def test_model_creation_with_defaults():
    model = Model()
    assert model is not None

def test_model_creation():

    # Create a new model
    model = Model(
        training      = False,
        device        = "cpu",
        signal_length = 100,
        hidden        = 16,
        predict_n     = 5,
        lr            = 0.2,
        max_iter      = 25,
        history_size  = 100
    )

    assert model.training == False
    assert model.device == "cpu"
    assert model.seq is not None
    assert model.criterion is not None
    assert model.optimizer is not None
    assert model.predict_n == 5

def test_get_prediction():
    data = BatchData()
    model = Model(predict_n=1, signal_length=SIGNAL_LENGTH, hidden=16)
    prediction = model._get_prediction(data.input.data.to("cpu"))
    assert type(prediction) == torch.Tensor
    assert list(prediction.shape) == [1, 1, 99]

def test_loss_computation():
    data = BatchData()
    model = Model(predict_n=1, signal_length=SIGNAL_LENGTH, hidden=16)
    prediction = model._get_prediction(data.input.data.to("cpu"))
    loss = model._compute_loss(prediction, data.target.data.to("cpu"))
    loss2 = model._compute_loss(prediction, data.target.data.to("cpu"))
    assert type(loss) == torch.Tensor
    assert loss >= 0.0 # MSE loss min is zero
    assert loss < 1e6 # We never want to end up to millions (exploding gradient)
    assert loss == loss2 # We have done nothing, loss should be still the same

def test_train_is_vebose(capfd):
    data = BatchData()
    model = Model(predict_n=1, signal_length=SIGNAL_LENGTH, hidden=16)
    model.train(*data.create_batch(SIGNAL_LENGTH)  , verbose=True)
    out, err = capfd.readouterr()
    assert "loss:" in out

    model.train(*data.create_batch(SIGNAL_LENGTH), verbose=False)
    out, err = capfd.readouterr()
    assert "loss:" not in out

# Test problem is simple and it should be easy to learn.
# Try to learn it, in which case losses should decrease.
# Since train interface function returns nothing, we must check
# convergence by outputting losses and then parsing the loss values.
def test_train_converges(capfd):
    def get_loss_avg_from_strings(losses):
        cumulative_loss = 0.0
        loss_count = 0
        for loss_count, loss_str in enumerate(losses):
            cumulative_loss += float(loss_str.split(':')[1])
        return float(cumulative_loss / (loss_count+1))

    data = BatchData()
    model = Model(predict_n=1, lr=0.05, max_iter=10, signal_length=SIGNAL_LENGTH, hidden=32)
    model.train(*data.create_batch(SIGNAL_LENGTH), verbose=True)
    out, err = capfd.readouterr()
    losses = out.splitlines()[1:] # 1: exclude [INFO] print
    early_loss_avg = get_loss_avg_from_strings(losses[0:5])
    later_loss_avg = get_loss_avg_from_strings(losses[10:15])
    assert early_loss_avg > later_loss_avg

def test_predict():
    data = BatchData()
    model = Model(predict_n=1, signal_length=SIGNAL_LENGTH, hidden=16)
    prediction = model.predict(*data.create_batch(SIGNAL_LENGTH))
    assert prediction.shape == (1, 1, 99)
    assert prediction.dtype == np.float64

def test_model_save_and_load(capfd):
    # Test saving
    model_filename = "./test_save.pt"
    model = Model(predict_n=1, signal_length=SIGNAL_LENGTH, hidden=16)
    model.save_model(model_filename)
    assert path.isfile(model_filename) == True

    # Test loading
    model = Model(load_path=model_filename, predict_n=1, signal_length=SIGNAL_LENGTH, hidden=16)
    out, err = capfd.readouterr()
    assert "[INFO] model has been loaded succesfully!" in out
    remove(model_filename) # clean up
