import pytest
import torch
import numpy as np
from os import path, remove
from failnet.model import Model, Sequence, Batch

class Data:
    def __init__(self, signal_length):
        self.n = 1
        self.signal_length = signal_length
        self.input = Batch(1, 1, 100-self.n)
        self.target = Batch(1, 1, 100-self.n)
        self.signal = np.arange(self.signal_length)
        self.input[0, 0] = self.signal[:-self.n]
        self.target[0, 0] = self.signal[self.n:]

# Test data set #1
# Single sequentally increasing signal from 0 to 100
data1 = Data(100)

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
    model = Model(predict_n=1, signal_length=100, hidden=16)
    prediction = model._get_prediction(data1.input.data.to("cpu"))
    assert type(prediction) == torch.Tensor
    assert list(prediction.shape) == [1, 1, 99]

def test_loss_computation():
    model = Model(predict_n=1, signal_length=100, hidden=16)
    prediction = model._get_prediction(data1.input.data.to("cpu"))
    loss = model._compute_loss(prediction, data1.target.data.to("cpu"))
    loss2 = model._compute_loss(prediction, data1.target.data.to("cpu"))
    assert type(loss) == torch.Tensor
    assert loss >= 0.0 # MSE loss min is zero
    assert loss < 1e6 # We never want to end up to millions (exploding gradient)
    assert loss == loss2 # We have done nothing, loss should be still the same

def test_train_is_vebose(capfd):
    model = Model(predict_n=1, signal_length=100, hidden=16)
    model.train(data1.input, data1.target, verbose=True)
    out, err = capfd.readouterr()
    assert "loss:" in out

    model.train(data1.input, data1.target, verbose=False)
    out, err = capfd.readouterr()
    assert "loss:" not in out

# Test problem is simple and it should be easy to learn.
# Try to learn it, in which case losses should decrease.
# Since train interface function returns nothing, we must check
# convergence by outputting losses and then parsing the loss values.
def test_train_converges(capfd):
    def get_loss_value_from_string(line):
        return float(line.split(':')[1])

    model = Model(predict_n=1, signal_length=100, hidden=64)
    model.train(data1.input, data1.target, verbose=True)
    out, err = capfd.readouterr()
    losses = out.splitlines()[1:] # 1: exclude [INFO] print
    loss_value1 = get_loss_value_from_string(losses[0])
    loss_value2 = get_loss_value_from_string(losses[1])
    loss_value3 = get_loss_value_from_string(losses[2])
    assert loss_value1 > loss_value2
    assert loss_value2 > loss_value3

def test_predict():
    model = Model(predict_n=1, signal_length=100, hidden=16)
    prediction = model.predict(data1.input, data1.target)
    assert prediction.shape == (1, 1, 99)
    assert prediction.dtype == np.float64

def test_model_save_and_load(capfd):
    # Test saving
    model_filename = "./test_save.pt"
    model = Model(predict_n=1, signal_length=100, hidden=16)
    model.save_model(model_filename)
    assert path.isfile(model_filename) == True

    # Test loading
    model = Model(load_path=model_filename, predict_n=1, signal_length=100, hidden=16)
    out, err = capfd.readouterr()
    assert "[INFO] model has been loaded succesfully!" in out
    remove(model_filename) # clean up

# TODO: tests for multiple signal cases
