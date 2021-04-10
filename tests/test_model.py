import pytest
import torch
import numpy as np
from . import config as cfg
from failnet.model import Model, Sequence, Batch

# Some test data
# TODO: put all this inside a class
n = 1
signal_length = 100
input_data_n1 = Batch(1, 1, 100-n)
target_data_n1 = Batch(1, 1, 100-n)
signal = np.arange(signal_length) # from 0 to 100
input_data_n1[0, 0] = signal[:-n]
target_data_n1[0, 0] = signal[n:]

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
    prediction = model._get_prediction(input_data_n1.data.to("cpu"))
    assert type(prediction) == torch.Tensor
    assert list(prediction.shape) == [1, 1, 99]

def test_loss_computation():
    model = Model(predict_n=1, signal_length=100, hidden=16)
    prediction = model._get_prediction(input_data_n1.data.to("cpu"))
    loss = model._compute_loss(prediction, target_data_n1.data.to("cpu"))
    print("data.type(): ", loss.data.type())
    assert type(loss) == torch.Tensor
    assert loss >= 0.0 # MSE loss min is zero
    assert loss < 1e6 # We probably never want to end up to millions (exploding gradient)

# TODO: tests for multiple signal cases
