import torch
import pytest
import numpy as np
from os import path, remove
from failnet.visualization import *

# Use different backend, so CI-pipeline doesn't get stuck
matplotlib.use("Agg")

def test_plot_initialization():
    # There is not much we can test here
    # Let's check that program doesn't crash
    assert init_plot() == True

def test_draw_signal_with_default_values():
    s = np.linspace(10,  100, num=10)
    assert draw_signal(s) == True

def test_draw_signal_with_values():
    s = np.linspace(10,  100, num=10)
    assert draw_signal(s, color="r") == True
    assert draw_signal(s, color='blue', linestyle='dashed', label="test") == True

def test_creates_plot_file():
    s1 = np.linspace(0.0, 1.0, num=10)
    s2 = np.linspace(1,   10,  num=10)
    s3 = np.linspace(10,  100, num=10)

    filename = "test_plot.svg"
    create_plot(filename, s1, s2, s3)
    assert path.isfile(filename) == True
    remove(filename) # clean up

def test_creates_plot_file_raises_exception_with_different_signal_lengths():
    s1 = np.linspace(0.0, 1.0, num=3)
    s2 = np.linspace(10,  100, num=999)

    with pytest.raises(Exception):
        create_plot(filename, s2, s1, s1)
    with pytest.raises(Exception):
        create_plot(filename, s1, s2, s1)
    with pytest.raises(Exception):
        create_plot(filename, s2, s2, s1)
