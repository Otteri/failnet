import torch
import pytest
import numpy as np
from failnet.detectors import *

def test_compare_single_value_raises_exception_when_different_array_lengths():
    with pytest.raises(Exception):
        compare_single_value(np.array([1, 2, 3]), np.array([0, 0]), epsilon=0)

def test_compare_single_value_raises_failure_when_error_greater_than_epsilon():
    assert compare_single_value(np.array([1.0]), np.array([2.0]), epsilon=1.1) == False
    assert compare_single_value(np.array([1.0]), np.array([2.0]), epsilon=1.0) == False
    assert compare_single_value(np.array([1.0]), np.array([2.0]), epsilon=0.9) == True
    assert compare_single_value(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), epsilon=0.1) == False
    assert compare_single_value(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]), epsilon=0.5) == True
    assert compare_single_value(np.array([1.0]), np.array([-2.0]), epsilon=4.0) == False
    assert compare_single_value(np.array([-1.0]), np.array([2.0]), epsilon=2.99) == True
    assert compare_single_value(np.array([-1.0]), np.array([-2.0]), epsilon=0.99) == True
    assert compare_single_value(np.array([1.0, 999, 1.0]), np.array([1.0, 1.0, -999]), epsilon=100) == True
