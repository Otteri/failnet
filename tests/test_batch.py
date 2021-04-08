import pytest
from failnet.model import Batch 

# Batch class tests

def test_batch_creation_with_args():
    batch = Batch(10, 5, 10)
    assert batch.data.ndim == 3

def test_batch_creation_withithout_args():
    batch = Batch()

def test_negative_batch_number():
    with pytest.raises(AssertionError):
        batch = Batch(-2, 5, 10)

def test_negative_signal_number():
    with pytest.raises(AssertionError):
        batch = Batch(99, -99, 10)

def test_negative_signal_length():
    with pytest.raises(AssertionError):
        batch2 = Batch(1, 50, -99)

def test_batch_assign():
    batch1 = Batch()
    batch2 = Batch(9, 99, 999)
    assert batch1.shape == (0, 0, 0)
    batch1 = batch2
    assert batch1.shape == (9, 99, 999)
