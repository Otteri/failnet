import torch
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

def test_batch_get_item_gives_correct_shape():
    batch = Batch(10, 5, 999)
    data = batch[9, 4]
    assert data.size() == torch.Size([999])

def test_batch_get_item_gives_consistent_dimensions():
    batch = Batch(3, 5, 2)
    data1 = batch[0, 0]
    data2 = batch[1, 4]
    assert data1.size() == torch.Size([2])
    assert data2.size() == torch.Size([2])

def test_batch_assign():
    batch1 = Batch()
    batch2 = Batch(9, 99, 999)
    assert batch1.shape == (0, 0, 0)
    batch1 = batch2
    assert batch1.shape == (9, 99, 999)

def test_batch_iter():
    batches = Batch(10, 2, 10)
    for i, batch in enumerate(batches):
        assert batch.shape == (2, 10)
    assert i+1 == 10 # Did we go through 10 batches?