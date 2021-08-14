import torch
import pytest
import numpy as np
from failnet.filters import filter_signal

# Checks that padding works
def test_signal_filtering_dimensions_stay_same():
    time = np.arange(0, 10, 0.1);
    sine_signal = torch.from_numpy(np.sin(time))
    input_shape = sine_signal.size()
    sine_filtered = filter_signal(sine_signal)
    output_shape = sine_filtered.size()    
    assert input_shape == output_shape

def test_signal_filtering_dimensions_stay_same():
    signal = torch.from_numpy(np.array([1, 2, 3, 4, 5]))
    signal_filtered = filter_signal(signal, n=3)
    assert signal_filtered[0] == 1
    assert signal_filtered[1] == pytest.approx(1.33, 0.1)
    assert signal_filtered[2] == 2
    assert signal_filtered[3] == 3
    assert signal_filtered[4] == 4
