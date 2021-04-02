import numpy as np
import config as cfg
import onnxruntime as ort
from detectors import compareSingleValue
from visualization import plot_signal

# Some test input data
n = cfg.predict_n
dummy_data = np.zeros((1, 1, cfg.signal_length))
dummy_input = dummy_data[..., :-n]

# Create NN
ort_session = ort.InferenceSession('failnet.onnx')

# Run through NN
model_predictions = ort_session.run(None, {'input_batch': dummy_input})

# Visualize result
model_predictions = np.array(model_predictions) # convert to numpy
prediction = model_predictions[0, 0, :]
plot_signal(prediction)
