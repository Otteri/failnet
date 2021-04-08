import numpy as np
import config as cfg
import onnxruntime as ort
from detectors import compare_single_value
from visualization import draw_signal, init_plot
# Env related imports
import gym
import pulsegen
# Plotting
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# Some test data that the model has been trying to learn
env = gym.make("PeriodicalSignal-v0", config_path="config.py")
n = cfg.predict_n
input_data = np.zeros((1, 1, cfg.signal_length-n))
target_data = np.zeros((1, 1, cfg.signal_length-n))
for i in range(0, 1):
    signal = env.record_rotation()
    input_data[i, 0] = signal[:-n]
    target_data[i, 0] = signal[n:]

# Create NN
ort_session = ort.InferenceSession('failnet.onnx')

# Run through NN
model_predictions = ort_session.run(None, {'input_batch': input_data})

# Visualize result
model_predictions = np.array(model_predictions) # convert to numpy
prediction = model_predictions[0, 0]
actual = target_data[0, 0, :]

init_plot()
draw_signal(prediction, color='r', label='prediction')
draw_signal(actual, color='b', label='actual')
plt.legend()
plt.show(block = True)

# Check if failures
is_failure = compare_single_value(actual, prediction, epsilon=2.0)
print("Failure(s): ", is_failure)
