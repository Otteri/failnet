import os
import sys
import gym
import pytest
import pulsegen
from failnet.train import *

# Adds tests/config.py to path, so test config will be loaded
# when calling `import config` and using `config_path`
@pytest.fixture
def config_path():
    config_dir = os.path.dirname(os.path.realpath("tests/config.py"))
    sys.path.append(config_dir)
    config_path = os.path.abspath("tests/config.py")
    return config_path

def test_periodicalsignal_env_generates_data(config_path):
    import config as cfg
    env = gym.make("PeriodicalSignal-v0", config_path=config_path)  
    train_input, train_target = get_data_batch(env, cfg, show_input=False)
    assert train_input.shape == (cfg.repetitions, 1, cfg.signal_length-cfg.predict_n)
    assert train_input.shape == (cfg.repetitions, 1, cfg.signal_length-cfg.predict_n)
