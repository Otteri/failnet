import os
import sys
import gym
import pytest
import pulsegen
from pathlib import Path
from failnet.train import train, get_data_batch

class MockArgs(object):
    def __init__(self):
        self.config_path = None
        self.tensorboard = False
        self.steps = 10
        self.show_input = False
        self.make_plots = False

# Use this in tests, it has correct config_path
args = MockArgs()

# Adds tests/config.py to path, so test config will be loaded
# when calling `import config` and using `config_path`
@pytest.fixture
def config_path():
    config_dir = os.path.dirname(os.path.realpath("tests/config.py"))
    sys.path.append(config_dir)
    config_path = os.path.abspath("tests/config.py")
    args.config_path = config_path
    return config_path

def test_periodicalsignal_env_generates_data(config_path):
    import config as cfg
    env = gym.make("PeriodicalSignal-v0", config_path=config_path)  
    train_input, train_target = get_data_batch(env, cfg, show_input=False)
    assert train_input.shape == (cfg.repetitions, 1, cfg.signal_length-cfg.predict_n)
    assert train_input.shape == (cfg.repetitions, 1, cfg.signal_length-cfg.predict_n)

def test_train_loop():
    import config as cfg
    Path(cfg.data_dir).mkdir(exist_ok=True)
    assert train(args, cfg) == True
