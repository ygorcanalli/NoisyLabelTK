# Set a seed value
import os
import random
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import backend as K

def set_seed(seed_value):

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
    time.sleep(1)
    ensure_seterministic()

def ensure_seterministic():
    seed_value = int(os.environ['PYTHONHASHSEED'])
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # 2. Set `python` built-in pseudo-random generator at a fixed value

    random.seed(seed_value)
    # 3. Set `numpy` pseudo-random generator at a fixed value

    np.random.seed(seed_value)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value

    tf.random.set_seed(seed_value)

def get_seed():
    return int(os.environ['PYTHONHASHSEED'])

def create_seed():
    return np.random.randint(0, 1000)

