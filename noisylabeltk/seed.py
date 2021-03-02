# Set a seed value
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def set_seed(seed_value):

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
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
    # 5. For layers that introduce randomness like dropout, make sure to set seed values
    #model.add(Dropout(0.25, seed=seed_value))
    #6 Configure a new global `tensorflow` session

    #session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    #sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    #K.set_session(sess)

def get_seed():
    return int(os.environ['PYTHONHASHSEED'])

