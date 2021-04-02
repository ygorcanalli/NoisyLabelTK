# Set a seed value
import os
import random
import numpy as np
import time
import tensorflow as tf
import contextlib
import joblib
from joblib import Parallel, delayed
import os
from tqdm import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


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

