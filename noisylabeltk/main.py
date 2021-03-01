from noisylabeltk.experiment import Experiment
import contextlib
import joblib
from joblib import Parallel, delayed
import os
from tqdm import tqdm

#project_name = 'ygorcanalli/LabelNoiseOnStructuredData'
project_name = 'ygorcanalli/sandbox'
tags = ['new_run']
n_jobs = 8
device = 'cpu'

if device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf


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

def create_and_run(parameters):
    exp = Experiment(parameters, project_name, tags)
    exp.run()

hyperparameters_range = {
    'num-layers': {
        'min': 1,
        'max': 3
    },
    'hidden-size': {
        'min': 8,
        'max': 128
    },
    'dropout': {
        'min': 0,
        'max': 0.2
    }
}

batch_size_list = [32]
epochs_list = [10]
dataset_list = ['german', 'breast-cancer']
model_list = ['simple-mlp']

transition_matrix_list = []

transition_matrix_list.append([[0.9, 0.1],
                               [0.1, 0.9]])
transition_matrix_list.append([[0.8, 0.2],
                               [0.1, 0.9]])
transition_matrix_list.append([[0.9, 0.1],
                               [0.2, 0.8]])
transition_matrix_list.append([[0.8, 0.2],
                               [0.2, 0.8]])
transition_matrix_list.append([[0.7, 0.3],
                               [0.2, 0.8]])
transition_matrix_list.append([[0.8, 0.2],
                               [0.3, 0.7]])
transition_matrix_list.append([[0.7, 0.3],
                               [0.3, 0.7]])

robust_method_list = [
    # (robust-method,  (loss-args, loss-kwargs))
    ('none', ([], {})),
    ('boot-soft', ([], {})),
    ('boot-hard', ([], {}))
]

for transition_matrix in transition_matrix_list:
    robust_method_list.append( ('forward', ( [transition_matrix] , {})) )
    robust_method_list.append( ('backward', ( [transition_matrix] , {})) )

noise_list = [
    # (noise, noise-args)
    ('none', [])
]

noise_rate_list = [0.1, 0.2, 0.3]
for noise_rate in noise_rate_list:
    noise_list.append( ('uniform', [noise_rate]) )

for transition_matrix in transition_matrix_list:
    noise_list.append( ('pairwise', [transition_matrix]) )

parameters_list = []
for batch_size in batch_size_list:
    for epochs in epochs_list:
        for dataset in dataset_list:
            for model in model_list:
                for noise, noise_args in noise_list:
                    for robust_method, (loss_args, loss_kwargs) in robust_method_list:
                        parameters = {
                            'batch_size': batch_size,
                            'epochs': epochs,
                            'dataset': dataset,
                            'model': model,
                            'noise': noise,
                            'noise-args': noise_args,
                            'robust-method': robust_method,
                            'loss-args': loss_args,
                            'loss-kwargs': loss_kwargs,
                            'hyperparameters-range': hyperparameters_range
                        }
                        parameters_list.append(parameters)

num_experiments = len(parameters_list)

with tqdm_joblib(tqdm(desc="%s progress" % project_name, total=num_experiments)) as progress_bar:
    Parallel(n_jobs=n_jobs)(delayed(create_and_run)(parameters) for parameters in parameters_list)
