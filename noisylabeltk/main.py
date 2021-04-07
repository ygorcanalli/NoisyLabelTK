from noisylabeltk.experiment import ExperimentBundle
from util import tqdm_joblib

from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

import os

project_name = 'ygorcanalli/sandbox'
n_jobs = 4
device = 'cpu'

if device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_uniform_transition_matrix(dataset, rate):
    sizes_map = {
        'german': 2,
        'breast-cancer': 2,
        'diabetes': 2,
        'synthetic-50-2-100K': 2,
        'synthetic-100-5-500K': 5,
        'synthetic-80-5-200K': 5,
    }
    size = sizes_map[dataset]
    T = np.identity(size)
    T = T * (1 - rate)
    T[T == 0] = rate / (size - 1)
    return T

def main():
    hyperparameters_trials = 50
    hyperparameters_epochs = 10
    hyperparameters_range = {
        'num-layers': {
            'min': 1,
            'max': 3
        },
        'hidden-size': {
            'min': 32,
            'max': 128
        },
        'dropout': {
            'min': 0.05,
            'max': 0.2
        },
        'search-space': {
            'num-layers': [1, 2, 3],
            'hidden-size': [32, 64, 128],
            'dropout': [0.05, 0.1, 0.15, 0.2]
        }
    }

    batch_size_list = [32]
    epochs_list = [20]
    dataset_list = ['synthetic-50-2-100K']
    model_list = ['simple-mlp']

    robust_methods = ['none', 'boot-soft', 'boot-hard', 'forward', 'backward']

    noise_list = []
    noise_rate_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30]

    for noise_rate in noise_rate_list:
        noise_list.append(('uniform', [noise_rate]))

    exp_bundles = []

    for dataset in dataset_list:
        #best_hyperparameters = None
        for batch_size in batch_size_list:
            for epochs in epochs_list:
                for model in model_list:
                    for noise, noise_args in noise_list:

                        robust_methods_list = []
                        for robust_method_name in robust_methods:
                            robust_method = {'name': robust_method_name, 'kwargs': None, 'args': None}
                            if robust_method_name == 'forward' or robust_method_name == 'backward':
                                noise_rate = noise_args[0]
                                transition_matrix = get_uniform_transition_matrix(dataset, noise_rate)
                                robust_method['args'] = [transition_matrix]
                            robust_methods_list.append(robust_method)

                        exp_bundle = ExperimentBundle(dataset, model, batch_size, epochs, \
                                                      robust_methods_list, \
                                                      project_name, noise, noise_args, \
                                                      hyperparameters_range, hyperparameters_trials, \
                                                      hyperparameters_epochs)
                        #if best_hyperparameters is None:
                        #    best_hyperparameters = exp_bundle.run_bundle()
                        #else:
                        #    exp_bundle.run_bundle(best_hyperparameters)
                        exp_bundles.append(exp_bundle)
    num_experiments = len(exp_bundles)

    with tqdm_joblib(tqdm(desc="%s progress" % project_name, total=num_experiments)) as progress_bar:
        Parallel(n_jobs=n_jobs)(delayed(exp_bundle.run_bundle)() for exp_bundle in exp_bundles)
if __name__ == "__main__":
    main()
