from noisylabeltk.seed import create_seed, set_seed, get_seed

set_seed(321)

from noisylabeltk.experiment import ExperimentBundle
import contextlib
import joblib
from joblib import Parallel, delayed
import os
from tqdm import tqdm

project_name = 'ygorcanalli/LabelNoise'
#project_name = 'ygorcanalli/sandbox'
tags = ["semdropout"]
n_jobs = 8
device = 'cpu'

if device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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

def main():
    hyperparameters_trials = 10
    hyperparameters_epochs = 10
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

    batch_size_list = [128]
    epochs_list = [20]
    dataset_list = ['synthetic-50-2-100K']
    model_list = ['simple-mlp']

    # transition_matrix_list = []
    #
    # transition_matrix_list.append([[0.9, 0.1],
    #                                [0.1, 0.9]])
    # transition_matrix_list.append([[0.8, 0.2],
    #                                [0.1, 0.9]])
    # transition_matrix_list.append([[0.9, 0.1],
    #                                [0.2, 0.8]])
    # transition_matrix_list.append([[0.8, 0.2],
    #                                [0.2, 0.8]])
    # transition_matrix_list.append([[0.7, 0.3],
    #                                [0.2, 0.8]])
    # transition_matrix_list.append([[0.8, 0.2],
    #                                [0.3, 0.7]])
    # transition_matrix_list.append([[0.7, 0.3],
    #                                [0.3, 0.7]])

    robust_methods = ['none', 'boot-soft', 'boot-hard', 'forward', 'backward']

    # for transition_matrix in transition_matrix_list:
    #     robust_method_list.append(('forward', ([transition_matrix], {})))
    #     robust_method_list.append(('backward', ([transition_matrix], {})))

    #noise_list = [
    #    # (noise, noise-args)
    #    ('none', [])
    #]
    noise_list = []
    noise_rate_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for noise_rate in noise_rate_list:
        noise_list.append(('uniform', [noise_rate]))

    # for transition_matrix in transition_matrix_list:
    #    noise_list.append( ('pairwise', [transition_matrix]) )

    parameters_list = []
    for batch_size in batch_size_list:
        for epochs in epochs_list:
            for dataset in dataset_list:
                for model in model_list:
                    for noise, noise_args in noise_list:

                        robust_methods_list = []
                        for robust_method_name in robust_methods:
                            robust_method = {'name': robust_method_name, 'kwargs': None, 'args': None}
                            if robust_method_name == 'forward' or robust_method_name == 'backward':
                                noise_rate = noise_args[0]
                                transition_matrix = [[1-noise_rate, noise_rate],
                                                     [noise_rate, 1-noise_rate]]
                                robust_method['args'] = [transition_matrix]
                            robust_methods_list.append(robust_method)

                        exp_bundle = ExperimentBundle(dataset, model, batch_size, epochs, \
                                                      robust_methods_list, \
                                                      project_name, noise, noise_args, \
                                                      hyperparameters_range, hyperparameters_trials, \
                                                      hyperparameters_epochs)
                        exp_bundle.run_bundle()

"""
    num_experiments = len(parameters_list)

    with tqdm_joblib(tqdm(desc="%s progress" % project_name, total=num_experiments)) as progress_bar:
        Parallel(n_jobs=n_jobs)(delayed(create_and_run)(parameters) for parameters in parameters_list)
"""
if __name__ == "__main__":
    main()
