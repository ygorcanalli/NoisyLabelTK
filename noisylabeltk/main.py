from noisylabeltk.seed import create_seed, set_seed, get_seed

set_seed(321)

from noisylabeltk.experiment import Experiment
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


def create_and_run(parameters):
    exp = Experiment(parameters, project_name, tags)
    exp.execute()

def main():
    hyperparameters_range = {
        'num_trials': 10,
        'num_epochs': 10,
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
    epochs_list = [10]
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

    robust_method_list = [
        # (robust-method,  (loss-args, loss-kwargs))
        ('none', ([], {})),
        ('boot-soft', ([], {})),
        ('boot-hard', ([], {})),
        ('forward', ([], {})),
        ('backward', ([], {}))
    ]

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
                        for robust_method, (loss_args, loss_kwargs) in robust_method_list:
                            if (robust_method == 'forward' or robust_method == 'backward') and not loss_args:
                                if noise == 'pairwise':
                                    loss_args = noise_args
                                elif noise == 'uniform':
                                    noise_rate = noise_args[0]
                                    transition_matrix = [[1-noise_rate, noise_rate],
                                                         [noise_rate, 1-noise_rate]]
                                    loss_args = [transition_matrix]

                            parameters = {
                                'batch-size': batch_size,
                                'epochs': epochs,
                                'dataset': dataset,
                                'model': model,
                                'noise': noise,
                                'noise-args': noise_args,
                                'robust-method': robust_method,
                                'loss-args': loss_args,
                                'loss-kwargs': loss_kwargs,
                                'hyperparameters-range': hyperparameters_range,
                                'seed': get_seed()
                            }
                            parameters_list.append(parameters)

    num_experiments = len(parameters_list)

    with tqdm_joblib(tqdm(desc="%s progress" % project_name, total=num_experiments)) as progress_bar:
        Parallel(n_jobs=n_jobs)(delayed(create_and_run)(parameters) for parameters in parameters_list)

if __name__ == "__main__":
    main()
