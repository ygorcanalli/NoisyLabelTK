from noisylabeltk.experiment import ExperimentBundle

import os

project_name = 'ygorcanalli/LabelNoise'
n_jobs = 8
device = 'gpu'

if device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
    dataset_list = ['income']
    model_list = ['simple-mlp']

    robust_methods = ['none', 'boot-soft', 'boot-hard', 'forward', 'backward']

    noise_list = []
    noise_rate_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for noise_rate in noise_rate_list:
        noise_list.append(('uniform', [noise_rate]))

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

if __name__ == "__main__":
    main()
