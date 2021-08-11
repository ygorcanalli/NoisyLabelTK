#%%

import numpy as np
from noisylabeltk.experiment import Experiment
from noisylabeltk.datasets import DatasetLoader
from noisylabeltk.metrics import evaluate_discrimination
from joblib import Parallel, delayed


import os

project_name = 'PESC-ATIS/FairnessDebug'
tags = ['creating metrics']
experiment_name = 'arbitrary_correction_batch'
n_jobs = 4
device = 'cpu'

if device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_p_list(fp_male, fn_male, fp_female, fn_female):
    P_list = [np.array([[1 - fp_male, fp_male],
                        [fn_male, 1 - fn_male]]),
              np.array([[1 - fp_female, fp_female],
                        [fn_female, 1 - fn_female]])]
    return P_list

def run(parameters):
    dataset_loader = DatasetLoader(parameters['dataset'], parameters['batch-size'], sensitive_labels=True)

    (train_ds, validation_ds, test_ds), num_features, num_classes = dataset_loader.load()

    exp = Experiment(num_features,
                     num_classes,
                     parameters,
                     project_name,
                     experiment_name,
                     tags=tags)


    exp.build_model(hyperparameters)
    exp.fit_model(train_ds, validation_ds)
    exp.evaluate(test_ds)
    exp.stop_tracking()
    #disc = evaluate_discrimination(dataset_loader, exp.model)
    #print(disc)
    #exp.neptune_run['metrics/discrimination'].log(disc)

hyperparameters = {
    'num_layers': 3,
    'dropout': 0.2
}
for i, size in enumerate([32, 64, 32]):
    hyperparameters['hidden_size_%d' % i] = size

datasets = ['income']
robust_methods = ['fair_forward']
repeat = 1


P_list_options = []
for fp_male in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
    for fn_female in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
        fn_male = fp_male/10
        fp_female = fn_female/10
        P_list = get_p_list(fp_male, fn_male, fp_female, fn_female)
        P_list_options.append(P_list)


#P_list = get_p_list(0.2,0.01,0.01,0.2)
#P_list_options = [P_list]

parameters_list = []
for i in range(repeat):
    for dataset in datasets:
        for robust_method in robust_methods:
            for P_list in P_list_options:
                parameters = {
                    'batch-size': 32,
                    'epochs': 10,
                    'dataset': dataset,
                    'model': 'simple-mlp',
                    'noise': None,
                    'noise-args': None,
                    'robust-method': 'fair-forward',
                    'loss-args': [P_list],
                    'loss-kwargs': None,
                }
                parameters_list.append(parameters)
                run(parameters)

Parallel(n_jobs=n_jobs)(delayed(run)(parameters) for parameters in parameters_list)
