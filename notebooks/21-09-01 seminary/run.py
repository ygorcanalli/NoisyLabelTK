#%%

import numpy as np
from noisylabeltk.experiment import Experiment
from noisylabeltk.datasets import DatasetLoader
from joblib import Parallel, delayed

import os

host = 'localhost'
port = 27017
database_name = 'fairness'
tags = ['verify inversion']
collection_name = '21_09_01_seminary'
n_jobs = 8
device = 'cpu'

if device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_p_list(fp_male, fn_male, fp_female, fn_female):
    P_list = [np.array([[1 - fp_male, fp_male],
                        [fn_male, 1 - fn_male]]),
              np.array([[1 - fp_female, fp_female],
                        [fn_female, 1 - fn_female]])]
    return P_list

def run(parameters, hyperparameters):

    dataset = DatasetLoader(parameters['dataset'], parameters['batch-size'], sensitive_labels=True).load()

    exp = Experiment(dataset['num_features'],
                     dataset['num_classes'],
                     parameters,
                     database_name,
                     collection_name,
                     tags=tags)

    exp.build_model(hyperparameters)
    exp.fit_model(dataset['train'], dataset['validation'], parameters['batch-size'])
    exp.evaluate(dataset['test'], parameters['batch-size'])
    exp.evaluate_discrimination(dataset['test'], parameters['batch-size'])
    exp.persist_metadata()


hyperparameters = {
    'num_layers': 3,
    'dropout': 0.2
}

for i, size in enumerate([32, 64, 32]):
    hyperparameters['hidden_size_%d' % i] = size

datasets = ['income', 'german']
robust_methods = ['fair_forward']
repeat = 1

parameters_list = []
for i in range(repeat):
    for ds in datasets:
        for robust_method in robust_methods:
            for unprotected_demotion in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                for protected_promotion in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    unprotected_promotion = unprotected_demotion / 10
                    protected_demotion = protected_promotion / 10
                    P_list = get_p_list(unprotected_demotion, unprotected_promotion, protected_demotion,
                                        protected_promotion)

                    parameters = {
                        'batch-size': 32,
                        'epochs': 10,
                        'dataset': ds,
                        'model': 'simple-mlp',
                        'noise': None,
                        'noise-args': None,
                        'robust-method': 'fair-forward',
                        'loss-args': [P_list],
                        'protected-promotion': protected_promotion,
                        'protected-demotion': protected_demotion,
                        'unprotected-promotion': unprotected_promotion,
                        'unprotected-demotion': unprotected_demotion,
                        'loss-kwargs': None,
                    }
                    parameters_list.append(parameters)
                    #run(parameters, hyperparameters)


Parallel(n_jobs=n_jobs)(delayed(run)(parameters, hyperparameters) for parameters in parameters_list)
