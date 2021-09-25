

import numpy as np
from noisylabeltk.experiment import Experiment
from noisylabeltk.datasets import DatasetLoader

#%%
import os

def get_p_list(privileged_demotion, privileged_promotion, protected_demotion,
    protected_promotion):
    P_list = [np.array([[1 - privileged_demotion, privileged_demotion],
                        [privileged_promotion, 1 - privileged_promotion]]),
              np.array([[1 - protected_demotion, protected_demotion],
                        [protected_promotion, 1 - protected_promotion]])]
    return P_list


host = 'localhost'
port = 27017
database_name = 'fairness'
tags = []
collection_name = 'genetic_optimization_debug'
n_jobs = 4
device = 'gpu'

if device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


dataset_name = 'income'
robust_method = 'fair-forward'
target_metric = 'Positives'

hyperparameters = {
    'num_layers': 3,
    'dropout': 0.2
}

for i, size in enumerate([64, 128, 64]):
    hyperparameters['hidden_size_%d' % i] = size

protected_promotion = 0
protected_demotion = 0
privileged_promotion = 0
privileged_demotion = 0
p_list = get_p_list(privileged_demotion, privileged_promotion, protected_demotion,protected_promotion)

parameters = {
    'batch-size': 32,
    'epochs': 10,
    'dataset': dataset_name,
    'model': 'simple-mlp',
    'noise': None,
    'noise-args': None,
    'robust-method': 'fair-forward',
    'loss-args': [p_list],
    'protected-promotion': protected_promotion,
    'protected-demotion': protected_demotion,
    'privileged-promotion': privileged_promotion,
    'privileged-demotion': privileged_demotion,
    'loss-kwargs': None,
}

dataset = DatasetLoader(parameters['dataset'], parameters['batch-size'], sensitive_labels=False).load()

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


auc = float(exp.run_entry['metrics']['AUC_overall'])
print(auc)
#%%


