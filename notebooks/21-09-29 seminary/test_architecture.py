

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


database_name = 'fairness'
device = 'cpu'

if device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


dataset_name = 'income'
robust_method = 'fair-forward'
target_metric = 'Positives'

hyperparameters = {
    'num_layers': 2,
    'dropout': 0.2
}

for i, size in enumerate([32, 64, 32]):
    hyperparameters['hidden_size_%d' % i] = size

#T = {"r_priv": 0.74, "p_priv": 0.18, "r_prot": 0.4, "p_prot": 0.38}
T = {"r_priv": 0, "p_priv": 0, "r_prot": 0, "p_prot": 0}
p_list = get_p_list(T['r_priv'], T['p_priv'], T['r_prot'], T['p_prot'])

parameters = {
    'batch-size': 32,
    'patience': 30,
    'epochs': 500,
    'dataset': dataset_name,
    'model': 'simple-mlp',
    'noise': None,
    'noise-args': None,
    'robust-method': 'fair-forward',
    'loss-args': [p_list],
    'protected-promotion': T['p_priv'],
    'protected-demotion': T['r_priv'],
    'privileged-promotion': T['p_prot'],
    'privileged-demotion': T['r_prot'],
    'loss-kwargs': None,
}

dataset = DatasetLoader(parameters['dataset'], parameters['batch-size'], sensitive_labels=True).load()

exp = Experiment(dataset['num_features'],
                 dataset['num_classes'],
                 parameters,
                 database_name)

exp.build_model(hyperparameters)
exp.fit_model(dataset['train'], verbose=1)
exp.evaluate(dataset['test'], verbose=1)
exp.evaluate_discrimination(dataset['test'])


auc = float(exp.run_entry['metrics']['test_AUC_overall'])
print(auc)
asd = abs(exp.run_entry['metrics']['test_Positives_balance'])
print(asd)
#%%


