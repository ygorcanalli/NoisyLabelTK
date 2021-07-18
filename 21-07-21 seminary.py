#%%

import numpy as np
from noisylabeltk.experiment import Experiment
from noisylabeltk.datasets import DatasetLoader
from noisylabeltk.metrics import evaluate_discrimination
import neptune.new as neptune


import os

#%%

project_name = 'PESC-ATIS/FairnessSeminary-21-07-21'
experiment_name = 'arbitrary_correction'
n_jobs = 1
device = 'cpu'

if device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#%%

def run(parameters):
    dataset_loader = DatasetLoader(parameters['dataset'], parameters['batch-size'])

    (train_ds, validation_ds, test_ds), num_features, num_classes = dataset_loader.load()

    exp = Experiment(num_features,
                     num_classes,
                     parameters,
                     project_name,
                     experiment_name)


    exp.build_model(hyperparameters)
    exp.fit_model(train_ds, validation_ds)


    exp.evaluate(test_ds)
    disc = evaluate_discrimination(dataset_loader, exp.model)
    exp.neptune_run['metrics/discrimination'].log(disc)
    exp.stop_tracking()

#%%

hyperparameters = {
    'num_layers': 3,
    'dropout': 0.2
}
for i, size in enumerate([32, 64, 32]):
    hyperparameters['hidden_size_%d' % i] = size

datasets = ['income']
robust_methods = ['fair_forward']
repeat = 1

#%%

fp_male = .2
fn_male = 0.05
fp_female = 0.01
fn_female = .25
P_list = [np.array([[1 - fp_male, fp_male],
                    [fn_male, 1 - fn_male]]),
          np.array([[1 - fp_female, fp_female],
                    [fn_female, 1 - fn_female]])]

#%%

for dataset in datasets:
    for robust_method in robust_methods:
        parameters = {
            'batch-size': 32,
            'epochs': 10,
            'dataset': dataset,
            'model': 'simple-mlp',
            'noise': None,
            'noise-args': None,
            'robust-method': robust_method,
            'loss-args': [P_list],
            'loss-kwargs': None,
        }

        for i in range(repeat):
            run(parameters)

#%%

project = neptune.get_project(
    name=project_name,
    api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ZGUxY2IwMy1kOTMzLTRjMTUtYjAxYy01MWE2MmMyYzQ0ZmYifQ==')

#%%

run_table_df = project.fetch_runs_table().to_pandas()

run_table_df.head()

#%%

run_table_df.rename(columns={'metrics/eval_accuracy': 'accuracy',
                     'model/params/dataset': 'dataset',
                     'model/params/robust-method': 'robust-method',
                     'metrics/eval_accuracy': 'accuracy',
                     'metrics/discrimination': 'discrimination',
                     'metrics/epoch/val_MatthewsCorrelationCoefficient': 'mcc'
                    }, inplace=True)

run_table_df.columns

#%%

run_table_df['acc/disc'] = run_table_df['accuracy']/run_table_df['discrimination']
run_table_df['mcc/disc'] = run_table_df['mcc']/run_table_df['discrimination']


group_criteria = ['dataset', 'robust-method']
agg_criteria = ({'accuracy': ['mean', 'std'],
                 'discrimination': ['mean', 'std'],
                 'mcc': ['mean', 'std'],
                 'acc/disc': ['mean'],
                 'mcc/disc': ['mean']})

results = run_table_df.groupby(group_criteria).agg(agg_criteria)
results

#%%

comparsion_path     = '../FairML/experiments/income/results/comparsion.png'

from IPython.display import Image
Image(filename=comparsion_path)

# results to income dataset

#%%

