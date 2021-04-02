#from noisylabeltk.seed import ensure_seterministic, get_seed
from tensorflow.keras.losses import categorical_crossentropy
from util import tqdm_joblib

from joblib import Parallel, delayed
from tqdm import tqdm
from noisylabeltk.datasets import DatasetLoader
from noisylabeltk.loss import make_loss
import noisylabeltk.models as models
import neptune.new as neptune
import optuna
from optuna.samplers import TPESampler
from neptune.new.integrations.tensorflow_keras import NeptuneCallback as NeptuneKerasCallback
from neptunecontrib.monitoring.optuna import NeptuneCallback as NeptuneOptunaCallback

class ExperimentBundle(object):

    def __init__(self, dataset, model, batch_size, epochs, \
                 robust_method_list, project_name, \
                 noise, noise_args, \
                 hyperparameters_range, num_trials, trial_epochs):

        self.dataset_name = dataset
        self.model_name = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.noise_name = noise
        self.noise_args = noise_args
        self.robust_method_list = robust_method_list
        self.project_name = project_name
        self.hyperparameters_range = hyperparameters_range
        self.num_trials = num_trials
        self.trial_epochs = trial_epochs
        self.train = None
        self.validation = None
        self.test = None
        self.num_features = None
        self.num_classes = None
        self.best_hyperparameters = None
        self.experiment_name = "%s-%s-%s" % (self.dataset_name, self.noise_name, str(self.noise_args))
    # TODO track hyperparameters tunning
    def _tune(self):

        def objective(trial):
            num_layers = trial.suggest_int("num_layers", self.hyperparameters_range['num-layers']['min'],\
                                           self.hyperparameters_range['num-layers']['max'])

            for i in range(num_layers):
                trial.suggest_int("hidden_size_{}".format(i), self.hyperparameters_range['hidden-size']['min'], \
                                  self.hyperparameters_range['hidden-size']['max'], log=True)

            trial.suggest_float("dropout", self.hyperparameters_range['dropout']['min'], \
                                self.hyperparameters_range['dropout']['max'])
            kwargs = trial.params
            model = models.create_model(self.model_name, self.num_features, self.num_classes, **kwargs)

            model.compile(optimizer='adam',
                               loss=categorical_crossentropy,
                               metrics=['accuracy'])

            history = model.fit(self.train, epochs=self.trial_epochs, verbose=0)

            eval_metrics = model.evaluate(self.validation, verbose=0)

            for j, metric in enumerate(eval_metrics):
                if model.metrics_names[j] == 'accuracy':
                    return metric

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.num_trials)

        self.best_hyperparameters = study.best_trial.params

    def _load_data(self):

        dataset_loader = DatasetLoader(self.dataset_name, self.batch_size)

        if self.noise_name is not None and self.noise_name != 'none' and self.noise_args is not None:
            (train_ds, validation_ds, test_ds), num_features, num_classes = dataset_loader.pollute_and_load(self.noise_name, *(self.noise_args))
        else:
            (train_ds, validation_ds, test_ds), num_features, num_classes = dataset_loader.load()

        self.train = train_ds
        self.validation = validation_ds
        self.test = test_ds
        self.num_features = num_features
        self.num_classes = num_classes

    def _run(self, robust_method, loss_args, loss_kwargs):

        parameters = {
            'batch-size': self.batch_size,
            'epochs': self.epochs,
            'dataset': self.dataset_name,
            'model': self.model_name,
            'noise': self.noise_name,
            'noise-args': self.noise_args,
            'robust-method': robust_method,
            'loss-args': loss_args,
            'loss-kwargs': loss_kwargs,
        }

        exp = Experiment(self.num_features, self.num_classes, parameters, self.project_name, self.experiment_name)
        exp.build_model(self.best_hyperparameters)
        exp.fit_model(self.train, self.validation)
        exp.evaluate(self.test)

    def run_bundle(self):
        self._load_data()
        self._tune()
        for robust_method in self.robust_method_list:
            self._run(robust_method['name'], robust_method['args'], robust_method['kwargs'])

class Experiment(object):

    def __init__(self, num_features, num_classes, parameters, project_name, experiment_name):

        self.parameters = parameters
        self.project_name = project_name
        self.name = experiment_name

        if 'noise_args' in self.parameters and self.parameters is not None:
            for i, arg in enumerate(self.parameters['noise-args']):
                self.run['parameters/noise_arg_%d' % i] = arg

        if 'loss_args' in self.parameters and self.parameters is not None:
            for i, arg in enumerate(self.parameters['loss-args']):
                self.run['parameters/loss_arg_%d' % i] = arg

        if 'loss_kwargs' in self.parameters and self.parameters is not None:
            for key, value in enumerate(self.parameters['loss-kwargs']):
                self.run['parameters/loss_arg_%d' % key] = value

        if 'robust-method' in self.parameters and self.parameters['robust-method'] is not None and \
                self.parameters['robust-method'] != 'none':
            args = []
            kwargs = {}
            if 'loss-args' in self.parameters and self.parameters['loss-args'] is not None:
                args = self.parameters['loss-args']
            if 'loss-kwargs' in self.parameters and self.parameters['loss-kwargs'] is not None:
                kwargs = self.parameters['loss-kwargs']

            self.loss_function = make_loss(self.parameters['robust-method'], *args, **kwargs)
        else:
            self.loss_function = make_loss('cross-entropy')

        self.num_features = num_features
        self.num_classes = num_classes
        self.run = None
        self.model = None

    def build_model(self, hyperparameters):

        if self.run is None:
            self._init_tracking()

        self.run['parameters/hyperparameters'] = hyperparameters

        self.model = models.create_model(self.parameters['model'], self.num_features, self.num_classes, **hyperparameters)
        self.model.compile(optimizer='adam',
                           loss=self.loss_function,
                           metrics=['accuracy'])

        self.model.summary(print_fn=lambda x: self.run['model_summary'].log(x))

    def fit_model(self, train, validation):

        neptune_cbk = NeptuneKerasCallback(run=self.run, base_namespace='metrics')
        history = self.model.fit(train, epochs=10,
                                 validation_data=validation,
                                 callbacks=[neptune_cbk],
                                 verbose=0)

    def evaluate(self, test):
        eval_metrics = self.model.evaluate(test, verbose=0)

        for j, metric in enumerate(eval_metrics):
            self.run['metrics/eval_' + self.model.metrics_names[j]].log(metric)

        self._stop_tracking()

    def _init_tracking(self):
        self.run = neptune.init(project=self.project_name, name=self.name)
        self.run["model/params"] = self.parameters
        self.run['parameters/num-features'] = self.num_features
        self.run['parameters/num-classes'] = self.num_classes

    def _stop_tracking(self):
        self.run.stop()
