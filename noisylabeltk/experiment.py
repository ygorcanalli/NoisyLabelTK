from noisylabeltk.seed import ensure_seterministic, get_seed

ensure_seterministic()

import sys
import traceback
from noisylabeltk.datasets import DatasetLoader
from noisylabeltk.loss import make_loss
import noisylabeltk.models as models
import neptune.new as neptune
import optuna
from optuna.samplers import TPESampler
from neptune.new.integrations.tensorflow_keras import NeptuneCallback as NeptuneKerasCallback
from neptunecontrib.monitoring.optuna import NeptuneCallback as NeptuneOptunaCallback

optuna.logging.set_verbosity(optuna.logging.WARN)

class Experiment(object):

    def __init__(self, parameters, project_name, tags):

        self.parameters = parameters
        self.name = "%s-%s-%s" % ( self.parameters['dataset'], self.parameters['noise'], self.parameters['robust-method'])
        self.tags = tags

        self.run = neptune.init(project=project_name, name=self.name)
        self.run["model/params"] = parameters

        if 'noise_args' in self.parameters and self.parameters is not None:
            for i, arg in enumerate(self.parameters['noise_args']):
                self.run['parameters/noise_arg_%d' % i] = arg

        if 'loss_args' in self.parameters and self.parameters is not None:
            for i, arg in enumerate(self.parameters['loss_args']):
                self.run['parameters/loss_arg_%d' % i] = arg

        if 'loss_kwargs' in self.parameters and self.parameters is not None:
            for key, value in enumerate(self.parameters['loss_kwargs']):
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

        self.train = None
        self.validation = None
        self.test = None
        self.num_features = None
        self.num_classes = None


    def _load_data(self):

        dataset_loader = DatasetLoader(self.parameters['dataset'], self.parameters['batch-size'])

        if 'noise' in self.parameters and self.parameters['noise'] is not None and \
                self.parameters['noise'] != 'none':
            args = []
            if 'noise-args' in self.parameters and self.parameters['noise-args'] is not None:
                args = self.parameters['noise-args']

            (train_ds, validation_ds, test_ds), num_features, num_classes = dataset_loader.pollute_and_load(self.parameters['noise'], *args)
        else:
            (train_ds, validation_ds, test_ds), num_features, num_classes = dataset_loader.load()

        self.train = train_ds
        self.validation = validation_ds
        self.test = test_ds
        self.num_features = num_features
        self.num_classes = num_classes

        self.run['parameters/num-features'] = num_features
        self.run['parameters/num-classes'] = num_classes

    def _build_model(self):

        self.model = models.create_model(self.parameters['model'], self.num_features, self.num_classes, **(self.best_hyperparameters))
        self.model.compile(optimizer='adam',
                           loss=self.loss_function,
                           metrics=['accuracy'])

        self.model.summary(print_fn=lambda x: self.run['model_summary'].log(x))

    def _fit_model(self):
        neptune_cbk = NeptuneKerasCallback(run=self.run, base_namespace='metrics')
        history = self.model.fit(self.train, epochs=10,
                                 validation_data=self.validation,
                                 callbacks=[neptune_cbk],
                                 verbose=1)

    def _evaluate(self):
        eval_metrics = self.model.evaluate(self.test, verbose=0)

        for j, metric in enumerate(eval_metrics):
            self.run['metrics/eval_' + self.model.metrics_names[j]].log(metric)

    def _tune(self):

        def objective(trial):
            num_layers = trial.suggest_int("num_layers", self.parameters['hyperparameters-range']['num-layers']['min'], self.parameters['hyperparameters-range']['num-layers']['max'])

            for i in range(num_layers):
                trial.suggest_int("hidden_size_{}".format(i), self.parameters['hyperparameters-range']['hidden-size']['min'], self.parameters['hyperparameters-range']['hidden-size']['max'], log=True)

            trial.suggest_float("dropout", self.parameters['hyperparameters-range']['dropout']['min'], self.parameters['hyperparameters-range']['dropout']['max'])
            kwargs = trial.params
            model = models.create_model(self.parameters['model'], self.num_features, self.num_classes, **kwargs)

            model.compile(optimizer='adam',
                               loss=self.loss_function,
                               metrics=['accuracy'])

            history = model.fit(self.train, epochs=self.parameters['hyperparameters-range']['num_epochs'], verbose=0)

            eval_metrics = model.evaluate(self.validation, verbose=0)

            for j, metric in enumerate(eval_metrics):
                if model.metrics_names[j] == 'accuracy':
                    return metric

        #monitor = NeptuneOptunaCallback(experiment=self.run)
        sampler = TPESampler(seed=get_seed())  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=self.parameters['hyperparameters-range']['num_trials'])

        self.run['parameters/best-hyperparameters'] = study.best_trial.params
        self.best_hyperparameters = study.best_trial.params

    def execute(self):
        self._load_data()
        self._tune()
        self._build_model()
        self._fit_model()
        self._evaluate()
        self.run.stop()

